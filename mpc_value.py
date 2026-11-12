"""Model predictive control with a learned terminal value for SustainDC.

Each step the controller plans the next H steps with the known models and the forecasts available at that time, then
charges the state reached at the end of the horizon with a terminal value V(s_H), the CO2 still to come from that state:

    min_{a_0..a_{H-1}}  sum_{k<H} CO2_k(s_k, a_k)  +  V(s_H)        (receding horizon: only a_0 is applied)

Without V the plan is myopic: work deferred past the horizon looks free and energy stored in the battery looks
worthless. V is learned by approximate policy iteration (train_mpc_value.py): run the MPC, fit V to its n-step
average-cost returns, run the MPC with the new V, and so on (see "Value learning" below).

Subsystems (the CO2 is additive and the subsystems are almost decoupled, so each has its own planner and terminal value):

* load shifting (ls): the deferred-task queue is simulated exactly (vectorized over candidate plans, same rules as
  envs/carbon_ls.py); the plan is searched with the cross-entropy method (CEM). V_ls(queue, context).
* battery (bat): exact dynamic programming over the stored energy (same transition as BatOracle). V_bat(usable stored energy, context).
* cooling (dc): holds a fixed setpoint (the tuned rule-based setpoint, 15 C by default); the energy model of the plan uses
  that setpoint.

Costs are in kg CO2. The energy model is IT + HVAC energy as a quadratic in CPU utilization, fitted each step from
baseline_policies.dc_power at the current weather and the cooling setpoint.

On-site PV (pv_config.capacity_kWp > 0). The mode is read from the env (pv_setup):
* automatic rule (include_pv unset): PV serves the DC first, its surplus charges the battery whatever the action. The
  battery DP uses battery_transition with the PV forecast; load shifting pays only the energy PV does not cover.
* PV actions (include_pv=True): the battery DP chooses among the 9 (battery, PV) actions with
  battery_dispatch_transition, the step's CO2 being grid import x CI - resold PV x pv_resell_credit x CI; load
  shifting pays the better of using the PV for the load and reselling it (grid_co2_kg).
PV forecast (pv_forecast): realistic = the true PV of the current step and the next 4 h (what the agents observe with
include_pv_obs), then yesterday's profile; perfect = the true PV. Values trained with PV add the PV context features
(current PV and next-4 h mean, / AC rating) to the context; their value.npz has pv_ctx = 1.

Forecasts (`forecast=`):
* realistic (default): what the agents observe. Carbon intensity (CI): the current value and the next 8 steps (the
  env's CI forecast), then yesterday's profile anchored on the last known value. Workload: the current and next step
  (also observed), then yesterday's profile anchored on the current value.
* perfect: the true future CI and workload (an optimistic reference).

Terminal value (`value=`):
* none: V = 0 (myopic MPC)
* naive: queued tasks run later at the past day's mean CI; stored battery energy displaces grid energy at that CI
* <path.npz>: learned value (train_mpc_value.py). Default form `per_unit`: the same structure as `naive`, but with
  learned prices, V_ls = sum_b q_b c_ls(b, |q|, context) and V_bat = -e c_bat(e, context) (e: usable stored energy), where c_ls is the expected
  CO2 of one queued task of age bucket b and c_bat the CI at which stored energy will be discharged, both fitted to
  Monte Carlo targets tracked through the MPC's own rollouts. Form `mlp` (ablation): unstructured MLPs fitted to n-step
  average-cost returns (see "Value learning"); the planner tends to exploit its errors (e.g. by hoarding work), because
  most of the return's variance comes from the context, not the queue.
In every case, when the horizon reaches the end of the episode the terminal cost becomes the episode-end cost used by
eval_compare.py (queued work at its marginal energy x CI; battery energy left has no value).

Value learning for the `mlp` form (average cost): the stage cost of subsystem i is its CO2 relative to doing nothing,
    d_ls  = CI * (E(executed utilization) - E(submitted utilization)) + drop penalty
    d_bat = CI * (battery's grid energy),
and V_i is the differential value of the MPC policy: V_i(s_t) ~ sum_{k<n} (d_{t+k} - rho_i) + V_i(s_{t+n}), with rho_i the
average stage cost. The constant offsets do not matter (all plans end at the same time); only how V depends on the
queue / stored energy does.

Policy spec for eval_compare.py:  mpc[:key=value,...], e.g.
    mpc:H=16,value=runs/mpc_value/az/value.npz,forecast=realistic=mpc_learned
keys: H (horizon steps, 16), value (none|naive|<npz>, naive), forecast (realistic|perfect), dc_setpoint (15),
      samples (CEM samples, 256), iters (CEM iterations, 3), explore (probability of a random action, 0).
"""
import json
import math

import numpy as np

from baseline_policies import NOOP, DAY, battery_dispatch_transition, battery_transition, dc_power, setpoint_action
from envs.bat_env_fwd_view import combine_bat_pv_action

AGE_BINS = 98          # queue counts by age in steps 0..96; bin 97 holds overdue tasks (older than 24 h)
OVERDUE = AGE_BINS - 1
MAX_UTIL_TASKS = 90    # the env never schedules above 90 % utilization
QUEUE_BUCKETS = (0, 16, 32, 48, 64, 80, OVERDUE, AGE_BINS)  # value features: 4 h age buckets + overdue
from sdc_eval.constants import DROPPED_TASK_CO2_FACTOR as DROP_PENALTY  # dropped task cost, x energy x CI (shared)
KNOWN_CI_STEPS = 8     # the env's CI forecast length (n_vars_ci)
FORECAST_DECAY = 16.0  # steps over which a forecast blends from the anchor value into yesterday's profile
BAT_ACTION_IDS = {'charge': 0, 'discharge': 1, 'idle': 2}
PV_ACTION_IDS = {'use': 0, 'battery': 1, 'resell': 2}  # include_pv=True (envs/bat_env_fwd_view.PV_ACTIONS)
KNOWN_PV_STEPS = 16    # PV the agents observe with include_pv_obs: the current step and the next 4 h


# ---------------------------------------------------------------------------------------------------------------------
# Load-shifting queue model (vectorized replica of CarbonLoadEnv.step)
# ---------------------------------------------------------------------------------------------------------------------
def queue_counts(ls):
    """Tasks in the env queue counted by age in 15-minute steps (bin OVERDUE: older than 24 h)."""
    q = np.zeros(AGE_BINS)
    for task in ls.tasks_queue:
        age_h = (ls.current_day - task['day']) * 24 + (ls.current_hour - task['hour'])
        q[min(int(round(age_h * 4)), OVERDUE)] += 1
    return q


def pop_oldest(q, n):
    """Remove n[i] tasks from the oldest end of each queue q[i] (in place)."""
    rev = q[:, ::-1]
    before = np.cumsum(rev, axis=1) - rev
    take = np.clip(n[:, None] - before, 0, rev)
    q -= take[:, ::-1]


def simulate_queue(q0, actions, workload, ns_frac, sh_frac, qmax):
    """Simulate the queue for candidate LS plans.

    q0: (AGE_BINS,) current counts; actions: (N, H) in {0 defer, 1 nothing, 2 process}; workload: (H,) submitted
    utilization. Returns utilization run (N, H), tasks dropped (N, H) and the final queues (N, AGE_BINS).
    """
    n, horizon = actions.shape
    q = np.repeat(np.asarray(q0, dtype=float)[None], n, axis=0)
    util = np.empty((n, horizon))
    dropped = np.zeros((n, horizon))
    zeros = np.zeros(n)
    for k in range(horizon):
        ns = math.ceil(workload[k] * ns_frac * 100)
        sh = math.floor(workload[k] * sh_frac * 100)
        cap = MAX_UTIL_TASKS - ns - sh
        odp = np.minimum(q[:, OVERDUE], cap) if cap > 0 else zeros
        q[:, OVERDUE] -= odp
        cap2 = cap - odp
        total = q.sum(axis=1)
        a = actions[:, k]
        add = np.where(a == 0, np.minimum(sh, qmax - total), 0.0)
        dropped[:, k] = np.where(a == 0, sh - add, 0.0)
        proc = np.where((a == 2) & (cap2 >= 1), np.minimum(np.minimum(sh, cap2), total), 0.0)
        if proc.any():
            pop_oldest(q, proc)
        util[:, k] = (ns + sh - add + proc + odp) / 100
        q[:, 0] += add
        # one step older
        q[:, OVERDUE] += q[:, OVERDUE - 1]
        q[:, 1:OVERDUE] = q[:, :OVERDUE - 1].copy()
        q[:, 0] = 0
    return util, dropped, q


# ---------------------------------------------------------------------------------------------------------------------
# Energy model, forecasts and value features
# ---------------------------------------------------------------------------------------------------------------------
def energy_coefficients(sdc, setpoint):
    """IT + HVAC energy (kWh per 15-min step) as c0 + c1 u + c2 u^2 of utilization u, at the current weather."""
    amb, wb = sdc.dc_env.ambient_temp, sdc.dc_env.wet_bulb
    u = np.array([0.1, 0.5, 0.9])
    e = np.array([sum(dc_power(sdc, x, setpoint, amb, wb)) / 1e3 * 0.25 for x in u])
    c2, c1, c0 = np.polyfit(u, e, 2)
    return np.array([c0, c1, c2])


def energy_kwh(coef, u):
    return coef[0] + coef[1] * u + coef[2] * u * u


def task_kwh(coef, u):
    """Marginal energy of one task (1 % utilization for one step)."""
    return (coef[1] + 2 * coef[2] * u) * 0.01


def blend_forecast(series, t, horizon, known):
    """Values at t..t+horizon-1: the true values for the first `known` steps (observed), then yesterday's profile
    anchored on the last known value, the anchor's offset fading out over FORECAST_DECAY steps."""
    out = np.array(series[t:t + known], dtype=float)
    last = t + known - 1
    offset = series[last] - series[last - DAY]
    for k in range(known, horizon):
        out = np.append(out, series[t + k - DAY] + offset * math.exp(-(k - known + 1) / FORECAST_DECAY))
    return out[:horizon]


def forecasts(sdc, horizon, mode):
    """CI (g/kWh) and submitted workload for the steps t..t+horizon-1, plus the 96 past values of each."""
    t_ci, t_wl = sdc.ci_m.time_step, sdc.workload_m.time_step
    ci_all, wl_all = sdc.ci_m.carbon_smooth, sdc.workload_m.cpu_smooth
    if mode == 'perfect':
        ci = np.array(ci_all[t_ci:t_ci + horizon], dtype=float)
        wl = np.array(wl_all[t_wl:t_wl + horizon], dtype=float)
    elif mode == 'realistic':
        ci = blend_forecast(ci_all, t_ci, horizon, min(horizon, 1 + KNOWN_CI_STEPS))
        wl = np.clip(blend_forecast(wl_all, t_wl, horizon, min(horizon, 2)), 0.0, 1.0)
    else:
        raise ValueError(f'Unknown forecast mode {mode!r}; use realistic or perfect')
    wl[0] = float(sdc.ls_env.workload)
    return ci, wl, np.asarray(ci_all[t_ci - DAY:t_ci], float), np.asarray(wl_all[t_wl - DAY:t_wl], float)


def pv_setup(sdc):
    """(mode, resell_credit, export_credit) of the env's on-site PV. mode: 'off' (no PV generated), 'auto' (the
    self-consumption rule) or 'agent' (include_pv=True: the battery agent's PV action)."""
    pv_m = getattr(sdc, 'pv_m', None)
    mode = getattr(sdc, 'pv_mode', 'auto')
    if pv_m is None or not pv_m.enabled or mode == 'off':
        return 'off', 0.0, False
    bat = sdc.bat_env
    return mode, float(getattr(bat, 'pv_resell_credit', 1.0)), bool(getattr(bat, 'pv_export_credit', False))


def pv_forecast(sdc, horizon, mode):
    """On-site PV energy (kWh per step) for the steps t..t+horizon-1. realistic: the true values for the steps the
    agents observe with include_pv_obs (current + next 4 h), then yesterday's profile at the same time of day
    (repeated for later days); perfect: the true values."""
    pv_m = sdc.pv_m
    series = np.asarray(pv_m.pv_kw, float) * 0.25
    t = pv_m.time_step
    k = np.arange(horizon)
    known = horizon if mode == 'perfect' else min(horizon, 1 + KNOWN_PV_STEPS)
    idx = np.where(k < known, t + k, t - DAY + k % DAY)
    out = series[idx % len(series)]
    out[0] = float(sdc.bat_env.pv_kw) * 0.25
    return out


def grid_co2_kg(e_kwh, ci, pv_kwh, pv=('off', 0.0, False)):
    """CO2 (kg) of serving e_kwh of DC energy with the battery idle, in steps with pv_kwh of PV (arrays broadcast).
    auto: PV first serves the DC (surplus curtailed, or credited if export_credit); agent: the better of 'use'
    (PV serves the DC, surplus curtailed) and 'resell' (all PV credited at resell_credit x CI)."""
    mode, credit, export = pv
    e = np.asarray(e_kwh, float)
    if mode == 'off':
        return e * ci / 1e3
    net = np.maximum(e - pv_kwh, 0.0)
    if mode == 'auto':
        cost = net - (np.maximum(pv_kwh - e, 0.0) if export else 0.0)
    else:
        cost = np.minimum(net, e - credit * pv_kwh)
    return cost * ci / 1e3


def marginal_task_co2_kg(coef, u, ci, pv_kwh, pv=('off', 0.0, False)):
    """CO2 (kg) of one more task (1 % utilization for one step) at utilization u, with the step's PV."""
    if pv[0] == 'off':
        return task_kwh(coef, u) * ci / 1e3
    return float(grid_co2_kg(energy_kwh(coef, u + 0.01), ci, pv_kwh, pv) - grid_co2_kg(energy_kwh(coef, u), ci, pv_kwh, pv))


def pv_context(pv_kwh, rating_kwh):
    """PV features of a state from the PV of its next 17 steps (kWh): current and next-4 h mean, / AC rating."""
    pv_kwh = np.asarray(pv_kwh, float)
    return np.array([pv_kwh[0], pv_kwh[1:1 + KNOWN_PV_STEPS].mean()]) / rating_kwh


def context_features(hour, ci_win, wl_win, ci_scale, pv_ctx=None):
    """Exogenous context of a state at a given hour: the CI and workload of the 96 steps before it, plus the PV
    features (pv_context) when the run has PV."""
    s = ci_scale
    h = 2 * np.pi * hour / 24
    hours_since_min = (DAY - int(np.argmin(ci_win))) / 4
    m = 2 * np.pi * hours_since_min / 24
    ctx = np.array([np.sin(h), np.cos(h), ci_win.mean() / s, ci_win.std() / s, ci_win[-1] / s, ci_win.min() / s,
                    ci_win.max() / s, (ci_win[-1] - ci_win[-5]) / s, np.sin(m), np.cos(m),
                    wl_win.mean(), wl_win[-1]])
    return ctx if pv_ctx is None else np.concatenate([ctx, pv_ctx])


N_BASE_CTX = 12  # context_features without PV


def queue_features(q):
    """(N, AGE_BINS) counts -> (N, 8): tasks per 4 h age bucket and overdue (/100), total (/1000)."""
    q = np.atleast_2d(q)
    buckets = [q[:, a:b].sum(axis=1) / 100 for a, b in zip(QUEUE_BUCKETS[:-1], QUEUE_BUCKETS[1:])]
    return np.column_stack(buckets + [q.sum(axis=1) / 1000])


# ---------------------------------------------------------------------------------------------------------------------
# Terminal values
# ---------------------------------------------------------------------------------------------------------------------
class MLP:
    """Numpy forward pass of a SiLU MLP with input/output standardization (weights from train_mpc_value.py)."""

    def __init__(self, params, prefix):
        self.layers = []
        i = 0
        while f'{prefix}W{i}' in params:
            self.layers.append((params[f'{prefix}W{i}'], params[f'{prefix}b{i}']))
            i += 1
        self.x_mean, self.x_std = params[f'{prefix}x_mean'], params[f'{prefix}x_std']
        self.y_mean, self.y_std = float(params[f'{prefix}y_mean']), float(params[f'{prefix}y_std'])

    def __call__(self, x):
        h = (np.atleast_2d(x) - self.x_mean) / self.x_std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = h / (1 + np.exp(-h))
        return h[:, 0] * self.y_std + self.y_mean


class ZeroValue:
    name = 'none'

    def ls(self, q, ctx, ci_win, coef, u):
        return np.zeros(len(q))

    def bat(self, usable, ctx, ci_win, eff_discharge):
        """Value (kg CO2, <= 0) of `usable` MWh stored above the battery's SoC floor."""
        return np.zeros(len(usable))


class NaiveValue(ZeroValue):
    """Queued tasks run later at the mean CI of the past day; stored energy (after discharge losses) displaces grid
    energy at that CI."""
    name = 'naive'

    def ls(self, q, ctx, ci_win, coef, u):
        return q.sum(axis=1) * task_kwh(coef, u) * ci_win.mean() / 1e3

    def bat(self, usable, ctx, ci_win, eff_discharge):
        return -np.asarray(usable, float) * eff_discharge * ci_win.mean()  # MWh x g/kWh = kg


class LearnedValue(ZeroValue):
    """Learned terminal value (train_mpc_value.py), in one of two forms:

    * per_unit (default): V_ls(q) = sum_b q_b * c_ls(b, |q|, ctx) and V_bat(e) = -e * c_bat(e, ctx), where e is the usable
      stored energy (MWh above the SoC floor), c_ls the expected CO2 (kg) of running one queued task of age bucket b and
      c_bat the grid CO2 (g) displaced per stored kWh when it is later discharged (CI x discharge efficiency). Both are
      learned from Monte Carlo targets tracked through the MPC's own rollouts.
    * mlp: unstructured MLPs V_ls(queue features, ctx), V_bat(e, ctx) fitted to n-step average-cost returns (ablation).
    """

    def __init__(self, path):
        params = dict(np.load(path))
        self.name = path
        self.form = str(params.get('form', 'mlp'))
        self.v_ls, self.v_bat = MLP(params, 'ls_'), MLP(params, 'bat_')
        self.ci_scale = float(params['ci_scale'])
        # Values trained with PV take the PV context features (files from before PV support have no flag).
        self.pv_ctx = bool(params.get('pv_ctx', False))
        self.scenario = json.loads(str(params['scenario'])) if 'scenario' in params else {}

    def _ctx(self, ctx):
        """The context this value was trained on: a value trained without PV ignores the PV features."""
        if self.pv_ctx and len(ctx) == N_BASE_CTX:
            raise ValueError(f'{self.name} was trained with on-site PV; evaluate it on a scenario with PV.')
        return ctx if self.pv_ctx else ctx[:N_BASE_CTX]

    def ls(self, q, ctx, ci_win, coef, u):
        ctx = self._ctx(ctx)
        q = np.atleast_2d(q)
        if self.form == 'mlp':
            x = np.hstack([queue_features(q), np.repeat(ctx[None], len(q), axis=0)])
            return self.v_ls(x)
        counts = queue_features(q)[:, :N_BUCKETS] * 100
        return (counts * self.v_ls(per_task_inputs(q.sum(axis=1), ctx)).reshape(len(q), N_BUCKETS)).sum(axis=1)

    def bat(self, usable, ctx, ci_win, eff_discharge):
        ctx = self._ctx(ctx)
        usable = np.asarray(usable, float)
        x = np.hstack([usable[:, None], np.repeat(ctx[None], len(usable), axis=0)])
        if self.form == 'mlp':
            return self.v_bat(x)
        return -usable * self.v_bat(x)  # c_bat already includes the discharge losses (train_mpc_value.py)


N_BUCKETS = len(QUEUE_BUCKETS) - 1


def per_task_inputs(total, ctx):
    """Inputs of the per-task value c_ls for every (queue, age bucket): one-hot bucket, queue length / 1000, context.
    total: (N,) queue lengths -> (N * N_BUCKETS, N_BUCKETS + 1 + len(ctx)), queue-major."""
    total = np.atleast_1d(np.asarray(total, float))
    n = len(total)
    onehot = np.tile(np.eye(N_BUCKETS), (n, 1))
    ctx = np.atleast_2d(ctx)
    ctx_rows = np.repeat(ctx, N_BUCKETS, axis=0) if len(ctx) == n else np.repeat(ctx, n * N_BUCKETS, axis=0)
    return np.hstack([onehot, np.repeat(total / 1000, N_BUCKETS)[:, None], ctx_rows])


def make_value(spec):
    if spec in (None, '', 'none'):
        return ZeroValue()
    if spec == 'naive':
        return NaiveValue()
    return LearnedValue(spec)


# ---------------------------------------------------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------------------------------------------------
class MPCPolicy:
    """Receding-horizon controller (eval_compare.py interface: reset(seed), act(sdc) -> {agent: action}).

    With record=True, `self.last` holds what train_mpc_value.py needs after each act(): the value features of the
    current state and the energy model used to compute the stage costs."""
    include_dc_setpoint = False
    include_pv_obs = False  # PV is read from the simulator (pv_forecast), not from the observations

    def __init__(self, H=16, value='naive', forecast='realistic', dc_setpoint=15.0, samples=256, iters=3,
                 elite_frac=0.1, smoothing=0.7, explore=0.0, bat_grid=101, record=False):
        self.H, self.forecast, self.dc_setpoint = int(H), forecast, float(dc_setpoint)
        self.value = make_value(value) if isinstance(value, (str, type(None))) else value
        self.samples, self.iters, self.n_elite = int(samples), int(iters), max(2, int(elite_frac * samples))
        self.smoothing, self.explore, self.bat_grid, self.record = smoothing, float(explore), int(bat_grid), record
        # The scenario a learned value was trained on, so eval_compare.py flags a different eval scenario (e.g. PV).
        self.train_config = dict(getattr(self.value, 'scenario', None) or {})

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.prev_plan = None
        self.last = None

    # -- load shifting: CEM over action sequences ----------------------------------------------------------------
    def plan_ls(self, q0, ci, wl, coef, ls, terminal, pv_kwh=None, pv=('off', 0.0, False)):
        """CEM over load-shifting action sequences. With PV, the energy of each step costs only what PV does not
        cover (grid_co2_kg, battery idle; the battery's use of PV is planned by plan_bat)."""
        horizon = len(ci)
        ns_frac, sh_frac, qmax = ls.non_shiftable_tasks_percentage, ls.shiftable_tasks_percentage, ls.queue_max_len

        def cost(plans):
            util, dropped, q_end = simulate_queue(q0, plans, wl, ns_frac, sh_frac, qmax)
            stage = grid_co2_kg(energy_kwh(coef, util), ci, pv_kwh, pv).sum(axis=1)
            drops = DROP_PENALTY * (dropped * task_kwh(coef, util) * ci).sum(axis=1) / 1e3
            return stage + drops + terminal(q_end, util[:, -1]), util

        fixed = [np.ones(horizon, int), np.zeros(horizon, int), np.full(horizon, 2)]
        if self.prev_plan is not None and len(self.prev_plan) > 1:
            prev = np.append(self.prev_plan[1:], 1)[:horizon]
            prev = np.pad(prev, (0, horizon - len(prev)), constant_values=1)
            fixed.insert(0, prev)
            probs = np.full((horizon, 3), 0.1)
            probs[np.arange(horizon), prev] = 0.8
        else:
            probs = np.full((horizon, 3), 1 / 3)
        best_plan, best_cost, best_util = None, np.inf, None
        for it in range(self.iters):
            cum = probs.cumsum(axis=1)
            plans = (self.rng.random((self.samples, horizon, 1)) > cum[None, :, :]).sum(axis=2).clip(0, 2)
            if it == 0:
                plans[:len(fixed)] = np.array(fixed)
            c, util = cost(plans)
            order = np.argsort(c)
            if c[order[0]] < best_cost:
                best_cost, best_plan, best_util = c[order[0]], plans[order[0]].copy(), util[order[0]].copy()
            elite = plans[order[:self.n_elite]]
            freq = np.stack([(elite == a).mean(axis=0) for a in range(3)], axis=1)
            probs = self.smoothing * freq + (1 - self.smoothing) * probs
        self.prev_plan = best_plan
        return int(best_plan[0]), best_util

    # -- battery: dynamic programming over stored energy ---------------------------------------------------------
    def plan_bat(self, stored, battery, ci, load_mwh, terminal, pv_mwh=None, pv=('off', 0.0, False)):
        """Exact DP over the stored energy (MWh) in the battery's usable window, with the env's Battery limits and
        PV rules. terminal(grid) is the value of the stored energy at the end of the horizon.
        Without PV or with the automatic rule (battery_transition), the actions are the 3 battery actions and the stage
        cost is the CO2 the battery adds to the step (plus the export credit its PV charging forgoes); with
        include_pv (pv mode 'agent', battery_dispatch_transition) they are the 9 (battery, PV) pairs and the stage cost
        is the step's CO2: grid import x CI - resold PV x resell_credit x CI. Costs that are the same for every
        action do not change the plan. Returns (battery action id, PV action id or None)."""
        mode, credit, export = pv
        pv_mwh = np.zeros(len(ci)) if pv_mwh is None else pv_mwh
        if mode == 'agent':
            pairs = [(b, p) for p in PV_ACTION_IDS for b in BAT_ACTION_IDS]

            def step(s, k, b, p):
                nxt, grid_e, resold = battery_dispatch_transition(s, battery, b, p, load_mwh[k], pv_mwh[k])
                return nxt, ci[k] * (grid_e - credit * resold)
        else:
            pairs = [(b, None) for b in BAT_ACTION_IDS]

            def step(s, k, b, p):
                nxt, de, from_pv = battery_transition(s, battery, b, load_mwh[k], pv_mwh[k])
                return nxt, ci[k] * (de + (from_pv if export else 0.0))

        cap = battery.capacity
        grid = np.linspace(battery.soc_min * cap, battery.soc_max * cap, self.bat_grid)
        v = terminal(grid)
        for k in range(len(ci) - 1, 0, -1):
            v = np.min([c + np.interp(nxt, grid, v) for nxt, c in (step(grid, k, b, p) for b, p in pairs)], axis=0)
        costs = {}
        for b, p in pairs:
            nxt, c = step(np.array([stored]), 0, b, p)
            costs[(b, p)] = float(c[0] + np.interp(nxt[0], grid, v))
        b, p = min(costs, key=costs.get)
        return BAT_ACTION_IDS[b], (None if p is None else PV_ACTION_IDS[p])

    def act(self, sdc):
        ls, bat = sdc.ls_env, sdc.bat_env
        battery = bat.battery
        floor = battery.soc_min * battery.capacity
        remaining = int(sdc.t_m.total_timesteps - sdc.t_m.current_timestep)
        horizon = max(1, min(self.H, remaining))
        at_end = remaining <= self.H
        ci, wl, ci_past, wl_past = forecasts(sdc, horizon, self.forecast)
        coef = energy_coefficients(sdc, self.dc_setpoint)
        ci_scale = getattr(self.value, 'ci_scale', None) or float(np.mean(sdc.ci_m.original_data))
        ci_win = np.concatenate([ci_past, ci])[-DAY:]
        wl_win = np.concatenate([wl_past, wl])[-DAY:]
        pv = pv_setup(sdc)
        if pv[0] == 'off':
            pv_kwh, ctx_pv_now, ctx_pv_end = np.zeros(horizon), None, None
        else:  # PV of the horizon, plus the 17 steps after it for the PV context of the state it ends in
            pv_all = pv_forecast(sdc, horizon + 1 + KNOWN_PV_STEPS, self.forecast)
            rating = sdc.pv_m.ac_rating_kw * 0.25
            pv_kwh = pv_all[:horizon]
            ctx_pv_now, ctx_pv_end = pv_context(pv_all, rating), pv_context(pv_all[horizon:], rating)
        ctx_end = context_features((ls.current_hour + horizon / 4) % 24, ci_win, wl_win, ci_scale, ctx_pv_end)
        q0 = queue_counts(ls)

        if at_end:  # episode-end cost, as charged by eval_compare.py
            def ls_terminal(q, u):
                return q.sum(axis=1) * task_kwh(coef, u) * ci_win.mean() / 1e3

            def bat_terminal(grid):
                return np.zeros(len(grid))
        else:
            def ls_terminal(q, u):
                return self.value.ls(q, ctx_end, ci_win, coef, u)

            def bat_terminal(grid):
                return self.value.bat(grid - floor, ctx_end, ci_win, battery.eff_discharge)

        a_ls, util = self.plan_ls(q0, ci, wl, coef, ls, ls_terminal, pv_kwh, pv)
        stored = float(battery.current_load)
        a_bat, a_pv = self.plan_bat(stored, battery, ci, energy_kwh(coef, util) / 1e3, bat_terminal,
                                    pv_kwh / 1e3, pv)
        if self.explore > 0:
            if self.rng.random() < self.explore:
                a_ls = int(self.rng.integers(3))
            if self.rng.random() < self.explore:
                a_bat = int(self.rng.integers(3))
            if a_pv is not None and self.rng.random() < self.explore:
                a_pv = int(self.rng.integers(len(PV_ACTION_IDS)))
        if self.record:
            self.last = {
                'queue': queue_features(q0)[0], 'q': q0, 'soc': stored - floor,  # usable stored energy (MWh)
                'ctx': context_features(ls.current_hour, ci_past, wl_past, ci_scale, ctx_pv_now),
                'coef': coef, 'ci': float(ci[0]), 'workload': float(wl[0]), 'pv': pv}
        return {'agent_ls': a_ls, 'agent_dc': setpoint_action(sdc.dc_env.raw_curr_stpt, self.dc_setpoint),
                'agent_bat': a_bat if a_pv is None else combine_bat_pv_action(a_bat, a_pv)}


PARAM_WORDS = {'none', 'naive', 'realistic', 'perfect'}


def make_mpc_policy(rest):
    """'H=16,value=naive,forecast=realistic' -> MPCPolicy."""
    kwargs = {}
    for kv in filter(None, rest.split(',')):
        k, _, v = kv.partition('=')
        kwargs[k] = v
    known = {'H', 'value', 'forecast', 'dc_setpoint', 'samples', 'iters', 'explore'}
    unknown = set(kwargs) - known
    if unknown:
        raise ValueError(f'Unknown mpc parameter(s) {unknown}; use {sorted(known)}')
    for k in ('H', 'samples', 'iters'):
        if k in kwargs:
            kwargs[k] = int(kwargs[k])
    for k in ('dc_setpoint', 'explore'):
        if k in kwargs:
            kwargs[k] = float(kwargs[k])
    return MPCPolicy(**kwargs)
