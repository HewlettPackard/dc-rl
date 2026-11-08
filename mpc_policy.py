"""Model predictive control (MPC) for SustainDC from the single-agent viewpoint: one controller picks the joint
(load shifting, cooling, battery) action, i.e. one of the 27 actions of SustainDCSingleAgentEnv's Discrete(27) space.

Model: a clone of the live simulator. SustainDC.step is deterministic once reset() has drawn the episode's randomness,
so the clone is an exact model of the data center, queue and battery. What it is told about the future exogenous
inputs (carbon intensity, workload, weather) depends on `forecast`:
  * oracle:      the true future. Perfect-forecast MPC, an optimistic reference for RL rather than a deployable
                 controller.
  * persistence: only past data. Every value after the current step is replaced by the same time of day averaged over
                 the last `forecast_days` days (the profile then repeats for later days). This is a deployable MPC;
                 the gap to `oracle` is the value of perfect forecasts.
The forecast replaces the series the managers read during a step (CI raw and normalized, workload, temperature and
wet bulb raw and normalized) from the step after the current one onward, so no future value reaches the planner.
(The RL agents observe 2 h of true future normalized CI; persistence does not use it.)

Optimizer: receding-horizon cross-entropy method (CEM) over joint-action sequences with move blocking. The horizon
is split into blocks of `block` steps that each hold one joint action; each block has its own categorical
distribution over the 27 joint actions (so correlated sub-actions, e.g. "defer + charge", are learned jointly).
Each replan:
  1. candidates = the previous best plan shifted by one block (warm start), the do-nothing plan, the 27 constant
     plans (first iteration only) and samples from the block distributions;
  2. every candidate is rolled out on a fresh clone of the planning model and scored (see cost below), optionally
     in `workers` parallel processes;
  3. the distributions move toward the empirical distribution of the elite (lowest-cost) candidates.
The first `replan` steps of the best plan found are executed, then the controller replans.

Cost (grams of CO2, minimized):
  sum over the horizon of   bat_CO2_footprint                     (grid CO2 incl. battery charging/discharging)
                          + drop_w    * task_g * tasks dropped      (deferred while the queue is full: work lost)
                          + overdue_w * task_g * overdue tasks      (tasks older than 24 h, per step)
  + terminal value:         task_g' * tasks still queued            (they must run later)
                          - battery stored energy (kWh) * CI'       (stored energy offsets later grid energy)
where task_g is the CO2 of running one task (1% of the DC for 15 min) at the marginal IT+HVAC energy and the
current CI, and CI' / task_g' use the mean (forecast) CI of the 24 h after the horizon. Without the terminal value
the controller would defer work past the horizon and drain the battery at its end for free.

Usage (from the repo root), next to the other baselines:
    python eval_compare.py --policy noop --policy mpc=mpc_oracle --policy mpc:forecast=persistence=mpc_persist \\
        --days_per_episode 3
    python eval_compare.py --policy mpc:horizon=96,block=8,samples=64,iters=4,workers=8=mpc_long ...
Parameters (key=value, comma separated): horizon, block, replan, samples, iters, elites, smoothing, drop_w,
overdue_w, forecast, forecast_days, workers, seed, verbose. See MPCPolicy.__init__.

Inside a SustainDCSingleAgentEnv, use the joint action directly:
    policy = MPCPolicy(); policy.reset(seed)
    action = policy.act_joint(env.env)   # Discrete(27) index; env must use action_type='discrete'
"""
import atexit
import copy
import hashlib
import io
import multiprocessing as mp
import pickle
import time

import numpy as np

from baseline_policies import dc_power
from sustaindc_single_env import N_SUB_ACTIONS, decode_joint_action

N_JOINT = int(np.prod(N_SUB_ACTIONS))  # 27
NOOP_JOINT = int(np.ravel_multi_index((1, 1, 2), N_SUB_ACTIONS))  # ls: do nothing, dc: hold, bat: idle
DAY = 96  # 15-minute steps per day
FORECASTS = ('oracle', 'persistence')

# Exogenous time series read by the managers during SustainDC.step, indexed by each manager's own time_step.
EXOGENOUS_SERIES = {
    'workload_m': ('cpu_smooth',),
    'ci_m': ('carbon_smooth', 'norm_carbon'),
    'weather_m': ('temperature_data', 'wet_bulb_data', 'norm_temp_data', 'norm_wet_bulb_data'),
}


def _zero_rewards(params):
    """Replaces SustainDC.calculate_reward on clones: the cost uses physical metrics, and the ls reward appends to the
    module-global energy history (utils/reward_creator.py), which rollouts must not touch."""
    return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------------------------------------------------
# Planning model
# ---------------------------------------------------------------------------------------------------------------------
class SimCloner:
    """Fast copies of a SustainDC instance for rollouts.

    deepcopy of a full SustainDC takes ~35 ms, almost all of it copying data that never changes during an episode:
    the managers' time series and the data center IT model (whose attributes are per-call scratch values, recomputed
    before being read). Those objects are shared with the clone through the deepcopy memo (~0.25 ms per clone).
    The shared set is rebuilt from the instance on every call, since reset() rebinds some of the arrays. Passing a
    memo that maps an object's id to a *different* object substitutes it in the clone (used for forecasts).
    """

    MIN_SHARED_SIZE = 1000

    @classmethod
    def shared_objects(cls, sdc):
        shared = {}
        for manager in (sdc.workload_m, sdc.ci_m, sdc.weather_m):
            for v in vars(manager).values():
                if isinstance(v, np.ndarray) and v.size >= cls.MIN_SHARED_SIZE:
                    shared[id(v)] = v
        for obj in (sdc.dc_env.dc, sdc.dc_env.DC_Config):
            shared[id(obj)] = obj
        if getattr(sdc, 'dashboard', None) is not None:
            shared[id(sdc.dashboard)] = sdc.dashboard
        return shared

    @classmethod
    def clone(cls, sdc, shared=None):
        sim = copy.deepcopy(sdc, dict(shared if shared is not None else cls.shared_objects(sdc)))
        sim.calculate_reward = _zero_rewards
        return sim


def persistence_forecast(series, t, days=1):
    """Copy of `series` where every value after index t is replaced by a forecast built from indices <= t only:
    the value at the same time of day averaged over the last `days` days, repeated for every later day."""
    out = np.array(series, dtype=float, copy=True)
    n_ahead = len(out) - t - 1
    if n_ahead <= 0:
        return out
    offsets = np.arange(1, DAY + 1)                                   # time of day relative to t: t+1 .. t+DAY
    src = t + offsets[None, :] - DAY * np.arange(1, int(days) + 1)[:, None]  # same time of day, 1..days days back
    profile = out[np.clip(src, 0, t)].mean(axis=0)                    # (before the data start: hold the oldest value)
    out[t + 1:] = profile[np.arange(n_ahead) % DAY]
    return out


def planning_model(sdc, forecast='oracle', forecast_days=1):
    """Clone of the live simulator whose future exogenous inputs follow the forecast. Rollouts clone this."""
    if forecast not in FORECASTS:
        raise ValueError(f'forecast must be one of {FORECASTS}, got {forecast!r}')
    shared = SimCloner.shared_objects(sdc)
    if forecast == 'persistence':
        for manager_name, attrs in EXOGENOUS_SERIES.items():
            manager = getattr(sdc, manager_name)
            for attr in attrs:
                arr = getattr(manager, attr)
                shared[id(arr)] = persistence_forecast(arr, manager.time_step, forecast_days)
    return SimCloner.clone(sdc, shared)


def marginal_task_kwh(sdc, utilization, setpoint, ambient, wet_bulb):
    """Marginal IT + HVAC energy (kWh) of one task: 1% of the DC's utilization for one 15-minute step."""
    lo, hi = max(utilization - 0.05, 0.0), min(utilization + 0.05, 1.0)
    p_lo = sum(dc_power(sdc, lo, setpoint, ambient, wet_bulb))
    p_hi = sum(dc_power(sdc, hi, setpoint, ambient, wet_bulb))
    return (p_hi - p_lo) / (hi - lo) / 100 / 1e3 * 0.25


def rollout_cost(model, shared, plan, params):
    """Cost (gCO2) of executing `plan` (one joint action per block) from a clone of the planning model."""
    sim = SimCloner.clone(model, shared)
    task_g = params['task_g']
    cost = 0.0
    for a in np.repeat(plan, params['block']):
        _, _, _, truncs, info = sim.step(decode_joint_action(a, 'discrete'))
        i = info['agent_ls']
        cost += i['bat_CO2_footprint']
        cost += params['drop_w'] * task_g * i['ls_tasks_dropped']
        cost += params['overdue_w'] * task_g * i['ls_overdue_penalty']
        if truncs['__all__']:
            break
    # Terminal value at the (forecast) mean CI of the next day. The planning model's CI series is the forecast.
    ci = sim.ci_m.carbon_smooth
    window = ci[sim.ci_m.time_step:sim.ci_m.time_step + DAY]
    ci_after = float(np.mean(window)) if len(window) else float(ci[-1])
    cost += params['task_kwh'] * ci_after * len(sim.ls_env.tasks_queue)
    cost -= float(sim.bat_env.battery.current_load) * 1e3 * ci_after  # stored MWh -> kWh
    return cost


# ---------------------------------------------------------------------------------------------------------------------
# Parallel rollouts
# ---------------------------------------------------------------------------------------------------------------------
def _static_key(obj):
    """Cache key of a static object sent to the workers: content hash for arrays (ids of freed arrays get reused),
    identity for the data center model objects (kept alive by the pool, so their ids stay unique)."""
    if isinstance(obj, np.ndarray):
        return 'a' + hashlib.blake2b(np.ascontiguousarray(obj).view(np.uint8), digest_size=16).hexdigest()
    return f'o{id(obj)}'


def _worker_main(conn):
    """Worker process: rebuilds the planning model from the live state it is sent and scores plans on it.

    Static objects (time series, data center model) arrive once and are cached; each replan only the mutable state is
    sent, as a pickle whose references to static objects are cache keys."""
    cache, model, shared = {}, None, None
    while True:
        msg = conn.recv()
        try:
            if msg[0] == 'state':
                _, new_statics, keep, state, forecast, forecast_days = msg
                cache.update(pickle.loads(new_statics))
                cache = {k: cache[k] for k in keep}
                unpickler = pickle.Unpickler(io.BytesIO(state))
                unpickler.persistent_load = cache.__getitem__
                model = planning_model(unpickler.load(), forecast, forecast_days)
                shared = SimCloner.shared_objects(model)
                conn.send(('ok', None))
            elif msg[0] == 'eval':
                _, plans, params = msg
                conn.send(('ok', [rollout_cost(model, shared, p, params) for p in plans]))
            else:
                break
        except Exception as e:  # report instead of leaving the parent waiting
            conn.send(('error', repr(e)))


class RolloutPool:
    """Persistent worker processes for rollouts. Each replan the live simulator's mutable state (a few kB) is sent to
    every worker, which builds the planning model (forecast included) itself; static objects are sent only when they
    change (new episode or new simulator). Each CEM iteration's plans are then split across the workers."""

    def __init__(self, n_workers):
        ctx = mp.get_context('spawn')  # fork is unsafe on macOS and with threaded libraries
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_main, args=(child,), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        self._sent = {}  # key -> static object currently cached by every worker (references keep ids unique)
        atexit.register(self.close)

    def set_state(self, sdc, forecast, forecast_days):
        statics = {id(obj): obj for obj in SimCloner.shared_objects(sdc).values()}
        keys = {i: _static_key(obj) for i, obj in statics.items()}
        buf = io.BytesIO()
        pickler = pickle.Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.persistent_id = lambda obj: keys.get(id(obj))
        pickler.dump(sdc)
        new = {keys[i]: obj for i, obj in statics.items() if keys[i] not in self._sent}
        self._sent = {keys[i]: obj for i, obj in statics.items()}
        msg = ('state', pickle.dumps(new, protocol=pickle.HIGHEST_PROTOCOL), list(self._sent), buf.getvalue(),
               forecast, forecast_days)
        for conn in self.conns:
            conn.send(msg)
        for conn in self.conns:
            self._recv(conn)

    @staticmethod
    def _recv(conn):
        status, payload = conn.recv()
        if status != 'ok':
            raise RuntimeError(f'MPC rollout worker failed: {payload}')
        return payload

    def evaluate(self, plans, params):
        chunks = [plans[i::len(self.conns)] for i in range(len(self.conns))]
        busy = []
        for conn, chunk in zip(self.conns, chunks):
            if chunk:
                conn.send(('eval', chunk, params))
                busy.append(conn)
        out = [None] * len(plans)
        for w, conn in enumerate(self.conns):
            if conn not in busy:
                continue
            out[w::len(self.conns)] = self._recv(conn)
        return out

    def close(self):
        for conn, proc in zip(self.conns, self.procs):
            if proc.is_alive():
                try:
                    conn.send(('close',))
                except (BrokenPipeError, OSError):
                    pass
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()
        self.conns, self.procs = [], []


# ---------------------------------------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------------------------------------
class MPCPolicy:
    """Receding-horizon CEM over the joint (ls, dc, bat) action. eval_compare policy interface:
    reset(seed) at the start of each episode, act(sdc) -> {agent: action} after each reset/step."""
    include_dc_setpoint = False
    train_config = {}

    def __init__(self, horizon=48, block=4, replan=None, samples=32, iters=3, elites=6, smoothing=0.7,
                 drop_w=10.0, overdue_w=1.0, forecast='oracle', forecast_days=1, workers=0, seed=0, verbose=False):
        """
        horizon:       planning horizon in 15-minute steps (rounded up to a whole number of blocks). 48 = 12 h.
        block:         steps each planned joint action is held (move blocking).
        replan:        steps executed from each plan before replanning (default: one block).
        samples:       candidate plans sampled per CEM iteration (on top of warm start / do-nothing / constant plans).
        iters:         CEM iterations per replan.
        elites:        lowest-cost candidates used to refit the block distributions.
        smoothing:     weight of the elite frequencies in the distribution update (1 = replace, 0 = never move).
        drop_w:        cost of a dropped task, in multiples of the CO2 of running it.
        overdue_w:     cost per step of each task older than 24 h, in multiples of the CO2 of running it.
        forecast:      'oracle' (true future exogenous inputs) or 'persistence' (past data only).
        forecast_days: days averaged by the persistence forecast.
        workers:       rollout processes (0 or 1: in-process). Results do not depend on it.
        seed:          added to the episode seed to seed the CEM sampling.
        """
        self.block = int(block)
        self.n_blocks = max(1, -(-int(horizon) // self.block))
        self.horizon = self.n_blocks * self.block
        self.replan = int(replan) if replan else self.block
        if not 1 <= self.replan <= self.horizon:
            raise ValueError(f'replan must be in [1, horizon={self.horizon}], got {self.replan}')
        if forecast not in FORECASTS:
            raise ValueError(f'forecast must be one of {FORECASTS}, got {forecast!r}')
        if int(forecast_days) < 1:
            raise ValueError(f'forecast_days must be >= 1, got {forecast_days}')
        self.samples, self.iters = int(samples), int(iters)
        self.elites = max(1, int(elites))
        self.smoothing = float(smoothing)
        self.drop_w, self.overdue_w = float(drop_w), float(overdue_w)
        self.forecast, self.forecast_days = forecast, int(forecast_days)
        self.workers = int(workers)
        self.seed_offset = int(seed)
        self.verbose = verbose
        self._pool = None
        self.reset(0)

    # -- episode ------------------------------------------------------------------------------------------------------
    def reset(self, seed):
        self.rng = np.random.default_rng(int(seed) + self.seed_offset)
        self.plan = None          # best plan from the last replan: joint action per block
        self.pending = []         # joint actions still to execute from the current plan
        self.stats = {'replans': 0, 'rollouts': 0, 'plan_seconds': 0.0}

    def act(self, sdc):
        return decode_joint_action(self.act_joint(sdc), 'discrete')

    def act_joint(self, sdc):
        """Next joint action as a Discrete(27) index (same encoding as SustainDCSingleAgentEnv, action_type='discrete')."""
        if not self.pending:
            self._replan(sdc)
        return int(self.pending.pop(0))

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # -- planning -----------------------------------------------------------------------------------------------------
    def _evaluator(self, sdc):
        """Build the planning model (here or in the workers) and return a function scoring a list of plans on it."""
        dc = sdc.dc_env
        task_kwh = marginal_task_kwh(sdc, float(sdc.ls_env.workload), float(dc.raw_curr_stpt),
                                     float(dc.ambient_temp), float(dc.wet_bulb))
        params = {'block': self.block, 'drop_w': self.drop_w, 'overdue_w': self.overdue_w, 'task_kwh': task_kwh,
                  'task_g': task_kwh * float(sdc.ci_m.carbon_smooth[sdc.ci_m.time_step])}  # current CI is known
        if self.workers > 1:
            if self._pool is None:
                self._pool = RolloutPool(self.workers)
            self._pool.set_state(sdc, self.forecast, self.forecast_days)
            return lambda plans: self._pool.evaluate(plans, params)
        model = planning_model(sdc, self.forecast, self.forecast_days)
        shared = SimCloner.shared_objects(model)
        return lambda plans: [rollout_cost(model, shared, p, params) for p in plans]

    def _replan(self, sdc):
        t0 = time.time()
        evaluate = self._evaluator(sdc)
        probs = self._initial_probs()
        seeds = [self._warm_start(), np.full(self.n_blocks, NOOP_JOINT)]
        seeds += [np.full(self.n_blocks, a) for a in range(N_JOINT)]
        best_plan, best_cost = None, np.inf
        seen = {}
        for it in range(self.iters):
            sampled = [self._sample(probs) for _ in range(self.samples)]
            plans = (seeds if it == 0 else [best_plan]) + sampled
            new = list({p.tobytes(): p for p in plans if p.tobytes() not in seen}.values())
            for p, c in zip(new, evaluate(new)):
                seen[p.tobytes()] = c
            costs = np.array([seen[p.tobytes()] for p in plans])
            order = np.argsort(costs, kind='stable')
            if costs[order[0]] < best_cost:
                best_cost, best_plan = float(costs[order[0]]), plans[order[0]].copy()
            elite = np.stack([plans[i] for i in order[:self.elites]])
            freq = np.stack([np.bincount(elite[:, b], minlength=N_JOINT) for b in range(self.n_blocks)]) / len(elite)
            probs = self.smoothing * freq + (1 - self.smoothing) * probs

        self.plan = best_plan
        self.pending = list(np.repeat(best_plan, self.block)[:self.replan].astype(int))
        self.stats['replans'] += 1
        self.stats['rollouts'] += len(seen)
        self.stats['plan_seconds'] += time.time() - t0
        if self.verbose:
            sub = [tuple(int(x) for x in np.unravel_index(a, N_SUB_ACTIONS)) for a in best_plan]
            print(f'  [mpc] t={sdc.t_m.current_timestep} cost={best_cost / 1e3:.1f} kg plan(ls,dc,bat)={sub} '
                  f'({len(seen)} rollouts, {time.time() - t0:.2f}s)', flush=True)

    def _initial_probs(self):
        """Uniform over the 27 joint actions, pulled toward the previous plan (shifted by the executed steps)."""
        probs = np.full((self.n_blocks, N_JOINT), 1.0 / N_JOINT)
        probs[np.arange(self.n_blocks), self._warm_start()] += 1.0
        return probs / probs.sum(axis=1, keepdims=True)

    def _warm_start(self):
        """Previous best plan shifted by the executed steps (rounded to blocks), padded with its last action."""
        if self.plan is None:
            return np.full(self.n_blocks, NOOP_JOINT)
        shift = min(-(-self.replan // self.block), self.n_blocks)
        return np.concatenate([self.plan[shift:], np.full(shift, self.plan[-1])]).astype(int)

    def _sample(self, probs):
        u = self.rng.random((self.n_blocks, 1))
        return (probs.cumsum(axis=1) < u).sum(axis=1).clip(0, N_JOINT - 1)


def make_mpc_policy(rest):
    """Build an MPCPolicy from 'key=value,...' (the part after 'mpc:' in an eval_compare spec)."""
    params = dict(kv.split('=', 1) for kv in rest.split(',') if kv) if rest else {}
    ints = {'horizon', 'block', 'replan', 'samples', 'iters', 'elites', 'seed', 'forecast_days', 'workers'}
    floats = {'smoothing', 'drop_w', 'overdue_w'}
    strings = {'forecast'}
    valid = ints | floats | strings | {'verbose'}
    unknown = set(params) - valid
    if unknown:
        raise ValueError(f'Unknown mpc parameter(s) {sorted(unknown)}; valid: {sorted(valid)}')
    kwargs = {}
    for k, v in params.items():
        if k in ints:
            kwargs[k] = int(v)
        elif k in floats:
            kwargs[k] = float(v)
        elif k == 'verbose':
            kwargs[k] = v.lower() in ('1', 'true', 'yes')
        else:
            kwargs[k] = v
    return MPCPolicy(**kwargs)
