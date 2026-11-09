"""Rule-based and perfect-forecast ("oracle") controllers for SustainDC, used by eval_compare.py to measure how much
CO2 reduction is achievable before training RL agents.

Each subsystem (load shifting `ls`, cooling `dc`, battery `bat`) has its own controller; ComposedPolicy combines one
controller per subsystem (unspecified subsystems do nothing). Controllers read the SustainDC object directly:

* rule-based controllers use only information an agent could have: the current state and the *past* carbon
  intensity (CI);
* oracle controllers additionally use the *true future* exogenous inputs (CI, workload, weather) of the episode,
  so they give an optimistic estimate of what is achievable (an upper bound on savings, up to the heuristics used).

Timing convention: when act() is called, the step about to be simulated uses the managers' *current* index
(ci_m.time_step, weather_m.time_step, workload_m.time_step); they advance by one after each step.

Policy specs (see eval_compare.py):
    rule:ls,dc,bat      rule-based controllers for the listed subsystems (others do nothing)
    oracle:ls,dc,bat    perfect-forecast controllers for the listed subsystems
    setpoint:<C>        hold the CRAC setpoint at <C> (moves toward it at the start), everything else does nothing
Parameters (thresholds, fixed setpoint) can be set with key=value after the subsystem list, e.g.
    rule:ls,dc,bat:ls_hi=0.6,ls_lo=0.4,dc_setpoint=18,bat_lo=0.3,bat_hi=0.7
"""
import numpy as np

import envs.datacenter as DataCenter

NOOP = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}
DAY = 96  # 15-minute steps per day
SETPOINT_GRID = (15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 21.6)


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------
def current_ci(sdc):
    return float(sdc.ci_m.carbon_smooth[sdc.ci_m.time_step])


def past_ci(sdc, n=DAY):
    t = sdc.ci_m.time_step
    return np.asarray(sdc.ci_m.carbon_smooth[max(0, t - n):t])


def future_ci(sdc, n=DAY):
    t = sdc.ci_m.time_step
    return np.asarray(sdc.ci_m.carbon_smooth[t:t + n])


def dc_power(sdc, load_frac, setpoint, ambient, wet_bulb):
    """IT and HVAC power (W) of the data center at a CPU load fraction and CRAC setpoint (same model as dc_gym.step)."""
    dc = sdc.dc_env
    cfg = dc.DC_Config
    cpu, fan, outlet = dc.dc.compute_datacenter_IT_load_outlet_temp([load_frac * 100] * cfg.NUM_RACKS, setpoint)
    it = sum(cpu) + sum(fan)
    ret = DataCenter.calculate_avg_CRAC_return_temp(cfg.RACK_RETURN_APPROACH_TEMP_LIST, outlet)
    _, ct_fan, _, chiller, _, _ = DataCenter.calculate_HVAC_power(setpoint, ret, ambient, it, cfg, wet_bulb_temp=wet_bulb)
    return it, ct_fan + chiller


def setpoint_action(current, target):
    """Cooling action (0: -1 C, 1: hold, 2: +1 C) moving the setpoint toward the target."""
    if target > current + 1e-9:
        return 2
    if target < current - 1e-9:
        return 0
    return 1


def expected_utilization(sdc, ls_action):
    """Approximate CPU utilization this step after the load-shifting action (used by the cooling controller)."""
    ls = sdc.ls_env
    w = float(ls.workload)
    if ls_action == 0:
        return w * (1 - ls.flexible_workload_ratio)
    if ls_action == 2:
        return min(0.9, w + min(len(ls.tasks_queue), int(w * ls.flexible_workload_ratio * 100)) / 100)
    return w


# ---------------------------------------------------------------------------------------------------------------------
# Load shifting
# ---------------------------------------------------------------------------------------------------------------------
def queue_can_take(sdc):
    """True if deferring this step's flexible tasks fits in the queue (otherwise the env drops the excess)."""
    ls = sdc.ls_env
    shiftable = int(np.floor(float(ls.workload) * ls.flexible_workload_ratio * 100))
    return len(ls.tasks_queue) + shiftable <= ls.queue_max_len


def oldest_task_age_hours(sdc):
    ls = sdc.ls_env
    if not ls.tasks_queue:
        return 0.0
    t = ls.tasks_queue[0]
    return (ls.current_day - t['day']) * 24 + (ls.current_hour - t['hour'])


class LSRule:
    """Defer flexible work when CI is above the `hi` quantile of the past day; run queued work when CI is below the
    `lo` quantile. Queue-aware: never defers more than the queue can hold (no dropped tasks) and runs the oldest work
    once it is `deadline` hours old, before the env would force it at whatever CI it happens to be (24 h)."""

    def __init__(self, hi=0.6, lo=0.4, deadline=20.0):
        self.hi, self.lo, self.deadline = hi, lo, deadline

    def init(self, sdc):
        pass

    def reference_ci(self, sdc):
        return past_ci(sdc)

    def act(self, sdc):
        ref = self.reference_ci(sdc)
        if len(ref) < DAY // 2:
            return 1
        ci = current_ci(sdc)
        queued = len(sdc.ls_env.tasks_queue) > 0
        if queued and oldest_task_age_hours(sdc) >= self.deadline:
            return 2
        if ci > np.quantile(ref, self.hi) and queue_can_take(sdc):
            return 0
        if queued and ci <= np.quantile(ref, self.lo):
            return 2
        return 1


class LSOracle(LSRule):
    """Same queue-aware rule, but against the true CI of the *next* day (deferred work must run within 24 h): defer when
    cheaper hours are coming (CI above the `hi` quantile of the next day), run queued work in the cheap hours (`lo`)."""

    def __init__(self, hi=0.5, lo=0.3, deadline=20.0):
        super().__init__(hi, lo, deadline)

    def reference_ci(self, sdc):
        return future_ci(sdc)


# ---------------------------------------------------------------------------------------------------------------------
# Cooling
# ---------------------------------------------------------------------------------------------------------------------
class DCFixed:
    """Hold a fixed CRAC setpoint (the setpoint starts at 18 C, so it first moves toward the target)."""

    def __init__(self, setpoint=18.0):
        self.target = float(setpoint)

    def init(self, sdc):
        pass

    def act(self, sdc, ls_action=1):
        return setpoint_action(sdc.dc_env.raw_curr_stpt, self.target)


class DCOracle:
    """Each step, move toward the setpoint that minimizes IT + HVAC power under the current load and weather.
    The cooling model has no thermal state (only the setpoint carries over), so this greedy choice is near-optimal."""

    def init(self, sdc):
        pass

    def act(self, sdc, ls_action=1):
        load = expected_utilization(sdc, ls_action)
        amb, wb = sdc.dc_env.ambient_temp, sdc.dc_env.wet_bulb
        best = min(SETPOINT_GRID, key=lambda s: sum(dc_power(sdc, load, s, amb, wb)))
        return setpoint_action(sdc.dc_env.raw_curr_stpt, best)


# ---------------------------------------------------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------------------------------------------------
def battery_transition(soc, cap, action, dc_load_mwh):
    """Vectorized replica of BatteryEnvFwd + Battery2 for one 15-minute step.

    soc: stored energy (MWh, array); action: 'charge' | 'discharge' | 'idle'; dc_load_mwh: this step's DC energy cap
    on discharge (dcload / 4). Returns (next soc, grid energy change in MWh: + when charging, - when discharging).
    """
    soc = np.asarray(soc, dtype=float)
    frac = soc / cap
    if action == 'charge':
        rate = np.round(0.5 * (1 - 1 / (1 + np.exp(-10 * (frac - 0.5)))), 4)
        t_u = rate * 15 / 60
        max_charge = np.minimum(cap * 0.1, (cap - soc) / (t_u + 0.04))
        added = np.minimum(max_charge, cap) * t_u
        return np.round(soc + added, 8), added
    if action == 'discharge':
        rate = np.maximum(0.5, 4 / (1 + np.exp(-10 * (frac - 0.25))))
        t_u = rate * 15 / 60
        max_dis = np.minimum(np.minimum(cap, soc / (0.01 + t_u)), dc_load_mwh)
        nxt = np.round(soc - np.minimum(max_dis, cap) * t_u, 8)
        delivered = np.where(max_dis < cap, max_dis * t_u, cap * t_u)
        return nxt, -delivered
    return soc, np.zeros_like(soc)


BAT_ACTIONS = {0: 'charge', 1: 'discharge', 2: 'idle'}


class BatRule:
    """Charge when CI is below the `lo` quantile of the past day, discharge above the `hi` quantile, else idle."""

    def __init__(self, lo=0.3, hi=0.7):
        self.lo, self.hi = lo, hi

    def init(self, sdc):
        pass

    def act(self, sdc):
        hist = past_ci(sdc)
        if len(hist) < DAY // 2:
            return 2
        ci = current_ci(sdc)
        if ci < np.quantile(hist, self.lo):
            return 0
        if ci > np.quantile(hist, self.hi):
            return 1
        return 2


class BatOracle:
    """Dynamic programming over the battery's stored energy with the true CI and an estimated DC load for the whole
    episode (load at a 18 C setpoint without load shifting). Minimizes sum_t CI_t * (grid energy added by the battery)."""

    def __init__(self, grid_points=201):
        self.g = grid_points

    def init(self, sdc):
        bat = sdc.bat_env
        self.cap = float(bat.max_bat_cap)
        self.t0 = sdc.t_m.current_timestep
        n = int(sdc.t_m.total_timesteps - sdc.t_m.current_timestep)
        ci = np.asarray(sdc.ci_m.carbon_smooth[sdc.ci_m.time_step:sdc.ci_m.time_step + n])
        wl = np.asarray(sdc.workload_m.cpu_smooth[sdc.workload_m.time_step:sdc.workload_m.time_step + n])
        amb = np.asarray(sdc.weather_m.temperature_data[sdc.weather_m.time_step:sdc.weather_m.time_step + n])
        wb = np.asarray(sdc.weather_m.wet_bulb_data[sdc.weather_m.time_step:sdc.weather_m.time_step + n])
        n = min(len(ci), len(wl), len(amb), len(wb))
        self.ci = ci[:n]
        self.load_mwh = np.array([sum(dc_power(sdc, float(wl[k]), 18.0, float(amb[k]), float(wb[k]))) / 1e6 / 4
                                  for k in range(n)])
        grid = np.linspace(0, self.cap, self.g)
        self.grid = grid
        V = np.zeros((n + 1, self.g))
        for k in range(n - 1, -1, -1):
            best = np.full(self.g, np.inf)
            for a in ('charge', 'discharge', 'idle'):
                nxt, de = battery_transition(grid, self.cap, a, self.load_mwh[k])
                best = np.minimum(best, self.ci[k] * de * 1e3 + np.interp(nxt, grid, V[k + 1]))
            V[k] = best
        self.V = V

    def act(self, sdc):
        k = int(sdc.t_m.current_timestep - self.t0)
        if k >= len(self.ci):
            return 2
        soc = float(sdc.bat_env.battery.current_load)
        cost = {}
        for a_id, a in BAT_ACTIONS.items():
            nxt, de = battery_transition(np.array([soc]), self.cap, a, self.load_mwh[k])
            cost[a_id] = float(self.ci[k] * de[0] * 1e3 + np.interp(nxt[0], self.grid, self.V[k + 1]))
        return min(cost, key=cost.get)


# ---------------------------------------------------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------------------------------------------------
class ComposedPolicy:
    """One controller per subsystem; subsystems without a controller do nothing. Same interface as eval_compare's
    policies: reset(seed) at the start of each episode, act(sdc) -> {agent: action}."""
    include_dc_setpoint = False
    train_config = {}

    def __init__(self, ls=None, dc=None, bat=None):
        self.ls, self.dc, self.bat = ls, dc, bat

    def reset(self, seed):
        self._initialized = False

    def act(self, sdc):
        if not self._initialized:
            for c in (self.ls, self.dc, self.bat):
                if c is not None:
                    c.init(sdc)
            self._initialized = True
        actions = dict(NOOP)
        if self.ls is not None:
            actions['agent_ls'] = int(self.ls.act(sdc))
        if self.dc is not None:
            actions['agent_dc'] = int(self.dc.act(sdc, actions['agent_ls']))
        if self.bat is not None:
            actions['agent_bat'] = int(self.bat.act(sdc))
        return actions


def make_baseline_policy(kind, rest):
    """Build a policy from a spec like 'rule:ls,dc:ls_hi=0.6,dc_setpoint=19' or 'oracle:bat' or 'setpoint:20'."""
    if kind == 'setpoint':
        return ComposedPolicy(dc=DCFixed(float(rest)))
    parts = rest.split(':')
    subsystems = [s for s in parts[0].split(',') if s] if parts[0] else ['ls', 'dc', 'bat']
    params = dict(kv.split('=') for kv in parts[1].split(',')) if len(parts) > 1 and parts[1] else {}
    p = {k: float(v) for k, v in params.items()}
    unknown = set(subsystems) - {'ls', 'dc', 'bat'}
    if unknown:
        raise ValueError(f"Unknown subsystem(s) {unknown}; use ls, dc, bat")
    if kind == 'rule':
        return ComposedPolicy(
            ls=LSRule(p.get('ls_hi', 0.6), p.get('ls_lo', 0.4), p.get('ls_deadline', 20.0)) if 'ls' in subsystems else None,
            dc=DCFixed(p.get('dc_setpoint', 18.0)) if 'dc' in subsystems else None,
            bat=BatRule(p.get('bat_lo', 0.3), p.get('bat_hi', 0.7)) if 'bat' in subsystems else None)
    if kind == 'oracle':
        return ComposedPolicy(
            ls=LSOracle(p.get('ls_hi', 0.5), p.get('ls_lo', 0.3), p.get('ls_deadline', 20.0)) if 'ls' in subsystems else None,
            dc=DCOracle() if 'dc' in subsystems else None,
            bat=BatOracle(int(p.get('bat_grid', 201))) if 'bat' in subsystems else None)
    raise ValueError(f"Unknown baseline policy kind {kind!r}")
