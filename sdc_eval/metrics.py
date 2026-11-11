"""Per-episode physical metrics: one registry of names and units, one accumulator, used by every algorithm.

Rewards are not comparable across setups (reward functions, reward sharing, the process-wide energy normalization
history), so algorithms are compared on these physical metrics. Training curves (rollout/*), in-training evaluation
(eval/*) and the final comparison (eval_compare.py) all compute them with EpisodeAccumulator, so a number means the
same thing everywhere.
"""
import hashlib
from collections import namedtuple

import numpy as np

from sdc_eval.constants import AGENTS, N_ACTIONS, STEP_HOURS, TASK_UTILIZATION

Metric = namedtuple('Metric', 'name unit lower_is_better description')

# Order matters: the first block is the historical eval_compare.py column order (episodes.csv stays compatible).
METRIC_SPECS = [
    Metric('co2', 'kgCO2', True, 'grid CO2 (energy drawn from the grid x carbon intensity)'),
    Metric('total_energy', 'MWh', True, 'grid energy incl. battery charging/discharging (net of on-site PV)'),
    Metric('pv_energy', 'MWh', False, 'on-site PV generation'),
    Metric('pv_curtailed', 'MWh', True, 'PV surplus curtailed (not used by the DC, stored or exported); '
                                          'before sdc_eval it included pv_exported'),
    Metric('it_energy', 'MWh', None, 'IT energy'),
    Metric('hvac_energy', 'MWh', True, 'cooling energy (cooling tower + chiller)'),
    Metric('water', 'L', True, 'cooling-tower water use'),
    Metric('tasks_dropped', 'tasks', True, 'tasks dropped by load shifting (queue full or too old)'),
    Metric('tasks_queued_end', 'tasks', True, 'tasks still deferred at the end of the episode'),
    Metric('avg_setpoint', 'C', None, 'mean CRAC setpoint'),
    Metric('work_submitted', 'util-steps', None, 'sum over steps of the original utilization (1.0 = full DC, 15 min)'),
    Metric('work_executed', 'util-steps', None, 'sum over steps of the utilization actually run'),
    Metric('co2_work_adjusted', 'kgCO2', True, 'co2 + unexecuted work at its marginal energy x the mean CI'),
    # Added with sdc_eval.
    Metric('pue', '-', True, '(IT + HVAC energy) / IT energy over the episode'),
    Metric('mean_ci', 'gCO2/kWh', None, 'mean grid carbon intensity over the episode (exogenous)'),
    Metric('effective_ci', 'gCO2/kWh', True, 'co2 / total_energy: carbon intensity of the energy actually drawn'),
    Metric('bat_soc_mean', '-', None, 'mean battery state of charge'),
    Metric('bat_discharged', 'MWh', None, 'energy delivered by the battery to the DC'),
    Metric('bat_charged', 'MWh', None, 'energy drawn to charge the battery (grid and/or PV, incl. losses)'),
    Metric('bat_cycles', 'cycles', None, 'equivalent full cycles: bat_discharged / capacity'),
    Metric('pv_self_consumed', 'MWh', False, 'PV used directly by the DC'),
    Metric('pv_exported', 'MWh', None, 'PV surplus exported to the grid (credited in co2 at the grid CI)'),
    Metric('task_age_mean', 'h', True, 'mean over steps of the average age of the deferred tasks'),
    Metric('task_age_max', 'h', True, 'oldest deferred task seen during the episode'),
    Metric('return_team', '-', False, 'sum of the mean of the three agents\' rewards (reward-function dependent)'),
    Metric('return_ls', '-', False, 'sum of the load-shifting agent\'s own reward'),
    Metric('return_dc', '-', False, 'sum of the cooling agent\'s own reward'),
    Metric('return_bat', '-', False, 'sum of the battery agent\'s own reward'),
] + [Metric(f'act_{a[6:]}_{k}', 'share', None, f'share of steps where {a} took action {k}')
     for a in AGENTS for k in range(N_ACTIONS)]

METRICS = [(m.name, m.unit) for m in METRIC_SPECS]  # (name, unit), the historical eval_compare.py format
METRIC_NAMES = [m.name for m in METRIC_SPECS]
SPEC_BY_NAME = {m.name: m for m in METRIC_SPECS}

# The metric used to choose the best model during training (lower is better), and the ones reported vs noop.
SELECTION_METRIC = 'co2_work_adjusted'
PAIRED_METRICS = ('co2_work_adjusted', 'co2', 'total_energy', 'water')


class EpisodeAccumulator:
    """Accumulates the metrics of one SustainDC episode from the merged per-step info dict.

        acc = EpisodeAccumulator()
        ... each step: acc.step(info['agent_ls'], actions, (sdc.ls_reward, sdc.dc_reward, sdc.bat_reward))
        metrics = acc.finalize(sdc)

    `actions` and `rewards` are optional (the training-env wrappers may not have them).
    """

    def __init__(self, record_inputs=True):
        self.m = {name: 0.0 for name in METRIC_NAMES}
        self.steps = 0
        self.ci_sum = self.amb_sum = 0.0
        self.record_inputs = record_inputs
        self.exo = []
        self.act_counts = np.zeros((len(AGENTS), N_ACTIONS))
        self.n_actions = 0
        self.has_rewards = False
        self.bat_capacity = 0.0
        self.last = None

    def step(self, i, actions=None, rewards=None):
        m = self.m
        m['co2'] += i['bat_CO2_footprint'] / 1e3
        m['total_energy'] += i['bat_total_energy_with_battery_KWh'] / 1e3
        m['pv_energy'] += i['pv_energy_KWh'] / 1e3
        m['pv_curtailed'] += i['pv_curtailed_KWh'] / 1e3
        m['pv_exported'] += i['pv_exported_KWh'] / 1e3
        m['it_energy'] += i['dc_ITE_total_power_kW'] * STEP_HOURS / 1e3
        m['hvac_energy'] += i['dc_HVAC_total_power_kW'] * STEP_HOURS / 1e3
        m['water'] += i['dc_water_usage']
        m['tasks_dropped'] += i['ls_tasks_dropped']
        m['avg_setpoint'] += i['dc_crac_setpoint']
        m['work_submitted'] += i['ls_original_workload']
        m['work_executed'] += i['ls_shifted_workload']
        m['bat_soc_mean'] += i['bat_SOC']
        m['bat_discharged'] += i['bat_energy_out_KWh'] / 1e3
        m['bat_charged'] += i['bat_energy_in_KWh'] / 1e3
        m['pv_self_consumed'] += i['pv_self_consumed_KWh'] / 1e3
        m['task_age_mean'] += i['ls_average_task_age']
        m['task_age_max'] = max(m['task_age_max'], i['ls_oldest_task_age'])
        self.bat_capacity = i['bat_max_bat_cap']
        self.ci_sum += i['bat_avg_CI']
        self.amb_sum += i['outside_temp']
        if self.record_inputs:
            self.exo.append((i['ls_original_workload'], i['outside_temp'], i['norm_CI'], i['day'], i['hour']))
        if actions is not None:
            for k, agent in enumerate(AGENTS):
                if agent in actions:
                    self.act_counts[k, int(actions[agent])] += 1
            self.n_actions += 1
        if rewards is not None:
            r_ls, r_dc, r_bat = (float(r) for r in rewards)
            m['return_ls'] += r_ls
            m['return_dc'] += r_dc
            m['return_bat'] += r_bat
            m['return_team'] += (r_ls + r_dc + r_bat) / 3
            self.has_rewards = True
        self.steps += 1
        self.last = i

    def finalize(self, sdc=None):
        """Episode metrics. `sdc` (the SustainDC env) is needed for co2_work_adjusted; without it, that metric is
        NaN (e.g. in vectorized training envs where the env object is in another process)."""
        m, steps, i = dict(self.m), self.steps, self.last
        m['tasks_queued_end'] = float(i['ls_tasks_in_queue'])
        m['avg_setpoint'] /= steps
        # Work deferred but never run (dropped or still queued at the end) did not consume energy. Charge it at the
        # marginal energy of running it and the episode's mean carbon intensity, so a policy cannot look greener by
        # not doing the work.
        unexecuted = max(m['work_submitted'] - m['work_executed'], 0.0)
        if sdc is not None:
            m['co2_work_adjusted'] = m['co2'] + unexecuted * marginal_kwh_per_util_step(
                sdc, m['work_submitted'] / steps, self.amb_sum / steps) * (self.ci_sum / steps) / 1e3
        else:
            m['co2_work_adjusted'] = float('nan')
        m['pue'] = (m['it_energy'] + m['hvac_energy']) / m['it_energy'] if m['it_energy'] else float('nan')
        m['mean_ci'] = self.ci_sum / steps
        m['effective_ci'] = m['co2'] / m['total_energy'] if m['total_energy'] else float('nan')  # kg/MWh = g/kWh
        m['bat_soc_mean'] /= steps
        m['bat_cycles'] = m['bat_discharged'] / self.bat_capacity if self.bat_capacity else 0.0
        m['task_age_mean'] /= steps
        if not self.has_rewards:
            for k in ('return_team', 'return_ls', 'return_dc', 'return_bat'):
                m[k] = float('nan')
        for k, agent in enumerate(AGENTS):
            for a in range(N_ACTIONS):
                m[f'act_{agent[6:]}_{a}'] = (self.act_counts[k, a] / self.n_actions) if self.n_actions else float('nan')
        m['steps'] = steps
        return m

    def fingerprint(self):
        """Hash of the exogenous inputs (workload, weather, CI, calendar): equal for two episodes iff every policy
        saw the same inputs."""
        return hashlib.sha1(np.round(np.array(self.exo, dtype=np.float64), 9).tobytes()).hexdigest()[:12]


def marginal_kwh_per_util_step(sdc, utilization, ambient):
    """Energy (kWh) to run 1.0 utilization-step (the whole DC for 15 min) at the margin: slope of IT + HVAC power
    at the given mean utilization, an 18 C setpoint and the given ambient temperature."""
    from baseline_policies import dc_power
    wb = sdc.dc_env.wet_bulb
    lo, hi = max(utilization - 0.05, 0.0), min(utilization + 0.05, 1.0)
    p_lo, p_hi = sum(dc_power(sdc, lo, 18.0, ambient, wb)), sum(dc_power(sdc, hi, 18.0, ambient, wb))
    return (p_hi - p_lo) / (hi - lo) / 1e3 * STEP_HOURS


def task_marginal_kwh(sdc, utilization, ambient):
    """Marginal energy (kWh) of one task (TASK_UTILIZATION for one step) at the given utilization and ambient
    temperature: the one definition used by the reward (co2_team_reward, per step), the MPC and the evaluation
    (co2_work_adjusted, at the episode's mean conditions)."""
    return marginal_kwh_per_util_step(sdc, utilization, ambient) * TASK_UTILIZATION


def aggregate(episodes):
    """Mean and std over episodes of every metric -> {name: (mean, std)}. NaNs are ignored."""
    out = {}
    for name in METRIC_NAMES:
        vals = np.array([e[name] for e in episodes], dtype=np.float64)
        ok = vals[~np.isnan(vals)]
        out[name] = (float(ok.mean()), float(ok.std())) if len(ok) else (float('nan'), float('nan'))
    return out
