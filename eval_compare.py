"""Compare policies (MARL, single-agent, baselines) on the same fixed set of SustainDC evaluation episodes.

Rewards are not comparable across setups (reward sharing, the global energy-normalization history), so every
policy is scored on physical metrics instead: CO2, IT / HVAC / total energy, water, dropped tasks and tasks
still queued at the end of the episode.

Each episode is fixed by its seed: `random` and `np.random` are seeded right before SustainDC.reset(), which is
where all of the simulator's randomness is drawn (start day/hour, workload/weather noise, initial task queue).
The harness checks that the exogenous inputs (workload, weather, carbon intensity) match across policies.

Run from the repo root, e.g.:
    python eval_compare.py --policy noop \\
        --policy sb3:results/sustaindc/ca/single_ppo/ppo_v1/seed-00001-job123 \\
        --policy harl:trained_models/sustaindc/ca/happo/happo3agents/seed-00001-2024-06-03-17-55-39 \\
        --n_episodes 5 --location ca --month 6

Policy specs:
    noop                          do nothing (ls: no shifting, dc: hold setpoint, bat: idle)
    random                        uniformly random actions (seeded per episode)
    sb3:<run_dir>[:<model>]       a train_single_agent.py run; <model> defaults to best_model
    harl:<run_dir>                a HARL run directory containing config.json and models/actor_agent*.pt;
                                  a glob (quote it) expands to one policy per run, labelled <label>_s<seed>
    rule:<subsystems>[:<params>]  rule-based controllers (past CI only), e.g. rule:ls,dc,bat or rule:bat:bat_lo=0.2
    oracle:<subsystems>[:<params>] perfect-forecast controllers (true future CI/workload/weather), e.g. oracle:ls,dc,bat
    setpoint:<C>                  hold the CRAC setpoint at <C>, everything else does nothing
Add =<label> to any spec to name it in the table, e.g. harl:<dir>=happo. See baseline_policies.py for the controllers.

Each policy is compared with `noop` episode by episode (paired), which removes the episode-to-episode variation:
the table reports the mean change and its 95% confidence interval over episodes. Runs labelled <label>_s<seed> are
also pooled per <label> (episode-wise mean over seeds).

Several result folders (e.g. from parallel jobs on the same seeds) can be merged afterwards:
    python eval_compare.py --combine results/eval_compare/a results/eval_compare/b --out_dir results/eval_compare/all
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import random
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import yaml

from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS, decode_joint_action
from utils.joint_obs import PV_OBS_DIM, build_joint_obs

ENV_CFG = 'harl/configs/envs_cfgs/sustaindc.yaml'
NOOP = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}
STEP_HOURS = 0.25  # 15-minute steps

# Config keys that define the simulated scenario. Policies may differ in observation settings, not in these.
SCENARIO_KEYS = ['location', 'month', 'days_per_episode', 'workload_file', 'cintensity_file', 'weather_file',
                 'flexible_load', 'max_bat_cap_Mw', 'dc_config_file', 'datacenter_capacity_mw', 'timezone_shift',
                 'pv_config', 'pv_export_credit', 'battery_config']

METRICS = [  # (name, unit)
    ('co2', 'kgCO2'),
    ('total_energy', 'MWh'),   # grid energy incl. battery charging/discharging (net of on-site PV)
    ('pv_energy', 'MWh'),      # on-site PV generation
    ('pv_curtailed', 'MWh'),   # PV surplus not used by the DC or the battery (curtailed or exported)
    ('it_energy', 'MWh'),
    ('hvac_energy', 'MWh'),
    ('water', 'L'),
    ('tasks_dropped', 'tasks'),
    ('tasks_queued_end', 'tasks'),
    ('avg_setpoint', 'C'),
    ('work_submitted', 'util-steps'),   # sum over steps of the original CPU utilization (1.0 = full DC for 15 min)
    ('work_executed', 'util-steps'),    # sum of the utilization actually run after load shifting
    ('co2_work_adjusted', 'kgCO2'),     # co2 + unexecuted work at its marginal energy x the episode's mean CI
]


# ---------------------------------------------------------------------------------------------------------------------
# Policies. Each one reads the SustainDC state after reset/step and returns {agent: int action}.
# ---------------------------------------------------------------------------------------------------------------------
class NoOpPolicy:
    include_dc_setpoint = False
    include_pv_obs = False
    train_config = {}

    def reset(self, seed):
        pass

    def act(self, sdc):
        return dict(NOOP)


class RandomPolicy(NoOpPolicy):
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, sdc):
        return {agent: int(self.rng.integers(3)) for agent in AGENTS}


class SB3Policy:
    def __init__(self, run_dir, model_name='best_model'):
        from train_single_agent import ALGOS
        with open(os.path.join(run_dir, 'config.json')) as f:
            cfg = json.load(f)
        self.train_config = cfg['env_args']
        self.include_dc_setpoint = bool(self.train_config.get('include_dc_setpoint', False))
        self.include_pv_obs = bool(self.train_config.get('include_pv_obs', False))
        self.action_type = self.train_config.get('action_type', 'multidiscrete')
        self.model = ALGOS[cfg['main_args']['algo']].load(os.path.join(run_dir, 'models', model_name), device='cpu')

    def reset(self, seed):
        pass

    def act(self, sdc):
        obs = build_joint_obs(sdc.ls_state, sdc.dc_state, sdc.bat_state, self.include_dc_setpoint, self.include_pv_obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return decode_joint_action(action, self.action_type)


class HARLPolicy:
    """Deterministic HARL actors loaded from a run directory, fed the same padded per-agent observations as in training."""
    ON_POLICY = ('happo', 'mappo', 'haa2c', 'hatrpo')
    OFF_POLICY = ('hasac', 'had3qn')

    def __init__(self, run_dir):
        from gymnasium import spaces
        from harl.algorithms.actors import ALGO_REGISTRY
        with open(os.path.join(run_dir, 'config.json')) as f:
            cfg = json.load(f)
        self.algo = cfg['main_args']['algo']
        if self.algo not in self.ON_POLICY + self.OFF_POLICY:
            raise ValueError(f"HARL algo {self.algo!r} not supported here (discrete-action algos: {self.ON_POLICY + self.OFF_POLICY})")
        self.train_config = cfg['env_args']
        self.include_dc_setpoint = bool(self.train_config.get('include_dc_setpoint', False))
        self.include_pv_obs = bool(self.train_config.get('include_pv_obs', False))
        self.agents = list(self.train_config['agents'])
        model_args, algo_args = cfg['algo_args']['model'], cfg['algo_args']['algo']
        self.recurrent = model_args.get('use_recurrent_policy', False) or model_args.get('use_naive_recurrent_policy', False)
        # RNN state shape only matters for recurrent on-policy actors; off-policy configs (e.g. HAD3QN) name their
        # layers base_hidden_sizes instead of hidden_sizes.
        hidden = model_args.get('hidden_sizes') or model_args.get('base_hidden_sizes') or [1]
        self.rnn_shape = (1, model_args.get('recurrent_n', 1), hidden[-1])

        # Supersuit pads every active agent's observation with zeros up to the largest one.
        n_pv = PV_OBS_DIM * int(self.include_pv_obs)
        obs_dims = {'agent_ls': 26 + n_pv, 'agent_dc': 14 + int(self.include_dc_setpoint) + n_pv, 'agent_bat': 13 + n_pv}
        self.pad_dim = max(obs_dims[a] for a in self.agents)
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.pad_dim,), dtype=np.float32)
        act_space = spaces.Discrete(3)

        self.actors = []
        for i, _ in enumerate(self.agents):
            actor = ALGO_REGISTRY[self.algo]({**model_args, **algo_args}, obs_space, act_space, device=torch.device('cpu'))
            state = torch.load(os.path.join(run_dir, 'models', f'actor_agent{i}.pt'), map_location='cpu')
            in_dim = next(v.shape[1] for k, v in state.items() if k.endswith('weight') and v.dim() == 2)
            if in_dim != self.pad_dim:
                raise ValueError(
                    f"{run_dir}: actor {i} expects {in_dim}-dim observations, but the current env gives "
                    f"{self.pad_dim} for this run's config. It was trained on an older observation layout and can't "
                    f"be evaluated on the current simulator (e.g. the 2024 trained_models/ use 14-dim observations).")
            actor.actor.load_state_dict(state)
            actor.actor.eval()
            self.actors.append(actor)

    def reset(self, seed):
        self.rnn_states = [np.zeros(self.rnn_shape, dtype=np.float32) for _ in self.agents]

    def _obs(self, sdc, agent):
        state = {'agent_ls': sdc.ls_state, 'agent_dc': sdc.dc_state, 'agent_bat': sdc.bat_state}[agent]
        if agent == 'agent_dc' and not self.include_dc_setpoint and sdc.include_dc_setpoint:
            state = np.delete(state, 14)  # the eval env appends the setpoint; this policy was trained without it
        padded = np.zeros((1, self.pad_dim), dtype=np.float32)
        padded[0, :len(state)] = state
        return padded

    @torch.no_grad()
    def act(self, sdc):
        actions = dict(NOOP)  # agents this run did not train keep the do-nothing action (as SustainDC does)
        masks = np.ones((1, 1), dtype=np.float32)
        for i, (agent, actor) in enumerate(zip(self.agents, self.actors)):
            obs = self._obs(sdc, agent)
            if self.algo in self.ON_POLICY:
                a, rnn = actor.act(obs, self.rnn_states[i], masks, None, deterministic=True)
                self.rnn_states[i] = rnn.cpu().numpy() if torch.is_tensor(rnn) else rnn
            elif self.algo == 'hasac':
                a = actor.get_actions(obs, None, stochastic=False)
            else:  # had3qn
                a = actor.get_actions(obs, epsilon_greedy=False)
            actions[agent] = int(np.asarray(a.cpu() if torch.is_tensor(a) else a).reshape(-1)[0])
        return actions


def split_label(spec):
    """'<spec>=<label>' -> (spec, label). Controller parameters also use '=' (rule:ls:ls_hi=0.8), so the label is the
    text after the last '=' unless that text is a number (then it is a parameter value and there is no label)."""
    head, sep, tail = spec.rpartition('=')
    if not sep:
        return spec, ''
    try:
        float(tail)
        return spec, ''
    except ValueError:
        return head, tail


def make_policy(spec):
    spec, label = split_label(spec)
    kind, _, rest = spec.partition(':')
    if kind == 'noop':
        policy = NoOpPolicy()
    elif kind == 'random':
        policy = RandomPolicy()
    elif kind == 'sb3':
        run_dir, _, model = rest.partition(':')
        policy = SB3Policy(run_dir, model or 'best_model')
    elif kind == 'harl':
        policy = HARLPolicy(rest)
    elif kind in ('rule', 'oracle', 'setpoint'):
        from baseline_policies import make_baseline_policy
        policy = make_baseline_policy(kind, rest)
    else:
        raise ValueError(f'Unknown policy spec {spec!r}')
    return label or spec, policy


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------------------------
def run_episode(sdc, policy, seed):
    """Run one seeded episode. Returns (metrics dict, fingerprint of the exogenous inputs)."""
    random.seed(seed)
    np.random.seed(seed)
    sdc.reset()
    policy.reset(seed)
    m = {name: 0.0 for name, _ in METRICS}
    exo = []
    steps = 0
    ci_sum = amb_sum = 0.0
    while True:
        _, _, _, truncs, info = sdc.step(policy.act(sdc))
        i = info['agent_ls']  # merged ls/dc/bat info
        m['co2'] += i['bat_CO2_footprint'] / 1e3
        m['total_energy'] += i['bat_total_energy_with_battery_KWh'] / 1e3
        m['pv_energy'] += i['pv_energy_KWh'] / 1e3
        m['pv_curtailed'] += (i['pv_curtailed_KWh'] + i['pv_exported_KWh']) / 1e3
        m['it_energy'] += i['dc_ITE_total_power_kW'] * STEP_HOURS / 1e3
        m['hvac_energy'] += i['dc_HVAC_total_power_kW'] * STEP_HOURS / 1e3
        m['water'] += i['dc_water_usage']
        m['tasks_dropped'] += i['ls_tasks_dropped']
        m['avg_setpoint'] += i['dc_crac_setpoint']
        m['work_submitted'] += i['ls_original_workload']
        m['work_executed'] += i['ls_shifted_workload']
        ci_sum += i['bat_avg_CI']
        amb_sum += i['outside_temp']
        exo.append((i['ls_original_workload'], i['outside_temp'], i['norm_CI'], i['day'], i['hour']))
        steps += 1
        if truncs['__all__']:
            break
    m['tasks_queued_end'] = float(i['ls_tasks_in_queue'])
    m['avg_setpoint'] /= steps
    # Work deferred but never run (dropped or still queued at the end) did not consume energy. Charge it at the
    # marginal energy of running it (IT + HVAC, at the mean utilization, 18 C, mean outdoor temperature) and the
    # episode's mean carbon intensity, so a policy cannot look greener by not doing the work.
    unexecuted = max(m['work_submitted'] - m['work_executed'], 0.0)
    m['co2_work_adjusted'] = m['co2'] + unexecuted * marginal_kwh_per_util_step(
        sdc, m['work_submitted'] / steps, amb_sum / steps) * (ci_sum / steps) / 1e3
    m['steps'] = steps
    fingerprint = hashlib.sha1(np.round(np.array(exo, dtype=np.float64), 9).tobytes()).hexdigest()[:12]
    return m, fingerprint


def marginal_kwh_per_util_step(sdc, utilization, ambient):
    """Energy (kWh) to run 1.0 utilization-step (the whole DC for 15 min) at the margin: slope of IT + HVAC power."""
    from baseline_policies import dc_power
    wb = sdc.dc_env.wet_bulb
    lo, hi = max(utilization - 0.05, 0.0), min(utilization + 0.05, 1.0)
    p_lo, p_hi = sum(dc_power(sdc, lo, 18.0, ambient, wb)), sum(dc_power(sdc, hi, 18.0, ambient, wb))
    return (p_hi - p_lo) / (hi - lo) / 1e3 * STEP_HOURS


def expand_specs(specs):
    """Expand harl:<glob> specs into one spec per run directory, labelled <label>_s<seed>."""
    out = []
    for spec in specs:
        body, label = split_label(spec)
        kind, _, rest = body.partition(':')
        if kind == 'harl' and any(ch in rest for ch in '*?['):
            dirs = sorted(d for d in glob.glob(rest) if os.path.isfile(os.path.join(d, 'config.json')))
            if not dirs:
                raise ValueError(f'No HARL run directories match {rest!r}')
            base = label or os.path.basename(os.path.dirname(dirs[0]))
            for d in dirs:
                with open(os.path.join(d, 'config.json')) as f:
                    seed = json.load(f)['algo_args']['seed']['seed']
                out.append(f'harl:{d}={base}_s{seed}')
        else:
            out.append(spec)
    return out


def paired_report(results, seeds, base='noop'):
    """Episode-by-episode comparison with the baseline: mean % change in CO2 / energy / water with a 95% CI (t-dist),
    plus pooled rows for runs labelled <label>_s<seed>. Returns a dict for summary.json."""
    from scipy import stats
    if base not in results:
        return {}
    b = results[base]

    def pct(eps, key):
        return np.array([100 * (e[key] / be[key] - 1) for e, be in zip(eps, b)])

    groups = {}
    for label in results:
        if label != base and '_s' in label and label.rsplit('_s', 1)[1].isdigit():
            groups.setdefault(label.rsplit('_s', 1)[0], []).append(label)
    rows = {label: results[label] for label in results if label != base}
    for g, labels in groups.items():
        if len(labels) > 1:  # pooled: per episode, average the metric over the runs (seeds) of the group
            rows[f'{g} (mean of {len(labels)} runs)'] = [
                {k: float(np.mean([results[l][i][k] for l in labels])) for k in results[labels[0]][i]}
                for i in range(len(seeds))]

    report = {}
    print(f'\nPaired comparison with {base} over {len(seeds)} episodes (mean change, 95% CI; negative = less):')
    print(f"  {'policy':44s}{'CO2 (work-adjusted)':>24s}{'CO2':>24s}{'total energy':>24s}{'water':>24s}{'dropped':>10s}{'queued end':>12s}")
    for label, eps in rows.items():
        cells, rep = [], {}
        for key in ('co2_work_adjusted', 'co2', 'total_energy', 'water'):
            d = pct(eps, key)
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
            rep[key] = {'mean_pct': float(d.mean()), 'ci95_pct': float(half), 'per_episode_pct': d.tolist()}
            cells.append(f'{d.mean():+7.2f}% ± {half:5.2f}')
        rep['tasks_dropped'] = float(np.mean([e['tasks_dropped'] for e in eps]))
        rep['tasks_queued_end'] = float(np.mean([e['tasks_queued_end'] for e in eps]))
        report[label] = rep
        print(f"  {label[:44]:44s}" + ''.join(f'{c:>24s}' for c in cells)
              + f"{rep['tasks_dropped']:10.0f}{rep['tasks_queued_end']:12.0f}")
    return report


def combine(dirs, out_dir):
    """Merge episodes.csv files from several eval_compare runs on the same seeds and report them together."""
    import csv
    results, fps, seeds = {}, {}, None
    for d in dirs:
        with open(os.path.join(d, 'episodes.csv')) as f:
            for row in csv.DictReader(f):
                label = row['policy']
                # Metrics added later (e.g. PV) are missing from older CSVs: read them as 0.
                results.setdefault(label, {})[int(row['seed'])] = {n: float(row.get(n, 0.0)) for n, _ in METRICS}
                fps.setdefault(int(row['seed']), set()).add(row['inputs_fingerprint'])
    seeds = sorted(next(iter(results.values())))
    if any(sorted(r) != seeds for r in results.values()):
        raise ValueError('Policies were evaluated on different seeds; cannot combine.')
    same_inputs = all(len(v) == 1 for v in fps.values())
    print(f"Combined {len(results)} policies from {len(dirs)} folders, {len(seeds)} episodes (seeds {seeds[0]}..{seeds[-1]}). "
          f"Exogenous inputs identical across policies: {'yes' if same_inputs else 'NO, comparison is not fair'}")
    results = {label: [r[s] for s in seeds] for label, r in results.items()}
    summary = {label: {name: (float(np.mean([e[name] for e in eps])), float(np.std([e[name] for e in eps])))
                       for name, _ in METRICS} for label, eps in results.items()}
    width = max(len(l) for l in summary) + 2
    print('\n' + 'policy'.ljust(width) + ''.join(f'{f"{n} ({u})":>26}' for n, u in METRICS))
    for label, s in summary.items():
        print(label.ljust(width) + ''.join(f'{f"{mu:.1f} ± {sd:.1f}":>26}' for mu, sd in s.values()))
    paired = paired_report(results, seeds)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'sources': dirs, 'seeds': seeds, 'inputs_identical': same_inputs, 'summary': summary,
                   'paired_vs_noop': paired}, f, indent=2)
    print(f'\nSaved {out_dir}/summary.json')


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument('--policy', action='append', help='Policy spec (repeatable), see above.')
    p.add_argument('--combine', nargs='+', default=None, help='Merge the episodes.csv of these eval_compare folders.')
    p.add_argument('--n_episodes', type=int, default=5)
    p.add_argument('--seed', type=int, default=1000, help='Episode i uses seed + i.')
    p.add_argument('--location', default=None)
    p.add_argument('--month', type=int, default=None)
    p.add_argument('--days_per_episode', type=int, default=None)
    p.add_argument('--pv_capacity_kWp', type=float, default=None,
                   help="Override the env config's on-site PV capacity (pv_config.capacity_kWp; 0 = no PV).")
    p.add_argument('--out_dir', default=None, help='Default: results/eval_compare/<timestamp>')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.combine:
        combine(args.combine, args.out_dir or os.path.join('results', 'eval_compare', 'combined'))
        return
    if not args.policy:
        p.error('--policy is required (or use --combine)')

    with open(ENV_CFG) as f:
        scenario = yaml.safe_load(f)
    for key in ['location', 'month', 'days_per_episode']:
        if getattr(args, key) is not None:
            scenario[key] = getattr(args, key)
    if args.pv_capacity_kWp is not None:
        scenario['pv_config'] = {**scenario.get('pv_config', {}), 'capacity_kWp': args.pv_capacity_kWp}
    seeds = [args.seed + i for i in range(args.n_episodes)]

    policies = [make_policy(s) for s in expand_specs(args.policy)]
    for label, pol in policies:
        diff = {k: (pol.train_config[k], scenario.get(k)) for k in SCENARIO_KEYS
                if k in pol.train_config and pol.train_config[k] != scenario.get(k)}
        if diff:
            print(f"Note: '{label}' was trained on a different scenario (trained vs eval): {diff}")

    print(f"Scenario: location={scenario['location']} month={scenario['month']} days={scenario['days_per_episode']} | "
          f"{len(seeds)} episodes, seeds {seeds[0]}..{seeds[-1]}")
    results, fingerprints = {}, {}
    for label, pol in policies:
        # Observation flags follow the policy; everything that shapes the episode comes from the scenario.
        sdc = SustainDC({**scenario, 'agents': list(AGENTS), 'include_dc_setpoint': pol.include_dc_setpoint,
                         'include_pv_obs': pol.include_pv_obs})
        results[label], fingerprints[label] = [], []
        for seed in seeds:
            m, fp = run_episode(sdc, pol, seed)
            results[label].append(m); fingerprints[label].append(fp)
            print(f"  {label} seed {seed}: CO2 {m['co2']:.1f} kg, dropped {m['tasks_dropped']:.0f}, "
                  f"queued at end {m['tasks_queued_end']:.0f}", flush=True)

    ref = next(iter(fingerprints.values()))
    same_inputs = all(fp == ref for fp in fingerprints.values())
    print('\nExogenous inputs identical across policies: ' + ('yes' if same_inputs else 'NO, comparison is not fair'))

    summary = {label: {name: (float(np.mean([e[name] for e in eps])), float(np.std([e[name] for e in eps])))
                       for name, _ in METRICS} for label, eps in results.items()}
    width = max(len(l) for l in summary) + 2
    header = ''.join(f'{f"{n} ({u})":>26}' for n, u in METRICS)
    print('\n' + 'policy'.ljust(width) + header)
    for label, s in summary.items():
        print(label.ljust(width) + ''.join(f'{f"{mu:.1f} ± {sd:.1f}":>26}' for mu, sd in s.values()))
    base = next((l for l, pol in policies if isinstance(pol, NoOpPolicy) and type(pol) is NoOpPolicy), None)
    if base:
        print(f'\nChange vs {base} (mean over episodes; negative = less):')
        for label, s in summary.items():
            if label == base:
                continue
            ch = [f"{n} {100 * (s[n][0] / summary[base][n][0] - 1):+.1f}%" for n in ('co2', 'total_energy', 'water')
                  if summary[base][n][0]]
            print(f"  {label}: {', '.join(ch)} | dropped {s['tasks_dropped'][0]:.0f} vs {summary[base]['tasks_dropped'][0]:.0f}, "
                  f"queued at end {s['tasks_queued_end'][0]:.0f} vs {summary[base]['tasks_queued_end'][0]:.0f}")

    paired = paired_report(results, seeds) if any(l == 'noop' for l, _ in policies) else {}

    out_dir = args.out_dir or os.path.join('results', 'eval_compare', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'episodes.csv'), 'w') as f:
        f.write('policy,seed,' + ','.join(n for n, _ in METRICS) + ',inputs_fingerprint\n')
        for label, eps in results.items():
            for seed, e, fp in zip(seeds, eps, fingerprints[label]):
                f.write(f'{label},{seed},' + ','.join(f'{e[n]:.6g}' for n, _ in METRICS) + f',{fp}\n')
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'scenario': scenario, 'seeds': seeds, 'policies': args.policy,
                   'inputs_identical': same_inputs, 'summary': summary, 'paired_vs_noop': paired}, f, indent=2)
    print(f'\nSaved {out_dir}/episodes.csv and summary.json')


if __name__ == '__main__':
    main()
