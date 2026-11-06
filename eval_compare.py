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
    harl:<run_dir>                a HARL run directory containing config.json and models/actor_agent*.pt
Add =<label> to any spec to name it in the table, e.g. harl:<dir>=happo.
"""
import argparse
import datetime
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
from utils.joint_obs import build_joint_obs

ENV_CFG = 'harl/configs/envs_cfgs/sustaindc.yaml'
NOOP = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}
STEP_HOURS = 0.25  # 15-minute steps

# Config keys that define the simulated scenario. Policies may differ in observation settings, not in these.
SCENARIO_KEYS = ['location', 'month', 'days_per_episode', 'workload_file', 'cintensity_file', 'weather_file',
                 'flexible_load', 'max_bat_cap_Mw', 'dc_config_file', 'datacenter_capacity_mw', 'timezone_shift']

METRICS = [  # (name, unit)
    ('co2', 'kgCO2'),
    ('total_energy', 'MWh'),   # grid energy incl. battery charging/discharging
    ('it_energy', 'MWh'),
    ('hvac_energy', 'MWh'),
    ('water', 'L'),
    ('tasks_dropped', 'tasks'),
    ('tasks_queued_end', 'tasks'),
    ('avg_setpoint', 'C'),
]


# ---------------------------------------------------------------------------------------------------------------------
# Policies. Each one reads the SustainDC state after reset/step and returns {agent: int action}.
# ---------------------------------------------------------------------------------------------------------------------
class NoOpPolicy:
    include_dc_setpoint = False
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
        self.action_type = self.train_config.get('action_type', 'multidiscrete')
        self.model = ALGOS[cfg['main_args']['algo']].load(os.path.join(run_dir, 'models', model_name), device='cpu')

    def reset(self, seed):
        pass

    def act(self, sdc):
        obs = build_joint_obs(sdc.ls_state, sdc.dc_state, sdc.bat_state, self.include_dc_setpoint)
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
        self.agents = list(self.train_config['agents'])
        model_args, algo_args = cfg['algo_args']['model'], cfg['algo_args']['algo']
        self.recurrent = model_args.get('use_recurrent_policy', False) or model_args.get('use_naive_recurrent_policy', False)
        self.rnn_shape = (1, model_args.get('recurrent_n', 1), model_args['hidden_sizes'][-1])

        # Supersuit pads every active agent's observation with zeros up to the largest one.
        obs_dims = {'agent_ls': 26, 'agent_dc': 14 + int(self.include_dc_setpoint), 'agent_bat': 13}
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
        if agent == 'agent_dc' and not self.include_dc_setpoint:
            state = state[:14]  # the eval env may append the setpoint; this policy was trained without it
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


def make_policy(spec):
    spec, _, label = spec.partition('=')
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
    while True:
        _, _, _, truncs, info = sdc.step(policy.act(sdc))
        i = info['agent_ls']  # merged ls/dc/bat info
        m['co2'] += i['bat_CO2_footprint'] / 1e3
        m['total_energy'] += i['bat_total_energy_with_battery_KWh'] / 1e3
        m['it_energy'] += i['dc_ITE_total_power_kW'] * STEP_HOURS / 1e3
        m['hvac_energy'] += i['dc_HVAC_total_power_kW'] * STEP_HOURS / 1e3
        m['water'] += i['dc_water_usage']
        m['tasks_dropped'] += i['ls_tasks_dropped']
        m['avg_setpoint'] += i['dc_crac_setpoint']
        exo.append((i['ls_original_workload'], i['outside_temp'], i['norm_CI'], i['day'], i['hour']))
        steps += 1
        if truncs['__all__']:
            break
    m['tasks_queued_end'] = float(i['ls_tasks_in_queue'])
    m['avg_setpoint'] /= steps
    m['steps'] = steps
    fingerprint = hashlib.sha1(np.round(np.array(exo, dtype=np.float64), 9).tobytes()).hexdigest()[:12]
    return m, fingerprint


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument('--policy', action='append', required=True, help='Policy spec (repeatable), see above.')
    p.add_argument('--n_episodes', type=int, default=5)
    p.add_argument('--seed', type=int, default=1000, help='Episode i uses seed + i.')
    p.add_argument('--location', default=None)
    p.add_argument('--month', type=int, default=None)
    p.add_argument('--days_per_episode', type=int, default=None)
    p.add_argument('--out_dir', default=None, help='Default: results/eval_compare/<timestamp>')
    args = p.parse_args()
    torch.set_num_threads(1)

    with open(ENV_CFG) as f:
        scenario = yaml.safe_load(f)
    for key in ['location', 'month', 'days_per_episode']:
        if getattr(args, key) is not None:
            scenario[key] = getattr(args, key)
    seeds = [args.seed + i for i in range(args.n_episodes)]

    policies = [make_policy(s) for s in args.policy]
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
        sdc = SustainDC({**scenario, 'agents': list(AGENTS), 'include_dc_setpoint': pol.include_dc_setpoint})
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

    out_dir = args.out_dir or os.path.join('results', 'eval_compare', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'episodes.csv'), 'w') as f:
        f.write('policy,seed,' + ','.join(n for n, _ in METRICS) + ',inputs_fingerprint\n')
        for label, eps in results.items():
            for seed, e, fp in zip(seeds, eps, fingerprints[label]):
                f.write(f'{label},{seed},' + ','.join(f'{e[n]:.6g}' for n, _ in METRICS) + f',{fp}\n')
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'scenario': scenario, 'seeds': seeds, 'policies': args.policy,
                   'inputs_identical': same_inputs, 'summary': summary}, f, indent=2)
    print(f'\nSaved {out_dir}/episodes.csv and summary.json')


if __name__ == '__main__':
    main()
