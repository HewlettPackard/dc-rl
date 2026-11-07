"""Evaluate a single-agent SB3 model on SustainDC and print the same summary metrics as eval_sustaindc.py.

Run from the repo root, e.g.:
    python eval_single_agent.py --run_dir results/sustaindc/ca/single_ppo/single_agent/seed-00001-...
    python eval_single_agent.py --run_dir <run_dir> --baseline   # also run the do-nothing policy for comparison
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np

from sustaindc_single_env import SustainDCSingleAgentEnv
from train_single_agent import ALGOS, load_obs_normalizer

DO_NOTHING = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}  # ls: do nothing, dc: keep setpoint, bat: idle


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    m = dict(co2=0.0, energy=0.0, water=0.0, dropped=0.0, queue=0.0, ret=0.0, steps=0)
    while True:
        obs, r, term, trunc, info = env.step(policy(obs))
        m['co2'] += info['bat_CO2_footprint']
        m['energy'] += info['bat_total_energy_with_battery_KWh']
        m['water'] += info['dc_water_usage']
        m['dropped'] += info['ls_tasks_dropped']
        m['queue'] += info['ls_tasks_in_queue']
        m['ret'] += r
        m['steps'] += 1
        if term or trunc:
            m['queue'] /= m['steps']
            return m


def summarize(name, episodes):
    mean = {k: np.mean([e[k] for e in episodes]) for k in episodes[0]}
    print(f"\n== {name} ({len(episodes)} episode(s), {int(mean['steps'])} steps each) ==")
    print(f"  Return:                 {mean['ret']:.2f}")
    print(f"  CO2 footprint:          {mean['co2'] / 1e3:.2f} kgCO2")
    print(f"  Total energy:           {mean['energy'] / 1e3:.2f} MWh")
    print(f"  Water usage:            {mean['water']:.2f} L")
    print(f"  Tasks dropped:          {mean['dropped']:.0f}")
    print(f"  Avg tasks in queue:     {mean['queue']:.2f}")
    return mean


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--run_dir', required=True)
    p.add_argument('--model', default='best_model', help='Model file in <run_dir>/models (without .zip).')
    p.add_argument('--n_episodes', type=int, default=1)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--baseline', action='store_true', help='Also evaluate the do-nothing policy on the same seeds.')
    args = p.parse_args()

    with open(os.path.join(args.run_dir, 'config.json')) as f:
        cfg = json.load(f)
    env = SustainDCSingleAgentEnv(dict(cfg['env_args']))
    model_path = os.path.join(args.run_dir, 'models', args.model)
    model = ALGOS[cfg['main_args']['algo']].load(model_path, device='cpu')
    normalize = load_obs_normalizer(model_path)

    seeds = [args.seed + i for i in range(args.n_episodes)]
    agent = summarize(f"{cfg['main_args']['algo']} ({args.model})",
                      [run_episode(env, lambda o: model.predict(normalize(o), deterministic=True)[0], s) for s in seeds])

    if args.baseline:
        if env.action_type == 'discrete':
            idx = int(np.ravel_multi_index(tuple(DO_NOTHING.values()), (3, 3, 3)))
            noop = lambda o: idx
        else:
            noop = lambda o: np.array(list(DO_NOTHING.values()))
        base = summarize('do-nothing baseline', [run_episode(env, noop, s) for s in seeds])
        print(f"\n  CO2 reduction vs baseline:    {100 * (1 - agent['co2'] / base['co2']):.2f}%")
        print(f"  Energy reduction vs baseline: {100 * (1 - agent['energy'] / base['energy']):.2f}%")


if __name__ == '__main__':
    main()
