"""Checks for the common evaluation suite (sdc_eval/).
Run from the repo root: python tests/verify_common_eval.py"""
import json, math, os, sys
sys.path.insert(0, os.getcwd())
import torch
import yaml
torch.set_num_threads(1)

from sdc_eval.constants import ENV_CFG
from sdc_eval.metrics import METRIC_NAMES
from sdc_eval.policies import make_policy
from sdc_eval.rollout import run_episode
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

with open(ENV_CFG) as f:
    SCENARIO = {**yaml.safe_load(f), 'days_per_episode': 1, 'agents': list(AGENTS)}
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def env_for(pol):
    return SustainDC({**SCENARIO, 'include_dc_setpoint': pol.include_dc_setpoint, 'include_pv_obs': pol.include_pv_obs})


print("1. The shared rollout reproduces the pre-refactor eval_compare.run_episode exactly")
with open('tests/golden/eval_metrics_pre_refactor.json') as f:
    golden = json.load(f)
for spec, by_seed in golden.items():
    _, pol = make_policy(spec)
    sdc = env_for(pol)
    for seed, ref in by_seed.items():
        m, fp = run_episode(sdc, pol, int(seed))
        diff = [k for k, v in ref['metrics'].items() if m[k] != v]
        check(f"{spec} seed {seed}: historical metrics bit-identical, same inputs", not diff and fp == ref['fingerprint'],
              f"differs: {diff}" if diff else "")

print("2. New metrics are present and sane")
_, noop = make_policy('noop')
m, _ = run_episode(env_for(noop), noop, 1000)
missing = [k for k in METRIC_NAMES if k not in m or (isinstance(m[k], float) and math.isnan(m[k]))]
check("every registered metric is computed (no NaN) for noop", not missing, f"missing/NaN: {missing}")
check("noop always takes the do-nothing action", m['act_ls_1'] == 1 and m['act_dc_1'] == 1 and m['act_bat_2'] == 1)
check("PUE > 1", m['pue'] > 1, f"{m['pue']:.3f}")
check("effective CI = co2 / energy", abs(m['effective_ci'] - m['co2'] / m['total_energy']) < 1e-9)
_, rnd = make_policy('random')
r1, _ = run_episode(env_for(rnd), rnd, 1000)
r2, _ = run_episode(env_for(rnd), rnd, 1000)
check("returns are reproducible (reward history cleared per episode)", r1['return_team'] == r2['return_team'],
      f"{r1['return_team']:.4f} vs {r2['return_team']:.4f}")
check("action shares sum to 1 per agent", all(abs(sum(r1[f'act_{a}_{k}'] for k in range(3)) - 1) < 1e-9
                                             for a in ('ls', 'dc', 'bat')))

print("3. SustainDC attaches the same episode metrics to the last step's info (used by the training loggers)")
import random
import numpy as np
from sdc_eval.rollout import reset_reward_history
_, rule = make_policy('rule:ls,dc,bat')
sdc = env_for(rule)
ref, _ = run_episode(sdc, rule, 1001)
random.seed(1001); np.random.seed(1001); reset_reward_history(); sdc.reset(); rule.reset(1001)
while True:
    _, _, _, truncs, info = sdc.step(rule.act(sdc))
    if truncs['__all__']:
        break
ep = info['agent_ls'].get('episode_metrics')
check("episode_metrics present on the last step", ep is not None)
check("...and equal to run_episode's metrics", ep is not None and all(ep[k] == ref[k] for k in ref),
      [k for k in ref if ep is not None and ep[k] != ref[k]])

print("4. Hyperparameter config: precedence, typo protection, flattening")
from sdc_eval.config import flatten, merge, parse_overrides
base = {'env': {'month': 6, 'pv_config': {'capacity_kWp': 0}}, 'algo': {'lr': 1e-3}}
cfg = merge(base, {'env': {'month': 3}, 'algo': {'lr': 1e-4, 'new_key': 1}}, open_sections=('algo',))
cfg = merge(cfg, parse_overrides(['env.pv_config.capacity_kWp=500', 'algo.lr=2e-4', 'algo.net=[64, 64]']),
            open_sections=('algo',))
check("config file then --set override the defaults",
      cfg == {'env': {'month': 3, 'pv_config': {'capacity_kWp': 500}}, 'algo': {'lr': 2e-4, 'new_key': 1, 'net': [64, 64]}},
      cfg)
try:
    merge(base, {'env': {'mnth': 3}})
    check("unknown env key raises", False)
except KeyError:
    check("unknown env key raises", True)
check("flatten for run_meta/HParams", flatten(cfg)['algo.net'] == '[64, 64]' and flatten(cfg)['env.pv_config.capacity_kWp'] == 500)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
