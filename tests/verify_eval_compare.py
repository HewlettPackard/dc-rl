"""Checks that eval_compare.py scores policies on reproducible, identical episodes.
Run from the repo root: python tests/verify_eval_compare.py"""
import os, sys
sys.path.insert(0, os.getcwd())
import yaml

import utils.reward_creator as rc
from eval_compare import NoOpPolicy, RandomPolicy, run_episode
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    SCENARIO = {**yaml.safe_load(f), 'days_per_episode': 1, 'agents': list(AGENTS)}
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


print("1. Same policy + same seed -> identical episode")
env = SustainDC(dict(SCENARIO))
m1, fp1 = run_episode(env, NoOpPolicy(), 1000)
m2, fp2 = run_episode(env, NoOpPolicy(), 1000)
check("noop metrics identical on rerun", m1 == m2, f"CO2 {m1['co2']:.3f} vs {m2['co2']:.3f}")
rc.energy_history.clear()  # the reward normalization history must not affect the metrics
m3, _ = run_episode(SustainDC(dict(SCENARIO)), NoOpPolicy(), 1000)
check("identical in a fresh env with a cleared reward history", m1 == m3)
r1, _ = run_episode(env, RandomPolicy(), 1000)
r2, _ = run_episode(env, RandomPolicy(), 1000)
check("seeded random policy identical on rerun", r1 == r2)

print("2. Different policies, same seed -> same exogenous inputs")
_, fp_rand = run_episode(env, RandomPolicy(), 1000)
check("noop and random see the same workload/weather/CI", fp1 == fp_rand, f"{fp1} vs {fp_rand}")
env_sp = SustainDC({**SCENARIO, 'include_dc_setpoint': False})
_, fp_nosp = run_episode(env_sp, RandomPolicy(), 1000)
check("observation flags don't change the inputs", fp1 == fp_nosp)
check("...while the actions do change the outcome", r1['co2'] != m1['co2'], f"{r1['co2']:.1f} vs {m1['co2']:.1f}")

print("3. Different seeds -> different episodes")
_, fp_other = run_episode(env, NoOpPolicy(), 1001)
check("seed 1001 differs from seed 1000", fp_other != fp1)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
