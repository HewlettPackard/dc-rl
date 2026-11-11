"""Checks for the CO2-aligned team reward. Run from the repo root: python tests/verify_co2_reward.py"""
import os, random, sys
sys.path.insert(0, os.getcwd())
import numpy as np

import utils.reward_creator as rc
from baseline_policies import make_baseline_policy
from eval_compare import NoOpPolicy, RandomPolicy, marginal_kwh_per_util_step, run_episode
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

CO2 = {'ls_reward': 'co2_team_reward', 'dc_reward': 'co2_team_reward', 'bat_reward': 'co2_team_reward'}
BASE = {'location': 'az', 'month': 6, 'days_per_episode': 3, 'flexible_load': 0.6, 'max_bat_cap_Mw': 1.0,
        'agents': list(AGENTS), **CO2}
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


print("1. Team reward equals the formula, for every agent")
e = SustainDC(dict(BASE)); e.seed(0); e.reset()
ok_same = ok_formula = True; drops = 0; last = None
while True:
    _, rew, _, tr, info = e.step({'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 2})  # always defer -> queue fills, drops
    i = info['__common__']
    task_g = rc.task_energy_kwh(i) * i['bat_avg_CI']
    expect = -(i['bat_CO2_footprint'] + 2.0 * i['ls_tasks_dropped'] * task_g) / 1e5
    ok_same &= len(set(rew.values())) == 1
    ok_formula &= abs(rew['agent_ls'] - expect) < 1e-12
    drops += i['ls_tasks_dropped']; last = (i, rew['agent_ls'])
    if tr['__all__']:
        break
check("all three agents get the same reward every step", ok_same)
check("reward == -(CO2 + 2x dropped) / 1e5", ok_formula, f"{drops:.0f} tasks dropped in the episode")
i, r = last
check("terminal step does not charge the unfinished queue (time limit: the learner bootstraps instead)",
      i['isterminal'] and i['ls_tasks_in_queue'] > 0 and abs(r - (-(i['bat_CO2_footprint'] + 2.0 * i['ls_tasks_dropped']
      * rc.task_energy_kwh(i) * i['bat_avg_CI']) / 1e5)) < 1e-12,
      f"{i['ls_tasks_in_queue']} tasks queued at the end")
exact = marginal_kwh_per_util_step(e, i['ls_shifted_workload'], i['outside_temp']) / 100
check("per-task energy = the evaluation's marginal energy at this step's conditions (shared definition)",
      abs(rc.task_energy_kwh(i) / exact - 1) < 0.05, f"{rc.task_energy_kwh(i):.3f} vs {exact:.3f} kWh per task")

print("2. Episode return + value of the work left queued ranks policies like work-adjusted CO2")
def ret_and_metric(policy, seed):
    # the reward does not charge work queued at the time limit; a learner's bootstrapped value does. Add that value
    # (the queue at its energy x the current CI) to the return to compare with the metric.
    random.seed(seed); np.random.seed(seed)
    sdc = SustainDC(dict(BASE)); total = 0.0
    sdc.reset(); policy.reset(seed)
    while True:
        _, rew, _, tr, info = sdc.step(policy.act(sdc)); total += rew['agent_ls']
        if tr['__all__']:
            c = info['__common__']
            total -= c['ls_tasks_in_queue'] * rc.task_energy_kwh(c) * c['bat_avg_CI'] / 1e5
            break
    m, _ = run_episode(SustainDC(dict(BASE)), policy, seed)
    return total, m['co2_work_adjusted']
pols = {'noop': NoOpPolicy(), 'random': RandomPolicy(),
        'rule_all': make_baseline_policy('rule', 'ls,dc,bat:ls_hi=0.6,ls_lo=0.5,dc_setpoint=15,bat_lo=0.4,bat_hi=0.8'),
        'oracle_all': make_baseline_policy('oracle', 'ls,dc,bat:ls_hi=0.6,ls_lo=0.5')}
for seed in (1000, 1001):
    vals = {k: ret_and_metric(p, seed) for k, p in pols.items()}
    by_return = sorted(vals, key=lambda k: -vals[k][0])
    # every pair whose work-adjusted CO2 differs by > 0.2% must be ordered the same way by the return (the reward
    # charges unfinished work slightly more conservatively than the metric, so near-ties may swap)
    bad = [(a, b) for a in vals for b in vals if vals[a][1] < vals[b][1] * (1 - 0.002) and not vals[a][0] > vals[b][0]]
    check(f"seed {seed}: same ranking (pairs > 0.2% apart)", not bad,
          " > ".join(f"{k} ({vals[k][0]:.1f}, {vals[k][1] / 1e3:.3f} t)" for k in by_return) + (f"; mismatched {bad}" if bad else ""))

print("3. Default reward")
d = SustainDC({k: v for k, v in BASE.items() if not k.endswith('_reward')})
check("the default config uses co2_team_reward for all three agents",
      d.ls_reward_method is d.dc_reward_method is d.bat_reward_method is rc.co2_team_reward)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
