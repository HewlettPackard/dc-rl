"""Checks for the MPC with a learned terminal value (mpc_value.py, train_mpc_value.py).
Run from the repo root: python tests/verify_mpc_value.py"""
import os, random, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import yaml

import mpc_value as M
from baseline_policies import battery_transition
from eval_compare import make_policy, split_label, run_episode, NoOpPolicy
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS
from train_mpc_value import fit_mlp, nstep_targets

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    SCENARIO = {**yaml.safe_load(f), 'location': 'az', 'days_per_episode': 4, 'agents': list(AGENTS)}
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


print("1. The planning model of the load-shifting queue reproduces the env exactly")
env = SustainDC(dict(SCENARIO))
random.seed(5); np.random.seed(5); env.reset()
ls = env.ls_env
rng = np.random.default_rng(0)
H, hist = 16, []
bad1 = bad_multi = n_multi = drops = overdue = 0
for t in range(380):
    # defer and never process for ~1.5 days (fills the queue, tasks go overdue), then random actions
    a = int(rng.choice(3, p=[0.6, 0.4, 0.0] if t < 150 else [0.45, 0.2, 0.35]))
    q0 = M.queue_counts(ls)
    _, wl, _, _ = M.forecasts(env, H, 'perfect')
    hist.append((q0, wl, a))
    u, d, qn = M.simulate_queue(q0, np.array([[a]]), wl[:1], ls.non_shiftable_tasks_percentage,
                                ls.shiftable_tasks_percentage, ls.queue_max_len)
    _, _, _, _, info = env.step({'agent_ls': a, 'agent_dc': 1, 'agent_bat': 2})
    i = info['agent_ls']
    drops += i['ls_tasks_dropped']
    overdue += i['ls_overdue_penalty'] > 0
    bad1 += abs(u[0, 0] - i['ls_shifted_workload']) > 1e-12 or d[0, 0] != i['ls_tasks_dropped'] \
        or not np.array_equal(qn[0], M.queue_counts(ls))
    if t >= H:
        q0h, wlh, _ = hist[t - H + 1]
        acts = np.array([[h[2] for h in hist[t - H + 1:t + 1]]])
        _, _, qh = M.simulate_queue(q0h, acts, wlh, ls.non_shiftable_tasks_percentage, ls.shiftable_tasks_percentage,
                                    ls.queue_max_len)
        n_multi += 1
        bad_multi += not np.array_equal(qh[0], M.queue_counts(ls))
check("one-step utilization, drops and queue match", bad1 == 0, f"{bad1} mismatches over 380 steps")
check(f"{H}-step queue predictions (perfect forecast) match", bad_multi == 0, f"{bad_multi}/{n_multi} mismatches")
check("the test covered a full queue and overdue tasks", drops > 0 and overdue > 0, f"dropped {drops:.0f}, "
      f"steps with overdue tasks {overdue}")

print("2. Energy model and forecasts")
coef = M.energy_coefficients(env, 15.0)
from baseline_policies import dc_power
true = [sum(dc_power(env, u, 15.0, env.dc_env.ambient_temp, env.dc_env.wet_bulb)) / 1e3 * 0.25 for u in (0.3, 0.7)]
err = max(abs(M.energy_kwh(coef, u) - e) / e for u, e in zip((0.3, 0.7), true))
check("quadratic energy model within 1% of dc_power", err < 0.01, f"max error {100 * err:.3f}%")
ci_p, wl_p, _, _ = M.forecasts(env, 24, 'perfect')
ci_r, wl_r, _, _ = M.forecasts(env, 24, 'realistic')
check("realistic forecast = truth for the observed steps (9 CI, 2 workload)",
      np.allclose(ci_r[:9], ci_p[:9]) and np.allclose(wl_r[:2], wl_p[:2]))
check("...and differs beyond them", not np.allclose(ci_r[9:], ci_p[9:]))

print("3. Battery DP plans with the exact battery transition")
pol = M.MPCPolicy(H=8, value='none')
pol.reset(0)
ci = np.array([100.0] * 4 + [500.0] * 4)
a = pol.plan_bat(0.0, 1.0, ci, np.full(8, 1.0), lambda g: np.zeros(len(g)))
check("charges ahead of a CI peak inside the horizon", a == M.BAT_ACTION_IDS['charge'], f"action {a}")
a = pol.plan_bat(0.5, 1.0, np.full(8, 300.0), np.full(8, 1.0), lambda g: np.zeros(len(g)))
check("with no terminal value, discharges stored energy (it looks worthless after the horizon)",
      a == M.BAT_ACTION_IDS['discharge'], f"action {a}")
a = pol.plan_bat(0.5, 1.0, np.full(8, 300.0), np.full(8, 1.0), lambda g: -g * 1000.0)
check("a terminal value that prices stored energy above the current CI keeps it", a != M.BAT_ACTION_IDS['discharge'])

print("4. Value-learning pieces")
d = np.arange(10, dtype=float)
v = np.zeros(10)
idx, y, rho = nstep_targets(d, v, 3, np.array([5, 10]))
check("n-step targets stay inside each episode", list(idx) == [0, 1, 5, 6], f"{list(idx)}")
check("n-step targets = sum of (d - rho) over n steps", np.allclose(y, [d[t:t + 3].sum() - 3 * rho for t in idx]))
x = np.random.default_rng(0).normal(size=(2000, 5))
yy = x[:, 0] ** 2 + np.sin(x[:, 1])
params, r2_tr, r2_val = fit_mlp(x, yy, np.arange(2000) >= 1800, epochs=30)
mlp = M.MLP({f'h_{k}': p for k, p in params.items()}, 'h_')
net_pred = mlp(x[:5])
check("MLP fits a smooth function", r2_val > 0.9, f"validation R2 {r2_val:.3f}")
check("numpy forward pass is finite and standardized back", np.all(np.isfinite(net_pred)))
np.savez('/tmp/_verify_mpc_value.npz', ci_scale=np.array(500.0),
         **{f'ls_{k}': p for k, p in fit_mlp(np.zeros((50, 20)) + np.arange(20), np.arange(50.0), np.zeros(50, bool),
                                              epochs=1)[0].items()},
         **{f'bat_{k}': p for k, p in fit_mlp(np.zeros((50, 13)) + np.arange(13), np.arange(50.0), np.zeros(50, bool),
                                               epochs=1)[0].items()})
lv = M.LearnedValue('/tmp/_verify_mpc_value.npz')
check("a saved value file loads with the input sizes the planner uses",
      lv.ls(np.zeros((3, M.AGE_BINS)), np.zeros(12), None, coef, 0.5).shape == (3,)
      and lv.bat(np.zeros(4), np.zeros(12), None).shape == (4,))

print("4b. Per-unit Monte Carlo targets")
from train_mpc_value import collect_episode, per_unit_targets
sc2 = {k: v for k, v in SCENARIO.items() if k != 'agents'}
ep, met = collect_episode({'scenario': {**sc2, 'days_per_episode': 2}, 'seed': 7,
                           'mpc': {'H': 8, 'value': 'naive', 'samples': 64, 'explore': 0.3}})
(xl, yl, wl_), (xb, yb) = per_unit_targets(ep)
qlen = ep['q'].sum(axis=1)
check("tracked tasks per state = queue length", np.isclose(wl_.sum(), qlen.sum()), f"{wl_.sum():.0f} vs {qlen.sum():.0f}")
bucket = xl[:, :M.N_BUCKETS].argmax(axis=1)
counts = (M.queue_features(ep['q'])[:, :M.N_BUCKETS] * 100).sum(axis=0)
check("tracked tasks per age bucket = queue counts",
      np.allclose(np.bincount(bucket, weights=wl_, minlength=M.N_BUCKETS), counts))
lo, hi = ep['task_cost'].min(), ep['task_cost'].max()
check("per-task targets lie within the per-step task costs (or the end cost)",
      np.all((yl >= min(lo, ep['end_task_cost']) - 1e-9) & (yl <= max(hi, ep['end_task_cost']) + 1e-9)))
check("battery targets lie in [0, max CI]", np.all((yb >= -1e-9) & (yb <= ep['ci'].max() + 1e-9)))
# synthetic battery: 1 MWh stored, discharged 0.5 at CI 400 then 0.5 at CI 200 -> mean 300
syn = {'q': np.zeros((4, M.AGE_BINS)), 'q_final': np.zeros(M.AGE_BINS), 'task_cost': np.zeros(4),
       'end_task_cost': 0.0, 'ctx': np.zeros((4, 12)), 'soc': np.array([1.0, 1.0, 0.5, 0.0]), 'soc_final': 0.0,
       'ci': np.array([100.0, 400.0, 200.0, 999.0]), 'cap': 1.0}
_, (_, yb_syn) = per_unit_targets(syn)
check("battery target = mean CI of the discharges of the stored energy (FIFO)",
      np.allclose(yb_syn[:3], [300.0, 300.0, 200.0]) and yb_syn[3] == 0.0, f"{yb_syn}")

print("5. eval_compare specs")
check("label after an mpc keyword value is not taken as a label",
      split_label('mpc:H=16,forecast=realistic') == ('mpc:H=16,forecast=realistic', ''))
check("explicit label", split_label('mpc:H=8,value=naive=mpc8') == ('mpc:H=8,value=naive', 'mpc8'))
check(".npz path is a value", split_label('mpc:value=runs/v.npz')[1] == '')
label, pol = make_policy('mpc:H=8,value=none,samples=64=myopic')
check("make_policy builds an MPCPolicy", isinstance(pol, M.MPCPolicy) and pol.H == 8 and label == 'myopic')

print("6. MPC episode runs and is reproducible")
short = SustainDC({**SCENARIO, 'days_per_episode': 1})
m1, fp1 = run_episode(short, M.MPCPolicy(H=8, value='naive', samples=64), 1000)
m2, fp2 = run_episode(short, M.MPCPolicy(H=8, value='naive', samples=64), 1000)
m0, fp0 = run_episode(short, NoOpPolicy(), 1000)
check("same seed -> identical metrics", m1 == m2)
check("same exogenous inputs as noop", fp1 == fp0)
check("no dropped tasks", m1['tasks_dropped'] == 0, f"{m1['tasks_dropped']:.0f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
