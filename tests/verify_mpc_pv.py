"""Checks for on-site PV in the learned-value MPC (mpc_value.py): the planning models against the env, the PV forecast,
old value files, and short MPC episodes in the three PV modes (off, automatic rule, include_pv PV actions).
Run from the repo root: python tests/verify_mpc_pv.py"""
import os, random, sys, tempfile
sys.path.insert(0, os.getcwd())
import numpy as np
import yaml

import mpc_value as M
from baseline_policies import battery_dispatch_transition
from envs.bat_env_fwd_view import PV_ACTIONS
from sdc_eval.policies import NoOpPolicy
from sdc_eval.rollout import run_episode
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    BASE = {**yaml.safe_load(f), 'location': 'az', 'days_per_episode': 2, 'agents': list(AGENTS)}
PV = {'pv_config': {'capacity_kWp': 500}}
BAT = {0: 'charge', 1: 'discharge', 2: 'idle'}
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make(extra, seed=1):
    random.seed(seed); np.random.seed(seed)
    env = SustainDC({**BASE, **extra}); env.reset()
    return env


print("1. battery_dispatch_transition reproduces the env step by step (include_pv, random battery x PV actions)")
for cap in (500, 3000):  # 3 MWp: PV surplus beyond the DC load, so curtailment and PV charging limits are exercised
    env = make({'pv_config': {'capacity_kWp': cap}, 'include_pv': True})
    rng = np.random.default_rng(2); err_s = err_g = err_r = 0.0; used = set()
    for _ in range(192):
        a_bat = int(rng.integers(9))
        act = {'agent_ls': int(rng.integers(3)), 'agent_dc': int(rng.integers(3)), 'agent_bat': a_bat}
        stored = env.bat_env.battery.current_load
        *_, info = env.step(act); i = info['agent_bat']
        b, p = BAT[a_bat % 3], PV_ACTIONS[a_bat // 3]
        used.add(p)
        load, pv_e = i['bat_total_energy_without_battery_KWh'] / 1e3, i['pv_energy_KWh'] / 1e3
        nxt, grid_e, resold = battery_dispatch_transition(np.array([stored]), env.bat_env.battery, b, p, load, pv_e)
        err_s = max(err_s, abs(nxt[0] - env.bat_env.battery.current_load))
        err_g = max(err_g, abs(grid_e[0] * 1e3 - i['grid_energy_KWh']))
        err_r = max(err_r, abs(resold[0] * 1e3 - i['pv_exported_KWh']))
    check(f"PV {cap} kWp: next stored energy, grid import and resold PV identical", err_s < 1e-9 and err_g < 1e-6
          and err_r < 1e-6 and len(used) == 3, f"max err {err_s:.1e} MWh, {err_g:.1e} kWh, {err_r:.1e} kWh")

print("2. grid_co2_kg (battery idle) matches the env")
env = make(PV)
pv = M.pv_setup(env)
err = 0.0
for _ in range(192):
    *_, info = env.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}); i = info['agent_bat']
    load, pv_e = i['bat_total_energy_without_battery_KWh'], i['pv_energy_KWh']
    err = max(err, abs(float(M.grid_co2_kg(load, i['bat_avg_CI'], pv_e, pv)) - i['bat_CO2_footprint'] / 1e3))
check("automatic rule: CO2 of an idle-battery step", pv[0] == 'auto' and err < 1e-6, f"max err {err:.1e} kg")
env = make({**PV, 'include_pv': True})
pv = M.pv_setup(env)
battery = env.bat_env.battery
err = 0.0
for load, pv_e, ci in ((0.4, 0.0, 300.0), (0.4, 0.1, 300.0), (0.1, 0.4, 250.0), (0.2, 0.2, 400.0)):
    s = np.array([battery.current_load])
    alts = [(battery_dispatch_transition(s, battery, 'idle', p, load, pv_e)) for p in ('use', 'resell')]
    best = min(ci * (g[0] - pv[1] * r[0]) for _, g, r in alts)  # kg: MWh x g/kWh
    err = max(err, abs(float(M.grid_co2_kg(load * 1e3, ci, pv_e * 1e3, pv)) - best))
check("include_pv: the better of use and resell", pv[0] == 'agent' and err < 1e-9, f"max err {err:.1e} kg")
check("no PV: pv_setup is 'off'", M.pv_setup(make({})) [0] == 'off'
      and M.pv_setup(make({**PV, 'include_pv': False}))[0] == 'off')

print("3. PV forecast")
env = make(PV)
for _ in range(40):
    env.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2})
t, series = env.pv_m.time_step, env.pv_m.pv_kw * 0.25
true = series[t:t + 40]
perfect, realistic = M.pv_forecast(env, 40, 'perfect'), M.pv_forecast(env, 40, 'realistic')
check("perfect = the true PV", np.allclose(perfect, true))
check("realistic = true PV for the observed 17 steps, then yesterday's profile",
      np.allclose(realistic[:17], true[:17]) and np.allclose(realistic[17:], series[t - 96 + 17:t - 96 + 40]))
check("step 0 = the PV the battery sees now", np.isclose(realistic[0], env.bat_env.pv_kw * 0.25))

print("4. Value files: an old (no-PV) value works with PV, a PV value refuses a no-PV scenario")
def fake_value(n_ctx, pv_ctx):
    rng = np.random.default_rng(0); p = {'ci_scale': np.array(300.0), 'form': np.array('per_unit')}
    for head, n_in in (('ls', M.N_BUCKETS + 1 + n_ctx), ('bat', 1 + n_ctx)):
        p.update({f'{head}_W0': rng.normal(size=(n_in, 1)) * 0.01, f'{head}_b0': np.zeros(1),
                  f'{head}_x_mean': np.zeros(n_in), f'{head}_x_std': np.ones(n_in),
                  f'{head}_y_mean': np.array(0.1), f'{head}_y_std': np.array(0.01)})
    if pv_ctx:
        p['pv_ctx'] = np.array(1)
    path = os.path.join(tempfile.mkdtemp(), 'value.npz'); np.savez(path, **p)
    return path
old, new = M.LearnedValue(fake_value(12, False)), M.LearnedValue(fake_value(14, True))
ctx14, ctx12 = np.zeros(14), np.zeros(12)
q = np.zeros((2, M.AGE_BINS)); q[:, 5] = 10
ok = len(old.ls(q, ctx14, None, None, None)) == 2 and len(new.bat(np.array([0.1, 0.2]), ctx14, None, 0.95)) == 2
try:
    new.ls(q, ctx12, None, None, None); refused = False
except ValueError:
    refused = True
check("old value ignores the PV features; PV value needs them", ok and refused)

print("5. Short MPC episodes (2 days, naive value) in the three PV modes")
for name, extra in (('no PV', {}), ('PV, automatic rule', PV), ('PV, PV actions', {**PV, 'include_pv': True})):
    env = SustainDC({**BASE, **extra})
    noop, _ = run_episode(env, NoOpPolicy(), 1000)
    pv_actions = []
    trace = lambda k, info, actions: pv_actions.append(info.get('pv_action', -1))
    m, _ = run_episode(env, M.MPCPolicy(H=16, value='naive', samples=64), 1000, trace=trace)
    detail = f"CO2 adj {m['co2_work_adjusted']:.0f} vs noop {noop['co2_work_adjusted']:.0f} kg"
    if extra.get('include_pv'):
        counts = np.bincount(np.array(pv_actions), minlength=3)
        detail += f"; PV actions use/battery/resell = {counts.tolist()}"
    check(f"{name}: runs and beats noop", m['co2_work_adjusted'] < noop['co2_work_adjusted'], detail)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
