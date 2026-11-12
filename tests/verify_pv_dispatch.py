"""Checks for the PV switch (include_pv) and the agent-dispatched PV actions (use / battery / resell). Run from the
repo root: python tests/verify_pv_dispatch.py

Part 1 tests the battery environment alone (needs numpy + gymnasium only). Part 2 tests SustainDC and the single-agent
wrapper and is skipped when the full simulator dependencies are not installed."""
import os, sys, random
sys.path.insert(0, os.getcwd())
import numpy as np

from envs.bat_env_fwd_view import PV_ACTIONS, BatteryEnvFwd, combine_bat_pv_action

results = []
CHARGE, DISCHARGE, IDLE = 0, 1, 2
USE, BATTERY, RESELL = 0, 1, 2


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make_bat(dispatch=True, credit=1.0):
    env = BatteryEnvFwd({'n_fwd_steps': 4, 'max_bat_cap': 1.0, 'max_dc_pw_MW': 2.0, 'dcload_max': 2.0,
                         'dcload_min': 0.1, 'battery_config': None})
    env.set_pv_dispatch(dispatch, resell_credit=credit)
    env.reset()
    return env


def step(env, bat_a, pv_a, load_mw=0.8, pv_kw=1000.0, ci=400.0):
    env.set_dcload(load_mw)
    env.set_pv(pv_kw)
    env.update_ci(ci, 0.0)
    return env.step(combine_bat_pv_action(bat_a, pv_a))[-1]


def balanced(i, load_kwh, ci):
    """PV and grid energy balances, and CO2 = grid x CI - export credit."""
    pv_ok = abs(i['pv_energy_KWh'] - (i['pv_self_consumed_KWh'] + i['pv_to_battery_KWh'] + i['pv_curtailed_KWh']
                                      + i['pv_exported_KWh'])) < 1e-6
    grid = load_kwh - i['pv_self_consumed_KWh'] - i['bat_energy_out_KWh'] + i['bat_energy_in_KWh'] - i['pv_to_battery_KWh']
    grid_ok = abs(i['grid_energy_KWh'] - max(grid, 0)) < 1e-6 and i['grid_energy_KWh'] >= 0
    co2_ok = abs(i['bat_CO2_footprint'] - (i['grid_energy_KWh'] * ci - i['pv_export_credit_gCO2'])) < 1e-6
    return pv_ok and grid_ok and co2_ok


print("1. Battery env: PV dispatch actions")
# Load 0.8 MW -> 200 kWh per step; PV 1000 kW -> 250 kWh; CI 400 g/kWh. 1 MWh battery at 0.5C -> 125 kWh/step max.
env = make_bat()
check("action space is Discrete(9) with dispatch", env.action_space.n == 9)
check("action = battery + 3 x PV, round trip", all(env.split_action(combine_bat_pv_action(b, p)) == (b, p)
                                                   for b in range(3) for p in range(3)))

i = step(make_bat(), IDLE, USE)
check("use: PV covers the load, surplus curtailed, no grid", i['pv_self_consumed_KWh'] == 200 and
      abs(i['pv_curtailed_KWh'] - 50) < 1e-9 and i['grid_energy_KWh'] == 0 and i['bat_CO2_footprint'] == 0, i['pv_a_t'])

i = step(make_bat(), IDLE, RESELL)
check("resell: all PV sold, load from the grid, CO2 credited at CI",
      i['pv_exported_KWh'] == 250 and i['pv_self_consumed_KWh'] == 0 and abs(i['grid_energy_KWh'] - 200) < 1e-9
      and abs(i['bat_CO2_footprint'] - (200 - 250) * 400) < 1e-6, f"CO2 {i['bat_CO2_footprint']:.0f} g")

i = step(make_bat(credit=0.5), IDLE, RESELL)
check("resell credit factor 0.5 halves the credit", abs(i['pv_export_credit_gCO2'] - 0.5 * 250 * 400) < 1e-6)

i = step(make_bat(), IDLE, BATTERY)
check("battery: PV charges the battery first, the rest supplies the load",
      abs(i['pv_to_battery_KWh'] - 125) < 1e-6 and abs(i['pv_self_consumed_KWh'] - 125) < 1e-6
      and abs(i['grid_energy_KWh'] - 75) < 1e-6 and i['bat_mode'] == 'pv_charge')

i = step(make_bat(), DISCHARGE, BATTERY)
check("battery + discharge: discharge overridden, PV charges", i['bat_mode'] == 'pv_charge'
      and i['bat_energy_out_KWh'] == 0 and abs(i['pv_to_battery_KWh'] - 125) < 1e-6)

i = step(make_bat(), CHARGE, BATTERY, pv_kw=200.0)  # 50 kWh of PV, battery takes 125 kWh: 75 from the grid
check("battery + charge: PV first, grid tops up", abs(i['pv_to_battery_KWh'] - 50) < 1e-6
      and abs(i['bat_energy_in_KWh'] - 125) < 1e-6 and abs(i['grid_energy_KWh'] - (200 + 75)) < 1e-6)

env = make_bat()
for _ in range(4):
    step(env, CHARGE, USE, pv_kw=0.0)
i = step(env, DISCHARGE, USE, pv_kw=600.0)  # net load 200 - 150 = 50 kWh
check("use + discharge: discharge capped at the load PV leaves", abs(i['bat_energy_out_KWh'] - 50) < 1e-6
      and i['grid_energy_KWh'] < 1e-6)
i = step(env, DISCHARGE, RESELL, pv_kw=600.0)
check("resell + discharge: battery serves the load while PV is sold", i['bat_energy_out_KWh'] > 0
      and i['pv_exported_KWh'] == 150 and balanced(i, 200, 400))

env = make_bat()
rng = np.random.default_rng(0)
ok = True
for _ in range(500):
    load, pv, ci = rng.uniform(0.1, 1.5), rng.choice([0.0, rng.uniform(0, 2000)]), rng.uniform(50, 600)
    i = step(env, int(rng.integers(3)), int(rng.integers(3)), load_mw=load, pv_kw=pv, ci=ci)
    ok &= balanced(i, load * 250, ci)
check("random dispatch: PV, grid and CO2 balances hold every step", ok)

print("2. Battery env: without dispatch (legacy)")
env = make_bat(dispatch=False)
check("action space stays Discrete(3)", env.action_space.n == 3)
i = step(env, IDLE, 0)
check("PV surplus absorbed automatically, pv_action = -1", i['pv_action'] == -1 and i['bat_mode'] == 'pv_charge'
      and abs(i['pv_to_battery_KWh'] - 50) < 1e-6)

print("3. SustainDC and the single-agent env")
try:
    import yaml
    from sustaindc_env import SustainDC
    from sustaindc_single_env import AGENTS, SustainDCSingleAgentEnv, decode_joint_action
except ImportError as e:
    print(f"  [SKIP] simulator dependencies not installed ({e})")
else:
    with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
        BASE = yaml.safe_load(f)
    BASE.update({'days_per_episode': 1, 'agents': list(AGENTS), 'pv_config': {'capacity_kWp': 1000}})

    def episode(cfg, seed=0, n_bat=9):
        random.seed(seed); np.random.seed(seed)
        env = SustainDC(cfg)
        env.reset()
        rng = np.random.default_rng(seed)
        infos = []
        while True:
            act = {'agent_ls': int(rng.integers(3)), 'agent_dc': int(rng.integers(3)), 'agent_bat': int(rng.integers(n_bat))}
            _, _, _, tr, info = env.step(act)
            infos.append(info['agent_ls'])
            if tr['__all__']:
                return env, infos

    env, infos = episode({**BASE, 'include_pv': True})
    check("include_pv=True: battery agent action space Discrete(9)", env.action_space[2].n == 9)
    check("include_pv=True: PV generated and all three PV actions used",
          sum(i['pv_energy_KWh'] for i in infos) > 0 and {i['pv_a_t'] for i in infos} == set(PV_ACTIONS))
    check("include_pv=True: balances hold every step",
          all(balanced(i, i['bat_total_energy_without_battery_KWh'], i['bat_avg_CI']) for i in infos))

    env, infos = episode({**BASE, 'include_pv': False}, n_bat=3)
    check("include_pv=False: no PV even with capacity_kWp > 0", env.action_space[2].n == 3
          and sum(i['pv_energy_KWh'] for i in infos) == 0)

    env, infos = episode({**BASE, 'include_pv': None}, n_bat=3)
    check("include_pv=None: legacy automatic PV", env.action_space[2].n == 3 and env.pv_mode == 'auto'
          and sum(i['pv_energy_KWh'] for i in infos) > 0 and all(i['pv_action'] == -1 for i in infos))

    try:
        SustainDC({**BASE, 'include_pv': True, 'pv_config': {'capacity_kWp': 0}})
        check("include_pv=True without capacity raises", False)
    except ValueError:
        check("include_pv=True without capacity raises", True)

    md = SustainDCSingleAgentEnv({**BASE, 'include_pv': True})
    d = SustainDCSingleAgentEnv({**BASE, 'include_pv': True, 'action_type': 'discrete'})
    check("single agent: MultiDiscrete([3,3,3,3]) / Discrete(81)", list(md.action_space.nvec) == [3, 3, 3, 3]
          and d.action_space.n == 81)
    decoded = {tuple(sorted(d.decode_action(a).items())) for a in range(81)}
    check("single agent: Discrete(81) decodes to 81 distinct combos", len(decoded) == 81)
    check("single agent: PV sub-action folded into agent_bat",
          decode_joint_action([1, 1, 2, 2], 'multidiscrete', include_pv=True)['agent_bat'] == 8)
    md.reset(seed=0)
    _, _, _, _, info = md.step(md.action_space.sample())
    check("single agent: steps with PV dispatch", info['pv_action'] in (0, 1, 2))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
