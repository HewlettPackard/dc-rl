"""Checks for the stationary Li-ion battery model (envs/battery_model.py) and its use in SustainDC and in the
baseline planner. Run from the repo root: python tests/verify_battery.py"""
import os, sys, random
sys.path.insert(0, os.getcwd())
import numpy as np, yaml

from envs.battery_model import Battery, DEFAULT_BATTERY_CONFIG
from baseline_policies import battery_transition
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    BASE = yaml.safe_load(f)
BASE.update({'days_per_episode': 2, 'agents': list(AGENTS)})
DT = 0.25
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def hours_to_charge(b, target):
    b.reset(); t = 0.0
    while b.get_battery_soc() < target - 1e-9 and t < 100:
        b.charge(DT); t += DT
    return t


print("1. Battery physics (1 MWh, defaults)")
b = Battery(1.0)
cfg = DEFAULT_BATTERY_CONFIG
check("defaults: 0.5C, 95% each way, SoC window 5-95%, taper from 90%",
      (b.c_rate, b.eff_charge, b.eff_discharge, b.soc_min, b.soc_max, b.taper_start) == (0.5, 0.95, 0.95, 0.05, 0.95, 0.90))
check("episodes start at the bottom of the SoC window", b.get_battery_soc() == cfg['soc_min'])
b.current_load = 0.5
check("symmetric power at mid SoC: 500 kW charge and discharge",
      np.isclose(b.charge_limit(0.5, DT), 0.125) and np.isclose(b.discharge_limit(0.5, DT), 0.125))
t_full = hours_to_charge(b, cfg['soc_max'])
check("about 2 hours from empty to full at 0.5C", 1.75 <= t_full <= 2.5, f"{t_full:.2f} h")
# Short steps so the power limit, not the remaining headroom, binds: 500 kW x 0.01 h = 5 kWh.
check("charging power tapers only above 90% SoC (full below, half at 92.5%, zero at 95%)",
      np.isclose(b.charge_limit(0.89, 0.01), 0.005) and np.isclose(b.charge_limit(0.925, 0.01), 0.0025)
      and b.charge_limit(0.95, 0.01) == 0)
b.current_load = 0.5
drawn = b.charge(DT)
check("charging stores 95% of the energy drawn", np.isclose(b.current_load, 0.5 + 0.95 * drawn))
before = b.current_load; out = b.discharge(DT)
check("discharging removes delivered / 95% from storage", np.isclose(before - b.current_load, out / 0.95))
b.reset(); b.current_load = 0.5; e_in = b.charge(DT); e_out = b.discharge(DT, max_out=e_in * 0.95 * 0.95)
check("round trip about 90%", np.isclose(e_out / e_in, 0.9025) and np.isclose(b.current_load, 0.5), f"{e_out / e_in:.4f}")
b.reset(); socs = []
rng = np.random.default_rng(0)
for _ in range(2000):
    (b.charge if rng.random() < 0.5 else b.discharge)(DT)
    socs.append(b.get_battery_soc())
check("SoC stays in the usable window", min(socs) >= cfg['soc_min'] - 1e-12 and max(socs) <= cfg['soc_max'] + 1e-12,
      f"[{min(socs):.3f}, {max(socs):.3f}]")
check("c_rate 0.25 (4-hour) takes twice as long", np.isclose(hours_to_charge(Battery(1.0, {'c_rate': 0.25}), 0.9), 2 * hours_to_charge(Battery(1.0), 0.9), atol=DT))
check("rates scale with capacity", np.isclose(Battery(3.0).charge_limit(1.5, DT), 3 * Battery(1.0).charge_limit(0.5, DT)))
for bad in ({'c_rate': 0}, {'eff_charge': 1.2}, {'soc_min': 0.95}, {'crate': 0.5}):
    try:
        Battery(1.0, bad); ok = False
    except ValueError:
        ok = True
    check(f"invalid battery_config {bad} rejected", ok)

print("2. In SustainDC")
random.seed(0); np.random.seed(0)
env = SustainDC({**BASE, 'battery_config': {'c_rate': 0.5}}); env.reset()
bat = env.bat_env.battery
check("env uses battery_config and the configured capacity", bat.c_rate == 0.5 and bat.capacity == BASE['max_bat_cap_Mw'])
t = 0
while env.bat_env.get_battery_soc() < bat.soc_max - 1e-9 and t < 96:
    *_, info = env.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 0}); t += 1
check("always-charge fills the battery in about 2 hours", 7 <= t <= 10, f"{t * DT:.2f} h")
i = info['agent_bat']
check("grid import = load + energy drawn to charge", np.isclose(i['grid_energy_KWh'], i['bat_total_energy_without_battery_KWh'] + i['bat_energy_in_KWh']))
ok = True
for _ in range(16):
    *_, info = env.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 1}); i = info['agent_bat']
    load = i['bat_total_energy_without_battery_KWh']
    ok &= i['bat_energy_out_KWh'] <= min(load, 125.0) + 1e-6 and np.isclose(i['grid_energy_KWh'], load - i['bat_energy_out_KWh'])
check("discharge capped at the load and at 500 kW; grid = load - delivered", ok)

print("3. Baseline planner replica matches the env step by step (random actions, with and without PV)")
for pv in (0, 1000):
    random.seed(1); np.random.seed(1)
    env = SustainDC({**BASE, 'location': 'ca', 'pv_config': {'capacity_kWp': pv}}); env.reset()
    rng = np.random.default_rng(1); err_s = err_g = 0.0
    for _ in range(192):
        act = {a: int(rng.integers(3)) for a in AGENTS}
        stored = env.bat_env.battery.current_load
        *_, info = env.step(act); i = info['agent_bat']
        load, pv_e = i['bat_total_energy_without_battery_KWh'] / 1e3, i['pv_energy_KWh'] / 1e3
        nxt, grid_delta, _ = battery_transition(np.array([stored]), env.bat_env.battery,
                                                {0: 'charge', 1: 'discharge', 2: 'idle'}[act['agent_bat']], load, pv_e)
        err_s = max(err_s, abs(nxt[0] - env.bat_env.battery.current_load))
        err_g = max(err_g, abs(grid_delta[0] * 1e3 - (i['grid_energy_KWh'] - max(load - pv_e, 0) * 1e3)))
    check(f"PV {pv} kWp: next stored energy and grid change identical", err_s < 1e-9 and err_g < 1e-6,
          f"max err {err_s:.1e} MWh, {err_g:.1e} kWh")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
