"""Checks for the cooling tower fan fix. Run from the repo root: python tests/verify_ct_fan_fix.py

The tower fan's rated power used to be set to the rated heat rejection (~1 MW for a 1 MW data center), and the tower
was sized for hard-coded peak temperatures (NY 30 C although its weather file reaches ~36 C), so on hot days the fan
alone could draw more than the IT load. Now the fan is rated at CT_FAN_POWER_FRACTION of the heat rejection and the
tower / chiller are sized from each location's weather file.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd

import envs.datacenter as D
from sustaindc_env import SustainDC
from utils.make_envs_pyenv import design_weather
from utils.utils_cf import obtain_paths

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


print("1. Fan rating and sizing")
for cap in [0.5, 1, 2]:
    e = SustainDC({'location': 'ny', 'month': 6, 'days_per_episode': 1, 'datacenter_capacity_mw': cap})
    cfg = e.dc_env.DC_Config
    check(f"capacity {cap} MW: fan rating == {cfg.CT_FAN_POWER_FRACTION:.3f} x rated heat rejection",
          abs(cfg.CT_FAN_REF_P - cfg.CT_FAN_POWER_FRACTION * cfg.CT_RATED_LOAD) < 1e-6,
          f"fan {cfg.CT_FAN_REF_P / 1e3:.1f} kW, heat {cfg.CT_RATED_LOAD / 1e3:.0f} kW")
e = SustainDC({'location': 'ny', 'month': 6, 'days_per_episode': 1}); cfg = e.dc_env.DC_Config
hvac_ub = e.dc_env.ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][1]
check("HVAC observation range uses the fan rating, not the heat load", hvac_ub < 0.5 * cfg.CT_RATED_LOAD,
      f"upper bound {hvac_ub / 1e3:.0f} kW")
for loc in ['az', 'ny']:
    d = pd.read_csv(f'data/Weather/{obtain_paths(loc)[1]}', skiprows=8, header=None).values[:, 6].astype(float)
    tower_t, chiller_wb = design_weather(loc)
    check(f"{loc}: tower designed for the 99.6th-percentile dry bulb of its weather file", abs(tower_t - np.percentile(d, 99.6)) < 1e-9,
          f"{tower_t:.1f} C (file max {d.max():.1f} C)")
    e = SustainDC({'location': loc, 'month': 6, 'days_per_episode': 1}); cfg = e.dc_env.DC_Config
    cond, chw = D.chiller_operating_temps(15.0, chiller_wb, cfg)
    avail = cfg.CHILLER_NOMINAL_CAP * D.chiller_available_capacity_ratio(cond, chw)
    check(f"{loc}: chiller covers the peak heat at its design wet bulb", avail >= cfg.CT_RATED_LOAD - 1e-6,
          f"available {avail / 1e3:.0f} kW >= {cfg.CT_RATED_LOAD / 1e3:.0f} kW at wet bulb {chiller_wb:.1f} C")

print("2. Fan power in July episodes (all locations, do-nothing)")
problems, rows = [], []
for loc in ['az', 'ca', 'ga', 'il', 'ny', 'tx', 'va', 'wa']:
    e = SustainDC({'location': loc, 'month': 6, 'days_per_episode': 3}); e.seed(1000); e.reset()
    rating = e.dc_env.DC_Config.CT_FAN_REF_P / 1e3
    fan, it, chiller = [], [], []
    while True:
        _, _, _, tr, info = e.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}); i = info['agent_dc']
        fan.append(i['dc_CT_total_power_kW']); it.append(i['dc_ITE_total_power_kW']); chiller.append(i['dc_Compressor_total_power_kW'])
        if tr['__all__']:
            break
    fan, it, chiller = map(np.array, (fan, it, chiller))
    rows.append(f"{loc} {fan.mean():.1f}/{fan.max():.1f} kW")
    if fan.max() > rating + 1e-9 or (fan / it).max() > 0.02:
        problems.append(f"{loc}: max fan {fan.max():.1f} kW, max fan/IT {(fan / it).max():.3f}")
check("fan never exceeds its rating or 2% of IT power", not problems, "; ".join(problems) or "mean/max: " + ", ".join(rows))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
