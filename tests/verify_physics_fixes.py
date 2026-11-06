"""Checks for the cooling-physics fixes. Run from the repo root: python tests/verify_physics_fixes.py

1. CRAC setpoint step scaling is symmetric (decreases accelerate like increases).
2. Cooling load equals the IT heat (steady-state energy balance); rack temperatures follow from it.
3. Server CPU and fan power follow the config curves (no inflated inlet-temperature sensitivity).
4. Chiller follows the EnergyPlus Chiller:Electric model: less efficient with warmer condenser water,
   more efficient with warmer chilled water (higher CRAC setpoint).
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, yaml

import envs.datacenter as D
from sustaindc_env import SustainDC

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


env = SustainDC({'location': 'ca', 'month': 6, 'days_per_episode': 2}); env.seed(0); env.reset()
d = env.dc_env; cfg = d.DC_Config


def op(load, stpt, amb=25.0, wb=18.0):
    cpu, fan, out = d.dc.compute_datacenter_IT_load_outlet_temp([load * 100] * cfg.NUM_RACKS, stpt)
    it = sum(cpu) + sum(fan)
    ret = D.calculate_avg_CRAC_return_temp(cfg.RACK_RETURN_APPROACH_TEMP_LIST, out)
    _, ct_fan, cool, chiller, _, _ = D.calculate_HVAC_power(stpt, ret, amb, it, cfg, wet_bulb_temp=wb)
    d.dc.heat_rejected_W = cool + chiller
    return dict(it=it, fan=sum(fan), cool=cool, ret=ret, out=np.array(out), inlet=np.array(d.dc.rackwise_inlet_temp),
                chiller=chiller, hvac=ct_fan + chiller, water=d.dc.calculate_cooling_tower_water_usage())


print("1. Setpoint step scaling is symmetric")
def run_actions(start, action, n=8):
    d.raw_curr_stpt = start; d.last_action = None; d.consecutive_actions = 0; d.action_scaling_factor = 1
    traj = []
    for _ in range(n):
        d.step(action); traj.append(round(d.raw_curr_stpt, 3))
    return traj
up, down = run_actions(15.0, 2), run_actions(21.6, 0)
up_steps, down_steps = np.diff([15.0] + up), np.diff([21.6] + down)
check("increase from 15: +1, +1, +1, +2, then capped", list(up_steps[:4]) == [1, 1, 1, 2] and up[4] == 21.6, f"{up}")
check("decrease from 21.6: -1, -1, -1, -2, then capped", list(np.round(down_steps[:4], 6)) == [-1, -1, -1, -2] and down[4] == 15.0, f"{down}")
hold = run_actions(18.0, 1, 5)
check("hold keeps the setpoint", hold == [18.0] * 5)
d.raw_curr_stpt = 18.0; d.last_action = None; d.consecutive_actions = 0; d.action_scaling_factor = 1
for _ in range(5):
    d.step(1)
d.step(2)
check("an increase after a run of holds moves +1", d.raw_curr_stpt == 19.0, f"{d.raw_curr_stpt}")

print("2. Energy balance")
dT = D.crac_air_temperature_rise(cfg)
ok = True; worst = 0
for load in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    for s in [15, 18, 21.6]:
        o = op(load, s)
        worst = max(worst, abs(o['cool'] / o['it'] - 1))
        ok &= abs(o['cool'] - o['it']) < 1e-6 and abs(o['ret'] - (s + dT)) < 1e-9
check("cooling load == IT power and return - setpoint == 1/(rho q c) at every load/setpoint", ok, f"air dT {dT:.2f} K, max |cool/IT - 1| = {worst:.1e}")
o = op(0.5, 18)
rise = o['out'] - o['inlet']
check("rack temperature rise positive and plausible (8-16 K)", rise.min() > 8 and rise.max() < 16, f"{rise.min():.1f}-{rise.max():.1f} K")

print("3. Server power follows the config")
rack = d.dc.racks_list[0]
c0, f0 = rack.compute_instantaneous_pwr_vecd(23.0, 0.0); c1, f1 = rack.compute_instantaneous_pwr_vecd(23.0, 100.0)
n = rack.num_CPUs; full = rack.full_load_pwr.mean(); idle = rack.idle_pwr.mean()
check("idle CPU power == configured idle power", abs(c0 / n - idle) < 1e-9, f"{c0 / n:.1f} W vs {idle:.0f} W")
check("full-load CPU power within 3% of nameplate", abs(c1 / n / full - 1) < 0.03, f"{c1 / n:.1f} W vs {full:.0f} W")
cl, _ = rack.compute_instantaneous_pwr_vecd(20.0, 100.0); ch, _ = rack.compute_instantaneous_pwr_vecd(26.6, 100.0)
check("CPU power rises < 3% of full load from 20 to 26.6 C inlet", (ch - cl) / n / full < 0.03, f"+{100 * (ch - cl) / n / full:.1f}%")
share = [op(L, 18)['fan'] / op(L, 18)['it'] for L in [0, 0.5, 1]]
check("server fans are a small share of IT power", max(share) < 0.15, " / ".join(f"{100 * x:.1f}%" for x in share))

print("4. Chiller (EnergyPlus Chiller:Electric)")
check("capacity ratio at the design point equals the first coefficient",
      abs(D.chiller_available_capacity_ratio(29.44, 6.667) - 0.944836) < 1e-12)
cop_wb = [op(0.5, 18, wb + 7, wb)['cool'] / op(0.5, 18, wb + 7, wb)['chiller'] for wb in [0, 10, 18, 24, 28, 32]]
check("COP falls as wet bulb rises", all(np.diff(cop_wb) < 0), " -> ".join(f"{x:.2f}" for x in cop_wb))
cop_s = [op(0.5, s)['cool'] / op(0.5, s)['chiller'] for s in [15, 17, 19, 21.6]]
check("COP rises with the CRAC setpoint (warmer chilled water)", all(np.diff(cop_s) > 0), " -> ".join(f"{x:.2f}" for x in cop_s))
check("COP in a realistic water-cooled range (3-9)", 3 < min(cop_wb + cop_s) and max(cop_wb + cop_s) < 9)
cond, chw = D.chiller_operating_temps(15.0, 32.0, cfg)
avail = cfg.CHILLER_NOMINAL_CAP * D.chiller_available_capacity_ratio(cond, chw)
check("sized chiller covers peak heat at the worst conditions", avail >= op(1.0, 21.6)['it'] - 1e-6,
      f"available {avail / 1e3:.0f} kW vs peak IT {op(1.0, 21.6)['it'] / 1e3:.0f} kW")

print("5. The cooling agent faces a real trade-off")
stpts = [15, 16, 17, 18, 19, 20, 21, 21.6]
def best(load, amb, wb):
    tot = {s: op(load, s, amb, wb)['it'] + op(load, s, amb, wb)['hvac'] for s in stpts}
    return min(tot, key=tot.get)
b_cold, b_hot = best(0.1, 5, 3), best(0.9, 44, 28)
check("cold weather / low load -> lowest setpoint best; hot / high load -> higher setpoint best", b_cold == 15 and b_hot > 15,
      f"best {b_cold} C vs {b_hot} C")

print("6. Water follows the heat rejected")
w5, w10 = op(0.5, 18)['water'], op(1.0, 18)['water']
h5, h10 = op(0.5, 18)['cool'] + op(0.5, 18)['chiller'], op(1.0, 18)['cool'] + op(1.0, 18)['chiller']
check("water / heat rejected constant", abs((w10 / h10) / (w5 / h5) - 1) < 1e-9, f"{w5:.0f} L vs {w10:.0f} L per 15 min")
check("~460 L per 15 min per MW rejected (evaporation + blowdown at 5 cycles + 1% drift)", abs(w10 / (h10 / 1e6) - 463.8) < 1, f"{w10 / (h10 / 1e6):.1f} L/MW")

print("7. Full episodes: every location, Jan / Jul / Dec, random actions")
problems = []
for loc in ['az', 'ca', 'ga', 'il', 'ny', 'tx', 'va', 'wa']:
    for month in [0, 6, 11]:
        try:
            e = SustainDC({'location': loc, 'month': month, 'days_per_episode': 3}); e.seed(1); e.reset()
            rng = np.random.default_rng(1)
            while True:
                _, rew, _, tr, info = e.step({k: int(rng.integers(3)) for k in ['agent_ls', 'agent_dc', 'agent_bat']})
                i = info['agent_dc']
                pue = i['dc_total_power_kW'] / i['dc_ITE_total_power_kW']
                if not (1.0 < pue < 2.0 and np.isfinite(list(rew.values())).all() and i['dc_water_usage'] > 0):
                    problems.append(f"{loc}/{month}: PUE {pue:.3f}, water {i['dc_water_usage']}"); break
                if tr['__all__']:
                    break
        except Exception as ex:
            problems.append(f"{loc}/{month}: {type(ex).__name__}: {ex}")
check("24 episodes run with PUE in (1, 2), finite rewards, positive water", not problems, "; ".join(problems[:3]))

print("8. HARL wrapper (training path) with the repo YAML")
from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
h = HARLSustainDCEnv(yaml.safe_load(open('harl/configs/envs_cfgs/sustaindc.yaml'))); h.reset(); steps = 0
while True:
    out = h.step(np.array([[sp.sample()] for sp in h.action_space])); steps += 1
    if all(out[3]):
        break
check("30-day episode completes", steps == 30 * 96, f"{steps} steps")

print(f"\n{sum(results)}/{len(results)} checks passed")
