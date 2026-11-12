"""Checks for the simulator fixes on this branch. Run from the repo root: python tests/verify_sustaindc_fixes.py"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, yaml

from utils.dc_config_reader import DC_Config
from sustaindc_env import SustainDC
import envs.datacenter as DataCenter
import utils.reward_creator as rc

BASE = {'location': 'ca', 'month': 6, 'days_per_episode': 2}
STEP_NOTHING = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make(**kw):
    cfg = dict(BASE); cfg.update(kw); return SustainDC(cfg)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc as e:
        return str(e)
    return None


def episode(env, seed, policy=None):
    env.seed(seed); env.reset(); rng = np.random.default_rng(seed); co2 = 0.0; infos = []
    while True:
        a = policy or {k: int(rng.integers(3)) for k in ['agent_ls', 'agent_dc', 'agent_bat']}
        _, rew, _, tr, info = env.step(a); co2 += info['agent_ls']['bat_CO2_footprint']; infos.append(info['agent_ls'])
        if tr['__all__']:
            return co2, infos


print("1. flexible_load reaches the load-shifting env")
for flex in [0.1, 0.4, 0.6]:
    e = make(flexible_load=flex)
    check(f"flexible_load={flex}", e.ls_env.flexible_workload_ratio == flex, f"ratio {e.ls_env.flexible_workload_ratio}")
e = make(flexible_load=0.6); e.seed(0); e.reset()
_, _, _, _, info = e.step({'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 2}); i = info['agent_ls']
deferred = i['ls_original_workload'] - i['ls_shifted_workload']
check("defer action defers ~60% of the load", abs(deferred / i['ls_original_workload'] - 0.6) < 0.05,
      f"orig {i['ls_original_workload']:.3f} -> executed {i['ls_shifted_workload']:.3f}")

print("2. Rack layout deterministic; seeded runs reproducible across env instances")
orders = {tuple(r[0]['full_load_pwr'] for r in DC_Config(dc_config_file='dc_config.json').RACK_CPU_CONFIG) for _ in range(20)}
c = DC_Config(dc_config_file='dc_config.json')
check("one rack ordering over 20 builds", len(orders) == 1, f"{len(orders)} orderings")
check("ordering matches the JSON file", list(orders)[0] == tuple(x[0] for x in c.DEFAULT_SERVER_POWER_CHARACTERISTICS))
a, _ = episode(make(), 3); b, _ = episode(make(), 3)
check("same seed, two new envs -> identical CO2", a == b, f"{a:.1f} vs {b:.1f}")

print("3. datacenter_capacity_mw scales the data center")
ref = {}
for cap in [0.5, 1, 2, 5]:
    cfg = DC_Config(dc_config_file='dc_config.json', datacenter_capacity_mw=cap)
    dc = DataCenter.DataCenter_ITModel(num_racks=cfg.NUM_RACKS, rack_supply_approach_temp_list=cfg.RACK_SUPPLY_APPROACH_TEMP_LIST,
                                       rack_CPU_config=cfg.RACK_CPU_CONFIG, max_W_per_rack=cfg.MAX_W_PER_RACK, DC_ITModel_config=cfg)
    cpu, fan, out22 = dc.compute_datacenter_IT_load_outlet_temp([100] * cfg.NUM_RACKS, 22.0)
    cpu2, fan2, _ = dc.compute_datacenter_IT_load_outlet_temp([100] * cfg.NUM_RACKS, 21.6)
    it22 = (sum(cpu) + sum(fan)) / 1e6; it216 = (sum(cpu2) + sum(fan2)) / 1e6
    check(f"capacity {cap} MW: full-load IT at 22C == capacity", abs(it22 - cap) < 1e-9, f"{it22:.4f} MW; at max agent setpoint 21.6C: {it216:.4f} MW")
    ref[cap] = out22
check("rack outlet temperatures independent of capacity", all(np.allclose(ref[k], ref[1]) for k in ref))
stats = {}
for cap in [0.5, 1, 5]:
    _, infos = episode(make(datacenter_capacity_mw=cap), 7, STEP_NOTHING)
    stats[cap] = {k: np.mean([x[k] for x in infos]) for k in ['dc_ITE_total_power_kW', 'dc_HVAC_total_power_kW', 'dc_water_usage', 'bat_CO2_footprint']}
for cap in [0.5, 5]:
    ratios = {k: stats[cap][k] / stats[1][k] for k in stats[1]}
    check(f"capacity {cap} vs 1 MW: IT, HVAC, water, CO2 all scale x{cap}", all(abs(r - cap) / cap < 0.01 for r in ratios.values()),
          ", ".join(f"{k.split('_')[1]} x{v:.3f}" for k, v in ratios.items()))
pue = {cap: (s['dc_ITE_total_power_kW'] + s['dc_HVAC_total_power_kW']) / s['dc_ITE_total_power_kW'] for cap, s in stats.items()}
check("PUE unchanged by capacity", max(pue.values()) - min(pue.values()) < 1e-3, ", ".join(f"{k} MW: {v:.4f}" for k, v in pue.items()))

print("4. max_bat_cap_Mw sets the battery capacity (MWh)")
auto = make()
check("None -> automatic sizing (1 h of peak DC power)", abs(auto.bat_env.max_bat_cap - auto.dc_env.ranges['max_battery_energy_Mwh']) < 1e-12,
      f"{auto.bat_env.max_bat_cap:.3f} MWh")
soc_traj = {}
for cap in [0.5, 1.0, 3.0, None]:
    e = make(max_bat_cap_Mw=cap)
    e.seed(0); e.reset(); socs = []
    for _ in range(96 * 4):  # 4 days of always-charge
        _, _, _, _, info = e.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 0})
        socs.append(info['agent_bat']['bat_SOC'])
    soc_traj[cap] = np.array(socs)
    stored = e.bat_env.battery.current_load
    check(f"max_bat_cap_Mw={cap}: capacity used, stored energy == SoC x capacity", (cap is None or e.bat_env.max_bat_cap == cap)
          and abs(stored - socs[-1] * e.bat_env.max_bat_cap) < 1e-6 and max(socs) <= 1 + 1e-9,
          f"capacity {e.bat_env.max_bat_cap:.3f} MWh, SoC after 1/2/4 days of charging {socs[95]:.3f}/{socs[191]:.3f}/{socs[-1]:.3f}, stored {stored:.3f} MWh")
check("SoC trajectory identical for every capacity (charging rate is relative to capacity)",
      all(np.allclose(soc_traj[k], soc_traj[1.0]) for k in soc_traj))
check("max_bat_cap_Mw=0 rejected", raises(lambda: make(max_bat_cap_Mw=0)) is not None)

print("5. individual_reward_weight mixes rewards")
for w in [1.0, 0.8, 0.0]:
    e = make(individual_reward_weight=w); e.seed(0); e.reset(); ok = True
    for _ in range(50):
        _, rew, _, _, _ = e.step({'agent_ls': 0, 'agent_dc': 2, 'agent_bat': 1})
        c = (1 - w) / 2; ls, dc, bat = e.ls_reward, e.dc_reward, e.bat_reward
        exp = {'agent_ls': w * ls + c * dc + c * bat, 'agent_dc': w * dc + c * ls + c * bat, 'agent_bat': w * bat + c * ls + c * dc}
        ok &= all(abs(rew[k] - exp[k]) < 1e-9 for k in exp)
    check(f"weight {w}: rewards match w*own + (1-w)/2*others", ok)
check("weight 1.5 rejected", raises(lambda: make(individual_reward_weight=1.5)) is not None)

print("6. Invalid actions raise instead of being treated as 'do nothing'")
e = make(); e.seed(0); e.reset()
for bad, label in [('defer', "string 'defer'"), (3, '3 (out of range)'), (-1, '-1'), (1.5, '1.5'), (None, 'None'), (np.array([0, 1]), 'two-element array')]:
    msg = raises(lambda: e.step({'agent_ls': bad, 'agent_dc': 1, 'agent_bat': 2}))
    check(f"rejects {label}", msg is not None, msg)
msg = raises(lambda: e.step({'agent_dc': 1, 'agent_bat': 2}))
check("rejects missing agent", msg is not None, msg)
for good, label in [(np.int64(1), 'np.int64'), (np.array([1]), 'array([1]) as from HARL'), (1.0, '1.0'), (np.array(2.0), '0-d float array')]:
    ok = raises(lambda: e.step({'agent_ls': good, 'agent_dc': 1, 'agent_bat': 2}), Exception) is None
    check(f"accepts {label}", ok)
e2 = SustainDC({**BASE, 'agents': ['agent_dc', 'agent_bat']}); e2.seed(0); e2.reset()
check("works with a subset of agents", raises(lambda: e2.step({'agent_dc': 2, 'agent_bat': 0}), Exception) is None)

print("7. CRAC setpoint resets between episodes")
e = make(); e.seed(0); e.reset()
for _ in range(10):
    e.step({'agent_ls': 1, 'agent_dc': 2, 'agent_bat': 2})
before = e.dc_env.raw_curr_stpt
e.reset()
check("setpoint back to initial value after reset", e.dc_env.raw_curr_stpt == e.dc_env.initial_stpt == 18,
      f"{before} before reset -> {e.dc_env.raw_curr_stpt} after")
dc_obs = e.dc_env.reset()[0]
raw = dc_obs[1] * e.dc_env.obs_delta[1] + e.dc_env.obs_min[1]
check("dc_env reset observation shows the reset setpoint", abs(raw - 18) < 1e-4, f"{raw:.2f}")

print("8. tou_reward works on fractional hours")
msg = None
try:
    vals = [rc.tou_reward({'hour': h, 'bat_total_energy_with_battery_KWh': 100.0}) for h in np.arange(0, 24, 0.25)]
except Exception as ex:
    msg = repr(ex)
check("all 96 quarter-hours", msg is None and len(vals) == 96, msg or f"price range {min(vals)/-100:.2f}-{max(vals)/-100:.2f}")
check("13.75 uses the 13:00 price", abs(rc.tou_reward({'hour': 13.75, 'bat_total_energy_with_battery_KWh': 1.0}) + 0.30) < 1e-12)
e = make(ls_reward='tou_reward', dc_reward='tou_reward', bat_reward='tou_reward')
check("full episode with tou_reward", raises(lambda: episode(e, 0), Exception) is None)

print("9. HARL wrapper end-to-end with the repo's YAML config (30-day episode, random actions)")
from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
args = yaml.safe_load(open('harl/configs/envs_cfgs/sustaindc.yaml'))
h = HARLSustainDCEnv(args); h.reset(); n = 0
def find_sdc(obj, depth=0):
    if isinstance(obj, SustainDC) or depth > 6:
        return obj
    for attr in ('env', 'aec_env', 'unwrapped', 'par_env'):
        if hasattr(obj, attr) and getattr(obj, attr) is not obj:
            r = find_sdc(getattr(obj, attr), depth + 1)
            if isinstance(r, SustainDC):
                return r
    return None
sdc = find_sdc(h)
while True:
    out = h.step(np.array([[sp.sample()] for sp in h.action_space])); n += 1
    if all(d for d in out[3]):
        break
check("episode completes", n == args['days_per_episode'] * 96, f"{n} steps; flexible_load {sdc.ls_env.flexible_workload_ratio if sdc else '?'}, battery {sdc.bat_env.max_bat_cap if sdc else '?'} MWh")

print("10. Dropped tasks are penalized in the default load-shifting reward")
e = make(flexible_load=0.6, days_per_episode=7, ls_reward='default_ls_reward', dc_reward='default_dc_reward', bat_reward='default_bat_reward'); e.seed(1); e.reset()  # the original rewards
drop_rewards, clean_rewards, drop_exact = [], [], []
while True:
    before = list(rc.energy_history)  # only default_ls_reward updates the history (once per step)
    _, _, _, tr, info = e.step({'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 2})
    i = info['agent_ls']
    d = i['ls_tasks_dropped']
    (drop_rewards if d > 0 else clean_rewards).append((e.ls_reward, d))
    if d > 0:  # same reward without the drops, on the same history: the drop must cost exactly -10 per task, clipped
        after = list(rc.energy_history)
        rc.energy_history.clear(); rc.energy_history.extend(before)
        r0 = rc.default_ls_reward({**i, 'ls_tasks_dropped': 0})
        rc.energy_history.clear(); rc.energy_history.extend(after)
        drop_exact.append(abs(e.ls_reward - np.clip(r0 - 10 * d, -10, 10)) < 1e-9)
    if tr['__all__']:
        break
check("steps with drops score the no-drop reward - 10 per task, clipped at -10",
      len(drop_rewards) > 0 and all(drop_exact),
      f"{len(drop_rewards)} steps with drops, {sum(d for _, d in drop_rewards)} tasks dropped, "
      f"rewards {min(r for r, _ in drop_rewards):.2f} to {max(r for r, _ in drop_rewards):.2f}" if drop_rewards else "no drops")
check("steps without drops are not floored", len(clean_rewards) > 0 and max(r for r, _ in clean_rewards) > -10,
      f"{len(clean_rewards)} steps, mean reward {np.mean([r for r, _ in clean_rewards]):.2f}")
import copy
hist = copy.copy(rc.energy_history)
def ls_reward_with(dropped):  # evaluate on a fixed energy history (each call appends to it)
    rc.energy_history.clear(); rc.energy_history.extend(hist)
    q = dict(i); q['ls_tasks_dropped'] = dropped; q['ls_overdue_penalty'] = 0; q['ls_oldest_task_age'] = 0
    return rc.default_ls_reward(q)
base, small = ls_reward_with(0), ls_reward_with(0.05)  # 0.05 "tasks" keeps the total inside the clip
check("penalty is -10 per dropped task before clipping", abs((small - base) - (-0.5)) < 1e-9, f"{base:.3f} -> {small:.3f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
