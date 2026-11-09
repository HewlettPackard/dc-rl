"""Checks for the on-site solar PV model and its integration in SustainDC. Run from the repo root:
python tests/verify_pv.py"""
import os, sys, random
sys.path.insert(0, os.getcwd())
import numpy as np, yaml

from envs.pv_model import ac_rating_kw, compute_pv_ac_kw
from utils.utils_cf import obtain_paths
from utils.joint_obs import PV_OBS_DIM, joint_obs_dim
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS, SustainDCSingleAgentEnv

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    BASE = yaml.safe_load(f)
BASE.update({'days_per_episode': 2, 'agents': list(AGENTS)})
# 1 MWp next to a ~1 MW data center: large enough that midday PV exceeds the load and there is a surplus.
PV = {'capacity_kWp': 1000}
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def rollout(cfg, seed, policy='random'):
    """Run one episode; return per-step (info, action, energy added/removed by the battery, obs, pre-step battery
    state: SoC and the most energy (kWh) the battery could take this step)."""
    random.seed(seed); np.random.seed(seed)
    env = SustainDC(cfg)
    random.seed(seed); np.random.seed(seed)
    env.reset()
    rng = np.random.default_rng(seed)
    steps = []
    while True:
        act = ({a: int(rng.integers(3)) for a in env.agents} if policy == 'random'
               else {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2})
        bat, benv = env.bat_env.battery, env.bat_env
        t_u = benv.charging_rate_modifier(bat) * 15 / 60
        pre = {'soc': bat.get_battery_soc(),
               'charge_limit_KWh': min(bat.calc_max_charge(t_u), bat.capacity) * bat.eff_c * t_u * 1e3}
        n_before = len(env.bat_env.energy_added_removed)
        obs, _, _, tr, info = env.step(act)
        added = env.bat_env.energy_added_removed[n_before:]
        steps.append((info['agent_ls'], act, added[0] if added else 0.0, obs, pre))
        if tr['__all__']:
            return env, steps


print("1. PV model physics (pvlib, EPW irradiance)")
# Typical PVWatts annual AC yields (kWh/kWp/yr) for a fixed latitude-tilt array.
ranges = {'az': (1650, 1900), 'ca': (1550, 1850), 'ny': (1250, 1500), 'wa': (1000, 1250)}
for loc, (lo, hi) in ranges.items():
    epw = obtain_paths(loc)[1]
    p = compute_pv_ac_kw(epw, {'capacity_kWp': 1.0})
    prof = p.reshape(365, 24).mean(axis=0)  # mean daily profile, local standard time (EPW row = hour ending h+1)
    check(f"{loc}: annual yield in PVWatts range", lo <= p.sum() <= hi, f"{p.sum():.0f} kWh/kWp")
    check(f"{loc}: zero at night, peak around solar noon", prof[:4].sum() == 0 and prof[21:].sum() == 0
          and 11 <= prof.argmax() <= 13, f"peak row-hour {prof.argmax()}")
    check(f"{loc}: AC output clipped at the inverter rating", p.max() <= ac_rating_kw({'capacity_kWp': 1.0}) + 1e-9)
epw = obtain_paths('ca')[1]
fixed, tracked = compute_pv_ac_kw(epw, {'capacity_kWp': 1.0}), compute_pv_ac_kw(epw, {'capacity_kWp': 1.0, 'tracking': 'single_axis'})
check("single-axis tracking yields more than fixed tilt", tracked.sum() > 1.05 * fixed.sum(),
      f"{tracked.sum():.0f} vs {fixed.sum():.0f}")
check("output scales linearly with capacity", np.allclose(compute_pv_ac_kw(epw, {'capacity_kWp': 250.0}), 250 * fixed))
check("capacity 0 gives no PV", not compute_pv_ac_kw(epw, {}).any())
try:
    compute_pv_ac_kw(epw, {'capacity_kwp': 1.0})
    check("unknown pv_config key raises", False)
except ValueError:
    check("unknown pv_config key raises", True)

print("2. PV in the environment: sync with the weather and the env clock")
env, steps = rollout({**BASE, 'location': 'ca', 'pv_config': PV}, seed=3)
pv_m, wm = env.pv_m, env.weather_m
check("PV series rolled with the same random day shift as the weather",
      np.array_equal(pv_m.pv_kw, np.roll(pv_m.original_pv_kw, wm.last_roll_days * 96)), f"roll {wm.last_roll_days} days")
hours = np.array([s[0]['hour'] for s in steps]); pv = np.array([s[0]['pv_power_kW'] for s in steps])
check("PV is zero at night on the env clock (hours 0-4, 21-24)", not pv[(hours < 4) | (hours >= 21)].any())
check("PV is produced around midday on the env clock", pv[(hours >= 11) & (hours < 14)].mean() > 0.1 * PV['capacity_kWp'])

print("3. Energy balance and carbon accounting (random actions, CA, 1 MWp)")
bal_err = flow_err = co2_err = 0.0
neg_grid = cap_violations = 0
idle_surplus = idle_stored = idle_grid_bad = dis_surplus = dis_surplus_ok = curtailed_steps = curtail_bad = 0
for info, act, added, _, pre in steps:
    load = info['bat_total_energy_without_battery_KWh']
    charge, discharge = max(added, 0.0), max(-added, 0.0)
    grid, pv_e = info['grid_energy_KWh'], info['pv_energy_KWh']
    bal_err = max(bal_err, abs(load + charge - (grid + info['pv_self_consumed_KWh'] + info['pv_to_battery_KWh'] + discharge)))
    flow_err = max(flow_err, abs(pv_e - (info['pv_self_consumed_KWh'] + info['pv_to_battery_KWh']
                                         + info['pv_curtailed_KWh'] + info['pv_exported_KWh'])))
    co2_err = max(co2_err, abs(info['bat_CO2_footprint'] - grid * info['bat_avg_CI']))
    neg_grid += grid < 0
    cap_violations += discharge > max(load - pv_e, 0) + 1e-6
    surplus = pv_e > load
    if act['agent_bat'] == 2:  # idle: never charges from the grid, but stores PV surplus while not full
        idle_grid_bad += abs(grid - max(load - pv_e, 0)) > 1e-6
        if surplus and pre['soc'] < 0.99:
            idle_surplus += 1
            idle_stored += info['pv_to_battery_KWh'] > 0 and grid == 0
    if act['agent_bat'] == 1 and surplus:  # discharge during a surplus: nothing discharged, surplus absorbed
        dis_surplus += 1
        dis_surplus_ok += discharge == 0 and info['bat_mode'] == 'pv_charge'
    if act['agent_bat'] != 0 and info['pv_curtailed_KWh'] > 1e-9:
        # PV is only curtailed when the battery took all it could this step
        curtailed_steps += 1
        curtail_bad += not (abs(info['pv_to_battery_KWh'] - pre['charge_limit_KWh']) < 1e-6 or pre['soc'] > 0.999)
check("load + charge = grid + PV self-use + PV to battery + discharge", bal_err < 1e-6, f"max err {bal_err:.2e} kWh")
check("PV = self-use + to battery + curtailed + exported", flow_err < 1e-6, f"max err {flow_err:.2e} kWh")
check("CO2 = grid import x CI", co2_err < 1e-6)
check("grid import never negative", neg_grid == 0)
check("battery never discharges more than the net (after PV) load", cap_violations == 0)
check("bat_total_energy_with_battery_KWh is the grid import",
      all(s[0]['bat_total_energy_with_battery_KWh'] == s[0]['grid_energy_KWh'] for s in steps))
check("idle stores PV surplus automatically (grid import 0)", idle_surplus > 0 and idle_stored == idle_surplus,
      f"{idle_stored}/{idle_surplus} idle surplus steps")
check("idle never charges from the grid", idle_grid_bad == 0)
check("discharge during a surplus absorbs PV instead", dis_surplus > 0 and dis_surplus_ok == dis_surplus,
      f"{dis_surplus_ok}/{dis_surplus} steps")
check("PV curtailed only when the battery took its full per-step charge", curtail_bad == 0,
      f"{curtailed_steps} curtailment steps")

print("4. PV lowers grid energy and CO2 (same seed, do-nothing actions)")
_, no_pv = rollout({**BASE, 'location': 'ca'}, seed=5, policy='noop')
_, with_pv = rollout({**BASE, 'location': 'ca', 'pv_config': PV}, seed=5, policy='noop')
co2_no, co2_pv = (sum(s[0]['bat_CO2_footprint'] for s in r) for r in (no_pv, with_pv))
it_no, it_pv = (sum(s[0]['dc_ITE_total_power_kW'] for s in r) for r in (no_pv, with_pv))
check("same exogenous inputs and DC power with and without PV", it_no == it_pv)
check("CO2 is lower with PV", co2_pv < co2_no, f"{co2_no / 1e3:.0f} -> {co2_pv / 1e3:.0f} kgCO2")
stored = sum(s[0]['pv_to_battery_KWh'] for s in with_pv)
soc_end = with_pv[-1][0]['bat_SOC']
check("do-nothing policy: the battery fills up from PV surplus", stored > 0 and soc_end > 0,
      f"{stored:.0f} kWh stored, final SoC {soc_end:.2f}")

print("5. Export credit")
_, exp = rollout({**BASE, 'location': 'ca', 'pv_config': PV, 'pv_export_credit': True}, seed=3)
check("nothing curtailed; surplus is exported", all(s[0]['pv_curtailed_KWh'] == 0 for s in exp)
      and sum(s[0]['pv_exported_KWh'] for s in exp) > 0)
check("CO2 = (grid import - export) x CI",
      max(abs(s[0]['bat_CO2_footprint'] - (s[0]['grid_energy_KWh'] - s[0]['pv_exported_KWh']) * s[0]['bat_avg_CI'])
          for s in exp) < 1e-6)

print("6. Observations (include_pv_obs)")
env_off = SustainDC({**BASE, 'pv_config': PV})
env_on = SustainDC({**BASE, 'pv_config': PV, 'include_pv_obs': True})
dims_off = [s.shape[0] for s in env_off.observation_space]
dims_on = [s.shape[0] for s in env_on.observation_space]
check("each agent's observation grows by 5", dims_on == [d + PV_OBS_DIM for d in dims_off], f"{dims_off} -> {dims_on}")
random.seed(1); np.random.seed(1)
obs = env_on.reset()
for _ in range(60):
    obs, *_ = env_on.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2})
check("observations match their spaces", all(obs[a].shape == env_on.observation_space[i].shape
                                             for i, a in enumerate(AGENTS)))
feats = [obs[a][-PV_OBS_DIM:] for a in AGENTS]
check("all agents see the same PV features, in [0, 1]",
      all(np.array_equal(f, feats[0]) for f in feats) and feats[0].min() >= 0 and feats[0].max() <= 1)
check("first PV feature = current PV / AC rating",
      np.isclose(feats[0][0], env_on.pv_m.get_current_pv_kw() / ac_rating_kw(PV), atol=1e-6))
single = SustainDCSingleAgentEnv({**BASE, 'pv_config': PV, 'include_pv_obs': True})
o, _ = single.reset(seed=0)
check("single-agent joint obs has the PV features at the end",
      o.shape == (joint_obs_dim(True, True),) == single.observation_space.shape
      and np.array_equal(o[-PV_OBS_DIM:], single.env.ls_state[-PV_OBS_DIM:]), f"{o.shape}")
from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
harl = HARLSustainDCEnv({**BASE, 'pv_config': PV, 'include_pv_obs': True})
_, s_obs, _ = harl.reset()
check("HARL shared obs has the joint size", s_obs[0].shape == (joint_obs_dim(BASE['include_dc_setpoint'], True),)
      == harl.share_observation_space[0].shape, f"{s_obs[0].shape}")

print("7. eval_compare metrics")
from eval_compare import NoOpPolicy, run_episode
m, _ = run_episode(SustainDC({**BASE, 'pv_config': PV}), NoOpPolicy(), 2000)
check("pv_energy metric is reported", m['pv_energy'] > 0, f"{m['pv_energy']:.2f} MWh")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
