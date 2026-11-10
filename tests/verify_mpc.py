"""Checks for mpc_policy.py: simulator clones are exact and isolated, the persistence forecast uses past data only,
parallel rollouts give the same results as serial ones, and the MPC beats doing nothing.
Run from the repo root: python tests/verify_mpc.py"""
import copy, os, random, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import yaml

import utils.reward_creator as rc
from eval_compare import NoOpPolicy, run_episode, split_label
from mpc_policy import (DAY, EXOGENOUS_SERIES, MPCPolicy, SimCloner, make_mpc_policy, persistence_forecast,
                        planning_model)
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS, decode_joint_action

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    SCENARIO = {**yaml.safe_load(f), 'days_per_episode': 1, 'agents': list(AGENTS)}
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def trajectory(sdc, actions):
    out = []
    for a in actions:
        _, _, _, _, info = sdc.step(decode_joint_action(a, 'discrete'))
        i = info['agent_ls']
        out.append((i['bat_CO2_footprint'], i['ls_tasks_in_queue'], i['bat_SOC'], i['dc_crac_setpoint']))
    return np.array(out)


def started_env(seed=3, steps=5):
    random.seed(seed); np.random.seed(seed)
    sdc = SustainDC(dict(SCENARIO)); sdc.reset()
    for _ in range(steps):
        sdc.step({'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 0})
    return sdc


def main():
    print("1. Clones are exact models and leave the live simulator untouched")
    sdc = started_env()
    actions = np.random.default_rng(0).integers(27, size=40)
    history_before = list(rc.energy_history)
    clone = SimCloner.clone(sdc)
    traj_clone = trajectory(clone, actions)
    check("rollout on a clone does not touch the global reward history", list(rc.energy_history) == history_before)
    check("clone shares the static time series", clone.ci_m.carbon_smooth is sdc.ci_m.carbon_smooth)
    check("clone has its own task queue", clone.ls_env.tasks_queue is not sdc.ls_env.tasks_queue)
    traj_live = trajectory(sdc, actions)
    check("live env after the rollout follows the same trajectory as the clone", np.allclose(traj_clone, traj_live),
          f"max diff {np.abs(traj_clone - traj_live).max():.3g}")

    print("2. Persistence forecast uses past data only")
    x = np.arange(10 * DAY, dtype=float)
    t = 3 * DAY + 10
    fc = persistence_forecast(x, t, days=1)
    check("values up to the current step unchanged", np.array_equal(fc[:t + 1], x[:t + 1]))
    check("next day = same time yesterday", np.array_equal(fc[t + 1:t + 1 + DAY], x[t + 1 - DAY:t + 1]))
    check("later days repeat the profile", np.array_equal(fc[t + 1 + DAY:t + 1 + 2 * DAY], fc[t + 1:t + 1 + DAY]))
    fc2 = persistence_forecast(x, t, days=2)
    check("2-day average", np.allclose(fc2[t + 1:t + 1 + DAY],
                                       (x[t + 1 - DAY:t + 1] + x[t + 1 - 2 * DAY:t + 1 - DAY]) / 2))
    check("early steps fall back to the oldest available value", np.all(persistence_forecast(x, 5)[6:DAY] <= 5))

    sdc = started_env()
    ci_live = sdc.ci_m.carbon_smooth.copy()
    model = planning_model(sdc, 'persistence')
    tc = sdc.ci_m.time_step
    check("live CI series untouched", np.array_equal(sdc.ci_m.carbon_smooth, ci_live))
    check("planning model CI is the forecast after now",
          np.array_equal(model.ci_m.carbon_smooth[tc + 1:tc + 1 + DAY], ci_live[tc + 1 - DAY:tc + 1]))
    check("planning model CI is the truth up to now", np.array_equal(model.ci_m.carbon_smooth[:tc + 1], ci_live[:tc + 1]))
    check("all exogenous series replaced (incl. PV)", all(
        getattr(getattr(model, m), a) is not getattr(getattr(sdc, m), a)
        for m, attrs in EXOGENOUS_SERIES.items() for a in attrs) and 'pv_m' in EXOGENOUS_SERIES)

    # Two simulators identical up to now but with a different future: persistence must plan the same, oracle not.
    twin = copy.deepcopy(sdc)
    for m, attrs in EXOGENOUS_SERIES.items():
        mgr = getattr(twin, m)
        for a in attrs:
            arr = getattr(mgr, a).copy()
            arr[mgr.time_step + 1:] = arr[mgr.time_step + 1:][::-1]  # scramble the future
            setattr(mgr, a, arr)
    plans = {}
    for fc_kind in ('persistence', 'oracle'):
        for name, env in (('real', sdc), ('twin', twin)):
            p = MPCPolicy(horizon=32, block=4, samples=16, iters=2, forecast=fc_kind)
            p.reset(7)
            p.act_joint(env)
            plans[fc_kind, name] = p.plan
    check("persistence: same plan whatever the true future",
          np.array_equal(plans['persistence', 'real'], plans['persistence', 'twin']))
    check("oracle: plan depends on the true future", not np.array_equal(plans['oracle', 'real'], plans['oracle', 'twin']))

    pv_scn = {**SCENARIO, 'pv_config': {'capacity_kWp': 500}}
    random.seed(3); np.random.seed(3)
    pv_env = SustainDC(dict(pv_scn)); pv_env.reset()
    tp = pv_env.pv_m.time_step
    pv_model = planning_model(pv_env, 'persistence')
    check("PV forecast is yesterday's profile",
          pv_env.pv_m.enabled and np.array_equal(pv_model.pv_m.pv_kw[tp + 1:tp + 1 + DAY], pv_env.pv_m.pv_kw[tp + 1 - DAY:tp + 1]))

    print("3. Parameter and spec parsing")
    p = make_mpc_policy('horizon=50,block=8,samples=4,iters=2')
    check("horizon rounds up to whole blocks", (p.horizon, p.n_blocks, p.replan) == (56, 7, 8))
    for bad in ('horizn=10', 'forecast=crystal_ball'):
        try:
            make_mpc_policy(bad); check(f"bad parameter rejected: {bad}", False)
        except ValueError:
            check(f"bad parameter rejected: {bad}", True)
    cases = {
        'mpc_joint': ('mpc_joint', ''), 'mpc_joint=lbl': ('mpc_joint', 'lbl'),
        'mpc_joint:forecast=persistence': ('mpc_joint:forecast=persistence', ''),
        'mpc_joint:horizon=96,forecast=persistence=lbl': ('mpc_joint:horizon=96,forecast=persistence', 'lbl'),
        'mpc:H=16,forecast=realistic': ('mpc:H=16,forecast=realistic', ''),  # mpc_value.py specs
        'mpc:H=8,value=naive=mpc8': ('mpc:H=8,value=naive', 'mpc8'),
        'mpc:value=runs/v.npz': ('mpc:value=runs/v.npz', ''),
        'rule:ls,dc,bat=lbl': ('rule:ls,dc,bat', 'lbl'),
        'rule:bat:bat_lo=0.2': ('rule:bat:bat_lo=0.2', ''),
        'rule:bat:bat_lo=0.2=lbl': ('rule:bat:bat_lo=0.2', 'lbl'),
        'sb3:results/run:best_model=ppo': ('sb3:results/run:best_model', 'ppo'),
    }
    wrong = {s: split_label(s) for s, want in cases.items() if split_label(s) != want}
    check("eval_compare spec labels", not wrong, f"wrong: {wrong}" if wrong else "")

    print("4. One-day episodes vs do-nothing (same seed)")
    env = SustainDC(dict(SCENARIO))
    noop, fp_noop = run_episode(env, NoOpPolicy(), 1000)
    for fc_kind in ('oracle', 'persistence'):
        mpc = MPCPolicy(horizon=32, block=4, samples=16, iters=2, forecast=fc_kind)
        m, fp = run_episode(env, mpc, 1000)
        check(f"{fc_kind}: same exogenous inputs as noop", fp == fp_noop)
        check(f"{fc_kind}: work-adjusted CO2 lower than noop", m['co2_work_adjusted'] < noop['co2_work_adjusted'],
              f"{m['co2_work_adjusted']:.1f} vs {noop['co2_work_adjusted']:.1f} kg (raw {m['co2']:.1f} vs {noop['co2']:.1f})")
        check(f"{fc_kind}: no dropped tasks", m['tasks_dropped'] == 0, f"{m['tasks_dropped']:.0f}")
        print(f"     planning: {mpc.stats['replans']} replans, {mpc.stats['rollouts']} rollouts, "
              f"{mpc.stats['plan_seconds']:.1f}s")

    print("5. Parallel rollouts match serial ones")
    serial = MPCPolicy(horizon=32, block=4, samples=16, iters=2, forecast='persistence')
    m_serial, _ = run_episode(env, serial, 1001)
    parallel = MPCPolicy(horizon=32, block=4, samples=16, iters=2, forecast='persistence', workers=4)
    try:
        m_par, _ = run_episode(env, parallel, 1001)
    finally:
        parallel.close()
    check("identical episode metrics with workers=4", m_par == m_serial)
    print(f"     planning time: serial {serial.stats['plan_seconds']:.1f}s, 4 workers {parallel.stats['plan_seconds']:.1f}s")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == '__main__':  # required: rollout workers are started with 'spawn', which re-imports this module
    sys.exit(0 if main() else 1)
