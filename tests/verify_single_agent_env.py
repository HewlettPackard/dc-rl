"""Checks for the single-agent SustainDC env. Run from the repo root: python tests/verify_single_agent_env.py"""
import os, sys, random
sys.path.insert(0, os.getcwd())
import numpy as np, yaml

import utils.reward_creator as rc
from sustaindc_single_env import SustainDCSingleAgentEnv
from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
from utils.joint_obs import JOINT_OBS_DIM

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    BASE = yaml.safe_load(f)
BASE.update({'days_per_episode': 1})
N_STEPS, SEED = 200, 7
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


print("1. Gymnasium / SB3 env checkers")
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check
for action_type in ['multidiscrete', 'discrete']:
    for name, fn in [('gymnasium', gym_check), ('sb3', sb3_check)]:
        env = SustainDCSingleAgentEnv({**BASE, 'action_type': action_type})
        try:
            fn(env, skip_render_check=True) if name == 'gymnasium' else fn(env, warn=True)
            check(f"{name} check_env ({action_type})", True)
        except Exception as e:  # noqa: BLE001
            check(f"{name} check_env ({action_type})", False, repr(e))

print("2. Spaces and action decoding")
env = SustainDCSingleAgentEnv(dict(BASE))
check("yaml turns the setpoint on", BASE.get('include_dc_setpoint') is True)
check("obs dim 30 with the setpoint", env.observation_space.shape == (JOINT_OBS_DIM + 1,))
off = SustainDCSingleAgentEnv({**BASE, 'include_dc_setpoint': False})
check("obs dim 29 with the setpoint off", off.observation_space.shape == (JOINT_OBS_DIM,))
check("MultiDiscrete([3,3,3])", list(env.action_space.nvec) == [3, 3, 3])
d = SustainDCSingleAgentEnv({**BASE, 'action_type': 'discrete'})
check("Discrete(27)", d.action_space.n == 27)
decoded = {tuple(d.decode_action(a).values()) for a in range(27)}
check("Discrete(27) decodes to all 27 distinct combos", len(decoded) == 27)
check("decode order is (ls, dc, bat)", d.decode_action(5) == {'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 2})
o_on, _ = env.reset(seed=0); o_off, _ = off.reset(seed=0)
check("setpoint flag only appends the last element", np.array_equal(o_on[:-1], o_off), "")
dc = env.env.dc_env
check("initial setpoint obs matches the env setpoint", np.isclose(o_on[-1], (dc.raw_curr_stpt - dc.min_temp) / (dc.max_temp - dc.min_temp)),
      f"obs {o_on[-1]:.3f}, setpoint {dc.raw_curr_stpt}")

print("2b. MARL side sees the setpoint too")
from sustaindc_env import SustainDC
sdc_on = SustainDC({**BASE, 'agents': ['agent_ls', 'agent_dc', 'agent_bat']})
sdc_off = SustainDC({**BASE, 'agents': ['agent_ls', 'agent_dc', 'agent_bat'], 'include_dc_setpoint': False})
check("cooling obs space 15 on / 14 off", sdc_on.observation_space[1].shape == (15,) and sdc_off.observation_space[1].shape == (14,))
harl_on = HARLSustainDCEnv(dict(BASE)); harl_off = HARLSustainDCEnv({**BASE, 'include_dc_setpoint': False})
check("HARL shared obs space 30 on / 29 off", harl_on.share_observation_space[0].shape == (30,) and harl_off.share_observation_space[0].shape == (29,))
check("HARL padded actor obs stays 26", harl_on.observation_space[1].shape == (26,))
sdc_on.reset(); o_dc = sdc_on.step({'agent_ls': 1, 'agent_dc': 2, 'agent_bat': 2})[0]['agent_dc']
d = sdc_on.dc_env
check("cooling obs[14] tracks the setpoint after an action", len(o_dc) == 15 and np.isclose(o_dc[14], (d.raw_curr_stpt - d.min_temp) / (d.max_temp - d.min_temp)),
      f"setpoint {d.raw_curr_stpt}")

print(f"3. Parity with the HARL (MARL) wrapper over {N_STEPS} steps, same seed and actions")
PARITY = {**BASE, "days_per_episode": 3}  # 288 steps > N_STEPS, so no truncation mid-test
rng = np.random.default_rng(SEED)
actions = rng.integers(0, 3, size=(N_STEPS, 3))

rc.energy_history.clear()
single = SustainDCSingleAgentEnv(dict(PARITY))
obs, _ = single.reset(seed=SEED)
s_obs, s_rew, s_soc = [obs], [], [None]
s_stpt_raw = [None]
for a in actions:
    obs, r, term, trunc, info = single.step(a)
    s_obs.append(obs); s_rew.append(r); s_soc.append(info['bat_SOC'])
    s_stpt_raw.append(info['dc_crac_setpoint'])

rc.energy_history.clear()
harl = HARLSustainDCEnv(dict(PARITY))
harl.seed(SEED - 1)  # HARL reset() increments the seed before seeding np.random
random.seed(SEED)    # HARL reset() does not seed `random` (used to sample the start day/hour)
_, sh, _ = harl.reset()
h_obs, h_rew = [sh[0]], []
for a in actions:
    _, sh, rew, _, _, _ = harl.step(np.array(a))
    h_obs.append(sh[0]); h_rew.append(rew[0][0])

obs_diff = max(np.abs(a - b).max() for a, b in zip(s_obs, h_obs))
rew_diff = np.abs(np.array(s_rew) - np.array(h_rew)).max()
check("joint obs == HARL shared obs", obs_diff < 1e-6, f"max diff {obs_diff:.2e}")
check("reward == HARL agent_ls reward", rew_diff < 1e-6, f"max diff {rew_diff:.2e}")
soc_ok = all(np.isclose(o[JOINT_OBS_DIM - 1], soc) for o, soc in zip(s_obs[1:], s_soc[1:]))
dcs = single.env.dc_env
stpt_ok = all(np.isclose(o[29], (v - dcs.min_temp) / (dcs.max_temp - dcs.min_temp)) for o, v in zip(s_obs[1:], s_stpt_raw[1:]))
check("obs[29] is the setpoint at every step", stpt_ok)
check("setpoint actually moves under random actions", np.ptp([o[29] for o in s_obs]) > 0, f"range {np.ptp([o[29] for o in s_obs]):.3f}")

check("obs[28] is the battery SoC (SOC bug fixed)", soc_ok)
check("battery SoC actually varies (not stuck at 0)", np.ptp([o[28] for o in h_obs]) > 0, f"range {np.ptp([o[28] for o in h_obs]):.3f}")

print("4. Episode ends with truncation (1 day = 96 steps)")
e = SustainDCSingleAgentEnv(dict(BASE)); e.reset(seed=1); n = 0
while True:
    _, _, term, trunc, _ = e.step(e.action_space.sample()); n += 1
    if term or trunc:
        break
check("truncated after 96 steps, never terminated", n == 96 and trunc and not term, f"{n} steps")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
