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
check("obs dim 29", env.observation_space.shape == (JOINT_OBS_DIM,))
check("MultiDiscrete([3,3,3])", list(env.action_space.nvec) == [3, 3, 3])
d = SustainDCSingleAgentEnv({**BASE, 'action_type': 'discrete'})
check("Discrete(27)", d.action_space.n == 27)
decoded = {tuple(d.decode_action(a).values()) for a in range(27)}
check("Discrete(27) decodes to all 27 distinct combos", len(decoded) == 27)
check("decode order is (ls, dc, bat)", d.decode_action(5) == {'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 2})
s = SustainDCSingleAgentEnv({**BASE, 'include_dc_setpoint': True})
o, _ = s.reset(seed=0)
check("include_dc_setpoint adds 1 dim in [0,1]", o.shape == (JOINT_OBS_DIM + 1,) and 0 <= o[-1] <= 1, f"{o[-1]:.3f}")

print(f"3. Parity with the HARL (MARL) wrapper over {N_STEPS} steps, same seed and actions")
PARITY = {**BASE, "days_per_episode": 3}  # 288 steps > N_STEPS, so no truncation mid-test
rng = np.random.default_rng(SEED)
actions = rng.integers(0, 3, size=(N_STEPS, 3))

rc.energy_history.clear()
single = SustainDCSingleAgentEnv(dict(PARITY))
obs, _ = single.reset(seed=SEED)
s_obs, s_rew, s_soc = [obs], [], [None]
for a in actions:
    obs, r, term, trunc, info = single.step(a)
    s_obs.append(obs); s_rew.append(r); s_soc.append(info['bat_SOC'])

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
