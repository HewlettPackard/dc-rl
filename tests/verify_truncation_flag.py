"""Checks that HARL treats the end of a SustainDC episode as a time-limit truncation, not a termination.
Run from the repo root: python tests/verify_truncation_flag.py"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, yaml

from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
from harl.utils.envs_tools import make_train_env
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    BASE = yaml.safe_load(f)
BASE.update({'days_per_episode': 1})
STEPS_PER_DAY = 96
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def noop(n_agents):
    return np.array([[1], [1], [2]][:n_agents])


print("1. Wrapper flags only the final (truncated) step, for every agent")
env = HARLSustainDCEnv(dict(BASE))
env.reset()
flags, steps = [], 0
while True:
    _, _, _, dones, infos, _ = env.step(noop(env.n_agents))
    steps += 1
    flags.append([info['bad_transition'] for info in infos])
    if all(dones):
        break
check("episode lasts one day", steps == STEPS_PER_DAY, f"{steps} steps")
check("no flag before the last step", not any(any(f) for f in flags[:-1]))
check("every agent flagged on the last step", all(flags[-1]), f"{flags[-1]}")
check("agents get separate info dicts", len({id(i) for i in infos}) == env.n_agents)

print("2. Vectorized env (as used in training) passes the flag and the pre-reset observation")
venv = make_train_env('sustaindc', 1, 1, dict(BASE))
venv.reset()
for t in range(STEPS_PER_DAY):
    _, _, _, dones, infos, _ = venv.step(np.stack([noop(env.n_agents)]))
check("vec env reports done at the time limit", np.all(dones[0]))
check("flag present in the vec env infos", all(i.get('bad_transition') for i in infos[0]))
check("original_obs kept for off-policy bootstrapping", 'original_obs' in infos[0][0])
venv.close()

print("3. On-policy critic returns bootstrap through a truncation (bad_mask 0) and stop at a termination (bad_mask 1)")
args = {'episode_length': 4, 'n_rollout_threads': 1, 'hidden_sizes': [8], 'recurrent_n': 1, 'gamma': 0.9,
        'gae_lambda': 0.95, 'use_gae': True, 'use_proper_time_limits': True}
space = env.share_observation_space[0]
for use_gae in (True, False):
    for bad, label in ((0.0, 'truncation'), (1.0, 'termination')):
        buf = OnPolicyCriticBufferEP({**args, 'use_gae': use_gae}, space)
        values = [5.0, 6.0, 7.0, 8.0]
        for t in range(4):
            last = t == 1  # episode ends after step 1 (the runner stores masks/bad_masks at t + 1)
            buf.insert(np.zeros((1, *space.shape)), np.zeros((1, 1, 8)), np.array([[values[t]]]), np.array([[1.0]]),
                       np.array([[0.0 if last else 1.0]]), np.array([[bad if last else 1.0]]))
        buf.compute_returns(np.array([[9.0]]))
        r1, r0 = buf.returns[1, 0, 0], buf.returns[0, 0, 0]
        if bad == 0.0:  # last step's target is its own value; step 0 bootstraps from V(s_1) = 6
            ok = np.isclose(r1, 6.0) and (np.isclose(r0, 1 + 0.9 * 6.0) if not use_gae else r0 > 1.0 + 1e-6)
        else:  # true termination: the last step's return is just its reward
            ok = np.isclose(r1, 1.0)
        check(f"{'GAE' if use_gae else 'MC'} {label}", ok, f"returns[0:2] = {r0:.3f}, {r1:.3f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
