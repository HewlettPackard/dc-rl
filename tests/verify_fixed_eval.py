"""Checks that the training-time evaluation is reproducible: same model -> same eval returns and metrics.
Run from the repo root: python tests/verify_fixed_eval.py"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from sustaindc_single_env import SustainDCSingleAgentEnv
from train_single_agent import load_obs_normalizer, vecnormalize_path

with open('harl/configs/envs_cfgs/sustaindc.yaml') as f:
    CFG = {**yaml.safe_load(f), 'days_per_episode': 1}
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


if __name__ == '__main__':
    train = VecNormalize(make_vec_env(SustainDCSingleAgentEnv, n_envs=2, seed=0, env_kwargs={'env_config': dict(CFG)},
                                      vec_env_cls=SubprocVecEnv))
    model = PPO('MlpPolicy', train, n_steps=96, batch_size=96, seed=0, device='cpu')
    model.learn(192)  # a few updates so the policy and the obs statistics are not trivial
    ev = VecNormalize(make_vec_env(SustainDCSingleAgentEnv, n_envs=3, seed=10_000, env_kwargs={'env_config': dict(CFG)},
                                   vec_env_cls=SubprocVecEnv), training=False, norm_reward=False)
    ev.obs_rms = train.obs_rms

    def evaluate(seed=True, clear=True):
        if seed:
            ev.seed(10_000)
        if clear:
            ev.env_method('reset_reward_history')
        r, _ = evaluate_policy(model, ev, n_eval_episodes=3, deterministic=True, return_episode_rewards=True)
        return r

    print("1. Fixed-seed evaluation")
    a, b = evaluate(), evaluate()
    check("same model -> identical eval returns", a == b, f"{np.round(a, 3)} vs {np.round(b, 3)}")
    c = evaluate(seed=False)
    check("without reseeding the episodes change", a != c)
    d = evaluate(clear=False)
    check("without clearing the reward history the returns drift", a != d, f"{np.round(a, 3)} vs {np.round(d, 3)}")
    e = evaluate()
    check("clearing restores the exact returns", a == e)

    print("2. Saved normalization is reused at test time")
    path = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'verify_fixed_eval_model')
    model.save(path); train.save(vecnormalize_path(path))
    norm = load_obs_normalizer(path)
    obs = np.random.randn(SustainDCSingleAgentEnv(dict(CFG)).observation_space.shape[0]).astype(np.float32)
    check("load_obs_normalizer matches the training VecNormalize", np.allclose(norm(obs), train.normalize_obs(obs)))
    check("models without saved stats are left unnormalized", load_obs_normalizer(path + '_none')(obs) is obs)
    train.close(); ev.close()

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
