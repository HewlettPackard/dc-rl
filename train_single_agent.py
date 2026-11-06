"""Train a single agent (full observation, joint action) on SustainDC with Stable-Baselines3.

Run from the repo root, e.g.:
    python train_single_agent.py --algo ppo --n_envs 8 --total_timesteps 5000000
    python train_single_agent.py --algo dqn --location ny --month 6

PPO hyperparameters default to harl/configs/algos_cfgs/happo.yaml so runs are comparable with the MARL ones.
"""
import argparse
import datetime
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import torch
import yaml
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sustaindc_single_env import SustainDCSingleAgentEnv

ENV_CFG = 'harl/configs/envs_cfgs/sustaindc.yaml'
HAPPO_CFG = 'harl/configs/algos_cfgs/happo.yaml'
ALGOS = {'ppo': PPO, 'a2c': A2C, 'dqn': DQN}

# Per-step info keys summed over an episode and logged (same metrics as harl/envs/sustaindc/sustaindc_logger.py).
EPISODE_SUM_METRICS = {
    'CO2_footprint_sum': 'bat_CO2_footprint',
    'net_energy_sum': 'bat_total_energy_with_battery_KWh',
    'water_usage': 'dc_water_usage',
    'ls_tasks_dropped': 'ls_tasks_dropped',
    'ite_power_sum': 'dc_ITE_total_power_kW',
    'hvac_power_sum': 'dc_HVAC_total_power_kW',
    'ct_power_sum': 'dc_CT_total_power_kW',
    'chiller_power_sum': 'dc_Compressor_total_power_kW',
}
EPISODE_MEAN_METRICS = {'ls_tasks_in_queue': 'ls_tasks_in_queue', 'bat_SOC': 'bat_SOC',
                        'dc_crac_setpoint': 'dc_crac_setpoint'}


class SustainDCMetricsCallback(BaseCallback):
    """Accumulate SustainDC info metrics per env and log their per-episode values to tensorboard."""

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._sums = [dict.fromkeys(EPISODE_SUM_METRICS, 0.0) for _ in range(n)]
        self._means = [dict.fromkeys(EPISODE_MEAN_METRICS, 0.0) for _ in range(n)]
        self._steps = [0] * n

    def _on_step(self):
        for i, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            for name, key in EPISODE_SUM_METRICS.items():
                self._sums[i][name] += float(info.get(key, 0.0))
            for name, key in EPISODE_MEAN_METRICS.items():
                self._means[i][name] += float(info.get(key, 0.0))
            self._steps[i] += 1
            if done:
                for name, v in self._sums[i].items():
                    self.logger.record_mean(f'sustaindc/{name}', v)
                for name, v in self._means[i].items():
                    self.logger.record_mean(f'sustaindc/{name}_mean', v / max(self._steps[i], 1))
                self._sums[i] = dict.fromkeys(EPISODE_SUM_METRICS, 0.0)
                self._means[i] = dict.fromkeys(EPISODE_MEAN_METRICS, 0.0)
                self._steps[i] = 0
        return True


def ppo_kwargs_from_happo(n_envs):
    """Map the HAPPO config onto SB3 PPO arguments."""
    with open(HAPPO_CFG) as f:
        cfg = yaml.safe_load(f)
    train, model, algo = cfg['train'], cfg['model'], cfg['algo']
    n_steps = train['episode_length']
    act = {'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU}[model['activation_func']]
    return dict(
        learning_rate=model['lr'],  # HARL keeps lr under 'model'
        n_steps=n_steps,
        batch_size=max(n_steps * n_envs // algo['actor_num_mini_batch'], 1),
        n_epochs=algo['ppo_epoch'],
        gamma=algo['gamma'],
        gae_lambda=algo['gae_lambda'],
        clip_range=algo['clip_param'],
        ent_coef=algo['entropy_coef'],
        max_grad_norm=algo['max_grad_norm'],
        policy_kwargs=dict(net_arch=list(model['hidden_sizes']), activation_fn=act),
    )


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--algo', choices=list(ALGOS), default='ppo')
    p.add_argument('--exp_name', default='single_agent')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--n_envs', type=int, default=8)
    p.add_argument('--total_timesteps', type=int, default=5_000_000)
    p.add_argument('--location', default=None, help='Override env config location (ny, ca, az, ga, ...).')
    p.add_argument('--month', type=int, default=None, help='Override env config month (0-11).')
    p.add_argument('--days_per_episode', type=int, default=None)
    p.add_argument('--action_type', choices=['multidiscrete', 'discrete'], default='multidiscrete',
                   help='Joint action encoding. DQN always uses discrete.')
    p.add_argument('--reward_type', choices=['agent_ls', 'ls_raw'], default='agent_ls')
    p.add_argument('--include_dc_setpoint', action='store_true')
    p.add_argument('--eval_freq', type=int, default=50_000, help='Evaluate every N total env steps.')
    p.add_argument('--n_eval_episodes', type=int, default=1)
    p.add_argument('--checkpoint_freq', type=int, default=500_000, help='Save a checkpoint every N total env steps.')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    with open(ENV_CFG) as f:
        env_config = yaml.safe_load(f)
    for key in ['location', 'month', 'days_per_episode']:
        if getattr(args, key) is not None:
            env_config[key] = getattr(args, key)
    env_config['action_type'] = 'discrete' if args.algo == 'dqn' else args.action_type
    env_config['reward_type'] = args.reward_type
    env_config['include_dc_setpoint'] = args.include_dc_setpoint

    stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_dir = os.path.join('results', 'sustaindc', env_config['location'], f'single_{args.algo}', args.exp_name,
                           f'seed-{args.seed:0>5}-{stamp}')
    os.makedirs(run_dir, exist_ok=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(SustainDCSingleAgentEnv, n_envs=args.n_envs, seed=args.seed,
                       env_kwargs={'env_config': dict(env_config)}, vec_env_cls=vec_cls)
    eval_env = make_vec_env(SustainDCSingleAgentEnv, n_envs=1, seed=args.seed + 10_000,
                            env_kwargs={'env_config': dict(env_config)}, vec_env_cls=DummyVecEnv)

    if args.algo == 'ppo':
        algo_kwargs = ppo_kwargs_from_happo(args.n_envs)
    else:
        algo_kwargs = dict(gamma=0.995)
    model = ALGOS[args.algo]('MlpPolicy', env, seed=args.seed, device=args.device, verbose=1,
                             tensorboard_log=os.path.join(run_dir, 'tb'), **algo_kwargs)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'main_args': vars(args), 'env_args': env_config,
                   'algo_args': {k: str(v) if k == 'policy_kwargs' else v for k, v in algo_kwargs.items()}},
                  f, indent=2)

    callbacks = CallbackList([
        SustainDCMetricsCallback(),
        EvalCallback(eval_env, n_eval_episodes=args.n_eval_episodes, deterministic=True,
                     eval_freq=max(args.eval_freq // args.n_envs, 1),
                     best_model_save_path=os.path.join(run_dir, 'models'),
                     log_path=os.path.join(run_dir, 'eval')),
        CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                           save_path=os.path.join(run_dir, 'models', 'checkpoints'), name_prefix=args.algo),
    ])
    print(f'Run dir: {run_dir}')
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, tb_log_name=args.algo)
    model.save(os.path.join(run_dir, 'models', 'final_model'))
    env.close(); eval_env.close()
    print(f'Saved final model to {run_dir}/models/final_model.zip')


if __name__ == '__main__':
    main()
