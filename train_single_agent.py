"""Train a single agent (full observation, joint action) on SustainDC with Stable-Baselines3.

Run from the repo root, e.g.:
    python train_single_agent.py --algo ppo --n_envs 8 --total_timesteps 5000000
    python train_single_agent.py --algo dqn --location ny --month 6

PPO hyperparameters default to harl/configs/algos_cfgs/happo.yaml so runs are comparable with the MARL ones.
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import yaml
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
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


def list_checkpoints(ckpt_dir, prefix):
    """Return [(num_timesteps, path)] of the model checkpoints in ckpt_dir, oldest first."""
    found = []
    for path in glob.glob(os.path.join(ckpt_dir, f'{prefix}_*_steps.zip')):
        m = re.search(r'_(\d+)_steps\.zip$', path)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def replay_buffer_path(ckpt_path):
    return ckpt_path[:-len('.zip')] + '_replay_buffer.pkl'


class ResumableCheckpointCallback(BaseCallback):
    """Save the model (and the replay buffer for off-policy algos) every `save_freq` env steps.

    Files are written to a temp name and renamed, so a job killed mid-save never leaves a corrupt checkpoint.
    Only the newest `keep` checkpoints are kept (DQN replay buffers are large).
    """

    def __init__(self, ckpt_dir, prefix, save_freq, keep=2):
        super().__init__()
        self.ckpt_dir, self.prefix, self.save_freq, self.keep = ckpt_dir, prefix, save_freq, keep

    def _on_training_start(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._last_bucket = self.num_timesteps // self.save_freq  # don't re-save right after resuming

    def _on_step(self):
        bucket = self.num_timesteps // self.save_freq
        if bucket > self._last_bucket:
            self._last_bucket = bucket
            self.save()
        return True

    def save(self):
        path = os.path.join(self.ckpt_dir, f'{self.prefix}_{self.num_timesteps}_steps.zip')
        if isinstance(self.model, OffPolicyAlgorithm):
            buf = replay_buffer_path(path)
            self.model.save_replay_buffer(buf + '.tmp')
            os.replace(buf + '.tmp', buf)
        # The model zip is written last: it marks the checkpoint as complete.
        self.model.save(path + '.tmp')
        os.replace(path + '.tmp', path)
        for _, old in list_checkpoints(self.ckpt_dir, self.prefix)[:-self.keep]:
            for f in (old, replay_buffer_path(old)):
                if os.path.exists(f):
                    os.remove(f)
        print(f'Checkpoint saved at {self.num_timesteps} steps: {path}', flush=True)


def restore_eval_history(eval_cb, npz_path, resumed_at):
    """Reload EvalCallback's history so a resumed run keeps evaluations.npz and the best-model threshold."""
    if not os.path.exists(npz_path):
        return
    d = np.load(npz_path)
    if len(d['timesteps']) == 0:
        return
    # best_model.zip on disk is the best over *all* past evals, even ones after the checkpoint we resume from.
    eval_cb.best_mean_reward = float(np.max(np.mean(d['results'], axis=1)))
    keep = d['timesteps'] <= resumed_at  # keep the logged history monotonic in timesteps
    eval_cb.evaluations_timesteps = d['timesteps'][keep].tolist()
    eval_cb.evaluations_results = d['results'][keep].tolist()
    eval_cb.evaluations_length = d['ep_lengths'][keep].tolist()


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
    p.add_argument('--include_dc_setpoint', action=argparse.BooleanOptionalAction, default=None,
                   help='Observe the current CRAC setpoint. Default: the value in the env yaml (on).')
    p.add_argument('--eval_freq', type=int, default=50_000, help='Evaluate every N total env steps.')
    p.add_argument('--n_eval_episodes', type=int, default=1)
    p.add_argument('--checkpoint_freq', type=int, default=250_000, help='Save a checkpoint every N total env steps.')
    p.add_argument('--results_root', default='results',
                   help='Top-level folder for runs: <root>/sustaindc/<location>/single_<algo>/<exp_name>/<run>.')
    p.add_argument('--run_id', default=None,
                   help='Run directory suffix (default: a timestamp). Reusing a run id resumes from its latest '
                        'checkpoint, e.g. the Slurm job id, so a preempted/requeued job continues where it stopped.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--torch_threads', type=int, default=1,
                   help='Torch threads for the learner. 1 leaves the other cores to the env processes.')
    args = p.parse_args()
    torch.set_num_threads(args.torch_threads)

    with open(ENV_CFG) as f:
        env_config = yaml.safe_load(f)
    for key in ['location', 'month', 'days_per_episode']:
        if getattr(args, key) is not None:
            env_config[key] = getattr(args, key)
    env_config['action_type'] = 'discrete' if args.algo == 'dqn' else args.action_type
    env_config['reward_type'] = args.reward_type
    if args.include_dc_setpoint is not None:
        env_config['include_dc_setpoint'] = args.include_dc_setpoint

    run_id = args.run_id or datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_dir = os.path.join(args.results_root, 'sustaindc', env_config['location'], f'single_{args.algo}', args.exp_name,
                           f'seed-{args.seed:0>5}-{run_id}')
    ckpt_dir = os.path.join(run_dir, 'models', 'checkpoints')
    os.makedirs(run_dir, exist_ok=True)

    if args.algo == 'ppo':
        algo_kwargs = ppo_kwargs_from_happo(args.n_envs)
    else:
        algo_kwargs = dict(gamma=0.995)
    config = {'main_args': vars(args), 'env_args': env_config,
              'algo_args': {k: str(v) if k == 'policy_kwargs' else v for k, v in algo_kwargs.items()}}

    checkpoints = list_checkpoints(ckpt_dir, args.algo)
    resumed_at = checkpoints[-1][0] if checkpoints else 0
    config_path = os.path.join(run_dir, 'config.json')
    if checkpoints and os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        if saved['env_args'] != json.loads(json.dumps(env_config)) or saved['algo_args'] != json.loads(json.dumps(config['algo_args'])):
            raise SystemExit(f'Refusing to resume {run_dir}: env/algo config differs from the saved config.json. '
                             f'Use a different --run_id for a new run.')
    else:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Offset env seeds on resume so the restarted envs don't replay the same episode starts.
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(SustainDCSingleAgentEnv, n_envs=args.n_envs, seed=args.seed + resumed_at,
                       env_kwargs={'env_config': dict(env_config)}, vec_env_cls=vec_cls)
    eval_env = make_vec_env(SustainDCSingleAgentEnv, n_envs=1, seed=args.seed + 10_000,
                            env_kwargs={'env_config': dict(env_config)}, vec_env_cls=DummyVecEnv)

    if checkpoints:
        ckpt = checkpoints[-1][1]
        print(f'Resuming from {ckpt} ({resumed_at} steps)', flush=True)
        model = ALGOS[args.algo].load(ckpt, env=env, device=args.device)
        if isinstance(model, OffPolicyAlgorithm) and os.path.exists(replay_buffer_path(ckpt)):
            model.load_replay_buffer(replay_buffer_path(ckpt))
    else:
        model = ALGOS[args.algo]('MlpPolicy', env, seed=args.seed, device=args.device, verbose=1,
                                 tensorboard_log=os.path.join(run_dir, 'tb'), **algo_kwargs)

    eval_cb = EvalCallback(eval_env, n_eval_episodes=args.n_eval_episodes, deterministic=True,
                           eval_freq=max(args.eval_freq // args.n_envs, 1),
                           best_model_save_path=os.path.join(run_dir, 'models'),
                           log_path=os.path.join(run_dir, 'eval'))
    if checkpoints:
        restore_eval_history(eval_cb, os.path.join(run_dir, 'eval', 'evaluations.npz'), resumed_at)
    callbacks = CallbackList([
        SustainDCMetricsCallback(),
        eval_cb,
        ResumableCheckpointCallback(ckpt_dir, args.algo, save_freq=args.checkpoint_freq),
    ])

    print(f'Run dir: {run_dir}', flush=True)
    remaining = args.total_timesteps - model.num_timesteps
    if remaining > 0:
        # reset_num_timesteps=False continues the step counter, LR/exploration schedules and the same tensorboard run.
        model.learn(total_timesteps=remaining, callback=callbacks, tb_log_name=args.algo,
                    reset_num_timesteps=not checkpoints)
    model.save(os.path.join(run_dir, 'models', 'final_model'))
    env.close(); eval_env.close()
    print(f'Saved final model to {run_dir}/models/final_model.zip')


if __name__ == '__main__':
    main()
