"""Train a single agent (full observation, joint action) on SustainDC with Stable-Baselines3.

Run from the repo root, e.g.:
    python train_single_agent.py --algo ppo --n_envs 8 --total_timesteps 5000000
    python train_single_agent.py --algo dqn --location ny --month 6

PPO hyperparameters default to harl/configs/algos_cfgs/happo.yaml so runs are comparable with the MARL ones;
--learning_rate, --ent_coef, --clip_range, --n_epochs and --vf_coef override them.
Observations and rewards are normalized with VecNormalize by default (--no-norm_obs / --no-norm_reward to turn off);
its statistics are saved next to every model as <model>_vecnormalize.pkl.
"""
import argparse
import datetime
import glob
import json
import os
import pickle
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from sustaindc_single_env import SustainDCSingleAgentEnv

ENV_CFG = 'harl/configs/envs_cfgs/sustaindc.yaml'
HAPPO_CFG = 'harl/configs/algos_cfgs/happo.yaml'
ALGOS = {'ppo': PPO, 'a2c': A2C, 'dqn': DQN}

# Per-step info keys summed over an episode and logged (same metrics as harl/envs/sustaindc/sustaindc_logger.py).
EPISODE_SUM_METRICS = {
    'CO2_footprint_sum': 'bat_CO2_footprint',
    'net_energy_sum': 'bat_total_energy_with_battery_KWh',
    'pv_energy_sum': 'pv_energy_KWh',
    'pv_curtailed_sum': 'pv_curtailed_KWh',
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
                self.logger.record_mean('sustaindc/ls_tasks_queued_end', float(info.get('ls_tasks_in_queue', 0.0)))
                self._sums[i] = dict.fromkeys(EPISODE_SUM_METRICS, 0.0)
                self._means[i] = dict.fromkeys(EPISODE_MEAN_METRICS, 0.0)
                self._steps[i] = 0
        return True


# Physical metrics logged at each evaluation (summed over the episode), as {name: (info key, scale)}.
EVAL_SUM_METRICS = {
    'co2_kg': ('bat_CO2_footprint', 1e-3),
    'energy_mwh': ('bat_total_energy_with_battery_KWh', 1e-3),
    'water': ('dc_water_usage', 1.0),
    'tasks_dropped': ('ls_tasks_dropped', 1.0),
    'setpoint_sum': ('dc_crac_setpoint', 1.0),
    'steps': (None, 1.0),
}


class FixedSeedEvalCallback(EvalCallback):
    """EvalCallback that scores the policy on the same episodes every time.

    Eval env i always replays seed `eval_seed + i` (one episode per env), and the reward normalization history of
    the eval processes is cleared first, so eval/mean_reward only changes when the policy does. It also logs
    physical metrics under eval/ (CO2, energy, water, dropped tasks, tasks still queued at the end, mean setpoint),
    appends them to <log_path>/metrics.csv, and saves the VecNormalize statistics with the best model.
    """

    def __init__(self, eval_env, eval_seed, **kwargs):
        super().__init__(eval_env, **kwargs)
        assert self.n_eval_episodes == eval_env.num_envs, 'use one eval env per evaluation episode'
        self.eval_seed = eval_seed

    @property
    def metrics_csv(self):
        return os.path.join(os.path.dirname(self.log_path), 'metrics.csv')

    def _on_step(self):
        if not (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            return True
        self.eval_env.seed(self.eval_seed)  # used by the reset at the start of evaluate_policy
        self.eval_env.env_method('reset_reward_history')
        self._acc = [dict.fromkeys(EVAL_SUM_METRICS, 0.0) for _ in range(self.eval_env.num_envs)]
        self._episodes = []
        best_before = self.best_mean_reward
        continue_training = super()._on_step()
        vec_norm = self.model.get_vec_normalize_env()
        if self.best_mean_reward > best_before and vec_norm is not None and self.best_model_save_path is not None:
            vec_norm.save(vecnormalize_path(os.path.join(self.best_model_save_path, 'best_model')))
        return continue_training

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        i, info = locals_['i'], locals_['info']
        acc = self._acc[i]
        for name, (key, scale) in EVAL_SUM_METRICS.items():
            acc[name] += 1.0 if key is None else float(info.get(key, 0.0)) * scale
        if not locals_['done']:
            return
        ep = {k: v for k, v in acc.items() if k not in ('setpoint_sum', 'steps')}
        ep['tasks_queued_end'] = float(info.get('ls_tasks_in_queue', 0.0))
        ep['avg_setpoint'] = acc['setpoint_sum'] / max(acc['steps'], 1)
        ep['reward'] = float(locals_['current_rewards'][i])  # eval rewards are not normalized
        self._episodes.append(ep)
        self._acc[i] = dict.fromkeys(EVAL_SUM_METRICS, 0.0)
        if len(self._episodes) == self.n_eval_episodes:
            # Recorded before EvalCallback dumps the logger, so they land on the same timestep as eval/mean_reward.
            means = {k: float(np.mean([e[k] for e in self._episodes])) for k in ep}
            for k, v in means.items():
                if k != 'reward':
                    self.logger.record(f'eval/{k}', v)
            write_header = not os.path.exists(self.metrics_csv)
            with open(self.metrics_csv, 'a') as f:
                if write_header:
                    f.write(','.join(['timesteps'] + list(means)) + '\n')
                f.write(','.join(f'{x:.6g}' for x in [self.num_timesteps] + list(means.values())) + '\n')


def vecnormalize_path(model_path):
    """VecNormalize statistics saved next to a model: <model>_vecnormalize.pkl."""
    base = model_path[:-len('.zip')] if model_path.endswith('.zip') else model_path
    return base + '_vecnormalize.pkl'


def load_obs_normalizer(model_path):
    """Return obs -> normalized obs for a saved model (identity if it was trained without VecNormalize)."""
    path = vecnormalize_path(model_path)
    if not os.path.exists(path):
        return lambda obs: obs
    with open(path, 'rb') as f:
        vec_norm = pickle.load(f)
    return vec_norm.normalize_obs


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
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            vn = vecnormalize_path(path)
            vec_norm.save(vn + '.tmp')
            os.replace(vn + '.tmp', vn)
        # The model zip is written last: it marks the checkpoint as complete.
        self.model.save(path + '.tmp')
        os.replace(path + '.tmp', path)
        for _, old in list_checkpoints(self.ckpt_dir, self.prefix)[:-self.keep]:
            for f in (old, replay_buffer_path(old), vecnormalize_path(old)):
                if os.path.exists(f):
                    os.remove(f)
        print(f'Checkpoint saved at {self.num_timesteps} steps: {path}', flush=True)


def restore_eval_history(eval_cb, npz_path, resumed_at):
    """Reload EvalCallback's history so a resumed run keeps evaluations.npz and the best-model threshold."""
    _trim_metrics_csv(os.path.join(os.path.dirname(npz_path), 'metrics.csv'), resumed_at)
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


def _trim_metrics_csv(path, resumed_at):
    """Drop metric rows logged after the checkpoint we resume from (they will be logged again)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.readlines()
    keep = lines[:1] + [l for l in lines[1:] if float(l.split(',')[0]) <= resumed_at]
    with open(path, 'w') as f:
        f.writelines(keep)


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
    p.add_argument('--learning_rate', type=float, default=None, help='Override the HAPPO learning rate (3e-5).')
    p.add_argument('--ent_coef', type=float, default=None, help='Override the HAPPO entropy coefficient (0.1).')
    p.add_argument('--clip_range', type=float, default=None, help='PPO only: override the HAPPO clip range (0.5).')
    p.add_argument('--n_epochs', type=int, default=None, help='PPO only: override the HAPPO epochs per update (5).')
    p.add_argument('--vf_coef', type=float, default=None, help='Override the value loss coefficient (SB3 default 0.5).')
    p.add_argument('--norm_obs', action=argparse.BooleanOptionalAction, default=True,
                   help='Normalize observations with running mean/std (VecNormalize).')
    p.add_argument('--norm_reward', action=argparse.BooleanOptionalAction, default=True,
                   help='Scale rewards by a running std of the discounted return (VecNormalize).')
    p.add_argument('--pv_capacity_kWp', type=float, default=None,
                   help="On-site PV capacity (pv_config.capacity_kWp; 0 = no PV). Default: the value in the env yaml.")
    p.add_argument('--include_pv', action=argparse.BooleanOptionalAction, default=None,
                   help="--include_pv: PV on and a 4th sub-action decides its use (use / battery / resell); needs "
                        "--pv_capacity_kWp > 0. --no-include_pv: no PV. Default: the env yaml (automatic PV rule).")
    p.add_argument('--include_pv_obs', action=argparse.BooleanOptionalAction, default=None,
                   help='Observe current and forecast PV. Default: the value in the env yaml (off).')
    p.add_argument('--eval_freq', type=int, default=50_000, help='Evaluate every N total env steps.')
    p.add_argument('--n_eval_episodes', type=int, default=5,
                   help='Episodes per evaluation, run in parallel; always the same seeds (eval_seed + i).')
    p.add_argument('--eval_seed', type=int, default=10_000,
                   help='First evaluation seed. Keep it away from the eval_compare.py test seeds (1000+).')
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
    if args.pv_capacity_kWp is not None:
        env_config['pv_config'] = {**env_config.get('pv_config', {}), 'capacity_kWp': args.pv_capacity_kWp}
    if args.include_pv_obs is not None:
        env_config['include_pv_obs'] = args.include_pv_obs
    if args.include_pv is not None:
        env_config['include_pv'] = args.include_pv

    run_id = args.run_id or datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_dir = os.path.join(args.results_root, 'sustaindc', env_config['location'], f'single_{args.algo}', args.exp_name,
                           f'seed-{args.seed:0>5}-{run_id}')
    ckpt_dir = os.path.join(run_dir, 'models', 'checkpoints')
    os.makedirs(run_dir, exist_ok=True)

    if args.algo == 'ppo':
        algo_kwargs = ppo_kwargs_from_happo(args.n_envs)
    else:
        algo_kwargs = dict(gamma=0.995)
    overridable = {'ppo': ('learning_rate', 'ent_coef', 'clip_range', 'n_epochs', 'vf_coef'),
                   'a2c': ('learning_rate', 'ent_coef', 'vf_coef'), 'dqn': ('learning_rate',)}[args.algo]
    for key in ('learning_rate', 'ent_coef', 'clip_range', 'n_epochs', 'vf_coef'):
        if getattr(args, key) is not None:
            if key not in overridable:
                raise SystemExit(f'--{key} does not apply to {args.algo}')
            algo_kwargs[key] = getattr(args, key)
    use_vecnorm = args.norm_obs or args.norm_reward
    norm_args = dict(norm_obs=args.norm_obs, norm_reward=args.norm_reward, gamma=algo_kwargs['gamma'])
    config = {'main_args': vars(args), 'env_args': env_config,
              'algo_args': {k: str(v) if k == 'policy_kwargs' else v for k, v in algo_kwargs.items()},
              'vecnormalize_args': norm_args if use_vecnorm else None}

    checkpoints = list_checkpoints(ckpt_dir, args.algo)
    resumed_at = checkpoints[-1][0] if checkpoints else 0
    config_path = os.path.join(run_dir, 'config.json')
    if checkpoints and os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
        if any(saved.get(k) != json.loads(json.dumps(config[k])) for k in ('env_args', 'algo_args', 'vecnormalize_args')):
            raise SystemExit(f'Refusing to resume {run_dir}: env/algo config differs from the saved config.json. '
                             f'Use a different --run_id for a new run.')
    else:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Offset env seeds on resume so the restarted envs don't replay the same episode starts.
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(SustainDCSingleAgentEnv, n_envs=args.n_envs, seed=args.seed + resumed_at,
                       env_kwargs={'env_config': dict(env_config)}, vec_env_cls=vec_cls)
    # One eval env per episode, each in its own process: the env seeds the process-wide `random`/`np.random`.
    eval_env = make_vec_env(SustainDCSingleAgentEnv, n_envs=args.n_eval_episodes, seed=args.eval_seed,
                            env_kwargs={'env_config': dict(env_config)},
                            vec_env_cls=SubprocVecEnv if args.n_eval_episodes > 1 else DummyVecEnv)
    if use_vecnorm:
        ckpt_vn = vecnormalize_path(checkpoints[-1][1]) if checkpoints else None
        if ckpt_vn and os.path.exists(ckpt_vn):
            env = VecNormalize.load(ckpt_vn, env)
        else:
            env = VecNormalize(env, **norm_args)
        # Eval uses the training statistics (synced before each evaluation) and reports raw rewards.
        eval_env = VecNormalize(eval_env, training=False, norm_obs=args.norm_obs, norm_reward=False)

    if checkpoints:
        ckpt = checkpoints[-1][1]
        print(f'Resuming from {ckpt} ({resumed_at} steps)', flush=True)
        model = ALGOS[args.algo].load(ckpt, env=env, device=args.device)
        if isinstance(model, OffPolicyAlgorithm) and os.path.exists(replay_buffer_path(ckpt)):
            model.load_replay_buffer(replay_buffer_path(ckpt))
    else:
        model = ALGOS[args.algo]('MlpPolicy', env, seed=args.seed, device=args.device, verbose=1,
                                 tensorboard_log=os.path.join(run_dir, 'tb'), **algo_kwargs)

    eval_cb = FixedSeedEvalCallback(eval_env, args.eval_seed, n_eval_episodes=args.n_eval_episodes, deterministic=True,
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
    if use_vecnorm:
        env.save(vecnormalize_path(os.path.join(run_dir, 'models', 'final_model')))
    env.close(); eval_env.close()
    print(f'Saved final model to {run_dir}/models/final_model.zip')


if __name__ == '__main__':
    main()
