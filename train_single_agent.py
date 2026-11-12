"""Train a single agent (full observation, joint action) on SustainDC with Stable-Baselines3.

Run from the repo root, e.g.:
    python train_single_agent.py --algo ppo --n_envs 8 --total_timesteps 5000000
    python train_single_agent.py --algo dqn --location ny --month 6

PPO hyperparameters default to harl/configs/algos_cfgs/happo.yaml so runs are comparable with the MARL ones;
--learning_rate, --ent_coef, --clip_range, --n_epochs and --vf_coef override them.
Observations and rewards are normalized with VecNormalize by default (--no-norm_obs / --no-norm_reward to turn off);
its statistics are saved next to every model as <model>_vecnormalize.pkl.

Logging and evaluation use the common suite (sdc_eval/), like every other trainer: rollout/, train/, health/, perf/
and eval/ scalars in <run>/metrics.csv and <run>/tb_common (SB3's own logs stay in <run>/tb), fixed-seed evaluation
every --eval_freq steps, and models/best_model = lowest work-adjusted CO2 (see sdc_eval/evaluator.py).
Hyperparameters: --config <yaml> with env:/algo: sections and --set algo.learning_rate=1e-4 overrides
(sdc_eval/config.py); 'algo' keys are SB3 constructor arguments plus net_arch and activation (tanh/relu).
Runs are written to <out_root>/sustaindc/<location>/single_<algo>/<exp_name>/seed-XXXXX-<run_id>, where out_root is
--out_root, else $SDC_OUT_ROOT, else ./results.
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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from sdc_eval.config import add_config_args, load_config, merge, parse_overrides
from sdc_eval.constants import ENV_CFG, EVAL_EVERY_STEPS, TRAIN_EVAL_SEED
from sdc_eval.evaluator import FixedSeedEvaluator
from sdc_eval.logger import RunLogger
from sdc_eval.paths import resolve_root
from sdc_eval.sb3 import CommonEvalCallback, CommonMetricsCallback
from sustaindc_single_env import SustainDCSingleAgentEnv

HAPPO_CFG = 'harl/configs/algos_cfgs/happo.yaml'
ALGOS = {'ppo': PPO, 'a2c': A2C, 'dqn': DQN}

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


def ppo_kwargs_from_happo(n_envs):
    """Map the HAPPO config onto SB3 PPO arguments."""
    with open(HAPPO_CFG) as f:
        cfg = yaml.safe_load(f)
    train, model, algo = cfg['train'], cfg['model'], cfg['algo']
    n_steps = train['episode_length']
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
        net_arch=list(model['hidden_sizes']),
        activation=model['activation_func'],
    )


ACTIVATIONS = {'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU}
POLICY_KEYS = ('net_arch', 'activation')  # 'algo' config keys that go to policy_kwargs instead of the constructor


def build_algo_kwargs(algo, hp):
    """Hyperparameters ('algo' config section) -> SB3 constructor kwargs. Unknown keys raise."""
    import inspect
    # Set by the trainer itself (--seed, --device, logging), not hyperparameters.
    reserved = {'self', 'policy', 'env', 'seed', 'device', 'verbose', 'tensorboard_log', 'policy_kwargs'}
    valid = {k for k in inspect.signature(ALGOS[algo].__init__).parameters if not k.startswith('_')} - reserved
    unknown = sorted(set(hp) - valid - set(POLICY_KEYS))
    if unknown:
        raise SystemExit(f'Unknown {algo} hyperparameters {unknown}. Valid: {sorted(valid | set(POLICY_KEYS))}')
    kwargs = {k: v for k, v in hp.items() if k not in POLICY_KEYS}
    policy_kwargs = {}
    if 'net_arch' in hp:
        policy_kwargs['net_arch'] = list(hp['net_arch'])
    if 'activation' in hp:
        policy_kwargs['activation_fn'] = ACTIVATIONS[hp['activation']]
    if policy_kwargs:
        kwargs['policy_kwargs'] = policy_kwargs
    return kwargs


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--algo', choices=list(ALGOS), default='ppo')
    add_config_args(p)
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
    p.add_argument('--eval_freq', type=int, default=EVAL_EVERY_STEPS, help='Evaluate every N total env steps.')
    p.add_argument('--n_eval_episodes', type=int, default=5,
                   help='Episodes per evaluation; always the same seeds (eval_seed + i).')
    p.add_argument('--eval_seed', type=int, default=TRAIN_EVAL_SEED,
                   help='First evaluation seed. Keep it away from the eval_compare.py test seeds (1000+).')
    p.add_argument('--eval_days', type=int, default=None,
                   help='Days per evaluation episode (default: the training days_per_episode).')
    p.add_argument('--save_decisions', action=argparse.BooleanOptionalAction, default=True,
                   help='Save the per-step decisions of every evaluation to <run>/eval/decisions/.')
    p.add_argument('--drop_tolerance', type=float, default=1.0,
                   help='Best-model guard: max increase over noop of the mean tasks dropped per eval episode.')
    p.add_argument('--checkpoint_freq', type=int, default=250_000, help='Save a checkpoint every N total env steps.')
    p.add_argument('--out_root', '--results_root', dest='out_root', default=None,
                   help='Top-level folder for runs: <root>/sustaindc/<location>/single_<algo>/<exp_name>/<run>. '
                        'Default: $SDC_OUT_ROOT if set, else ./results.')
    p.add_argument('--run_id', default=None,
                   help='Run directory suffix (default: a timestamp). Reusing a run id resumes from its latest '
                        'checkpoint, e.g. the Slurm job id, so a preempted/requeued job continues where it stopped.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--torch_threads', type=int, default=1,
                   help='Torch threads for the learner. 1 leaves the other cores to the env processes.')
    args = p.parse_args()
    torch.set_num_threads(args.torch_threads)

    # Hyperparameters: defaults < --config < explicit flags < --set (sdc_eval/config.py).
    with open(ENV_CFG) as f:
        env_config = yaml.safe_load(f)
    env_config['action_type'] = args.action_type
    env_config['reward_type'] = args.reward_type
    hp = {'env': env_config, 'algo': ppo_kwargs_from_happo(args.n_envs) if args.algo == 'ppo' else dict(gamma=0.995)}
    hp = merge(hp, load_config(args.config), open_sections=('algo',))
    for key in ['location', 'month', 'days_per_episode', 'include_dc_setpoint', 'include_pv_obs']:
        if getattr(args, key) is not None:
            hp['env'][key] = getattr(args, key)
    if args.pv_capacity_kWp is not None:
        hp['env']['pv_config'] = {**hp['env'].get('pv_config', {}), 'capacity_kWp': args.pv_capacity_kWp}
    if args.include_pv is not None:
        hp['env']['include_pv'] = args.include_pv
    overridable = {'ppo': ('learning_rate', 'ent_coef', 'clip_range', 'n_epochs', 'vf_coef'),
                   'a2c': ('learning_rate', 'ent_coef', 'vf_coef'), 'dqn': ('learning_rate',)}[args.algo]
    for key in ('learning_rate', 'ent_coef', 'clip_range', 'n_epochs', 'vf_coef'):
        if getattr(args, key) is not None:
            if key not in overridable:
                raise SystemExit(f'--{key} does not apply to {args.algo}')
            hp['algo'][key] = getattr(args, key)
    hp = merge(hp, parse_overrides(args.overrides), open_sections=('algo',))
    env_config = hp['env']
    if args.algo == 'dqn':
        env_config['action_type'] = 'discrete'
    algo_kwargs = build_algo_kwargs(args.algo, hp['algo'])

    run_id = args.run_id or datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_dir = os.path.join(resolve_root(args.out_root), 'sustaindc', env_config['location'], f'single_{args.algo}', args.exp_name,
                           f'seed-{args.seed:0>5}-{run_id}')
    ckpt_dir = os.path.join(run_dir, 'models', 'checkpoints')
    os.makedirs(run_dir, exist_ok=True)

    use_vecnorm = args.norm_obs or args.norm_reward
    norm_args = dict(norm_obs=args.norm_obs, norm_reward=args.norm_reward, gamma=algo_kwargs.get('gamma', 0.99))
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
    if use_vecnorm:
        ckpt_vn = vecnormalize_path(checkpoints[-1][1]) if checkpoints else None
        if ckpt_vn and os.path.exists(ckpt_vn):
            env = VecNormalize.load(ckpt_vn, env)
        else:
            env = VecNormalize(env, **norm_args)

    if checkpoints:
        ckpt = checkpoints[-1][1]
        print(f'Resuming from {ckpt} ({resumed_at} steps)', flush=True)
        model = ALGOS[args.algo].load(ckpt, env=env, device=args.device)
        if isinstance(model, OffPolicyAlgorithm) and os.path.exists(replay_buffer_path(ckpt)):
            model.load_replay_buffer(replay_buffer_path(ckpt))
    else:
        model = ALGOS[args.algo]('MlpPolicy', env, seed=args.seed, device=args.device, verbose=1,
                                 tensorboard_log=os.path.join(run_dir, 'tb'), **algo_kwargs)

    hparams = {'algo': hp['algo'], 'env': env_config, 'vecnormalize': norm_args if use_vecnorm else None,
               'train': {k: vars(args)[k] for k in ('n_envs', 'total_timesteps', 'seed', 'eval_freq', 'n_eval_episodes')}}
    run_logger = RunLogger(run_dir, f'single_{args.algo}', 'off_policy' if args.algo == 'dqn' else 'on_policy',
                           env_cfg=env_config, seed=args.seed, resume_from_step=resumed_at if checkpoints else None,
                           hparams=hparams)
    evaluator = FixedSeedEvaluator(env_config, run_dir, run_logger, n_episodes=args.n_eval_episodes,
                                   seed=args.eval_seed, drop_tolerance=args.drop_tolerance,
                                   days_per_episode=args.eval_days, save_decisions=args.save_decisions)
    if checkpoints:
        evaluator.trim_after(resumed_at)
    eval_cb = CommonEvalCallback(evaluator, env_config, args.eval_freq, os.path.join(run_dir, 'models'),
                                 vecnormalize_path)
    callbacks = CallbackList([
        CommonMetricsCallback(run_logger, is_dqn=args.algo == 'dqn'),
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
    if eval_cb.last_eval_step != model.num_timesteps:
        eval_cb.init_callback(model)
        eval_cb.evaluate()  # the final model is evaluated too (it may become the best)
    if evaluator.best is None:
        print('Note: no evaluation passed the best-model guard (tasks dropped vs noop), so there is no '
              'models/best_model; eval_compare.py falls back to final_model.', flush=True)
    run_logger.close(evaluator.best)
    env.close()
    print(f'Saved final model to {run_dir}/models/final_model.zip')


if __name__ == '__main__':
    main()
