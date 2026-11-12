"""Stable-Baselines3 adapter: logs SB3 training (PPO, A2C, DQN) with the common schema (sdc_eval.logger).

    CommonMetricsCallback    rollout/* from each finished episode, train/* + health/* from SB3's own diagnostics
                             (renamed to the common keys) plus gradient norms, value/advantage/Q statistics, perf/*
    CommonEvalCallback       FixedSeedEvaluator every eval_freq env steps; saves models/best_model when it improves
"""
import math
import os
import time

import numpy as np
import torch

from sdc_eval.logger import ENTROPY_MAX, entropy_norm
from sdc_eval.policies import SB3Policy

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover
    BaseCallback = object

# SB3 logger key -> common key. Entropy is handled separately (SB3 logs -entropy as entropy_loss).
SB3_KEYS = {
    'train/policy_gradient_loss': 'train/policy_loss',
    'train/value_loss': 'train/value_loss',
    'train/loss': None,  # PPO/A2C: total loss; DQN: TD loss (mapped below)
    'train/approx_kl': 'health/approx_kl',
    'train/clip_fraction': 'health/clip_fraction',
    'train/clip_range': 'health/clip_range',
    'train/explained_variance': 'health/explained_variance',
    'train/learning_rate': 'health/lr',
    'train/n_updates': 'health/n_updates',
    'train/std': 'health/action_std',
    'rollout/exploration_rate': 'health/epsilon',
}
JOINT_ENTROPY_MAX = 3 * ENTROPY_MAX  # joint action over 3 agents x 3 actions (MultiDiscrete or Discrete(27))


class CommonMetricsCallback(BaseCallback):
    def __init__(self, run_logger, is_dqn=False, q_stats_every=10_000):
        super().__init__()
        self.rl = run_logger
        self.is_dqn = is_dqn
        self.q_stats_every = q_stats_every
        self._grad_norms = []
        self._orig_clip = None
        self._last_q_stats = -math.inf
        self._t_rollout_end = None

    # Gradient norms: SB3 calls torch.nn.utils.clip_grad_norm_ on every gradient step and discards the returned
    # (pre-clip) norm; record it.
    def _on_training_start(self):
        self.rl._last = (time.time(), self.num_timesteps)  # perf/sps from the first update on
        self._orig_clip = torch.nn.utils.clip_grad_norm_
        orig, norms = self._orig_clip, self._grad_norms

        def recording_clip(*args, **kwargs):
            n = orig(*args, **kwargs)
            norms.append(float(n))
            return n
        torch.nn.utils.clip_grad_norm_ = recording_clip

    def _on_training_end(self):
        if self._orig_clip is not None:
            torch.nn.utils.clip_grad_norm_ = self._orig_clip
        self._flush_train()
        self.rl.flush()

    def _on_step(self):
        for info in self.locals['infos']:
            ep = info.get('episode_metrics')
            if ep is not None:
                self.rl.episode(ep, self.num_timesteps)
                if 'episode' in info:  # Monitor wrapper: the return of the reward actually optimized
                    self.rl.scalar('rollout/return', info['episode']['r'], self.num_timesteps)
        if self.is_dqn and self.num_timesteps - self._last_q_stats >= self.q_stats_every:
            self._flush_train()
            if self._q_stats():
                self._last_q_stats = self.num_timesteps
        return True

    def _on_rollout_start(self):
        # SB3 records train() diagnostics into its logger; they are still there when the next rollout starts.
        self._flush_train()
        if self._t_rollout_end is not None:
            self.rl.perf(self.num_timesteps, update_time=time.time() - self._t_rollout_end)

    def _on_rollout_end(self):
        self._t_rollout_end = time.time()
        buf = getattr(self.model, 'rollout_buffer', None)
        if buf is not None and getattr(buf, 'full', False):
            self.rl.scalars('health', {
                'value_mean': float(np.mean(buf.values)), 'adv_mean': float(np.mean(buf.advantages)),
                'adv_std': float(np.std(buf.advantages)), 'return_target_mean': float(np.mean(buf.returns)),
            }, self.num_timesteps)

    def _flush_train(self):
        vals = dict(self.model.logger.name_to_value)
        step = self.num_timesteps
        for k, v in vals.items():
            if k == 'train/loss':
                self.rl.scalar('train/q_loss' if self.is_dqn else 'train/total_loss', v, step)
            elif k == 'train/entropy_loss':
                self.rl.scalar('health/entropy', -v, step)
                self.rl.scalar('health/entropy_norm', -v / JOINT_ENTROPY_MAX, step)
            elif SB3_KEYS.get(k):
                self.rl.scalar(SB3_KEYS[k], v, step)
        if self._grad_norms:
            g = np.array(self._grad_norms)
            self.rl.scalars('health', {'grad_norm': g.mean(), 'grad_norm_max': g.max()}, step)
            self._grad_norms.clear()

    @torch.no_grad()
    def _q_stats(self):
        buf = self.model.replay_buffer
        if buf.size() < 256:
            return False
        batch = buf.sample(256, env=self.model._vec_normalize_env)
        q = self.model.q_net(batch.observations)
        q_next = self.model.q_net_target(batch.next_observations).max(dim=1)[0]
        target = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.model.gamma * q_next
        q_taken = q.gather(1, batch.actions.long()).flatten()
        self.rl.scalars('health', {
            'q_mean': q.max(dim=1)[0].mean().item(), 'q_max': q.max().item(),
            'target_q_mean': target.mean().item(), 'td_error_abs': (q_taken - target).abs().mean().item(),
            'buffer_size': buf.size() * buf.n_envs,
        }, self.num_timesteps)
        return True


class CommonEvalCallback(BaseCallback):
    """Fixed-seed evaluation every `eval_freq` env steps (counted over all training envs)."""

    def __init__(self, evaluator, env_config, eval_freq, models_dir, vecnormalize_path):
        super().__init__()
        self.evaluator, self.env_config, self.eval_freq = evaluator, env_config, eval_freq
        self.models_dir, self.vecnormalize_path = models_dir, vecnormalize_path
        self._last_bucket = None
        self.last_eval_step = None

    def _on_training_start(self):
        self._last_bucket = self.num_timesteps // self.eval_freq  # don't re-evaluate right after resuming

    def _save(self, name):
        path = os.path.join(self.models_dir, name)
        self.model.save(path)
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            vec_norm.save(self.vecnormalize_path(path))

    def evaluate(self):
        vec_norm = self.model.get_vec_normalize_env()
        policy = SB3Policy.from_live(self.model, self.env_config, vec_norm.normalize_obs if vec_norm else None)
        out = self.evaluator.evaluate(policy, self.num_timesteps, save_fn=lambda tag: self._save('best_model'))
        self.last_eval_step = self.num_timesteps
        print(f"Eval at {self.num_timesteps} steps: CO2 (work-adj.) {out['co2_work_adjusted']:.1f} kg "
              f"({out['co2_work_adjusted_pct_vs_noop']:+.2f}% vs noop), dropped {out['tasks_dropped']:.0f}", flush=True)

    def _on_step(self):
        bucket = self.num_timesteps // self.eval_freq
        if bucket > self._last_bucket:
            self._last_bucket = bucket
            self.evaluate()
        return True
