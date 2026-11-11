"""HARL adapter: logs HARL training (on- and off-policy runners) with the common schema (sdc_eval.logger) and runs
the common fixed-seed evaluation (sdc_eval.evaluator) in place of HARL's own evaluation.

The runners call:
    tracker = HarlTracker(runner, family)        in __init__, after the run directory and the networks exist
    tracker.rollout_step(rewards, infos, steps)  after every vectorized env step  -> rollout/*
    tracker.on_policy_update(...)                after every on-policy update      -> train/*, health/*, perf/*
    tracker.off_policy_update(...)               after off-policy updates          -> train/*, health/*, perf/*
    tracker.maybe_evaluate(steps, save_fn)       every eval_every_steps            -> eval/*, models/ = best
    tracker.finish(steps, save_fn, save_final_fn)

New HARL config keys (section eval:, defaults below; set them with --set eval.eval_every_steps=200000):
"""
import math
import os

import numpy as np
import torch

from sdc_eval.constants import EVAL_EVERY_STEPS, TRAIN_EVAL_SEED
from sdc_eval.evaluator import FixedSeedEvaluator
from sdc_eval.logger import ENTROPY_MAX, RunLogger
from sdc_eval.metrics import SELECTION_METRIC
from sdc_eval.policies import HARLPolicy

EVAL_DEFAULTS = {
    'eval_every_steps': EVAL_EVERY_STEPS,  # common fixed-seed evaluation every N env steps (all threads)
    'n_eval_episodes': 5,                  # episodes per evaluation, seeds TRAIN_EVAL_SEED + i
    'eval_days': None,                     # days per evaluation episode (None: the training days_per_episode)
    'drop_tolerance': 1.0,                 # best-model guard: max mean tasks dropped per episode above noop
    'legacy_eval': False,                  # also run HARL's original (not fixed-seed) evaluation
    'save_decisions': True,                # per-step decisions of every evaluation -> eval/decisions/
}
__doc__ += '\n'.join(f'    {k}: {v!r}' for k, v in EVAL_DEFAULTS.items()) + '\n'


def add_eval_defaults(algo_args):
    for k, v in EVAL_DEFAULTS.items():
        algo_args.setdefault('eval', {}).setdefault(k, v)


def _f(x):
    """float() of python numbers, numpy values and (0-d) tensors."""
    if torch.is_tensor(x):
        return float(x.detach().float().mean().cpu())
    return float(np.mean(x))


class HarlTracker:
    def __init__(self, runner, family):
        a, ev = runner.algo_args, runner.algo_args['eval']
        self.runner = runner
        self.agents = list(runner.env_args['agents'])
        resumed = getattr(runner, 'resume_path', None) is not None
        self.resume_steps = int(getattr(runner, 'resume_steps', 0) or 0)
        self.rl = RunLogger(runner.run_dir, runner.args['algo'], family, env_cfg=runner.env_args,
                            seed=a['seed']['seed'], writer=runner.writter,
                            resume_from_step=self.resume_steps if resumed else None,
                            hparams={k: v for k, v in a.items() if k != 'logger'} | {'env': runner.env_args})
        self.evaluator = FixedSeedEvaluator(runner.env_args, runner.run_dir, self.rl, n_episodes=ev['n_eval_episodes'],
                                            seed=TRAIN_EVAL_SEED, drop_tolerance=ev['drop_tolerance'],
                                            days_per_episode=ev['eval_days'],
                                            save_decisions=ev['save_decisions'])
        if resumed:
            self.evaluator.trim_after(self.resume_steps)
        self.every = int(ev['eval_every_steps'])
        self.last_bucket = self.resume_steps // self.every
        self.last_eval_step = None
        self.n_threads = a['train']['n_rollout_threads']
        self.ep_return = np.zeros(self.n_threads)
        self.rl._last = None

    # ----------------------------------------------------------------------------------------------------------------
    def rollout_step(self, rewards, infos, env_steps):
        """rewards: (n_threads, n_agents, 1); infos: per thread, a list of per-agent info dicts."""
        self.ep_return += np.mean(np.asarray(rewards, dtype=np.float64), axis=(1, 2))
        for t, info in enumerate(infos):
            ep = info[0].get('episode_metrics') if len(info) else None
            if ep is not None:
                self.rl.episode(ep, env_steps)
                self.rl.scalar('rollout/return', self.ep_return[t], env_steps)  # mean over agents of the optimized reward
                self.ep_return[t] = 0.0

    # ----------------------------------------------------------------------------------------------------------------
    def _agent_lr(self, actor):
        for name in ('actor_optimizer', 'optimizer'):
            opt = getattr(actor, name, None)
            if opt is not None:
                return opt.param_groups[0]['lr']
        return float('nan')

    def on_policy_update(self, actor_infos, critic_info, change, rollout_stats, env_steps, update_time):
        """actor_infos: per-agent train_info dicts returned by the actors; critic_info: the critic's; change: per agent
        {'approx_kl', 'clip_fraction'} between the policies before and after the update; rollout_stats: advantage /
        value / explained-variance statistics computed from the buffers before the update."""
        r = self.runner
        per_agent = {}
        for i, (name, info) in enumerate(zip(self.agents, actor_infos)):
            short = name.replace('agent_', '')
            d = {}
            for k, v in info.items():
                if k == 'policy_loss':
                    self.rl.scalar(f'train/agent_{short}/policy_loss', _f(v), env_steps)
                    d['policy_loss'] = _f(v)
                elif k == 'dist_entropy':
                    d['entropy'] = _f(v)
                    d['entropy_norm'] = _f(v) / ENTROPY_MAX
                elif k == 'actor_grad_norm':
                    d['grad_norm'] = _f(v)
                elif k == 'ratio':
                    d['ratio_mean'] = _f(v)
                elif k == 'kl':  # HATRPO's own KL (trust-region constraint)
                    d['trpo_kl'] = _f(v)
                else:
                    d[k] = _f(v)
            d.update(change[i])
            d['lr'] = self._agent_lr(r.actor[i])
            per_agent[short] = d
            self.rl.scalars(f'health/agent_{short}', {k: v for k, v in d.items() if k != 'policy_loss'}, env_steps)
        # Means over agents: one curve per algorithm, comparable with the single-agent baselines.
        for k in ('entropy_norm', 'approx_kl', 'clip_fraction', 'grad_norm'):
            vals = [d[k] for d in per_agent.values() if k in d and not math.isnan(d[k])]
            if vals:
                self.rl.scalar(f'health/{k}', float(np.mean(vals)), env_steps)
        losses = [d['policy_loss'] for d in per_agent.values() if 'policy_loss' in d]
        if losses:
            self.rl.scalar('train/policy_loss', float(np.mean(losses)), env_steps)
        self.rl.scalar('train/value_loss', _f(critic_info.get('value_loss', float('nan'))), env_steps)
        self.rl.scalar('health/critic_grad_norm', _f(critic_info.get('critic_grad_norm', float('nan'))), env_steps)
        self.rl.scalar('health/critic_lr', r.critic.critic_optimizer.param_groups[0]['lr'], env_steps)
        self.rl.scalars('health', rollout_stats, env_steps)
        self.rl.perf(env_steps, update_time=update_time)

    def off_policy_update(self, infos, env_steps, update_time=None):
        """infos: {'train/...' or 'health/...': value} gathered by the off-policy runner."""
        for k, v in infos.items():
            self.rl.scalar(k, _f(v), env_steps)
        self.rl.perf(env_steps, update_time=update_time)

    # ----------------------------------------------------------------------------------------------------------------
    def live_policy(self):
        r = self.runner
        return HARLPolicy.from_live(r.actor, r.args['algo'], r.env_args, r.algo_args['model'],
                                    pad_dim=r.envs.observation_space[0].shape[0])

    def evaluate(self, env_steps, save_fn):
        r = self.runner
        if hasattr(r, 'prep_rollout'):
            r.prep_rollout()
        out = self.evaluator.evaluate(self.live_policy(), env_steps, save_fn=lambda tag: save_fn())
        self.last_eval_step = env_steps
        self.rl.flush()
        print(f"Eval at {env_steps} steps: CO2 (work-adj.) {out[SELECTION_METRIC]:.1f} kg "
              f"({out[f'{SELECTION_METRIC}_pct_vs_noop']:+.2f}% vs noop), dropped {out['tasks_dropped']:.0f}"
              + (' -> new best' if self.evaluator.best and self.evaluator.best['env_steps'] == env_steps else ''),
              flush=True)
        return out

    def maybe_evaluate(self, env_steps, save_fn):
        bucket = env_steps // self.every
        if bucket > self.last_bucket:
            self.last_bucket = bucket
            return self.evaluate(env_steps, save_fn)
        return None

    def finish(self, env_steps, save_fn, save_final_fn):
        """End of training: save models_final/, evaluate the final policy (it may become the best), close the log."""
        save_final_fn()
        if self.last_eval_step != env_steps:
            self.evaluate(env_steps, save_fn)
        if self.evaluator.best is None:
            print('Note: no evaluation passed the best-model guard (tasks dropped vs noop), so models/ holds no '
                  'best model; eval_compare.py falls back to models_final/.', flush=True)
        self.rl.close(self.evaluator.best)


def explained_variance(values, returns):
    var = np.var(returns)
    return float('nan') if var == 0 else float(1 - np.var(returns - values) / var)
