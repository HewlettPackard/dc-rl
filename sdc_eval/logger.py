"""RunLogger: one logging schema for every algorithm.

Every scalar goes to TensorBoard (<run>/tb_common) and to <run>/metrics.csv (long format: env_steps, wall_time,
key, value), keyed by environment steps so curves of different algorithms line up. Namespaces:

    rollout/<metric>   physical metrics of each completed training episode (sdc_eval.metrics), plus return/ep_len
    eval/<metric>      fixed-seed evaluation: mean over seeds (and <metric>_std, <metric>_pct_vs_noop)
    train/<loss>       algorithm losses: policy_loss, value_loss, q_loss, actor_loss, alpha_loss, ...
    health/<diag>      training-health diagnostics: entropy(_norm), approx_kl, clip_fraction, explained_variance,
                       grad norms, lr, Q statistics, alpha/epsilon, action shares, nan_or_inf ...
                       per agent where it applies: health/agent_ls/entropy
    perf/<x>           sps, wall_time, update_time, gpu_mem_mb

env_steps counts environment steps summed over parallel envs (one SustainDC step = 1, whatever the number of
agents), i.e. the number of 15-minute transitions simulated for training.
"""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time

from sdc_eval.constants import N_ACTIONS, SCENARIO_KEYS

ENTROPY_MAX = math.log(N_ACTIONS)  # entropy of a uniform policy over the 3 actions of each agent


def git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def scenario_hash(env_cfg):
    """Short hash of the scenario-defining config keys: two runs with the same hash trained on the same problem."""
    sc = {k: env_cfg.get(k) for k in SCENARIO_KEYS}
    return hashlib.sha1(json.dumps(sc, sort_keys=True, default=str).encode()).hexdigest()[:10]


class RunLogger:
    def __init__(self, run_dir, algo, family, env_cfg=None, seed=None, writer=None, resume_from_step=None,
                 hparams=None):
        """
        family: 'on_policy', 'off_policy', 'mpc' or 'baseline' (decides which health/ keys are expected).
        writer: an existing TensorBoard SummaryWriter to reuse (e.g. HARL's); otherwise one is created.
        resume_from_step: when resuming a run, metrics.csv rows logged after this step are dropped (they will be
            logged again by the resumed run).
        hparams: the run's resolved hyperparameters (nested dict, e.g. {'algo': {...}, 'env': {...}}); saved
            flattened in run_meta.json and, by close(), in TensorBoard's HParams tab with the best eval results.
        """
        from sdc_eval.config import flatten
        self.hparams = flatten(hparams or {})
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        if writer is None:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(run_dir, 'tb_common'))
        self.writer = writer
        self.csv_path = os.path.join(run_dir, 'metrics.csv')
        if resume_from_step is not None and os.path.exists(self.csv_path):
            self._trim_csv(resume_from_step)
        new = not os.path.exists(self.csv_path)
        self._csv_file = open(self.csv_path, 'a', newline='')
        self._csv = csv.writer(self._csv_file)
        if new:
            self._csv.writerow(['env_steps', 'wall_time', 'key', 'value'])
        self.t0 = time.time()
        self._last = None  # (time, env_steps) of the previous perf() call
        self.nan_count = 0
        meta_path = os.path.join(run_dir, 'run_meta.json')
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        meta.update({
            'algo': algo, 'family': family, 'seed': seed, 'run_dir': os.path.abspath(run_dir),
            'git_sha': git_sha(), 'command': ' '.join(sys.argv),
            'scenario_hash': scenario_hash(env_cfg) if env_cfg else None,
            'scenario': {k: env_cfg.get(k) for k in SCENARIO_KEYS} if env_cfg else None,
            'obs': {k: env_cfg.get(k) for k in ('include_dc_setpoint', 'include_pv_obs', 'agents')} if env_cfg else None,
            'rewards': {k: env_cfg.get(k) for k in ('ls_reward', 'dc_reward', 'bat_reward', 'individual_reward_weight')}
            if env_cfg else None,
            'hparams': self.hparams,
        })
        meta.setdefault('created', time.strftime('%Y-%m-%d %H:%M:%S'))
        meta.setdefault('starts', []).append({'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                              'resumed_from_step': resume_from_step})
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    def _trim_csv(self, step):
        with open(self.csv_path) as f:
            rows = list(csv.reader(f))
        keep = [rows[0]] + [r for r in rows[1:] if float(r[0]) <= step]
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerows(keep)

    def scalar(self, key, value, env_steps):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            if key.startswith(('train/', 'health/')):
                self.nan_count += 1
                self.writer.add_scalar('health/nan_or_inf', self.nan_count, env_steps)
            if math.isnan(value):
                return
        self.writer.add_scalar(key, value, env_steps)
        self._csv.writerow([int(env_steps), round(time.time() - self.t0, 2), key, f'{value:.8g}'])

    def scalars(self, prefix, values, env_steps):
        """Log a dict {name: value} under prefix/ (prefix may be '' for full keys)."""
        for k, v in values.items():
            self.scalar(f'{prefix}/{k}' if prefix else k, v, env_steps)

    def episode(self, metrics, env_steps, prefix='rollout'):
        """Physical metrics of one completed episode (EpisodeAccumulator.finalize)."""
        self.scalars(prefix, {k: v for k, v in metrics.items() if k != 'steps'}, env_steps)
        self.scalar(f'{prefix}/ep_len', metrics.get('steps', float('nan')), env_steps)

    def perf(self, env_steps, update_time=None):
        now = time.time()
        vals = {'wall_time': now - self.t0}
        if self._last is not None and env_steps > self._last[1]:
            vals['sps'] = (env_steps - self._last[1]) / max(now - self._last[0], 1e-9)
        if update_time is not None:
            vals['update_time'] = update_time
        try:
            import torch
            if torch.cuda.is_available():
                vals['gpu_mem_mb'] = torch.cuda.max_memory_allocated() / 2 ** 20
        except Exception:
            pass
        self._last = (now, env_steps)
        self.scalars('perf', vals, env_steps)

    def flush(self):
        self._csv_file.flush()
        self.writer.flush()

    def close(self, best_eval=None):
        """best_eval: FixedSeedEvaluator.best; its metrics go to the HParams tab next to the hyperparameters."""
        if self.hparams:
            from sdc_eval.metrics import PAIRED_METRICS, SELECTION_METRIC
            m = (best_eval or {}).get('metrics', {})
            keys = [SELECTION_METRIC, f'{SELECTION_METRIC}_pct_vs_noop', 'co2', 'tasks_dropped', 'tasks_queued_end'] + \
                   [f'{k}_pct_vs_noop' for k in PAIRED_METRICS[1:]]
            metrics = {f'hparam/best_{k}': float(m.get(k, float('nan'))) for k in keys}
            metrics['hparam/best_env_steps'] = float((best_eval or {}).get('env_steps', float('nan')))
            hp = {k: ('None' if v is None else v) for k, v in self.hparams.items()}
            try:
                self.writer.add_hparams(hp, metrics, run_name='hparams')
            except Exception as e:  # never fail a finished run over the HParams summary
                print(f'Warning: could not write the TensorBoard HParams summary: {e}')
            with open(os.path.join(self.run_dir, 'run_meta.json')) as f:
                meta = json.load(f)
            meta['best_eval'] = best_eval
            with open(os.path.join(self.run_dir, 'run_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2, default=str)
        self.flush()
        self._csv_file.close()


def entropy_norm(entropy):
    """Entropy as a fraction of the uniform policy's (1 = uniform, 0 = deterministic), comparable across algos."""
    return float(entropy) / ENTROPY_MAX
