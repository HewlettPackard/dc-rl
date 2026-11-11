"""FixedSeedEvaluator: the in-training evaluation and best-model rule shared by every algorithm.

Each evaluation runs the current policy on the same fixed episodes (TRAIN_EVAL_SEED + i, disjoint from the final
FINAL_EVAL_SEED episodes used for reporting), with the same rollout and metrics as eval_compare.py. The do-nothing
policy is evaluated once on the same episodes, so every evaluation also reports the paired % change vs noop.

Best model: lowest mean co2_work_adjusted (CO2 with unexecuted work charged at its marginal energy), among the
evaluations whose mean dropped tasks per episode stay within `drop_tolerance` of noop's. The framework passes a
save_fn(tag) that writes its own checkpoint format; it is called with 'best' when an evaluation sets a new best.

Writes <run>/eval/evals.csv (one row per evaluation, every metric's mean) and <run>/eval/best_eval.json.
"""
import json
import os

import numpy as np

from sdc_eval.constants import AGENTS, TRAIN_EVAL_SEED, seeds as seed_range
from sdc_eval.metrics import METRIC_NAMES, PAIRED_METRICS, SELECTION_METRIC, aggregate
from sdc_eval.policies import NoOpPolicy
from sdc_eval.rollout import run_episodes


def run_spec_episode(job):
    """Pool worker: build the policy from its spec (sdc_eval.policies.make_policy) and run one seeded episode."""
    import torch
    from sdc_eval.policies import make_policy
    from sustaindc_env import SustainDC
    torch.set_num_threads(1)
    scenario, spec, seed, record = (job + (False,))[:4]
    _, policy = make_policy(spec)
    sdc = SustainDC({**scenario, 'agents': list(AGENTS), 'include_dc_setpoint': bool(policy.include_dc_setpoint),
                     'include_pv_obs': bool(policy.include_pv_obs)})
    from sdc_eval.decisions import DecisionRecorder
    from sdc_eval.rollout import run_episode
    rec = DecisionRecorder(sdc) if record else None
    m, fp = run_episode(sdc, policy, seed, trace=rec)
    return m, fp, (rec.rows if record else None)


class FixedSeedEvaluator:
    def __init__(self, scenario, run_dir, logger=None, n_episodes=5, seed=TRAIN_EVAL_SEED, drop_tolerance=1.0,
                 days_per_episode=None, save_decisions=True):
        """
        scenario: the training env config (dict). Observation flags are taken from each evaluated policy.
        days_per_episode: optionally shorten evaluation episodes (default: the training scenario's).
        drop_tolerance: max increase over noop of the mean tasks dropped per episode for a model to be selectable.
        save_decisions: write every evaluation's per-step decisions to <run>/eval/decisions/<env_steps>.csv.gz
            (sdc_eval.decisions) and the best one's to <run>/eval/decisions_best.csv.gz.
        """
        self.save_decisions = save_decisions
        self.scenario = {k: v for k, v in scenario.items()}
        if days_per_episode is not None:
            self.scenario['days_per_episode'] = days_per_episode
        self.seeds = seed_range(seed, n_episodes)
        self.run_dir = run_dir
        self.dir = os.path.join(run_dir, 'eval')
        os.makedirs(self.dir, exist_ok=True)
        self.logger = logger
        self.drop_tolerance = drop_tolerance
        self._envs = {}
        self.best_path = os.path.join(self.dir, 'best_eval.json')
        self.best = None
        if os.path.exists(self.best_path):  # resumed run
            with open(self.best_path) as f:
                self.best = json.load(f)
        self._noop = None

    def _env(self, policy):
        from sustaindc_env import SustainDC
        key = (bool(policy.include_dc_setpoint), bool(policy.include_pv_obs))
        if key not in self._envs:
            self._envs[key] = SustainDC({**self.scenario, 'agents': list(AGENTS), 'include_dc_setpoint': key[0],
                                         'include_pv_obs': key[1]})
        return self._envs[key]

    @property
    def noop(self):
        if self._noop is None:
            pol = NoOpPolicy()
            self._noop, _ = run_episodes(self._env(pol), pol, self.seeds)
        return self._noop

    def evaluate(self, policy, env_steps, save_fn=None, spec=None, pool=None):
        """Evaluate a policy and log eval/*. Returns {metric: mean}.

        policy: a policy object (sdc_eval.policies interface), run in this process; or
        spec + pool: a policy spec string (as for eval_compare.py), each episode run by a worker of a
            multiprocessing pool (for slow policies such as MPC).
        """
        if spec is not None:
            jobs = [(self.scenario, spec, s, self.save_decisions) for s in self.seeds]
            out = pool.map(run_spec_episode, jobs) if pool is not None else [run_spec_episode(j) for j in jobs]
            episodes = [m for m, _, _ in out]
            decisions = [(s, rows) for s, (_, _, rows) in zip(self.seeds, out)]
        elif self.save_decisions:
            episodes, _, decisions = run_episodes(self._env(policy), policy, self.seeds, record=True)
        else:
            episodes, _ = run_episodes(self._env(policy), policy, self.seeds)
        decisions_path = None
        if self.save_decisions:
            from sdc_eval.decisions import frame
            os.makedirs(os.path.join(self.dir, 'decisions'), exist_ok=True)
            decisions_path = os.path.join(self.dir, 'decisions', f'{int(env_steps):012d}.csv.gz')
            frame(decisions, env_steps=int(env_steps)).to_csv(decisions_path, index=False)
        agg = aggregate(episodes)
        means = {k: v[0] for k, v in agg.items()}
        out = dict(means)
        for k in PAIRED_METRICS:
            pct = [100 * (e[k] / b[k] - 1) for e, b in zip(episodes, self.noop) if b[k]]
            out[f'{k}_pct_vs_noop'] = float(np.mean(pct)) if pct else float('nan')
        for k in (SELECTION_METRIC, 'co2', 'total_energy', 'tasks_dropped'):
            out[f'{k}_std'] = agg[k][1]

        noop_dropped = float(np.mean([e['tasks_dropped'] for e in self.noop]))
        eligible = means['tasks_dropped'] <= noop_dropped + self.drop_tolerance
        is_best = eligible and (self.best is None or means[SELECTION_METRIC] < self.best['metrics'][SELECTION_METRIC])
        out['eligible'] = float(eligible)
        if is_best:
            self.best = {'env_steps': int(env_steps), 'selection_metric': SELECTION_METRIC, 'seeds': self.seeds,
                         'metrics': out}
            with open(self.best_path, 'w') as f:
                json.dump(self.best, f, indent=2)
            if decisions_path is not None:
                import shutil
                shutil.copy(decisions_path, os.path.join(self.dir, 'decisions_best.csv.gz'))
            if save_fn is not None:
                save_fn('best')
        if self.best is not None:
            out[f'best_{SELECTION_METRIC}'] = self.best['metrics'][SELECTION_METRIC]
            out[f'best_{SELECTION_METRIC}_pct_vs_noop'] = self.best['metrics'][f'{SELECTION_METRIC}_pct_vs_noop']

        if self.logger is not None:
            self.logger.scalars('eval', out, env_steps)
            self.logger.flush()
        self._append_csv(env_steps, out, is_best)
        return out

    def _append_csv(self, env_steps, out, is_best):
        path = os.path.join(self.dir, 'evals.csv')
        keys = sorted(out)
        new = not os.path.exists(path)
        with open(path, 'a') as f:
            if new:
                f.write('env_steps,is_best,' + ','.join(keys) + '\n')
            f.write(f'{int(env_steps)},{int(is_best)},' + ','.join(f'{out[k]:.8g}' for k in keys) + '\n')

    def trim_after(self, env_steps):
        """On resume: drop evaluations logged after the checkpoint we resume from."""
        path = os.path.join(self.dir, 'evals.csv')
        if not os.path.exists(path):
            return
        with open(path) as f:
            lines = f.readlines()
        with open(path, 'w') as f:
            f.writelines(lines[:1] + [l for l in lines[1:] if int(l.split(',', 1)[0]) <= env_steps])
        if self.best is not None and self.best['env_steps'] > env_steps:
            self.best = None  # the saved best model came from the discarded future; it will be re-selected
            os.remove(self.best_path)


__all__ = ['FixedSeedEvaluator', 'METRIC_NAMES']
