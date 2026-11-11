"""One seeded evaluation episode of a policy on SustainDC. Shared by eval_compare.py, the in-training evaluator,
the MPC value training and the trace recorder.

Each episode is fixed by its seed: `random` and `np.random` are seeded right before SustainDC.reset(), which is
where all of the simulator's randomness is drawn (start day/hour, workload/weather noise, initial task queue). The
reward normalization history is cleared too, so the returns (not only the physical metrics) are reproducible.
"""
import random

import numpy as np

from sdc_eval.metrics import EpisodeAccumulator


def reset_reward_history():
    """Clear the process-wide energy history that the default rewards are z-scored against."""
    import utils.reward_creator as rc
    rc.energy_history.clear()


def run_episode(sdc, policy, seed, trace=None):
    """Run one seeded episode. Returns (metrics dict, fingerprint of the exogenous inputs).

    trace: optional callable(step_index, info, actions) called after every step (e.g. to record per-step traces).
    """
    random.seed(seed)
    np.random.seed(seed)
    reset_reward_history()
    sdc.reset()
    policy.reset(seed)
    acc = EpisodeAccumulator()
    while True:
        actions = policy.act(sdc)
        _, _, _, truncs, info = sdc.step(actions)
        i = info['agent_ls']  # merged ls/dc/bat info
        acc.step(i, actions, (sdc.ls_reward, sdc.dc_reward, sdc.bat_reward))
        if trace is not None:
            trace(acc.steps - 1, i, actions)
        if truncs['__all__']:
            break
    return acc.finalize(sdc), acc.fingerprint()


def run_episodes(sdc, policy, seeds):
    """run_episode over several seeds -> (list of metrics, list of fingerprints)."""
    out = [run_episode(sdc, policy, s) for s in seeds]
    return [m for m, _ in out], [fp for _, fp in out]
