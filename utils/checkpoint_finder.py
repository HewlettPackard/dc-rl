import os
from typing import Optional
import logging

from ray.tune.utils.util import is_nan
import pandas as pd


def get_best_checkpoint(trial_dir: str, metric: Optional[str] = 'episode_reward_mean', mode: Optional[str] = 'max') -> str:
    """
    Gets best persistent checkpoint path of provided trial.

    Args:
        trial: The log directory of a trial instance which contains
                all the checkpoints
        metric: The metric by which the best checkpoint would be chosen, like
                'episode_reward_mean'
        mode: One of [min, max].

    Returns:
        string (path to checkpoint)
    """
    
    trial_dir = os.path.abspath(trial_dir)
    df_metrics = pd.read_json(os.path.join(trial_dir, 'result.json'), lines=True)
    df_metrics = pd.json_normalize(df_metrics["sampler_results"])

    def get_path_and_metric(checkpoint, metric):
        checkpoint_nr = int(checkpoint[-6:]) - 1

        if checkpoint_nr < 0 or checkpoint_nr >= len(df_metrics):
            return (checkpoint, None)  # Returning None if out of bounds

        return (checkpoint, df_metrics.iloc[checkpoint_nr][metric])

    checkpoint_paths = [
        get_path_and_metric(cp, metric)
        for cp in next(os.walk(trial_dir))[1]
    ]

    checkpoint_paths = [(cp, metric_val) for (cp, metric_val) in checkpoint_paths if not pd.isna(metric_val)]

    if not checkpoint_paths:
        logging.Logger(name='No Checkpoint').error("No checkpoints have been found for trial: " + trial_dir)
        return

    a = -1 if mode == "max" else 1
    best_path_metrics = sorted(checkpoint_paths, key=lambda x: a * x[1])

    return os.path.join(trial_dir, best_path_metrics[0][0])
