import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig

from geo_dcrl import (
    HierarchicalDCRLCombinatorial,
    DEFAULT_CONFIG
)
# from utils.rllib_callbacks import CustomCallbacks
from create_trainable import create_wrapped_trainable

NUM_WORKERS = 4
NAME = "HDCRL-Combinatorial"
RESULTS_DIR = './results/'
NUM_TRAINING_STEPS = 300_000_000
CONFIG = (
        PPOConfig()
        .environment(
            env=HierarchicalDCRLCombinatorial,
            env_config=DEFAULT_CONFIG
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=NUM_WORKERS,
            rollout_fragment_length=2,
            # observation_filter='MeanStdFilter'
            )
        .training(
            gamma=0.5,
            lr=1e-5,
            kl_coeff=0.2,
            clip_param=0.1,
            entropy_coeff_schedule=0.2,  # [[0,0.02],[NUM_TRAINING_STEPS,0.0]],
            use_gae=True,
            train_batch_size=2048,
            num_sgd_iter=5,
            model={'fcnet_hiddens': [32, 32]},
            shuffle_sequences=True
        )
        .resources(num_gpus=0)
        .debugging(seed=0)
    )

if __name__ == '__main__':
    os.environ["RAY_DEDUP_LOGS"] = "0"
    # ray.init(logging_level='debug', num_cpus=NUM_WORKERS+1)
    ray.init(ignore_reinit_error=True)

    tune.Tuner(
        create_wrapped_trainable(PPO),
        param_space=CONFIG.to_dict(),
        run_config=air.RunConfig(
            stop={"timesteps_total": NUM_TRAINING_STEPS},
            verbose=0,
            local_dir=RESULTS_DIR,
            # storage_path=RESULTS_DIR,
            name=NAME,
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=5,
                num_to_keep=5,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max"
            ),
        )
    ).fit()

