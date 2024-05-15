import os
import argparse
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.utils.filter import MeanStdFilter

from heirarchical_env import (
    HeirarchicalDCRL, 
    HeirarchicalDCRLWithHysterisis,
    HeirarchicalDCRLWithHysterisisMultistep, 
    DEFAULT_CONFIG
)
from utils.rllib_callbacks import CustomCallbacks
from create_trainable import create_wrapped_trainable

NUM_WORKERS = 4

# Create a parser
parser = argparse.ArgumentParser(description='Run training with specified environment.')
# Add an argument for the experiment name
parser.add_argument('--name', type=str, default='SingleStep', choices=['SingleStep', 'MultiStep'],
                    help='Name of the experiment. Options are SingleStep and MultiStep.')

# Parse the arguments
args = parser.parse_args()

# Use the provided name, or default to 'SingleStep' if no name was provided
NAME = args.name

env_name = {
    "SingleStep": HeirarchicalDCRLWithHysterisis,
    "MultiStep": HeirarchicalDCRLWithHysterisisMultistep
}[NAME]

RESULTS_DIR = './results/'

CONFIG = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config=DEFAULT_CONFIG
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=NUM_WORKERS,
            # rollout_fragment_length=10,
            # observation_filter='MeanStdFilter'
            )
        .training(
            gamma=0.99,
            # gamma=0.0, 
            lr=1e-4,
            kl_coeff=0.2,
            clip_param=0.1,
            entropy_coeff=0.0,
            use_gae=True,
            train_batch_size=2048,
            num_sgd_iter=50,
            model={'fcnet_hiddens': [64, 64]}, 
            shuffle_sequences=True
        )
        .resources(num_gpus=0)
        .debugging(seed=0)
    )

if __name__ == '__main__':
    os.environ["RAY_DEDUP_LOGS"] = "0"
    # ray.init(logging_level='debug', num_cpus=NUM_WORKERS+1)
    # ray.init(local_mode=True, ignore_reinit_error=True)
    ray.init(ignore_reinit_error=True, num_cpus=NUM_WORKERS+1)
    
    tune.Tuner(
        create_wrapped_trainable(PPO),
        param_space=CONFIG.to_dict(),
        run_config=air.RunConfig(
            stop={"timesteps_total": 100_000_000},
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
