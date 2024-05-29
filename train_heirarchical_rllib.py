import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.utils.filter import MeanStdFilter
import argparse
from gymnasium.spaces import Discrete, Tuple
import logging
from ray.rllib.utils.test_utils import check_learning_achieved

from heirarchical_env_rllib import (
    HeirarchicalDCRL_RLLib, 
    DEFAULT_CONFIG
)

from create_trainable import create_wrapped_trainable

NUM_WORKERS = 4
NAME = "test"
RESULTS_DIR = './results/'


parser = argparse.ArgumentParser()
parser.add_argument("--flat", action="store_true")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.0, help="Reward at which we stop training."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    #ray.init(local_mode=args.local_mode)
    ray.init(local_mode=True, ignore_reinit_error=True)

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.flat:
        results = tune.Tuner(
            "PPO",
            run_config=air.RunConfig(stop=stop),
            param_space=(
                PPOConfig()
                .environment(HeirarchicalDCRL_RLLib)
                .rollouts(num_rollout_workers=0)
                .framework(args.framework)
            ).to_dict(),
        ).fit()
    else:
        heirDCRL = HeirarchicalDCRL_RLLib(DEFAULT_CONFIG)

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id.startswith("high_level"):
                return "high_level_policy"
            elif agent_id == "DC1":
                return "DC1_policy"
            elif agent_id == "DC2":
                return "DC2_policy"
            elif agent_id == "DC3":
                return "DC3_policy"

        config = (
            PPOConfig()
            .environment(
                env=HeirarchicalDCRL_RLLib,
                env_config=DEFAULT_CONFIG
            )
            .framework(args.framework)
            .rollouts(num_rollout_workers=0)
            .training(entropy_coeff=0.01)
            .multi_agent(
                policies={
                    "high_level_policy": (
                        None,
                        heirDCRL.observation_space,
                        heirDCRL.action_space,
                        PPOConfig.overrides(gamma=0.9),
                    ),
                    "DC1_policy": (
                        None,
                        heirDCRL.dc_observation_space,
                        heirDCRL.datacenters['DC1'].action_space,
                        PPOConfig.overrides(gamma=0.0),
                    ),
                    "DC2_policy": (
                        None,
                        heirDCRL.dc_observation_space,
                        heirDCRL.datacenters['DC2'].action_space,
                        PPOConfig.overrides(gamma=0.0),
                    ),
                    "DC3_policy": (
                        None,
                        heirDCRL.dc_observation_space,
                        heirDCRL.datacenters['DC3'].action_space,
                        PPOConfig.overrides(gamma=0.0),
                    ),
                },
                policy_mapping_fn=policy_mapping_fn,
            )
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        )

        results = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop, verbose=1),
        ).fit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
