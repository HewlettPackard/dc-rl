from typing import Union

import ray
from ray import air, tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms import Algorithm, AlgorithmConfig

from create_trainable import create_wrapped_trainable

def train(
    algorithm: Union[str, Algorithm], 
    config: AlgorithmConfig,
    results_dir: str = './results',
    name: str = 'latest_experiment',
    overrides: dict = {}
    ):
    """
    Starts the training process for a given algorithm.

    Args:
        algorithm (rllib.algorithm): RL algorithm to use for training.
        config (algorithm_config): Algorithm training configuration.
        results_dir (string): Directory to save the results
        overrides (dict): Extra configuration

    """

    # Create a dummy environment to get obs. and action space
    dummy_env = config.env(config.env_config)
    ls_env, dc_env, bat_env = dummy_env.ls_env, dummy_env.dc_env, dummy_env.bat_env 

    config = config.multi_agent(
                policies={
                    "agent_ls": PolicySpec(
                        None,
                        ls_env.observation_space,
                        ls_env.action_space,
                        config={"agent_id" : 0},
                    ),
                    "agent_dc": PolicySpec(
                        None,
                        dc_env.observation_space,
                        dc_env.action_space,
                        config={"agent_id" : 1},
                    ),
                    "agent_bat": PolicySpec(
                        None,
                        bat_env.observation_space,
                        bat_env.action_space,
                        config={"agent_id" : 2},
                    ),
                },
                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            )

    # Only include agents as specified in env_config['agents']
    for agent in list(config.policies.keys()):
        if agent not in config.env_config['agents']:
            config.policies.pop(agent)

    # Reassign agent ids
    for i, policy in enumerate(config.policies.values()):
        policy.config['agent_id'] = i
    
    config = config.to_dict()
    config.update(overrides)

   
    tune.Tuner(
        create_wrapped_trainable(algorithm),
        param_space=config,
        run_config=air.RunConfig(stop={"timesteps_total": 100_000_000},
            verbose=0,
            local_dir=results_dir,
            name=name,
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=5,
                num_to_keep=5,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max"
            ),
        )
    ).fit()