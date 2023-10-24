"""
Creates algorithm configuration for PPO and starts training process
"""

import os

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from dcrl_eplus_env import DCRLeplus
from dcrl_env import DCRL
from utils.rllib_callbacks import CustomCallbacks

from typing import Union

import ray
from ray import air, tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms import Algorithm, AlgorithmConfig

from create_trainable import create_wrapped_trainable
from gymnasium.spaces import Dict, Discrete
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.models.centralized_critic_models import YetAnotherTorchCentralizedCriticModel

import numpy as np

class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(1))

        # set the opponent actions into the observation
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array(
            [action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]
        )
        to_update[:, -1:] = opponent_actions
        
        
def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""

    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": 0,  # filled in by FillInActions
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": 0,  # filled in by FillInActions
        },
    }
    return new_obs


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

    ModelCatalog.register_custom_model("cc_model", YetAnotherTorchCentralizedCriticModel)
    
    # Create a dummy environment to get obs. and action space
    dummy_env = config.env(config.env_config)
    ls_env, dc_env = dummy_env.ls_env, dummy_env.dc_env

    observer_space = Dict(
        {
            "own_obs": ls_env.observation_space,
            # These two fields are filled in by the CentralCriticObserver, and are
            # not used for inference, only for training.
            "opponent_obs": dc_env.observation_space,
            "opponent_action": Discrete(1),
        }
    )
    
    config = config.multi_agent(
                policies={
                    "agent_ls": PolicySpec(
                        None,
                        observer_space,
                        ls_env.action_space,
                        config={"agent_id" : 0},
                    ),
                    "agent_dc": PolicySpec(
                        None,
                        observer_space,
                        dc_env.action_space,
                        config={"agent_id" : 1},
                    )
                },
                policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "agent_ls" if agent_id == 0  else "agent_dc",
                observation_fn=central_critic_observer,
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
        run_config=air.RunConfig(stop={"timesteps_total": 100_000_000_000},
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
    
if __name__ == '__main__':
    os.environ["RAY_DEDUP_LOGS"] = "0"

    # ray.init(ignore_reinit_error=True)
    ray.init(local_mode=True, ignore_reinit_error=True)

        
    # Data collection config
    TIMESTEP_PER_HOUR = 4
    COLLECTED_DAYS = 7
    NUM_AGENTS = 2
    NUM_WORKERS = 1


    NAME = "test"
    RESULTS_DIR = './results'

    CONFIG = (
            PPOConfig()
            .environment(
                env=DCRL,
                env_config={
                    # Agents active
                    'agents': ['agent_ls', 'agent_dc'],

                    # Datafiles
                    'location': 'ny',
                    'cintensity_file': 'NYIS_NG_&_avgCI.csv',
                    'weather_file': 'USA_NY_New.York-Kennedy.epw',
                    'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',

                    # Battery capacity
                    'max_bat_cap_Mw': 0.05,
                    
                    # Collaborative weight in the reward
                    'individual_reward_weight': 0.8,
                    
                    # Flexible load ratio
                    'flexible_load': 0.5,
                    
                    # Specify reward methods
                    'ls_reward': 'default_ls_reward',
                    'dc_reward': 'default_dc_reward',
                    'bat_reward': 'default_bat_reward'
                }
            )
            .framework("torch")
            .rollouts(num_rollout_workers=NUM_WORKERS,
                    rollout_fragment_length='auto')
            .training(
                gamma=0.99, 
                lr=1e-5, 
                lr_schedule=[[0, 3e-5], [10000000, 1e-6]],
                kl_coeff=0.3, 
                clip_param=0.02,
                entropy_coeff=0.05,
                use_gae=True, 
                train_batch_size=96*2 * NUM_WORKERS * NUM_AGENTS,
                model={"custom_model": "cc_model"},
                _enable_learner_api=False,
                shuffle_sequences=True
            )
            .callbacks(CustomCallbacks)
            .resources(num_cpus_per_worker=1, num_gpus=0)
        )

    train(
        algorithm="PPO",
        config=CONFIG,
        results_dir=RESULTS_DIR,
        name=NAME,
    )