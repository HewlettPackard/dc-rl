"""
Creates algorithm configuration for PPO and starts training process
"""

#%%

import os
from typing import Union

import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import (Postprocessing,
                                                 compute_advantages)
from ray.rllib.examples.models.centralized_critic_models import \
    TorchCentralizedCriticModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

from create_trainable import create_wrapped_trainable
from dcrl_env import DCRL
# from train import train
from dcrl_eplus_env import DCRLeplus
from utils.rllib_callbacks import CustomCallbacks

NAME = "test"
RESULTS_DIR = './results'

# Data collection config
TIMESTEP_PER_HOUR = 4
COLLECTED_DAYS = 7
NUM_AGENTS = 2
NUM_WORKERS = 1

ModelCatalog.register_custom_model("cc_model", TorchCentralizedCriticModel)


OPPONENT_OBS = "agent_dc"
OPPONENT_ACTION = "agent_ls"



class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch["agent_dc"],
        train_batch[OPPONENT_ACTION],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):
    pytorch = True
    if (pytorch and hasattr(policy, "compute_central_vf")) or (
        not pytorch and policy.loss_initialized()
    ):
        assert other_agent_batches is not None
        if policy.config["enable_connectors"]:
            [(_, _, opponent_batch)] = list(other_agent_batches.values())
        else:
            [(_, opponent_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        if pytorch:
            sample_batch[SampleBatch.VF_PREDS] = (
                policy.compute_central_vf(
                    convert_to_torch_tensor(
                        sample_batch[SampleBatch.CUR_OBS], policy.device
                    ),
                    convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
                    convert_to_torch_tensor(
                        sample_batch[OPPONENT_ACTION], policy.device
                    ),
                )
                .cpu()
                .detach()
                .numpy()
            )

    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch

class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return centralized_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )

class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
            return CCPPOTorchPolicy

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
        CentralizedCritic,
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

    
    CONFIG = (
            PPOConfig()
            .environment(
                env=DCRL if not os.getenv('EPLUS') else DCRLeplus,
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