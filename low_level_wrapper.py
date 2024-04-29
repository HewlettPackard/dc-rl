import os
import random
import warnings
    
import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_tf

from harl.runners import RUNNER_REGISTRY
from harl.utils.configs_tools import get_defaults_yaml_args
from dcrl_env import DCRL
from hierarchical_workload_optimizer import WorkloadOptimizer
from utils.base_agents import *

# Boilerplate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf1, *_ = try_import_tf()
tf1.disable_v2_behavior()

class LowLevelActorBase:
    "Base class for trained low level agents within a DCRL environment"

    def __init__(self, config: Dict = {}):
        
        self.do_nothing_actors = {
            "agent_ls": BaseLoadShiftingAgent(), 
            "agent_dc": BaseHVACAgent(), 
            "agent_bat": BaseBatteryAgent()
        }
    
    def compute_actions(self, observations: Dict, **kwargs) -> Dict:
        actions = {env_id: {} for env_id in observations}

        for env_id, obs in observations.items():
            for agent_id in obs:
                actions[env_id][agent_id] = self.do_nothing_actors[agent_id].do_nothing_action()

        return actions

class LowLevelActorHARL(LowLevelActorBase):
    
    def __init__(self, config, active_agents: list = []):
        super().__init__(config)

        config = config['harl']
        algo_args, env_args = get_defaults_yaml_args(config["algo"], config["env"])
        algo_args['train']['n_rollout_threads'] = 1
        algo_args['eval']['n_eval_rollout_threads'] = 1
        algo_args['train']['model_dir'] = config['model_dir']
    
        self.ll_actors = RUNNER_REGISTRY[config["algo"]](config, algo_args, env_args)
        self.active_agents = active_agents
        
    def compute_actions(self, observations, **kwargs):
        actions = {}
        
        eval_rnn_states = np.zeros((1, 1, 1), dtype=np.float32)
        eval_masks = np.ones((1, 1), dtype=np.float32)
        
        expected_length = self.ll_actors.actor[0].obs_space.shape[0]

        for agent_idx, (agent_id, agent_obs) in enumerate(observations.items()):
            if agent_id in self.active_agents:
                additional_length = expected_length - len(agent_obs)
                
                # Create an array of 1's with the required additional length
                ones_to_add = np.ones(additional_length, dtype=agent_obs.dtype)

                # Concatenate the current array with the array of 1's
                agent_obs = np.concatenate((agent_obs, ones_to_add))

                # eval_rnn_states and eval_masks is only being used on RNN
                # Obtain the action of each actor
                # TODO: Make sure that we are asking the correct actor for their agent_id and agent_idx
                action, _ = self.ll_actors.actor[agent_idx].act(agent_obs, eval_rnn_states, eval_masks, deterministic=True)
        
                actions[agent_id] = action.numpy()[0]
            else:
                actions[agent_id] = self.do_nothing_actors[agent_id].do_nothing_action()

        return actions

class LowLevelActorRLLIB(LowLevelActorBase):

    def __init__(self, config, active_agents: list = []):
        super().__init__(config)

        self.active_agents = active_agents

        self.policies = {}
        for agent_id in self.active_agents:
            with tf1.Session().as_default():
                self.policies[agent_id] = Policy.from_checkpoint(
                    f'{config["rllib"]["checkpoint_path"]}/policies/{agent_id}'
                    )

        self.is_maddpg = config["rllib"]["is_maddpg"]
        # self.actor = Algorithm.from_checkpoint(config['checkpoint_path'])
        
    def compute_actions(self, observations, **kwargs):
        actions = {}
        
        for agent_id, obs in observations.items():
            if agent_id in self.active_agents:
                action = self.policies[agent_id].compute_single_action(obs)[0]
                if self.is_maddpg:
                    # MADDPG returns logits instead of discrete actions
                    actions[agent_id] = np.argmax(action, axis=-1)
            else:
                actions[agent_id] = self.do_nothing_actors[agent_id].do_nothing_action()

        return actions