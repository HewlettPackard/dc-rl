import os
import sys
import random
import numpy as np
import json

import warnings
import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete, Tuple, Discrete

from dcrl_env_harl_partialobs import DCRL as DCRLPartObs

sys.path.insert(0, f'{os.path.dirname(os.path.abspath(__file__))}/HARL')
from harl.runners import RUNNER_REGISTRY
from harl.utils.configs_tools import get_defaults_yaml_args
from utils.base_agents import *

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
    )

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = {
    # NY config
    'config1' : {
        'location': 'ny',
        'cintensity_file': 'NY_NG_&_avgCI.csv',
        'weather_file': 'USA_NY_New.York-LaGuardia.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc3.json',
        'datacenter_capacity_mw' : 1,
        'timezone_shift': 8,
        'month': 7,
        'days_per_episode': 30
        },

    # GA config
    'config2' : {
        'location': 'ga',
        'cintensity_file': 'GA_NG_&_avgCI.csv',
        'weather_file': 'USA_GA_New.York-LaGuardia.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc2.json',
        'datacenter_capacity_mw' : 1,
        'timezone_shift': 0,
        'month': 7,
        'days_per_episode': 30
        },

    # WA config
    'config3' : {
        'location': 'ca',
        'cintensity_file': 'CA_NG_&_avgCI.csv',
        'weather_file': 'USA_CA_San.Jose-Mineta.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 0.9,
        'timezone_shift': 16,
        'month': 7,
        'days_per_episode': 30
        },
    
    # List of active low-level agents
    'active_agents': ['agent_dc'],

    # config for loading trained low-level agents
    'low_level_actor_config': {
        'harl': {
            'algo' : 'happo',
            'env' : 'dcrl',
            'exp_name' : 'll_actor',
            'model_dir': f'{CURR_DIR}/seed-00001-2024-05-01-21-50-12/models',
            'saved_config' : f'{CURR_DIR}/seed-00001-2024-05-01-21-50-12/config.json'
            }
        }
    }

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
        
        with open(config['saved_config'], encoding='utf-8') as file:
            saved_config = json.load(file)
        algo_args, env_args = saved_config['algo_args'], saved_config['env_args']
        
        algo_args, env_args = get_defaults_yaml_args(config["algo"], config["env"])
        algo_args['train']['n_rollout_threads'] = 1
        algo_args['eval']['n_eval_rollout_threads'] = 1
        algo_args['train']['model_dir'] = config['model_dir']
    
        self.ll_actors = RUNNER_REGISTRY[saved_config["main_args"]["algo"]](config, algo_args, env_args)
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

class HARL_HeirarchicalDCRL(gym.Env):
    
    def __init__(self, config): 

        self.config = config

        # Init all datacenter environments
        DC1 = DCRLPartObs(config['config1'])
        DC2 = DCRLPartObs(config['config2'])
        DC3 = DCRLPartObs(config['config3'])

        self.datacenters = {
            'DC1': DC1,
            'DC2': DC2,
            'DC3': DC3,
        }

        # Load trained lower level agent
        self.lower_level_actor = LowLevelActorHARL(
            config['low_level_actor_config'],
            config['active_agents']
            )
        
        self.low_level_observations = {}
        self.low_level_infos = {}

        # Define observation and action space
        self.observation_space = Dict({dc: Box(0, 10000, [5]) for dc in self.datacenters})
        self.action_space = Dict({
            "sender": Discrete(3),
            "receiver": Discrete(3),
            "workload_to_move": Box(0., 1., [1])
        })

def main():
    """Main function."""
    pass


if __name__ == "__main__":
    main()