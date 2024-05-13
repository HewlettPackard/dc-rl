import os
import sys
import random
import json
import numpy as np

from tqdm import tqdm

import torch
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'heterogeneous_dcrl')))  # default foldername if you git clone heterogeneous_dcrl
from harl.runners import RUNNER_REGISTRY

from heirarchical_env import HeirarchicalDCRLWithHysterisis
from heirarchical_env import DEFAULT_CONFIG, CURR_DIR
from heirarchical_env import LowLevelActorHARL
from hierarchical_workload_optimizer import WorkloadOptimizer
from dcrl_env_harl_partialobs import DCRL as DCRLPartObs

from utils.base_agents import *


DEFAULT_CONFIG['low_level_actor_config'] = {
        'harl': {
            'algo' : 'haa2c',
            'env' : 'dcrl',
            'exp_name' : 'll_actor',
            'model_dir': f'{CURR_DIR}/seed-00001-2024-05-01-21-50-12/models',
            'saved_config' : f'{CURR_DIR}/seed-00001-2024-05-01-21-50-12/config.json'
            }
        }
    
class LowLevelActorHARL_v2(LowLevelActorHARL):
    
    def __init__(self, config, active_agents: list = []):
        self.do_nothing_actors = {
            "agent_ls": BaseLoadShiftingAgent(), 
            "agent_dc": BaseHVACAgent(),
            "agent_bat": BaseBatteryAgent()
        }

        config = config['harl']
        
        with open(config['saved_config'], encoding='utf-8') as file:
            saved_config = json.load(file)
        algo_args, env_args = saved_config['algo_args'], saved_config['env_args']

        algo_args['train']['n_rollout_threads'] = 1
        algo_args['eval']['n_eval_rollout_threads'] = 1
        algo_args['train']['model_dir'] = config['model_dir']
    
        self.ll_actors = RUNNER_REGISTRY[saved_config["main_args"]["algo"]](config, algo_args, env_args)
        self.active_agents = active_agents

class HARL_HierarchicalDCRL(HeirarchicalDCRLWithHysterisis):
    
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
        self.lower_level_actor = LowLevelActorHARL_v2(
            config['low_level_actor_config'],
            config['active_agents']
            )
        
        self.low_level_observations = {}
        self.low_level_infos = {}

        # Define observation and action space
        self.observation_space = Dict({dc: Box(0, 10000, [5]) for dc in self.datacenters})

        self.num_datacenters = len(self.datacenters)
        self.num_actions = int((self.num_datacenters-1)*(self.num_datacenters)/2)
        # Update the action with a signed value
        self.action_space = Box(-1, 1, shape=(self.num_actions,))
 
    def reset(self, seed=None, options=None):
        self.not_computed_workload = 0
        # Set seed if we are not in rllib
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            # tf1.random.set_random_seed(0)

        self.low_level_observations = {}
        self.heir_obs = {}
        infos = {}

        # Reset environments and store initial observations and infos
        for env_id, env in self.datacenters.items():
            obs, info, available_actions = env.reset()
            self.low_level_observations[env_id] = obs
            self.low_level_infos[env_id] = info
            
            self.heir_obs[env_id] = self.get_dc_variables(env_id)
        
        self.start_index_manager = env.workload_m.time_step
        self.simulated_days = env.days_per_episode

        # Initialize metrics
        self.metrics = {
            env_id: {
                'bat_CO2_footprint': [],
                'bat_total_energy_with_battery_KWh': [],
                'ls_tasks_dropped': [],
                'dc_water_usage': [],
            }
            for env_id in self.datacenters
        }

        self.all_done = {env_id: False for env_id in self.datacenters}
        
        return self.heir_obs, self.low_level_infos
    
    def step(self, actions):
        
        # actions is an array of floats in the range (-1.0, 1.0) of "ordered data centers" in the format
        # [DC1<->DC2, DC1<->DC3, ..., DC1<->DCN... DC2<->DC3...DC2<->DCN....DC(N-1)<->DCN] where N = self.num_datacenters
        # have to return as a list of same length as actions with each element being a dictionary of sender, receiver and workload_to_move
        
        def convert_actions(actions):
            new_actions = {}
            cntr = 1
            for i in range(self.num_datacenters-1):
                offset = int(self.num_datacenters*i - i*(i+1)/2)
                for j in range(self.num_datacenters-(i+1)):
                    # depending on the sign choose the sender and receiver
                    # if positive, i is sender and j+i+1 is receiver
                    # if negative, j+i+1 is sender and i is receiver
                    if actions[offset+j] >= 0:
                        new_actions[f'transfer_{cntr}'] = \
                            {'sender': i, 'receiver': j+i+1, \
                                'workload_to_move': np.array([actions[offset+j]])}
                    else:
                        new_actions[f'transfer_{cntr}'] = \
                            {'sender': j+i+1, 'receiver': i, \
                                'workload_to_move': np.array([-actions[offset+j]])}
                    cntr += 1    
            return new_actions
        
        actions = convert_actions(actions)

        return super().step(actions)

def main():
    """Main function."""
    # env = HARL_HierarchicalDCRL(DEFAULT_CONFIG)
    env = HARL_HierarchicalDCRL(DEFAULT_CONFIG)
    done = False
    obs, _ = env.reset(seed=0)
    total_reward = 0
    
    greedy_optimizer = WorkloadOptimizer(env.datacenters.keys())
    
    max_iterations = 4*24*30
    # Antonio: Keep in mind that each environment is set to have days_per_episode=30. You can modify this parameter to simulate the whole year
    with tqdm(total=max_iterations) as pbar:
        while not done:
    
            # Random actions
            # actions = env.action_space.sample()
            # actions = np.random.uniform(-1, 1, env.action_space.shape)  # for environement v2
            
            # Do nothing
            # actions = {'sender': 0, 'receiver': 0, 'workload_to_move': np.array([0.0])}
            # actions = np.zeros(env.action_space.shape)  # for environement v2

            # One-step greedy
            # ci = [obs[dc][-1] for dc in env.datacenters]
            # actions = {'sender': np.argmax(ci), 'receiver': np.argmin(ci), 'workload_to_move': np.array([1.])}                                                                    
            
            # Multi-step Greedy
            actions = np.zeros(env.action_space.shape)  # for environement v2
            _, transfer_matrix = greedy_optimizer.compute_adjusted_workload(obs)
            for send_key,val_dict in transfer_matrix.items():
                for receive_key,val in val_dict.items():
                    if val!=0:
                        i,k = int(send_key[-1])-1, int(receive_key[-1])-1
                        if i > k:  # the ordering is not right; reverse it and als; also assuming val!= also weeds out i = k case
                            i,k = k,i
                            multiplier = -1
                        j = k-i-1
                        offset = int(env.num_datacenters*i - i*(i+1)/2)
                        actions[offset+j] = val*multiplier
            actions = actions.clip(-1,1)
            
            obs, reward, _, truncated, _ = env.step(actions)
            done = truncated
            total_reward += reward

            # Update the progress bar
            pbar.update(1)

    # After simulation, calculate average metrics for each environment
    average_metrics = {
        env_id: {metric: sum(values) / len(values) for metric, values in env_metrics.items()}
        for env_id, env_metrics in env.metrics.items()
    }

    # Print average metrics for each environment
    for env_id, env_metrics in average_metrics.items():
        print(f"Average Metrics for {env.datacenters[env_id].location}:")
        for metric, value in env_metrics.items():
            print(f"\t{metric}: {value:,.2f}")
        print()  # Blank line for readability

    # Sum metrics across datacenters
    print("Summed metrics across all DC:")
    total_metrics = {}
    for metric in env_metrics:
        total_metrics[metric] = 0.0
        for env_id in average_metrics:
            total_metrics[metric] += average_metrics[env_id][metric]

        print(f'\t{metric}: {total_metrics[metric]:,.2f}')

    print("Total reward = ", total_reward)


if __name__ == "__main__":
    main()