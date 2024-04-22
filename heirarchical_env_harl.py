import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'HARL')))

from harl.runners import RUNNER_REGISTRY
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
import torch

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from ray.rllib.algorithms.algorithm import Algorithm

from dcrl_env import DCRL
from hierarchical_workload_optimizer import WorkloadOptimizer
import matplotlib.pyplot as plt

DEFAULT_CONFIG = {
    # AZ config
    'config1' : {
        'agents': ['agent_ls', 'agent_dc', 'agent_bat'],
        'location': 'az',
        'cintensity_file': 'AZPS_NG_&_avgCI.csv',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc3.json',
        'datacenter_capacity_mw' : 1.1,
        'timezone_shift': 8,
        'month': 8,
        'days_per_episode': 30,
        },

    # NY config
    'config2' : {
        'agents': ['agent_ls', 'agent_dc', 'agent_bat'],
        'location': 'ny',
        'cintensity_file': 'NYIS_NG_&_avgCI.csv',
        'weather_file': 'USA_NY_New.York-Kennedy.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc2.json',
        'datacenter_capacity_mw' : 1,
        'timezone_shift': 0,
        'month': 8,
        'days_per_episode': 30
        },

    # WA config
    'config3' : {
        'agents': ['agent_ls', 'agent_dc', 'agent_bat'],
        'location': 'wa',
        'cintensity_file': 'WAAT_NG_&_avgCI.csv',
        'weather_file': 'USA_WA_Port.Angeles-Fairchild.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 0.9,
        'timezone_shift': 16,
        'month': 8,
        'days_per_episode': 30
        },
    
}

harl_args = {
    'algo' : 'happo',
    'env' : 'dcrl',
    'exp_name' : 'll_actor',
    'model_dir': '/lustre/guillant/HARL/results/dcrl/ny/happo/installtest/seed-00001-2024-04-22-20-59-21/models',
}

class HeirarchicalDCRL(gym.Env):

    def __init__(self, config):

        # Init all datacenter environments
        DC1 = DCRL(config['config1'])
        DC2 = DCRL(config['config2'])
        DC3 = DCRL(config['config3'])

        self.environments = {
            'DC1': DC1,
            'DC2': DC2,
            'DC3': DC3,
        }

        # Load trained agent from harl
        # Create the instance of the runner and Restore with model_dir param
        algo_args, env_args = get_defaults_yaml_args(harl_args["algo"], harl_args["env"])
        algo_args['train']['n_rollout_threads'] = 1
        algo_args['eval']['n_eval_rollout_threads'] = 1
        algo_args['train']['model_dir'] = harl_args['model_dir']
        
        self.ll_actors = RUNNER_REGISTRY[harl_args["algo"]](harl_args, algo_args, env_args)

        self.low_level_observations = {}

        # Define observation and action space
        self.action_space = Dict({dc: Box(0, 1, [1]) for dc in self.environments})
        
    def reset(self):
        self.low_level_observations = {}
        heir_obs = {}
        infos = {}

        # Reset environments and store initial observations and infos
        for env_id, env in self.environments.items():
            obs, info = env.reset()
            self.low_level_observations[env_id] = obs
            heir_obs[env_id] = env.get_hierarchical_variables()
            infos[env_id] = info
        
        # Initialize metrics
        self.metrics = {
            env_id: {
                'bat_CO2_footprint': [],
                'bat_total_energy_with_battery_KWh': [],
                'ls_tasks_dropped': [],
                'dc_water_usage': [],
                'original_workload': [],
                'top_level_workload': [],
                'low_level_workload': []
            } 
            for env_id in self.environments
        }   

        self.all_done = {env_id: False for env_id in self.environments}

        return heir_obs, infos
        
    def calc_reward(self,):
        pass
    
    @torch.no_grad()
    def step(self, actions, original_workload):
        
        # Update workload for all DCs
        for env_id, adj_workload in actions.items():
            self.environments[env_id].set_hierarchical_workload(round(adj_workload, 6))

        # Compute actions for each agent in each environment
        low_level_actions = {env_id: {} for env_id in self.environments}
        
        eval_rnn_states = np.zeros((1, 1, 1), dtype=np.float32)
        eval_masks = np.ones((1, 1), dtype=np.float32)
        
        expected_length = self.ll_actors.actor[0].obs_space.shape[0] # 19
        
        for env_id, env_obs in self.low_level_observations.items():
            if self.all_done[env_id]:  # Skip if environment is done
                continue

            for agent_idx, (agent_id, agent_obs) in enumerate(env_obs.items()):
                policy_id = agent_id  # Customize policy ID if necessary
                additional_length = expected_length - len(agent_obs)
                 # Create an array of 1's with the required additional length
                ones_to_add = np.ones(additional_length, dtype=agent_obs.dtype)
                # Concatenate the current array with the array of 1's
                agent_obs = np.concatenate((agent_obs, ones_to_add))

                # eval_rnn_states and eval_masks is only being used on RNN
                # Obtain the action of each actor
                # TODO: Make sure that we are asking the correct actor for their agent_id and agent_idx
                action, _ = self.ll_actors.actor[agent_idx].act(agent_obs, eval_rnn_states, eval_masks, deterministic=True)
                
                low_level_actions[env_id][agent_id] = action.numpy()[0]


        # Step through each environment with computed low_level_actions
        for env_idx, env_id in enumerate(self.environments):
            if self.all_done[env_id]:
                continue
            new_obs, rewards, terminated, truncated, info = self.environments[env_id].step(low_level_actions[env_id])
            self.low_level_observations[env_id] = new_obs
            self.all_done[env_id] = terminated['__all__'] or truncated['__all__']

            # Update metrics for each environment
            env_metrics = self.metrics[env_id]
            env_metrics['bat_CO2_footprint'].append(info['agent_bat']['bat_CO2_footprint'])
            env_metrics['bat_total_energy_with_battery_KWh'].append(info['agent_bat']['bat_total_energy_with_battery_KWh'])
            env_metrics['ls_tasks_dropped'].append(info['agent_ls']['ls_tasks_dropped'])
            env_metrics['dc_water_usage'].append(info['agent_dc']['dc_water_usage'])
            env_metrics['original_workload'].append(original_workload[env_idx])
            env_metrics['top_level_workload'].append(actions[env_id])
            env_metrics['low_level_workload'].append(info['agent_ls']['ls_shifted_workload'])
            # env_metrics['low_level_workload'].append(info['agent_ls']['ls_original_workload'])

        done = any(self.all_done.values())

        # Get observations for the next step
        heir_obs = {}
        if not done:
            for env_id, env in self.environments.items():
                heir_obs[env_id] = env.get_hierarchical_variables()

        return heir_obs, done


if __name__ == '__main__':
    env = HeirarchicalDCRL(DEFAULT_CONFIG)
    done = False
    obs, _ = env.reset()
    top_level_actor = WorkloadOptimizer(env.environments.keys())
    
    max_iterations = 4*24*DEFAULT_CONFIG['config1']['days_per_episode']
    # Antonio: Keep in mind that each environment is set to have days_per_episode=30. You can modify this parameter to simulate the whole year
    with tqdm(total=max_iterations) as pbar:
        while not done:
            # breakpoint()
            # Random actions
            # actions = {key: val[0] for key, val in env.action_space.sample().items()}

            # Do nothing 
            # actions = {dc: state[1] for dc, state in obs.items()}

            # Rule-based
            original_workload = [state[1] for state in obs.values()]
            actions, _ = top_level_actor.compute_adjusted_workload(obs)
            obs, done = env.step(actions, original_workload)

            # Update the progress bar
            pbar.update(1)

    # After simulation, calculate average metrics for each environment
    average_metrics = {
        env_id: {metric: sum(values) / len(values) for metric, values in env_metrics.items()}
        for env_id, env_metrics in env.metrics.items()
    }

    # Print average metrics for each environment
    for env_id, env_metrics in average_metrics.items():
        print(f"Average Metrics for {env_id}:")
        for metric, value in env_metrics.items():
            print(f"        {metric}: {value:.2f}")
        print()  # Blank line for readability

    total_metrics = {}
    for metric in env_metrics:
        total_metrics[metric] = 0.0
        for env_id in average_metrics:
            total_metrics[metric] += average_metrics[env_id][metric]

        print(f'{metric}: {total_metrics[metric]:,.2f}')
        
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # Adjust size as needed

    # Convert index to days for the x-axis
    time_steps_per_day = 96  # 15 minutes per timestep, 96 timesteps per day
    x_labels = np.arange(0, len(env.metrics['DC1']['original_workload']) + 1) / time_steps_per_day
    max_days = x_labels[-1]
    
    # Loop through each configuration and plot
    for i, (env_id, env_metric) in enumerate(env.metrics.items()):
        axs[i].plot(x_labels[:-1], env_metric['original_workload'], label='Original Workload', color='tab:blue')
        axs[i].plot(x_labels[:-1], env_metric['top_level_workload'], label='Geographical Load Shifting', color='tab:green')
        axs[i].plot(x_labels[:-1], env_metric['low_level_workload'], label='Temporal Load Shifting', color='tab:red')
        
        axs[i].set_title(f'Location: {env.environments[env_id].location.upper()}', fontsize=14)
        axs[i].set_xlabel('Days', fontsize=12)
        axs[i].set_ylabel('Workload', fontsize=12)
        axs[i].legend(fontsize=10)
        
        # Set x-axis limits
        axs[i].set_xlim(0, max_days)
        axs[i].set_ylim(0, 1)

        # Enable grid
        axs[i].grid(True, which='both', linestyle=':', linewidth=0.75, color='gray', alpha=0.7)

        # Set the fontsize for the tick labels
        axs[i].tick_params(axis='both', which='major', labelsize=10)


    plt.tight_layout()
    plt.savefig('workload_evaluation.png', dpi=300)