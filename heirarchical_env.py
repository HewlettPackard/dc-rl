import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from ray.rllib.algorithms.algorithm import Algorithm

from dcrl_env import DCRL
from hierarchical_workload_optimizer import WorkloadOptimizer

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
        'month': 0,
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
        'month': 0,
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
        'month': 0,
        'days_per_episode': 30
        },
    
    'checkpoint_path': 'results/test/MADDPGStable_DCRL_41ad0_00000_0_2024-04-11_20-08-26/checkpoint_011585',
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

        # Load trained agent
        self.lower_level_actor = Algorithm.from_checkpoint(config['checkpoint_path'])

        if 'MADDPG' in config['checkpoint_path']:
            for env in self.environments.values():
                env.actions_are_logits = True

        self.low_level_observations = {}
        
        # Default agents in case we do not have a trained low-level agent
        self.base_agents = {
            "agent_ls": BaseLoadShiftingAgent(), 
            "agent_dc": BaseHVACAgent(), 
            "agent_bat": BaseBatteryAgent()
            }

        # Define observation and action space
        self.action_space = Dict({dc: Box(0, 1, [1]) for dc in self.environments})
        
    def reset(self):
        self.low_level_observations = {}
        self.heir_obs = {}
        infos = {}

        # Reset environments and store initial observations and infos
        for env_id, env in self.environments.items():
            obs, info = env.reset()
            self.low_level_observations[env_id] = obs
            self.heir_obs[env_id] = np.array(env.get_hierarchical_variables())
            infos[env_id] = info
        
        # Initialize metrics
        self.metrics = {
            env_id: {
                'bat_CO2_footprint': [],
                'bat_total_energy_with_battery_KWh': [],
                'ls_tasks_dropped': [],
                'dc_water_usage': [],
            } 
            for env_id in self.environments
        }   

        self.all_done = {env_id: False for env_id in self.environments}
        
        return self.heir_obs, infos
        
    def calc_reward(self):
        reward = 0
        for dc in self.rewards:
            reward += self.rewards[dc]['agent_bat']
        return reward

    def step(self, actions):

        # Update workload for all DCs
        for env_id, adj_workload in actions.items():
            if isinstance(adj_workload, np.ndarray):
                adj_workload = adj_workload[0]

            self.environments[env_id].set_hierarchical_workload(round(adj_workload, 6))

        # Compute actions for each agent in each environment
        low_level_actions = {env_id: {} for env_id in self.environments}
        
        for env_id, env_obs in self.low_level_observations.items():
            if self.all_done[env_id]:  # Skip if environment is done
                continue

            for agent_id, agent_obs in env_obs.items():
                policy_id = agent_id  # Customize policy ID if necessary
                action = self.lower_level_actor.compute_single_action(agent_obs, policy_id=policy_id)
                low_level_actions[env_id][agent_id] = action


        # Step through each environment with computed low_level_actions
        self.rewards = {}
        for env_id in self.environments:
            if self.all_done[env_id]:
                continue

            new_obs, rewards, terminated, truncated, info = self.environments[env_id].step(low_level_actions[env_id])
            self.low_level_observations[env_id] = new_obs
            self.all_done[env_id] = terminated['__all__'] or truncated['__all__']

            self.rewards[env_id] = rewards

            # Update metrics for each environment
            env_metrics = self.metrics[env_id]
            env_metrics['bat_CO2_footprint'].append(info['agent_bat']['bat_CO2_footprint'])
            env_metrics['bat_total_energy_with_battery_KWh'].append(info['agent_bat']['bat_total_energy_with_battery_KWh'])
            env_metrics['ls_tasks_dropped'].append(info['agent_ls']['ls_tasks_dropped'])
            env_metrics['dc_water_usage'].append(info['agent_dc']['dc_water_usage'])

        done = any(self.all_done.values())

        # Get observations for the next step
        if not done:
            self.heir_obs = {}
            for env_id, env in self.environments.items():
                self.heir_obs[env_id] = np.array(env.get_hierarchical_variables())

        return self.heir_obs, self.calc_reward(), False, done, {}

if __name__ == '__main__':
    env = HeirarchicalDCRL(DEFAULT_CONFIG)
    done = False
    obs, _ = env.reset()
    greedy_optimizer = WorkloadOptimizer(env.environments.keys())
    
    max_iterations = 4*24*30
    # Antonio: Keep in mind that each environment is set to have days_per_episode=30. You can modify this parameter to simulate the whole year
    with tqdm(total=max_iterations) as pbar:
        while not done:
    
            # Random actions
            # actions = {key: val[0] for key, val in env.action_space.sample().items()}

            # Do nothing 
            actions = {dc: state[1] for dc, state in obs.items()}

            # Rule-based
            # actions, _ = greedy_optimizer.compute_adjusted_workload(obs)

            obs, reward, terminated, truncated, info = env.step(actions)
            done = truncated

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