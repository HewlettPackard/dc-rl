from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from ray.rllib.algorithms.algorithm import Algorithm

from dcrl_env import DCRL
from hierarchical_workload_optimizer import WorkloadOptimizer

config1 = {
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
    }

# NY config
config2 = {
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
    }

# WA config
config3 = {
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
    }

class HeirarchicalDCRL(gym.Env):

    def __init__(self, config={}):

        DC1 = DCRL(config1)
        DC2 = DCRL(config2)
        DC3 = DCRL(config3)

        # Define the environments
        self.environments = {
            'DC1': DC1,
            'DC2': DC2,
            'DC3': DC3,
        }

        checkpoint_path = 'results/test/MADDPGStable_DCRL_41ad0_00000_0_2024-04-11_20-08-26/checkpoint_011585'
        self.lower_level_actor = Algorithm.from_checkpoint(checkpoint_path)

        if 'MADDPG' in checkpoint_path:
            for env in self.environments.values():
                env.actions_are_logits = True

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
            } 
            for env_id in self.environments
        }   

        self.all_done = {env_id: False for env_id in self.environments}

        return heir_obs, infos
        
    def calc_reward(self,):

        return 

    def step(self, actions):

        # Update workload for all DCs
        for env_id, adj_workload in actions.items():
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
        for env_id in self.environments:
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

        done = any(self.all_done.values())

        # Get observations for the next step
        heir_obs = {}
        if not done:
            for env_id, env in self.environments.items():
                heir_obs[env_id] = env.get_hierarchical_variables()

        return heir_obs, done

if __name__ == '__main__':
    env = HeirarchicalDCRL()
    done = False
    obs, _ = env.reset()
    top_level_actor = WorkloadOptimizer(env.environments.keys())
    
    max_iterations = 4*24*365
    # Antonio: Keep in mind that each environment is set to have days_per_episode=30. You can modify this parameter to simulate the whole year
    with tqdm(total=max_iterations) as pbar:
        while not done:
            # breakpoint()
            # Random actions
            # actions = {key: val[0] for key, val in env.action_space.sample().items()}

            # Do nothing 
            # actions = {dc: state[1] for dc, state in obs.items()}

            # Rule-based 
            actions, _ = top_level_actor.compute_adjusted_workload(obs)
            obs, done = env.step(actions)

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