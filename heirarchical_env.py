import os
import random
import warnings
    
import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete, Tuple

from dcrl_env import DCRL
from hierarchical_workload_optimizer import WorkloadOptimizer
from low_level_wrapper import LowLevelActorRLLIB, LowLevelActorHARL

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
    )

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = {
    # AZ config
    'config1' : {
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
        'location': 'wa',
        'cintensity_file': 'BPAT_NG_&_avgCI.csv',
        'weather_file': 'USA_WA_Port.Angeles-Fairchild.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 0.9,
        'timezone_shift': 16,
        'month': 0,
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
            'model_dir': f'{CURR_DIR}/seed-00001-2024-04-22-20-59-21/models',
            },
        'rllib': {
            'checkpoint_path': f'{CURR_DIR}/maddpg/checkpoint_000000/',
            'is_maddpg': True
        }
    },
}

class HeirarchicalDCRL(gym.Env):

    def __init__(self, config):

        # Init all datacenter environments
        DC1 = DCRL(config['config1'])
        DC2 = DCRL(config['config2'])
        DC3 = DCRL(config['config3'])

        self.datacenters = {
            'DC1': DC1,
            'DC2': DC2,
            'DC3': DC3,
        }

        # Load trained lower level agent
        # self.lower_level_actor = LowLevelActorHARL(
        #     config['low_level_actor_config'],
        #     config['active_agents']
        #     )
        
        self.lower_level_actor = LowLevelActorRLLIB(
            config['low_level_actor_config'], 
            config['active_agents']
            )
        
        self.low_level_observations = {}

        # Define observation and action space
        self.observation_space = Dict({dc: Box(0, 10000, [4]) for dc in self.datacenters})
        self.action_space = MultiDiscrete([3, 3])

    def reset(self, seed=None, options=None):
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # tf1.random.set_random_seed(0)

        self.low_level_observations = {}
        self.heir_obs = {}
        infos = {}

        # Reset environments and store initial observations and infos
        for env_id, env in self.datacenters.items():
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
            for env_id in self.datacenters
        }   

        self.all_done = {env_id: False for env_id in self.datacenters}
        
        return self.heir_obs, infos
    
    def step(self, actions):

        # Shift workloads between datacenters according to 
        # the actions provided by the agent. This will return a dict with 
        # recommend workloads for all DCs
        if not isinstance(actions, dict):
            actions = self.compute_adjusted_workloads(actions)

        # Set workload for all DCs accordingly
        for env_id, adj_workload in actions.items():
            if isinstance(adj_workload, np.ndarray):
                adj_workload = adj_workload[0]

            self.datacenters[env_id].set_hierarchical_workload(round(adj_workload, 6))

        # Compute actions for each agent in each environment
        low_level_actions = {}
        
        for env_id, env_obs in self.low_level_observations.items():
            # Skip if environment is done
            if self.all_done[env_id]:
                continue

            low_level_actions[env_id] = self.lower_level_actor.compute_actions(env_obs)

        # Step through each environment with computed low_level_actions
        self.low_level_infos = {}
        for env_id in self.datacenters:
            if self.all_done[env_id]:
                continue

            new_obs, rewards, terminated, truncated, info = self.datacenters[env_id].step(low_level_actions[env_id])
            self.low_level_observations[env_id] = new_obs
            self.all_done[env_id] = terminated['__all__'] or truncated['__all__']

            self.low_level_infos[env_id] = info

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
            for env_id, env in self.datacenters.items():
                self.heir_obs[env_id] = np.array(env.get_hierarchical_variables())

        return self.heir_obs, self.calc_reward(), False, done, {}

    def compute_adjusted_workloads(self, actions) -> Dict:
        # Translate the recommended workload transfer to actual workload.
        # This will return a dict with the new workload for the sender and the receiver

        datacenters = list(self.datacenters.keys())
        sender, receiver = [datacenters[i] for i in actions]

        s_capacity, s_workload, *_ = self.heir_obs[sender]
        r_capacity, r_workload, *_ = self.heir_obs[receiver]

        if sender == receiver:
            return {sender: s_workload, receiver: r_workload}
        
        # Convert percentage workload to mwh
        s_mwh = s_capacity * s_workload
        r_mwh = r_capacity * r_workload

        # Calculate the amount to move
        mwh_to_move = min(s_mwh, r_capacity - r_mwh)
        s_mwh -= mwh_to_move
        r_mwh += mwh_to_move

        # Convert back to percentage workload
        s_workload  = s_mwh / s_capacity
        r_workload = r_mwh / r_capacity

        return {sender: s_workload, receiver: r_workload}

    def calc_reward(self):
        reward = 0
        for dc in self.low_level_infos:
            reward += self.low_level_infos[dc]['agent_bat']['bat_CO2_footprint']
        return -1 * reward / 1e6


class HeirarchicalDCRLWithHysterisis(HeirarchicalDCRL):

    def __init__(self, config):
        super().__init__(config)
        self.action_space = Tuple([MultiDiscrete([3, 3]), Box(0., 1., [1])])

    def compute_adjusted_workloads(self, actions) -> dict:
        if isinstance(actions, tuple):
            actions = np.concatenate(actions)
            
        datacenters = list(self.datacenters.keys())
        sender, receiver = [datacenters[int(i)] for i in actions[:2]]

        s_capacity, s_workload, *_ = self.heir_obs[sender]
        r_capacity, r_workload, *_ = self.heir_obs[receiver]

        if sender == receiver:
            return {sender: s_workload, receiver: s_workload}
        
        # Convert percentage workload to mwh
        s_mwh = s_capacity * s_workload
        r_mwh = r_capacity * r_workload

        # Calculate the amount to move
        max_mwh_to_move = s_mwh*actions[2] if len(actions) == 3 else s_mwh
        mwh_to_move = min(max_mwh_to_move, r_capacity - r_mwh)
        s_mwh -= mwh_to_move
        r_mwh += mwh_to_move

        # Convert back to percentage workload
        s_workload = s_mwh / s_capacity
        r_workload = r_mwh / r_capacity

        self.set_hysterisis(mwh_to_move, sender, receiver)

        return {sender: s_workload, receiver: r_workload}

    def set_hysterisis(self, mwh_to_move: float, sender: str, receiver: str):
        PENALTY = 0.2
        
        cost_of_moving_mw = mwh_to_move * PENALTY

        self.datacenters[sender].dc_env.set_workload_hysterisis(cost_of_moving_mw)
        self.datacenters[receiver].dc_env.set_workload_hysterisis(cost_of_moving_mw)

    def calc_reward(self):
        return super().calc_reward()


if __name__ == '__main__':

    env = HeirarchicalDCRL(DEFAULT_CONFIG)
    # env = HeirarchicalDCRLWithHysterisis(DEFAULT_CONFIG)
    
    done = False
    obs, _ = env.reset(seed=0)
    total_reward = 0
    
    greedy_optimizer = WorkloadOptimizer(env.datacenters.keys())
    
    max_iterations = 4*24*30
    # Antonio: Keep in mind that each environment is set to have days_per_episode=30. You can modify this parameter to simulate the whole year
    with tqdm(total=max_iterations) as pbar:
        while not done:
    
            # Random actions
            actions = env.action_space.sample()

            # Do nothing
            # actions = {dc: state[1] for dc, state in obs.items()}

            # One-step greedy
            # ci = [obs[dc][-1] for dc in env.datacenters]
            # actions = np.array([np.argmax(ci), np.argmin(ci)])
            
            # Multi-step Greedy
            actions, _ = greedy_optimizer.compute_adjusted_workload(obs)

            obs, reward, terminated, truncated, info = env.step(actions)
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