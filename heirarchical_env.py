import os
import random
import warnings

import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete

from dcrl_env_harl_partialobs import DCRL
from hierarchical_workload_optimizer import WorkloadOptimizer
from low_level_wrapper import LowLevelActorRLLIB, LowLevelActorHARL

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
    )

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = {
    # NY config
    'config1' : {
        'location': 'tx',
        'cintensity_file': 'TX_NG_&_avgCI.csv',
        'weather_file': 'USA_AZ_Phoenix-Sky.Harbor.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc3.json',
        'datacenter_capacity_mw' : 1.0,
        'timezone_shift': 8,
        'month': 7,
        'days_per_episode': 30,
        'partial_obs': True,
        'nonoverlapping_shared_obs_space': True
        },

    # GA config
    'config2' : {
        'location': 'ga',
        'cintensity_file': 'GA_NG_&_avgCI.csv',
        'weather_file': 'USA_GA_Atlanta-Hartsfield-Jackson.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc2.json',
        'datacenter_capacity_mw' : 1.0,
        'timezone_shift': 0,
        'month': 7,
        'days_per_episode': 30,
        'partial_obs': True,
        'nonoverlapping_shared_obs_space': True
        },

    # WA config
    'config3' : {
        'location': 'va',
        'cintensity_file': 'VA_NG_&_avgCI.csv',
        'weather_file': 'UUSA_VA_Leesburg.Exec.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 1.0,
        'timezone_shift': 16,
        'month': 7,
        'days_per_episode': 30,
        'partial_obs': True,
        'nonoverlapping_shared_obs_space': True
        },
    
    # List of active low-level agents
    'active_agents': ['agent_dc'],

    # config for loading trained low-level agents
    'low_level_actor_config': {
        'harl': {
            'algo' : 'haa2c',
            'env' : 'dcrl',
            'exp_name' : 'll_actor',
            'model_dir': f'{CURR_DIR}/seed-00001-2024-05-01-21-50-12',
            },
        'rllib': {
            'checkpoint_path': f'{CURR_DIR}/maddpg/checkpoint_000000/',
            'is_maddpg': True
        }
    },
}

class HeirarchicalDCRL(gym.Env):

    def __init__(self, config):

        self.config = config

        # Init all datacenter environments
        DC1 = DCRL(config['config1'])
        DC2 = DCRL(config['config2'])
        DC3 = DCRL(config['config3'])

        self.datacenters = {
            'DC1': DC1,
            'DC2': DC2,
            'DC3': DC3,
        }

        self.datacenter_ids = list(self.datacenters.keys())
        
        # Load trained lower level agent
        self.lower_level_actor = LowLevelActorHARL(
            config['low_level_actor_config'],
            config['active_agents']
            )
        
        # self.lower_level_actor = LowLevelActorRLLIB(
        #     config['low_level_actor_config'], 
        #     config['active_agents']
        #     )
        
        self.low_level_observations = {}
        self.low_level_infos = {}
        
        self._max_episode_steps = 4*24*DEFAULT_CONFIG['config1']['days_per_episode']

        # Define observation and action space
        self.observation_space = Dict({dc: Box(-10, 10, [5]) for dc in self.datacenters})

        # Define the components of a single transfer action
        transfer_action = Dict({
            'sender': Discrete(3), 
            'receiver': Discrete(3),
            'workload_to_move': Box(low=0.0, high=1.0, shape=(1,), dtype=float)
        })

        # Define the action space for two transfers
        self.action_space = Dict({
            'transfer_1': transfer_action,
        })

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
            obs, info, _ = env.reset()
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
        
        self.overassigned_workload = self.safety_enforcement(actions)

        # Compute actions for each dc_id in each environment
        low_level_actions = {}
        
        # We need to update the low level observations with the new workloads for each datacenter.
        # So, for each DC in the environment, we need to update the workload on agent_ls and on agent_dc.
        # Now, we are hardcoding the positon of that variables in the arrays and modifiying them directly.
        # This is not ideal, but it is a quick fix for now.
        for datacenter_id in self.datacenters:
            curr_workload = self.datacenters[datacenter_id].workload_m.get_current_workload()
            # On agent_ls, the workload is the 5th element of the array (sine/cos hour day, workload, queue, etc)
            # On agent_dc, the workload is the 10th element of the array
            self.low_level_observations[datacenter_id]['agent_ls'][4] = curr_workload
            self.low_level_observations[datacenter_id]['agent_dc'][9] = curr_workload
        
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
            for env_id in self.datacenters:
                self.heir_obs[env_id] = self.get_dc_variables(env_id)

        return self.heir_obs, self.calc_reward(), False, done, {}

    def get_dc_variables(self, dc_id: str) -> np.ndarray:
        dc = self.datacenters[dc_id]

        # TODO: check if the variables are normalized with the same values or with min_max values
        obs = {
            'dc_capacity': dc.datacenter_capacity_mw,
            'curr_workload': dc.workload_m.get_current_workload(),
            'weather': dc.weather_m.get_current_weather(),
            'total_power_kw': self.low_level_infos[dc_id]['agent_dc'].get('dc_total_power_kW', 0),
            'ci': dc.ci_m.get_current_ci(),
        }

        return obs

    def workload_mapper(self, origin_dc, target_dc, action):
        """
        Translates the workload values from origin dc scale to target dc scale
        """
        assert (action >= 0) & (action <= 1), "action should be a positive fraction"
        return action * (origin_dc.datacenter_capacity_mw / target_dc.datacenter_capacity_mw)

    def safety_enforcement(self, actions: dict):
        
        # Sort dictionary by workload_to_move
        actions = dict(sorted(actions.items(), key = lambda x: x[1]['workload_to_move'], reverse=True))

        # base_workload_on_next_step for all dcs
        self.base_workload_on_next_step = {dc : self.datacenters[dc].workload_m.get_n_step_future_workload(n=1) for dc in self.datacenters}
        self.base_workload_on_curr_step = {dc : self.datacenters[dc].workload_m.get_n_step_future_workload(n=0) for dc in self.datacenters}
        # for _, base_workload in self.base_workload_on_next_step.items():
        #     assert (base_workload >= 0) & (base_workload <= 1), "base_workload next_step should be positive and a fraction"
        # for _, base_workload in self.base_workload_on_curr_step.items():
        #     assert (base_workload >= 0) & (base_workload <= 1), "base_workload curr_step should be positive and a fraction"
        
        overassigned_workload = []
        for _, action in actions.items():

            sender = self.datacenter_ids[action['sender']]
            receiver = self.datacenter_ids[action['receiver']]
            workload_to_move = action['workload_to_move'][0]

            sender_capacity = self.datacenters[sender].datacenter_capacity_mw
            receiver_capacity = self.datacenters[receiver].datacenter_capacity_mw

            # determine the effective workload to be moved and update 
            workload_to_move_mwh = workload_to_move * self.base_workload_on_curr_step[sender] * sender_capacity
            receiver_available_mwh = (1.0 - self.base_workload_on_next_step[receiver]) * receiver_capacity
            effective_movement_mwh = min(workload_to_move_mwh, receiver_available_mwh)
            
            self.base_workload_on_curr_step[sender] -= effective_movement_mwh / sender_capacity
            self.base_workload_on_next_step[receiver] += effective_movement_mwh / receiver_capacity

            # set hysterisis
            self.set_hysterisis(effective_movement_mwh, sender, receiver)
            
            # keep track of overassigned workload
            overassigned_workload.append(
                (
                    sender, 
                    receiver,
                    (workload_to_move_mwh - effective_movement_mwh) / receiver_capacity
                )
            )
        
        # update individual datacenters with the base_workload_on_curr_step
        for dc, base_workload in self.base_workload_on_curr_step.items():
            self.datacenters[dc].workload_m.set_n_step_future_workload(n = 0, workload = base_workload)

        # update individual datacenters with the base_workload_on_next_step
        for dc, base_workload in self.base_workload_on_next_step.items():
            self.datacenters[dc].workload_m.set_n_step_future_workload(n = 1, workload = base_workload)
        
        return overassigned_workload
    
    def set_hysterisis(self, mwh_to_move: float, sender: str, receiver: str):
        PENALTY = 0.6
        
        cost_of_moving_mw = mwh_to_move * PENALTY

        self.datacenters[sender].dc_env.set_workload_hysterisis(cost_of_moving_mw)
        self.datacenters[receiver].dc_env.set_workload_hysterisis(cost_of_moving_mw)

    def calc_reward(self) -> float:
        reward = 0
        for dc in self.low_level_infos:
            reward += self.low_level_infos[dc]['agent_bat']['bat_CO2_footprint']
        return -1 * reward / 1e6

class HeirarchicalDCRLWithHysterisisMultistep(HeirarchicalDCRL):

    def __init__(self, config):
        super().__init__(config)
        
        # Define the components of a single transfer action
        transfer_action = Dict({
            'sender': Discrete(3),  # sender
            'receiver': Discrete(3),  # receiver
            'workload_to_move': Box(low=0.0, high=1.0, shape=(1,), dtype=float)  # workload_to_move
        })

        # Define the action space for two transfers
        self.action_space = Dict({
            'transfer_1': transfer_action,
            'transfer_2': transfer_action
        })


    
if __name__ == '__main__':

    env = HeirarchicalDCRL(DEFAULT_CONFIG)
    
    done = False
    obs, _ = env.reset(seed=0)
    total_reward = 0
    
    greedy_optimizer = WorkloadOptimizer(list(env.datacenters.keys()))
    
    max_iterations = 4*24*30
    # Antonio: Keep in mind that each environment is set to have days_per_episode=30. You can modify this parameter to simulate the whole year
    with tqdm(total=max_iterations) as pbar:
        while not done:
    
            # Random actions
            actions = env.action_space.sample()
            
            # Do nothing
            """
            actions = {
                'transfer_1': {
                    'sender': 0,
                    'receiver': 0,
                    'workload_to_move': np.array([0.0])
                    }
                }
            """

            # One-step greedy
            """
            ci = [obs[dc][-1] for dc in env.datacenters]
            actions = {
                'transfer_1': {
                       'sender': np.argmax(ci), 
                       'receiver': np.argmin(ci), 
                       'workload_to_move': np.array([1.])
                       }
                }
            """

            # Multi-step Greedy
            # actions = greedy_optimizer.compute_actions(obs)
                
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