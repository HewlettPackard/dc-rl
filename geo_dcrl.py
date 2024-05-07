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
from hierarchical_workload_optimizer import WorkloadOptimizer
from dcrl_env_harl_partialobs import DCRL as DCRLPartObs

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'heterogeneous_dcrl')))  # default foldername if you git clone heterogeneous_dcrl
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
            'algo' : 'haa2c',
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
        
    def reset(self, seed=None, options=None):

        # Set seed if we are not in rllib
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self.low_level_observations = {}
        self.heir_obs = {}
        infos = {}

        # Reset environments and store initial observations and infos
        for env_id, env in self.datacenters.items():
            obs, info, available_actions = env.reset()
            # small change for now to convert list to dictionary
            obs = {'agent_ls': obs[0], 'agent_dc': obs[1], 'agent_bat': obs[2]}
            self.low_level_observations[env_id] = obs
            self.low_level_infos[env_id] = info
            
            self.heir_obs[env_id] = self.get_dc_variables(env_id)
            
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

        # Shift workloads between datacenters according to 
        # the actions provided by the agent.
        datacenter_ids = list(self.datacenters.keys())
        sender, receiver = datacenter_ids[actions['sender']], datacenter_ids[actions['receiver']]
        if sender != receiver:
            adjusted_workloads = self.compute_adjusted_workloads(actions)
            
            # Set reduced workload for the sender
            self.set_hierarchical_workload(sender, adjusted_workloads[sender])
            
            # Move the workload to the receiver
            self.move_hierarchical_workload(receiver, adjusted_workloads[receiver])

        # Compute actions for each dc_id in each environment
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
            for env_id in self.datacenters:
                self.heir_obs[env_id] = self.get_dc_variables(env_id)

        return self.heir_obs, self.calc_reward(), False, done, {}
    
    def set_hierarchical_workload(self, dc_id: str, workload: float):
        # self.datacenters[dc_id].workload_m.set_current_workload(workload)
        workload_m = self.datacenters[dc_id].workload_m
        workload_m.cpu_smooth[workload_m.time_step] = workload

    def move_hierarchical_workload(self, dc_id: str, workload: float):

        workload_m = self.datacenters[dc_id].workload_m
        remaining_workload = workload

        # Minimum number of timesteps to spread the workload over
        # it could extend beyond this incase where the capacity is not available
        N = 4

        # We move the workload starting from the next time step
        i = 1
        while remaining_workload > 0:
            workload_at_i = workload_m.cpu_smooth[workload_m.time_step + i]
            workload_to_move = min(1 - workload_at_i, workload / N, remaining_workload)
            workload_m.cpu_smooth[workload_m.time_step + i] += workload_to_move
            remaining_workload -= workload_to_move
            i += 1

            # Break if we are close to the end of the episode
            if workload_m.time_step + i >= len(workload_m.cpu_smooth):
                break
    
    def compute_adjusted_workloads(self, actions) -> dict:

        datacenters = list(self.datacenters.keys())
        sender = datacenters[actions['sender']] 
        receiver = datacenters[actions['receiver']]

        s_capacity, s_workload, *_ = self.heir_obs[sender]
        r_capacity, r_workload, *_ = self.heir_obs[receiver]

        # Convert percentage workload to mwh
        s_mwh = s_capacity * s_workload
        r_mwh = r_capacity * r_workload

        # Calculate the amount to move
        mwh_to_move = s_mwh * actions['workload_to_move'][0]
        s_mwh -= mwh_to_move
        r_mwh += mwh_to_move

        # Convert back to percentage workload
        s_workload = s_mwh / s_capacity
        r_workload = r_mwh / r_capacity

        self.set_hysterisis(mwh_to_move, sender, receiver)

        return {sender: s_workload, receiver: r_workload}
    
    def set_hysterisis(self, mwh_to_move: float, sender: str, receiver: str):
        PENALTY = 0.6
        
        cost_of_moving_mw = mwh_to_move * PENALTY

        self.datacenters[sender].dc_env.set_workload_hysterisis(cost_of_moving_mw)
        self.datacenters[receiver].dc_env.set_workload_hysterisis(cost_of_moving_mw)
    
    def calc_reward(self):
        reward = 0
        for dc in self.low_level_infos:
            reward += self.low_level_infos[dc]['agent_bat']['bat_CO2_footprint']
        return -1 * reward / 1e6
    
    def get_dc_variables(self, dc_id: str) -> np.ndarray:
        dc = self.datacenters[dc_id]

        obs = {
            'dc_capacity': dc.datacenter_capacity_mw,
            'current_workload': dc.workload_m.get_current_workload(),
            'weather': dc.weather_m.get_current_weather(),
            'total_power_kw': self.low_level_infos[dc_id]['agent_dc'].get('dc_total_power_kW', 0),
            'ci': dc.ci_m.get_current_ci(),
        }

        return np.array(list(obs.values()))


def main():
    """Main function."""
    env = HARL_HeirarchicalDCRL(DEFAULT_CONFIG)
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
            
            # Do nothing
            # actions = {'sender': 0, 'receiver': 0, 'workload_to_move': np.array([0.0])}

            # One-step greedy
            # ci = [obs[dc][-1] for dc in env.datacenters]
            # actions = {'sender': np.argmax(ci), 'receiver': np.argmin(ci), 'workload_to_move': np.array([1.])}
            
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


if __name__ == "__main__":
    main()