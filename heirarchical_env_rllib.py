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
from heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
    )

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


class HeirarchicalDCRL_RLLib(HeirarchicalDCRL):

    def __init__(self, config):

        super().__init__(config)

        # DC1 = DCRL(config['config1'])

        # self.datacenters = {
        #     'DC1': DC1,
        # }
        self.observation_space = Tuple({dc: self.dc_observation_space for dc in self.datacenters})

    def reset(self, seed=None, options=None):
        
        # Set seed if we are not in rllib
        self.steps_remaining_at_level = 5
        self.cumulative_reward = 0
        seed = 0
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            # tf1.random.set_random_seed(0)

        self.low_level_observations = {}
        self.heir_obs = {}

        # Reset environments and store initial observations and infos
        for env_id, env in self.datacenters.items():
            obs, info, _ = env.reset()
            self.low_level_observations[env_id] = obs
            self.low_level_infos[env_id] = info
            
            self.heir_obs[env_id] = self.get_dc_variables(env_id)
        
        self.start_index_manager = env.workload_m.time_step
        self.simulated_days = env.days_per_episode
        self.total_computed_workload = 0

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
        #print("HEIRARCHICAL DCRL RLLIB RESET")
        obs = {}
        obs['high_level_agent'] = self.heir_obs
        self.all_done = {env_id: False for env_id in self.datacenters}
        #print(self.flatten_obs(self.heir_obs))
       
        #return self.flatten_obs(self.heir_obs), self.low_level_infos
        return obs, self.low_level_infos
    
    def flatten_obs(self, obs):
        f_obs = {}
        for values in obs.values():
            for key, value in values.items():
                if key not in f_obs:
                    f_obs[key] = np.array(value).flatten()
                else:
                    f_obs[key] += np.array(value).flatten()
        return f_obs

    def step(self, action_dict):
        #assert len(action_dict) == 1, action_dict
        print("ACTION DICT", action_dict)
        if "high_level_agent" in action_dict:
            return self._high_level_step(action_dict["high_level_agent"])
        else:
            return self._low_level_step(action_dict)

    def _high_level_step(self, action):
        print("HIGH LEVEL STEP")
        logger.debug("High level agent sets goal")

        self.steps_remaining_at_level = 5
        self.num_high_level_steps += 1
        self.cumulative_reward = 0
        self.overassigned_workload = self.safety_enforcement(action)
        obs, rew = {}, {}
        for env_id in self.datacenters:
            obs[env_id] = self.get_dc_variables(env_id)
            rew[env_id] = 0
        done = truncated = {"__all__": False}
        print("printobs", obs)
        return obs, rew, done, truncated, {}

    def _low_level_step(self, action):
        print("LOW LEVEL STEP")
        # Compute actions for each dc_id in each environment
        self.steps_remaining_at_level -= 1
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
        heir_obs = {}
        rew = self.calc_reward()
        
        for env_id in self.datacenters:
            heir_obs[env_id] = self.get_dc_variables(env_id)
        
        if self.steps_remaining_at_level == 0 or done:
            red = {}
            done = {"__all__": True}
            truncated = {"__all__": False}
            rew["high_level_agent"] = self.cumulative_reward
            heir_obs["high_level_agent"] = self.flatten_obs(heir_obs)
        return heir_obs, rew, False, done, {}


    def calc_reward(self) -> float:
        reward = {}
        for dc, id in zip(self.low_level_infos, self.datacenters):
            reward[id] = -1 * self.low_level_infos[dc]['agent_bat']['bat_CO2_footprint'] / 1e6
            self.cumulative_reward += reward[id]
        return reward

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

    env = HeirarchicalDCRL_RLLib(DEFAULT_CONFIG)
    
    done = False
    obs, info = env.reset(seed=0)
    total_reward = 0

    greedy_optimizer = WorkloadOptimizer(list(env.datacenters.keys()))
    
    agent = 2
    max_iterations = 4*24*30
    
    with tqdm(total=max_iterations) as pbar:
        while not done:
            
            # Random actions
            if agent == 0:
                actions = env.action_space.sample()
            
            # Do nothing
            elif agent == 1:
                actions = {
                    'transfer_1': {
                        'sender': 0,
                        'receiver': 0,
                        'workload_to_move': np.array([0.0])
                        }
                    }

            # One-step greedy
            elif agent == 2:
                print("OBS", obs)
                if "high_level_agent" in obs:
                    print("HIGH LEVEL AGENT")
                    f_obs = obs['high_level_agent']
                    ci = f_obs['ci']
                else:
                    ci = [obs[dc]['ci'] for dc in env.datacenters]
                actions = {
                    'transfer_1': {
                        'sender': np.argmax(ci),
                        'receiver': np.argmin(ci),
                        'workload_to_move': np.array([1.])
                        }
                    }

            # Multi-step Greedy
            else:
                actions = greedy_optimizer.compute_actions(obs)
            
            obs, reward, terminated, truncated, info = env.step(actions)
            done = truncated
            #total_reward += reward

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
    print("Total computed workload = ", env.total_computed_workload)