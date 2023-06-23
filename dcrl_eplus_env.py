import os
from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import ray

from dcrl_env import EnvConfig
from ray.rllib.env import EnvContext
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from utils import make_envs, reward_creator
from utils.base_agents import (BaseBatteryAgent, BaseHVACAgent,
                               BaseLoadShiftingAgent)
from utils.managers import CI_Manager, Time_Manager, Workload_Manager
from utils.utils_cf import get_init_day, obtain_paths

#import logging
import warnings

class DCRLeplus(MultiAgentEnv):
    #logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    def __init__(self, env_config: Union[dict, EnvContext] = {}):
        '''
        Args:
            env_config (dict): Dictionary containing parameters as defined in EnvConfig
        '''
        super().__init__()

        # Initialize the environment config
        env_config = EnvConfig(env_config)

        # create environments and agents
        self.agents = env_config['agents']
        self.location = env_config['location']
        
        self.ci_file = env_config['cintensity_file']
        self.weather_file = env_config['weather_file']
        self.workload_file = env_config['workload_file']

        self.indv_reward = env_config['individual_reward_weight']
        self.collab_reward = (1 - self.indv_reward) / 2
        
        self.flexible_load = env_config['flexible_load']
        
        # Assign month according to worker index, if available
        if hasattr(env_config, 'worker_index'):
            month = int((env_config.worker_index - 1) % 12)
        else:
            month = env_config.get('month', 0)
            
        self._agent_ids = set(self.agents)
        
        ci_loc, wea_loc = obtain_paths(self.location)
        
        ls_reward_method = 'default_ls_reward' if not 'ls_reward' in env_config.keys() else env_config['ls_reward']
        self.ls_reward_method = reward_creator.get_reward_method(ls_reward_method)

        dc_reward_method =  'default_dc_reward' if not 'dc_reward' in env_config.keys() else env_config['dc_reward']
        self.dc_reward_method = reward_creator.get_reward_method(dc_reward_method)
        
        bat_reward_method = 'default_bat_reward' if not 'bat_reward' in env_config.keys() else env_config['bat_reward']
        self.bat_reward_method = reward_creator.get_reward_method(bat_reward_method)
        
        self.ls_env = make_envs.make_ls_env(month)
        self.dc_env = make_envs.make_dc_env(month, self.location) 
        self.bat_env = make_envs.make_bat_fwd_env(month)

        self._obs_space_in_preferred_format = True
        
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})
        self.base_agents = {}
        flexible_load = 0
        
        # Create the observation/action space if the agent is used for training.
        # Otherwise, create the base agent for the environment.
        if "agent_ls" in self.agents:
            self.observation_space["agent_ls"] = self.ls_env.observation_space
            self.action_space["agent_ls"] = self.ls_env.action_space
            flexible_load = self.flexible_load
        else:
            self.base_agents["agent_ls"] = BaseLoadShiftingAgent()
            
        if "agent_dc" in self.agents:
            self.observation_space["agent_dc"] = self.dc_env.observation_space
            self.action_space["agent_dc"] = self.dc_env.action_space
        else:
            self.base_agents["agent_dc"] = BaseHVACAgent()
            
        if "agent_bat" in self.agents:
            self.observation_space["agent_bat"] = self.bat_env.observation_space
            self.action_space["agent_bat"] = self.bat_env.action_space
        else:
            self.base_agents["agent_bat"] = BaseBatteryAgent()
            
        # Create the managers: date/hour/time manager, workload manager, and CI manager.
        self.init_day = get_init_day(month)
        self.t_m = Time_Manager(init_day=self.init_day)
        self.workload_m = Workload_Manager(workload_filename=self.workload_file, flexible_workload_ratio=flexible_load, init_day=self.init_day)
        self.ci_m = CI_Manager(init_day=self.init_day, location=ci_loc, filename=self.ci_file)

        # This actions_are_logits is True only for MADDPG, because RLLib defines MADDPG only for continuous actions.
        self.actions_are_logits = env_config.get("actions_are_logits", False)


    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Environment options.

        Returns:
            states (dict): Dictionary of states.
            infos (dict): Dictionary of infos.
        """
        self.terminateds = set()
        self.truncateds = set()

        self.ls_terminated = False
        self.dc_terminated = False
        self.bat_terminated = False
        self.ls_truncated = False
        self.dc_truncated = False
        self.bat_truncated = False
        self.ls_reward = 0
        self.dc_reward = 0
        self.bat_reward = 0

        # Reset the managers
        t_i = self.t_m.reset()
        workload, day_workload = self.workload_m.reset()
        ci_i, ci_i_future = self.ci_m.reset()
        
        # Update the workload of the load shifting environment
        self.ls_env.update_workload(day_workload, workload)
        
        # Reset all the environments
        ls_s, self.ls_info = self.ls_env.reset()
        self.dc_state, self.dc_info = self.dc_env.reset()
        bat_s, self.bat_info = self.bat_env.reset()
        
        # Update the shared observation space
        batSoC = bat_s[1]

        # ls_state -> [time (sine/cosine enconded), original ls observation, current+future normalized CI, current workload, energy variables from DC, battery SoC]
        var_to_LS_energy = [self.dc_state[i] for i in [4, 5, 6, 8]]  # 4: External Temperature, 5: Internal Temperature, 6: Setpoint, 8: Total energy consumption
        self.ls_state = np.hstack((t_i, ls_s, ci_i_future, workload, var_to_LS_energy, batSoC))
        
        # bat_state -> [time (sine/cosine enconded), battery SoC, current+future normalized CI]
        self.bat_state = np.hstack((t_i, bat_s, ci_i_future))

        states = {}
        infos = {}
        # Update states and infos considering the agents defined in the environment config self.agents.
        if "agent_ls" in self.agents:
            states["agent_ls"] = self.ls_state
            infos["agent_ls"] = self.ls_info
            
        if "agent_dc" in self.agents:
            states["agent_dc"] = self.dc_state
            infos["agent_dc"] = self.dc_info
            
        if "agent_bat" in self.agents:
            states["agent_bat"] = self.bat_state
            infos["agent_bat"] = self.bat_info
            
        return states, infos

    def step(self, action_dict: MultiAgentDict):
        """
        Step the environment.

        Args:
            action_dict (MultiAgentDict): Dictionary of actions of each agent defined in self.agents.
  
        Returns:
            obs (dict): Dictionary of observations/states.
            rews (dict): Dictionary of rewards.
            terminated (dict): Dictionary of terminated flags.
            truncated (dict): Dictionary of truncated flags.
            infos (dict): Dictionary of infos.
        """
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        terminated["__all__"] = False
        truncated["__all__"] = False

        # Step in the managers
        day, hour, t_i, terminal = self.t_m.step()
        workload, day_workload = self.workload_m.step()
        ci_i, ci_i_future = self.ci_m.step()

        # Transform the actions if the algorithm uses continuous action space. Like RLLib with MADDPG.
        if self.actions_are_logits:
            for k, v in action_dict.items():
                if isinstance(v, np.ndarray):
                    action_dict[k] = np.random.choice(np.arange(len(v)), p=v)
        
        # Extract the action from the action dictionary.
        # If the agent is declared, use the action from the action dictionary.
        # If the agent is not declared, use the default action (do nothing) of the base agent.
        if "agent_ls" in self.agents:
            action = action_dict["agent_ls"]
        else:
            action = self.base_agents["agent_ls"].do_nothing_action()
        
        # Now, update the load shifting environment/agent first.
        self.ls_env.update_workload(day_workload, workload)
        
        # Do a step
        self.ls_state, _, self.ls_terminated, self.ls_truncated, self.ls_info = self.ls_env.step(action)

         # Now, the data center environment/agent.
        if "agent_dc" in self.agents:
            action = action_dict["agent_dc"]
        else:
            action = self.base_agents["agent_dc"].do_nothing_action()

        # Update the data center environment/agent.
        shifted_wkld = self.ls_info['ls_shifted_workload']
        self.dc_env.set_shifted_wklds(shifted_wkld)

        # Do a step in the data center environment
        # By default, the reward is ignored. The reward is calculated after the battery env step with the total energy usage.
        self.dc_state, _, self.dc_terminated, self.dc_truncated, self.dc_info = self.dc_env.step(action)
        
        # We will use the dc_terminated as flag because EnergyPlus has an internal flag to end the environment
        terminal = self.dc_terminated
        
        # Finally, the battery environment/agent.
        if "agent_bat" in self.agents:
            action = action_dict["agent_bat"]
        else:
            action = self.base_agents["agent_bat"].do_nothing_action()
            
        self.bat_env.set_dcload(self.dc_info['dc_ITE_total_power_kW']/1e3)  # The DC load is updated with the total power in MW.
        self.bat_state = self.bat_env.update_state()  # The state is updated with DC load
        self.bat_env.update_ci(ci_i, ci_i_future[0])  # Update the CI with the current CI, and the normalized current CI.
        
        # Do a step in the battery environment
        self.bat_state, _, self.bat_terminated, self.bat_truncated, self.bat_info = self.bat_env.step(action)
        
        # Update the state of the bat state
        batSoC = self.bat_state[1]
        self.bat_state = np.hstack((t_i, self.bat_state, ci_i_future))

        # Update the shared variables
        self.dc_state[-3:] = [shifted_wkld, ci_i_future[0], batSoC]

        # We need to update the LS state with the DC energy variables and the final battery SoC.
        var_to_LS_energy = [self.dc_state[i] for i in [4, 5, 6, 8]]
        self.ls_state = np.hstack((t_i, self.ls_state, ci_i_future, workload, var_to_LS_energy, batSoC))
        
        # params should be a dictionary with all of the info requiered plus other aditional information like the external temperature, the hour, the day of the year, etc.
        # Merge the self.bat_info, self.ls_info, self.dc_info in one dictionary called info_dict
        info_dict = {**self.bat_info, **self.ls_info, **self.dc_info}
        add_info = {"day": day, "hour": hour, "day_workload": day_workload, "norm_CI": ci_i_future[0]}
        reward_params = {**info_dict, **add_info}
        self.ls_reward, self.dc_reward, self.bat_reward = self.calculate_reward(reward_params)
        
        # If agent_ls is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_ls" in self.agents:
            obs['agent_ls'] = self.ls_state
            rew["agent_ls"] = self.indv_reward * self.ls_reward + self.collab_reward * self.bat_reward + self.collab_reward * self.dc_reward
            terminated["agent_ls"] = terminal
            info["agent_ls"] = self.ls_info

        # If agent_dc is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_dc" in self.agents:
            obs["agent_dc"] = self.dc_state 
            rew["agent_dc"] = self.indv_reward * self.dc_reward + self.collab_reward * self.ls_reward + self.collab_reward * self.bat_reward
            terminated["agent_dc"] = terminal
            info["agent_dc"] = self.dc_info

         # If agent_bat is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_bat" in self.agents:
            obs["agent_bat"] =  self.bat_state
            rew["agent_bat"] = self.indv_reward * self.bat_reward + self.collab_reward * self.dc_reward + self.collab_reward * self.ls_reward
            terminated["agent_bat"] = terminal
            info["agent_bat"] = self.bat_info

        if terminal:
            terminated["__all__"] = True
            truncated["__all__"] = True
            for agent in self.agents:
                truncated[agent] = True
                
        return obs, rew, terminated, truncated, info

    def calculate_reward(self, params):
        """
        Calculate the individual reward for each agent.

        Args:
            params (dict): Dictionary of parameters to calculate the reward.

        Returns:
            ls_reward (float): Individual reward for the load shifting agent.
            dc_reward (float): Individual reward for the data center agent.
            bat_reward (float): Individual reward for the battery agent.
        """
        
        ls_reward = self.ls_reward_method(params)
        dc_reward = self.dc_reward_method(params)
        bat_reward = self.bat_reward_method(params)
        return ls_reward, dc_reward, bat_reward
    
if __name__ == '__main__':

    env = DCRLeplus()

    # Set seeds for reproducibility    
    np.random.seed(0)
    env.ls_env.action_space.seed(0)
    env.dc_env.action_space.seed(0)
    env.bat_env.action_space.seed(0)

    done = False
    env.reset()
    step = 0
    reward_ls = reward_dc = reward_bat = 0
    
    while not done:
        action_dict = {
            'agent_ls': env.ls_env.action_space.sample(),
            'agent_dc': np.int64(env.dc_env.action_space.sample()),
            'agent_bat': env.bat_env.action_space.sample()
        }

        _, rewards, terminated, _, _ = env.step(action_dict)
        done = terminated['__all__']
        
        reward_ls += rewards['agent_ls']
        reward_dc += rewards['agent_dc']
        reward_bat += rewards['agent_bat']

        print(f"Step {step}, rewards = ", rewards)
        step += 1
    
    print("End of episode.")
    print("Load shifting agent reward = ", reward_ls)
    print("DC agent reward = ", reward_dc)
    print("Battery agent reward = ", reward_bat)