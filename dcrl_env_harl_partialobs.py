import os
from typing import Optional, Tuple, Union

import gymnasium
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from utils import reward_creator
from utils.base_agents import (BaseBatteryAgent, BaseHVACAgent,
                               BaseLoadShiftingAgent)
from utils.make_envs_pyenv import (make_bat_fwd_env, make_dc_pyeplus_env,
                                   make_ls_env)
from utils.managers import (CI_Manager, Time_Manager, Weather_Manager,
                            Workload_Manager)
from utils.utils_cf import get_energy_variables, get_init_day, obtain_paths


class EnvConfig(dict):

    # Default configuration for this environment. New parameters should be
    # added here
    DEFAULT_CONFIG = {
        # Agents active
        'agents': ['agent_ls', 'agent_dc', 'agent_bat'],

        # Datafiles
        'location': 'ny',
        'cintensity_file': 'NYIS_NG_&_avgCI.csv',
        'weather_file': 'USA_NY_New.York-Kennedy.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        
        # Capacity (MW) of the datacenter
        'datacenter_capacity_mw': 1,
        
        # Timezone shift
        'timezone_shift': 0,
        
        # Days per simulated episode
        'days_per_episode': 30,
        
        # Maximum battery capacity
        'max_bat_cap_Mw': 2,
        
        # Data center configuration file
        'dc_config_file': 'dc_config.json',
        
        # weight of the individual reward (1=full individual, 0=full collaborative, default=0.8)
        'individual_reward_weight': 0.8,
        
        # flexible load ratio of the total workload
        'flexible_load': 0.1,
        
        # Specify reward methods. These are defined in utils/reward_creator.
        'ls_reward': 'default_ls_reward',
        'dc_reward': 'default_dc_reward',
        'bat_reward': 'default_bat_reward',

        # Evaluation flag that is required by the load-shifting environment
        # To be set only during offline evaluation
        'evaluation': False,

        # Set this to True if an agent (like MADDPG) returns continuous actions,
        "actions_are_logits": False
    }

    def __init__(self, raw_config):
        dict.__init__(self, self.DEFAULT_CONFIG.copy())

        # Override defaults with the passed config
        for key, val in raw_config.items():
            self[key] = val


class DCRL(gym.Env):
    def __init__(self, env_config):
        '''
        Args:
            env_config (dict): Dictionary containing parameters as defined in 
                            EnvConfig above
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
        
        self.max_bat_cap_Mw = env_config['max_bat_cap_Mw']
        self.indv_reward = env_config['individual_reward_weight']
        self.collab_reward = (1 - self.indv_reward) / 2
        
        self.flexible_load = env_config['flexible_load']

        self.datacenter_capacity_mw = env_config['datacenter_capacity_mw']
        self.dc_config_file = env_config['dc_config_file']
        self.timezone_shift = env_config['timezone_shift']
        self.days_per_episode = env_config['days_per_episode']
        
        # Assign month according to worker index, if available
        if hasattr(env_config, 'worker_index'):
            self.month = int((env_config.worker_index - 1) % 12)
        else:
            self.month = env_config.get('month')

        self.evaluation_mode = env_config['evaluation']

        self._agent_ids = set(self.agents)

        ci_loc, wea_loc = obtain_paths(self.location)
        
        ls_reward_method = 'default_ls_reward' if not 'ls_reward' in env_config.keys() else env_config['ls_reward']
        self.ls_reward_method = reward_creator.get_reward_method(ls_reward_method)

        dc_reward_method =  'default_dc_reward' if not 'dc_reward' in env_config.keys() else env_config['dc_reward']
        self.dc_reward_method = reward_creator.get_reward_method(dc_reward_method)
        
        bat_reward_method = 'default_bat_reward' if not 'bat_reward' in env_config.keys() else env_config['bat_reward']
        self.bat_reward_method = reward_creator.get_reward_method(bat_reward_method)
        
        n_vars_energy, n_vars_battery = 0, 0  # for partial observability (for p.o.)
        n_vars_ci = 8
        self.ls_env = make_ls_env(month=self.month, test_mode=self.evaluation_mode, n_vars_ci=n_vars_ci, \
                                    n_vars_energy=n_vars_energy, n_vars_battery=n_vars_battery, queue_max_len=1000)
        self.dc_env, _ = make_dc_pyeplus_env(month=self.month+1, location=ci_loc, max_bat_cap_Mw=self.max_bat_cap_Mw, use_ls_cpu_load=True, \
                                            datacenter_capacity_mw=self.datacenter_capacity_mw, dc_config_file=self.dc_config_file, add_cpu_usage=False)  # for p.o.
        self.bat_env = make_bat_fwd_env(month=self.month, max_bat_cap_Mwh=self.dc_env.ranges['max_battery_energy_Mwh'], 
                                        max_dc_pw_MW=self.dc_env.ranges['Facility Total Electricity Demand Rate(Whole Building)'][1]/1e6, 
                                        dcload_max=self.dc_env.ranges['Facility Total Electricity Demand Rate(Whole Building)'][1],
                                        dcload_min=self.dc_env.ranges['Facility Total Electricity Demand Rate(Whole Building)'][0],
                                        n_fwd_steps=n_vars_ci)

        self.bat_env.dcload_max = self.dc_env.power_ub_kW / 4 # Assuming 15 minutes timestep. Kwh
        
        self.bat_env.dcload_min = self.dc_env.power_lb_kW / 4 # Assuming 15 minutes timestep. Kwh
        
        self._obs_space_in_preferred_format = True
        
        self.observation_space = []
        self.action_space = []
        
        self.base_agents = {}
        flexible_load = 0
        
        # Create the observation/action space if the agent is used for training.
        # Otherwise, create the base agent for the environment.
        if "agent_ls" in self.agents:
            self.observation_space.append(self.ls_env.observation_space)
            self.action_space.append(self.ls_env.action_space)
            flexible_load = self.flexible_load
        else:
            self.base_agents["agent_ls"] = BaseLoadShiftingAgent()
            
        if "agent_dc" in self.agents:
            self.observation_space.append(self.dc_env.observation_space)
            self.action_space.append(self.dc_env.action_space)
        else:
            self.base_agents["agent_dc"] = BaseHVACAgent()
            
        if "agent_bat" in self.agents:
            self.observation_space.append(self.bat_env.observation_space)
            self.action_space.append(self.bat_env.action_space)
        else:
            self.base_agents["agent_bat"] = BaseBatteryAgent()
            
        # ls_state[0:10]->10 variables
        # dc_state[4:9] & bat_state[5]->5+1 variables

        # Create the managers: date/hour/time manager, workload manager, weather manager, and CI manager.
        self.init_day = get_init_day(self.month)
        self.t_m = Time_Manager(self.init_day, timezone_shift=self.timezone_shift, days_per_episode=self.days_per_episode)
        self.workload_m = Workload_Manager(init_day=self.init_day, workload_filename=self.workload_file, timezone_shift=self.timezone_shift)
        self.weather_m = Weather_Manager(init_day=self.init_day, location=wea_loc, filename=self.weather_file, timezone_shift=self.timezone_shift)
        self.ci_m = CI_Manager(init_day=self.init_day, location=ci_loc, filename=self.ci_file, future_steps=n_vars_ci, timezone_shift=self.timezone_shift)

        # This actions_are_logits is True only for MADDPG, because RLLib defines MADDPG only for continuous actions.
        self.actions_are_logits = env_config.get("actions_are_logits", False)
                
    

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
    
    def reset(self):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Environment options.

        Returns:
            states (dict): Dictionary of states.
            infos (dict): Dictionary of infos.
        """
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
        t_i = self.t_m.reset() # Time manager
        workload = self.workload_m.reset() # Workload manager
        temp, norm_temp, wet_bulb, norm_wet_bulb = self.weather_m.reset() # Weather manager
        ci_i, ci_i_future = self.ci_m.reset() # CI manager. ci_i -> CI in the current timestep.
        
        # Set the external ambient temperature to data center environment
        self.dc_env.set_ambient_temp(temp, wet_bulb)
        
        # Update the workload of the load shifting environment
        self.ls_env.update_workload(workload)
        
        # Reset all the environments
        ls_s, self.ls_info = self.ls_env.reset()
        self.dc_state, self.dc_info = self.dc_env.reset()
        bat_s, self.bat_info = self.bat_env.reset()
        
        # ls_state -> [time (sine/cosine enconded), original ls observation, current+future normalized CI]
        self.ls_state = np.float32(np.hstack((t_i, ls_s, ci_i_future)))  # for p.o.
        
        # dc state -> [time (sine/cosine enconded), original dc observation, current normalized CI]  # p.o.
        self.dc_state = np.float32(np.hstack((t_i, self.dc_state, ci_i_future[0])))  # p.o.
        
        # bat_state -> [time (sine/cosine enconded), battery SoC, current+future normalized CI]
        self.bat_state = np.float32(np.hstack((t_i, bat_s, ci_i_future)))

        # states should be a dictionary with agent names as keys and their observations as values
        states = {}
        self.infos = {}
        # Update states and infos considering the agents defined in the environment config self.agents.
        if "agent_ls" in self.agents:
            states["agent_ls"] = self.ls_state
        self.infos["agent_ls"] = self.ls_info
        if "agent_dc" in self.agents:
            states["agent_dc"] = self.dc_state
        self.infos["agent_dc"] = self.dc_info
        if "agent_bat" in self.agents:
            states["agent_bat"] = self.bat_state
        self.infos["agent_bat"] = self.bat_info

        # Common information
        self.infos['__common__'] = {}
        self.infos['__common__']['time'] = t_i
        self.infos['__common__']['workload'] = workload
        self.infos['__common__']['weather'] = temp
        self.infos['__common__']['ci'] = ci_i
        self.infos['__common__']['ci_future'] = ci_i_future
        
        # Store the states
        self.infos['__common__']['states'] = {}
        self.infos['__common__']['states']['agent_ls'] = self.ls_state
        self.infos['__common__']['states']['agent_dc'] = self.dc_state
        self.infos['__common__']['states']['agent_bat'] = self.bat_state
        
        available_actions = None
                
        return states, self.infos, available_actions 

            

    def step(self, action_dict):
        """
        Step the environment.

        Args:
            action_dict: Dictionary of actions of each agent defined in self.agents.
  
        Returns:
            obs (dict): Dictionary of observations/states.
            rews (dict): Dictionary of rewards.
            terminated (dict): Dictionary of terminated flags.
            truncated (dict): Dictionary of truncated flags.
            infos (dict): Dictionary of infos.
        """
        obs, rew, terminateds, truncateds, info = {}, {}, {}, {}, {}
        terminateds["__all__"] = False
        truncateds["__all__"] = False
        
        # Step in the managers
        day, hour, t_i, terminal = self.t_m.step()
        workload = self.workload_m.step()
        temp, norm_temp, wet_bulb, norm_wet_bulb = self.weather_m.step()
        ci_i, ci_i_future = self.ci_m.step()
        
        # Extract the action from the action dictionary.
        # If the agent is declared, use the action from the action dictionary.
        # If the agent is not declared, use the default action (do nothing) of the base agent.
        if "agent_ls" in self.agents:
            action = action_dict["agent_ls"]
        else:
            action = self.base_agents["agent_ls"].do_nothing_action()
            
        # Now, update the load shifting environment/agent first.
        self.ls_env.update_workload(workload)
        
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
        self.dc_env.set_ambient_temp(temp, wet_bulb)
        
        # Do a step in the data center environment
        # By default, the reward is ignored. The reward is calculated after the battery env step with the total energy usage.
        # dc_state -> [self.ambient_temp, zone_air_therm_cooling_stpt, zone_air_temp, hvac_power, it_power]
        self.dc_state, _, self.dc_terminated, self.dc_truncated, self.dc_info = self.dc_env.step(action)

        # Finally, the battery environment/agent.
        if "agent_bat" in self.agents:
            action = action_dict["agent_bat"]
        else:
            action = self.base_agents["agent_bat"].do_nothing_action()
            
        # The battery environment/agent is updated.
        self.bat_env.set_dcload(self.dc_info['dc_total_power_kW'] / 1e3) # The DC load is updated with the total power in MW.
        self.bat_state = self.bat_env.update_state() # The state is updated with DC load
        self.bat_env.update_ci(ci_i, ci_i_future[0]) # Update the CI with the current CI, and the normalized current CI.
        
        # Do a step in the battery environment
        self.bat_state, _, self.bat_terminated, self.bat_truncated, self.bat_info = self.bat_env.step(action)
        
        # ls_state -> [time (sine/cosine enconded), original ls observation, current+future normalized CI]
        self.ls_state = np.float32(np.hstack((t_i, self.ls_state, ci_i_future)))  # for p.o.
        
        # Update the shared variables
        # dc state -> [time (sine/cosine enconded), original dc observation, current normalized CI]
        self.dc_state = np.float32(np.hstack((t_i, self.dc_state, ci_i_future[0])))  # for p.o.
        
        # Update the state of the bat state
        # bat_state -> [time (sine/cosine enconded), battery SoC, current+future normalized CI]
        self.bat_state = np.float32(np.hstack((t_i, self.bat_state, ci_i_future)))

        # params should be a dictionary with all of the info requiered plus other aditional information like the external temperature, the hour, the day of the year, etc.
        # Merge the self.bat_info, self.ls_info, self.dc_info in one dictionary called info_dict
        info_dict = {**self.bat_info, **self.ls_info, **self.dc_info}
        add_info = {"outside_temp": temp, "day": day, "hour": hour, "norm_CI": ci_i_future[0]}
        reward_params = {**info_dict, **add_info}
        self.ls_reward, self.dc_reward, self.bat_reward = self.calculate_reward(reward_params)
        
        # If agent_ls is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_ls" in self.agents:
            obs['agent_ls'] = self.ls_state
            rew["agent_ls"] = self.indv_reward * self.ls_reward + self.collab_reward * self.bat_reward + self.collab_reward * self.dc_reward
            terminateds["agent_ls"] = False
            truncateds["agent_ls"] = False
        info["agent_ls"] = {**self.dc_info, **self.ls_info, **self.bat_info, **add_info}

        # If agent_dc is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_dc" in self.agents:
            obs["agent_dc"] = self.dc_state
            rew["agent_dc"] = self.indv_reward * self.dc_reward + self.collab_reward * self.ls_reward + self.collab_reward * self.bat_reward
            terminateds["agent_dc"] = False
            truncateds["agent_dc"] = False
        info["agent_dc"] = {**self.dc_info, **self.ls_info, **self.bat_info, **add_info}

         # If agent_bat is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_bat" in self.agents:
            obs["agent_bat"] = self.bat_state
            rew["agent_bat"] = self.indv_reward * self.bat_reward + self.collab_reward * self.dc_reward + self.collab_reward * self.ls_reward
            terminateds["agent_bat"] = False
            truncateds["agent_bat"] = False
        info["agent_bat"] = {**self.dc_info, **self.ls_info, **self.bat_info, **add_info}

        info["__common__"] = reward_params
        if terminal:
            terminateds["__all__"] = False
            truncateds["__all__"] = True
            for agent in self.agents:
                truncateds[agent] = True
            
        # Common information
        self.infos['__common__'] = {}
        self.infos['__common__']['time'] = t_i
        self.infos['__common__']['workload'] = workload
        self.infos['__common__']['weather'] = temp
        self.infos['__common__']['ci'] = ci_i
        self.infos['__common__']['ci_future'] = ci_i_future
    
        # Store the states
        self.infos['__common__']['states'] = {}
        self.infos['__common__']['states']['agent_ls'] = self.ls_state
        self.infos['__common__']['states']['agent_dc'] = self.dc_state
        self.infos['__common__']['states']['agent_bat'] = self.bat_state

        return obs, rew, terminateds, truncateds, info

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
    
    def render(self):
        pass

    def close(self):
        self.env.close()  # pylint: disable=no-member
        
    def get_avail_actions(self):
        if self.discrete:  # pylint: disable=no-member
            avail_actions = []
            for agent_id in range(self.n_agents):  # pylint: disable=no-member
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n
    
    def state(self):
        states = tuple(
            self.scenario.observation(  # pylint: disable=no-member
                self.world.agents[self._index_map[agent]], self.world  # pylint: disable=no-member
            ).astype(np.float32)
            for agent in self.possible_agents  # pylint: disable=no-member
        )
        return np.concatenate(states, axis=None)
    
    def get_hierarchical_variables(self):
        return self.datacenter_capacity, self.workload_m.get_current_workload(), self.weather_m.get_current_weather(), self.ci_m.get_current_ci()  # pylint: disable=no-member
        
    def set_hierarchical_workload(self, workload):
        self.workload_m.set_current_workload(workload)
        
    def get_available_capacity(self, time_steps: int) -> float:
        """
        Calculate the available capacity of the datacenter over the next time_steps.

        Args:
            time_steps (int): Number of 15-minute time steps to consider.

        Returns:
            float: The available capacity in MW.
        """
        # Initialize the available capacity
        available_capacity = 0

        # Retrieve the current workload and the datacenter's total capacity
        current_step = self.workload_m.time_step
        max_capacity = self.datacenter_capacity_mw

        # Calculate the remaining capacity over each time step
        for step in range(1, time_steps + 1):
            # Make sure we're not exceeding the total steps available in workload data
            if current_step + step < len(self.workload_m.cpu_smooth):
                current_workload = self.workload_m.cpu_smooth[current_step + step]
                remaining_capacity = (1 - current_workload)*max_capacity
                available_capacity += max(0, remaining_capacity)
                # print(f"Step: {step}, Current Workload: {current_workload}, Remaining Capacity: {remaining_capacity}, Available Capacity: {available_capacity}")

        # Return the average capacity over the given time steps
        return available_capacity