import os
import random
from typing import Optional, Tuple, Union

import torch
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from utils import reward_creator
from utils.base_agents import (BaseBatteryAgent, BaseHVACAgent,
                               BaseLoadShiftingAgent)
from utils.rbc_agents import RBCBatteryAgent
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


class SustainDC(gym.Env):
    def __init__(self, env_config):
        '''
        Initialize the SustainDC environment.

        Args:
            env_config (dict): Dictionary containing parameters as defined in 
                               EnvConfig above.
        '''
        super().__init__()

        # Initialize the environment config
        env_config = EnvConfig(env_config)

        # Create environments and agents
        self.agents = env_config['agents']
        self.rbc_agents = env_config.get('rbc_agents', [])
        
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
        
        n_vars_energy, n_vars_battery = 0, 0  # For partial observability (for p.o.)
        n_vars_ci = 2
        self.ls_env = make_ls_env(month=self.month, test_mode=self.evaluation_mode, n_vars_ci=n_vars_ci, 
                                  n_vars_energy=n_vars_energy, n_vars_battery=n_vars_battery, queue_max_len=1000)
        self.dc_env, _ = make_dc_pyeplus_env(month=self.month + 1, location=ci_loc, max_bat_cap_Mw=self.max_bat_cap_Mw, use_ls_cpu_load=True, 
                                             datacenter_capacity_mw=self.datacenter_capacity_mw, dc_config_file=self.dc_config_file, add_cpu_usage=False)
        self.bat_env = make_bat_fwd_env(month=self.month, max_bat_cap_Mwh=self.dc_env.ranges['max_battery_energy_Mwh'], 
                                        max_dc_pw_MW=self.dc_env.ranges['Facility Total Electricity Demand Rate(Whole Building)'][1] / 1e6, 
                                        dcload_max=self.dc_env.ranges['Facility Total Electricity Demand Rate(Whole Building)'][1],
                                        dcload_min=self.dc_env.ranges['Facility Total Electricity Demand Rate(Whole Building)'][0],
                                        n_fwd_steps=n_vars_ci)

        self.bat_env.dcload_max = self.dc_env.power_ub_kW / 4  # Assuming 15 minutes timestep. Kwh
        
        self.bat_env.dcload_min = self.dc_env.power_lb_kW / 4  # Assuming 15 minutes timestep. Kwh
        
        self._obs_space_in_preferred_format = True
        
        self.observation_space = []
        self.action_space = []
        
        # Do nothing controllers
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
        self.ranges_day = [max(0, self.init_day - 7), min(364, self.init_day + 7)]
        self.t_m = Time_Manager(self.init_day, timezone_shift=self.timezone_shift, days_per_episode=self.days_per_episode)
        self.workload_m = Workload_Manager(init_day=self.init_day, workload_filename=self.workload_file, timezone_shift=self.timezone_shift)
        self.weather_m = Weather_Manager(init_day=self.init_day, location=wea_loc, filename=self.weather_file, timezone_shift=self.timezone_shift)
        self.ci_m = CI_Manager(init_day=self.init_day, location=ci_loc, filename=self.ci_file, future_steps=n_vars_ci, timezone_shift=self.timezone_shift)

        # This actions_are_logits is True only for MADDPG if continuous actions is used on the algorithm.
        self.actions_are_logits = env_config.get("actions_are_logits", False)

    def seed(self, seed=None):
        """
        Set the random seed for the environment.

        Args:
            seed (int, optional): Random seed.
        """
        if seed is None:
            seed = 1
        
        # Set seed for numpy
        np.random.seed(seed)
        
        # Set seed for Python's random module
        random.seed(seed)
        
        # Set seed for torch (if using PyTorch)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Seed the environment
        if hasattr(self, 'action_space') and hasattr(self.action_space, 'seed'):
            self.action_space.seed(seed)
        if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(seed)
        
        
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
        random_init_day = random.randint(max(0, self.ranges_day[0]), min(364, self.ranges_day[1]))
        random_init_hour = random.randint(0, 23)
        
        t_i = self.t_m.reset(init_day=random_init_day, init_hour=random_init_hour)
        workload = self.workload_m.reset(init_day=random_init_day, init_hour=random_init_hour)  # Workload manager
        temp, norm_temp, wet_bulb, norm_wet_bulb = self.weather_m.reset(init_day=random_init_day, init_hour=random_init_hour)  # Weather manager
        ci_i, ci_i_future = self.ci_m.reset(init_day=random_init_day, init_hour=random_init_hour)  # CI manager. ci_i -> CI in the current timestep.
        
        # Set the external ambient temperature to data center environment
        self.dc_env.set_ambient_temp(temp, wet_bulb)
        
        # Update the workload of the load shifting environment
        self.ls_env.update_workload(workload)
        
        # Reset all the environments
        ls_s, self.ls_info = self.ls_env.reset()
        self.dc_state, self.dc_info = self.dc_env.reset()
        bat_s, self.bat_info = self.bat_env.reset()
        
        # ls_state -> [time (sine/cosine enconded), original ls observation, current+future normalized CI]
        self.ls_state = np.float32(np.hstack((t_i, ls_s, ci_i_future)))  # For p.o.


        # dc state -> [time (sine/cosine enconded), original dc observation, current normalized CI, next_workload]  # p.o.
        # self.dc_state = np.float32(np.hstack((t_i, self.dc_state, self.workload_m.get_next_workload())))  # For p.o.
        # next_workload = self.workload_m.get_next_workload()
        # is_high_workload = int(next_workload > 0.75)
        # is_low_workload = int(next_workload < 0.25)
        hour = random_init_hour
        # is_workload_increasing = int(next_workload > workload * 1.1)
        # is_workload_decreasing = int(next_workload < workload * 0.9)
        # is_peak_workload = int(next_workload > 0.9)
        # is_night_time =int( hour < 6 or hour > 18)
        # # Update the shared variables
        # # dc state -> [time (sine/cosine enconded), original dc observation, current normalized CI]
        # # self.dc_state = np.float32(np.hstack((t_i, self.dc_state, next_workload, is_high_workload)))  # For p.o.
        # self.dc_state = np.float32(np.hstack((t_i,
        #                                       self.dc_state,
        #                                       next_workload,
        #                                       is_high_workload,
        #                                       is_low_workload,
        #                                       is_workload_increasing,
        #                                       is_workload_decreasing,
        #                                       is_peak_workload,
        #                                       is_night_time
        #                                       )))
        
        next_workload = self.workload_m.get_next_workload()
        next_out_temperature = self.weather_m.get_next_temperature()
        next_wet_bulb = self.weather_m.get_next_wetbulb()

        is_high_workload = int(next_workload > 0.75)
        is_low_workload = int(next_workload < 0.25)

        is_workload_increasing = int(next_workload > workload * 1.1)
        is_workload_decreasing = int(next_workload < workload * 0.9)
        is_peak_workload = int(next_workload > 0.9)
        is_night_time = int(hour < 6 or hour > 20)
        is_midday = int(hour > 10 and hour < 18)
        # Update the shared variables
        # dc state -> [time (sine/cosine enconded), original dc observation, current normalized CI]
        # self.dc_state = np.float32(np.hstack((t_i, self.dc_state, next_workload, is_high_workload)))  # For p.o.
        # self.dc_state = np.float32(np.hstack((t_i,
        #                                       self.dc_state,
        #                                       next_workload,
        #                                       next_out_temperature,
        #                                       is_workload_increasing,
        #                                       is_workload_decreasing,
        #                                       is_night_time,
        #                                       is_midday
        #                                       )))
        
        self.ls_state = np.float32(np.hstack((next_workload,
                                              next_out_temperature)))  # For p.o.
                
        pump_speed = (self.dc_info.get('dc_coo_mov_flow_actual', 0.05) - self.dc_env.min_pump_speed) / (self.dc_env.max_pump_speed-self.dc_env.min_pump_speed)
        supply_temp = (self.dc_info.get('dc_supply_liquid_temp', 27) - self.dc_env.min_supply_temp) / (self.dc_env.max_supply_temp-self.dc_env.min_supply_temp)

        self.dc_state = np.float32(np.hstack((
                                              next_workload,
                                              next_out_temperature,
                                              pump_speed,
                                              supply_temp
                                              
                                              )))
        
        # bat_state -> [time (sine/cosine enconded), battery SoC, current+future normalized CI]
        # self.bat_state = np.float32(np.hstack((t_i, bat_s, ci_i_future)))
        self.bat_state = np.float32(np.hstack((next_workload,
                                              next_out_temperature)))

        # States should be a dictionary with agent names as keys and their observations as values
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
        
        return states

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
        self.bat_env.set_dcload(self.dc_info['dc_total_power_kW'] / 1e3)  # The DC load is updated with the total power in MW.
        self.bat_state = self.bat_env.update_state()  # The state is updated with DC load
        self.bat_env.update_ci(ci_i, ci_i_future[0])  # Update the CI with the current CI, and the normalized current CI.
        
        # Do a step in the battery environment
        self.bat_state, _, self.bat_terminated, self.bat_truncated, self.bat_info = self.bat_env.step(action)
        
        # ls_state -> [time (sine/cosine enconded), original ls observation, current+future normalized CI]
        # self.ls_state = np.float32(np.hstack((t_i, self.ls_state, ci_i_future)))  # For p.o.
                
        next_workload = self.workload_m.get_next_workload()
        next_out_temperature = self.weather_m.get_next_temperature()
        next_wet_bulb = self.weather_m.get_next_wetbulb()
        is_high_workload = int(next_workload > 0.75)
        is_low_workload = int(next_workload < 0.25)

        is_workload_increasing = int(next_workload > workload * 1.1)
        is_workload_decreasing = int(next_workload < workload * 0.9)
        is_peak_workload = int(next_workload > 0.9)
        is_night_time = int(hour < 6 or hour > 20)
        is_midday = int(hour > 10 and hour < 18)
        # Update the shared variables
        # dc state -> [time (sine/cosine enconded), original dc observation, current normalized CI]
        # self.dc_state = np.float32(np.hstack((t_i, self.dc_state, next_workload, is_high_workload)))  # For p.o.
        # self.dc_state = np.float32(np.hstack((t_i,
        #                                       self.dc_state,
        #                                       next_workload,
        #                                       next_out_temperature,
        #                                       is_workload_increasing,
        #                                       is_workload_decreasing,
        #                                       is_night_time,
        #                                       is_midday
        #                                       )))
        
        self.ls_state = np.float32(np.hstack((next_workload,
                                              next_out_temperature)))  # For p.o.
        
        pump_speed = np.round((self.dc_info.get('dc_coo_mov_flow_actual', 0.05) - self.dc_env.min_pump_speed) / (self.dc_env.max_pump_speed-self.dc_env.min_pump_speed), 6)
        supply_temp = np.round((self.dc_info.get('dc_supply_liquid_temp', 27) - self.dc_env.min_supply_temp) / (self.dc_env.max_supply_temp-self.dc_env.min_supply_temp), 6)


        # pump_speed = (np.round(self.dc_info['dc_coo_mov_flow_actual'], 6) - 0.05) / (0.5-0.05)
        # supply_temp = (np.round(self.dc_info['dc_supply_liquid_temp'], 6) - 15) / (45-15)
        self.dc_state = np.float32(np.hstack((
                                              next_workload,
                                              next_out_temperature,
                                              pump_speed,
                                              supply_temp
                                              
                                              )))
        # Update the state of the bat state
        # bat_state -> [time (sine/cosine enconded), battery SoC, current+future normalized CI]
        # self.bat_state = np.float32(np.hstack((t_i, self.bat_state, ci_i_future)))
        self.bat_state = np.float32(np.hstack((next_workload,
                                              next_out_temperature)))
        # Params should be a dictionary with all of the info required plus other additional information like the external temperature, the hour, the day of the year, etc.
        # Merge the self.bat_info, self.ls_info, self.dc_info in one dictionary called info_dict
        info_dict = {**self.bat_info, **self.ls_info, **self.dc_info}
        add_info = {"outside_temp": temp, "day": day, "hour": hour, "norm_CI": ci_i_future[0], "forecast_CI": ci_i_future}
        reward_params = {**info_dict, **add_info}
        self.ls_reward, self.dc_reward, self.bat_reward = self.calculate_reward(reward_params)
        
        # If agent_ls is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_ls" in self.agents:
            obs['agent_ls'] = self.ls_state
            rew["agent_ls"] = self.ls_reward  #self.indv_reward * self.ls_reward + self.collab_reward * self.bat_reward + self.collab_reward * self.dc_reward
            terminateds["agent_ls"] = False
            truncateds["agent_ls"] = False
        info["agent_ls"] = {**self.dc_info, **self.ls_info, **self.bat_info, **add_info}

        # If agent_dc is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_dc" in self.agents:
            obs["agent_dc"] = self.dc_state
            rew["agent_dc"] = self.dc_reward # self.indv_reward * self.dc_reward + self.collab_reward * self.ls_reward + self.collab_reward * self.bat_reward
            terminateds["agent_dc"] = False
            truncateds["agent_dc"] = False
        info["agent_dc"] = {**self.dc_info, **self.ls_info, **self.bat_info, **add_info}

         # If agent_bat is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_bat" in self.agents:
            obs["agent_bat"] = self.bat_state
            rew["agent_bat"] = self.bat_reward # self.indv_reward * self.bat_reward + self.collab_reward * self.dc_reward + self.collab_reward * self.ls_reward
            terminateds["agent_bat"] = False
            truncateds["agent_bat"] = False
        info["agent_bat"] = {**self.dc_info, **self.ls_info, **self.bat_info, **add_info}

        info["__common__"] = reward_params
        if terminal:
            terminateds["__all__"] = False
            truncateds["__all__"] = True
            for agent in self.agents:
                truncateds[agent] = True
            
            # Terminate the FMU from the data center environment
            # self.dc_env.reset_fmu()
        
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
        """
        Render the environment.
        """
        pass

    def close(self):
        """
        Close the environment.
        """
        self.env.close()  # pylint: disable=no-member
        
    def get_avail_actions(self):
        """
        Get the available actions for the agents.

        Returns:
            list: List of available actions for each agent.
        """
        if self.discrete:  # pylint: disable=no-member
            avail_actions = []
            for agent_id in range(self.n_agents):  # pylint: disable=no-member
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """
        Get the available actions for a specific agent.

        Args:
            agent_id (int): Agent ID.

        Returns:
            list: List of available actions for the agent.
        """
        return [1] * self.action_space[agent_id].n
    
    def state(self):
        """
        Get the state of the environment.

        Returns:
            np.ndarray: State of the environment.
        """
        states = tuple(
            self.scenario.observation(  # pylint: disable=no-member
                self.world.agents[self._index_map[agent]], self.world  # pylint: disable=no-member
            ).astype(np.float32)
            for agent in self.possible_agents  # pylint: disable=no-member
        )
        return np.concatenate(states, axis=None)