import os
import sys
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

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for servers without display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # For reading images
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from collections import deque

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
        n_vars_ci = 8
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
        
        # Plots for the rendering
        # Load and scale icons for the visualization using Matplotlib
        self.datacenter_icon = mpimg.imread('/lustre/guillant/dc-rl/icons/data_center_icon2.png')
        self.temperature_icon = mpimg.imread('/lustre/guillant/dc-rl/icons/thermostat_icon.png')
        self.battery_icon = mpimg.imread('/lustre/guillant/dc-rl/icons/battery_icon.png')
        self.background_image = mpimg.imread('/lustre/guillant/dc-rl/icons/background_v2.png')

        # Resize images if necessary
        from PIL import Image
        self.datacenter_icon = self.resize_image(self.datacenter_icon, (1024, 1024))
        self.temperature_icon = self.resize_image(self.temperature_icon, (50, 50))
        self.battery_icon = self.resize_image(self.battery_icon, (50, 50))
        self.background_image = self.resize_image(self.background_image, (1600, 900))  # Adjust the size as needed

        self.BG_COLOR = (1.0, 1.0, 1.0)  # White background in normalized RGB
        self.BAR_COLOR = (105/255, 179/255, 162/255)  # Normalize RGB values to [0,1]
        self.FONT_COLOR = 'black'
        self.fontsize = 20  # Default font size
        
        self.carbon_intensity_history = deque(maxlen=96+1)  # Store the carbon intensity history for plotting
        self.external_temperature_history = deque(maxlen=96+1)  # Store the carbon intensity history for plotting


    def resize_image(self, image, size):
        from PIL import Image
        img = Image.fromarray((image * 255).astype('uint8'))  # Convert to PIL Image
        img = img.resize(size, Image.LANCZOS)
        return np.array(img) / 255.0  # Convert back to numpy array

    def seed(self, seed=None):
        """
        Set the random seed for the environment.

        Args:
            seed (int, optional): Random seed.
        """
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
        
        # dc state -> [time (sine/cosine enconded), original dc observation, current normalized CI]  # p.o.
        self.dc_state = np.float32(np.hstack((t_i, self.dc_state, ci_i_future[0])))  # p.o.
        
        # bat_state -> [time (sine/cosine enconded), battery SoC, current+future normalized CI]
        self.bat_state = np.float32(np.hstack((t_i, bat_s, ci_i_future)))

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
        
        return states, self.infos

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
        self.ls_state = np.float32(np.hstack((t_i, self.ls_state, ci_i_future)))  # For p.o.
        
        # Update the shared variables
        # dc state -> [time (sine/cosine enconded), original dc observation, current normalized CI]
        self.dc_state = np.float32(np.hstack((t_i, self.dc_state, ci_i_future[0])))  # For p.o.
        
        # Update the state of the bat state
        # bat_state -> [time (sine/cosine enconded), battery SoC, current+future normalized CI]
        self.bat_state = np.float32(np.hstack((t_i, self.bat_state, ci_i_future)))

        # Params should be a dictionary with all of the info required plus other additional information like the external temperature, the hour, the day of the year, etc.
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
        
        # Update self.infos with the agents information
        self.infos["agent_ls"] = info["agent_ls"]
        self.infos["agent_dc"] = info["agent_dc"]
        self.infos["agent_bat"] = info["agent_bat"]
        
        
        # Append values for the render method
        self.carbon_intensity_history.append(ci_i)
        self.external_temperature_history.append(temp)
        
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

    # def render(self, mode='human'):
    #     """
    #     Render the environment.
    #     """
    #     if mode == 'pygame':
    #         self.render_pygame()
        
    #     elif mode == 'plots':
    #         # Plot HVAC and other loads as a bar chart
    #         self.ax[0, 0].clear()
    #         self.ax[0, 0].bar(['Fan Load', 'Compressor Load'], 
    #                         [self.dc_env.CRAC_Fan_load, self.dc_env.Compressor_load])
    #         self.ax[0, 0].set_title("HVAC System Loads")

    #         # Plot CPU Power across racks as a line graph
    #         self.ax[0, 1].clear()
    #         self.ax[0, 1].plot(np.sum(self.dc_env.rackwise_cpu_pwr)/1e6, label="CPU Power (MW)")
    #         self.ax[0, 1].set_title("Rack-wise CPU Power")
    #         self.ax[0, 1].legend()

    #         # Plot Battery State of Charge and HVAC Load
    #         self.ax[1, 0].clear()
    #         self.ax[1, 0].plot(self.bat_env.info['bat_SOC']*100, label="Battery SoC (%)")
    #         self.ax[1, 0].plot(self.dc_env.HVAC_load/1e6, label="HVAC Power (MW)")
    #         self.ax[1, 0].set_title("Battery & HVAC Power")
    #         self.ax[1, 0].legend()

    #         # Update temperature plot
    #         self.ax[1, 1].clear()
    #         self.ax[1, 1].plot(np.sum(self.dc_env.rackwise_outlet_temp), label="Outlet Temp (C)")
    #         self.ax[1, 1].set_title("Rack-wise Outlet Temperature (C)")
    #         self.ax[1, 1].legend()

    #         # Redraw and pause briefly to simulate live updating
    #         plt.pause(0.05)
    #         plt.draw()

    #     elif mode == 'animation':
    #         print('Rendering the environment is not supported.')
    #         raise NotImplementedError
    #     else:
    #         raise NotImplementedError
        
        
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
    
    def get_soc_color(self, value):
        """
        Compute the color for the battery SoC bar, interpolating from red to green.
        
        Args:
            value (float): Battery SoC percentage (0 to 100).
        
        Returns:
            tuple: RGB color tuple.
        """
        # Normalize the value to [0, 1]
        normalized_value = value / 100.0
        red = 1 - normalized_value    # Red decreases as SoC increases
        green = normalized_value      # Green increases as SoC increases
        blue = 0                      # Blue remains constant
        return (red, green, blue)

    def draw_bar(self, ax, label, value, position, max_value=100, color=None):
        bar_width = 1.5  # Matplotlib uses relative widths for bars
        filled_width = (value / max_value)
        
        # Draw the empty bar to indicate the complete range
        ax.barh(position, 1, height=bar_width, color='lightgray', edgecolor='black', align='center')
        
        # Use the provided color or the default bar color
        if color is None:
            color = self.BAR_COLOR  # Default color
        
        # Draw the filled portion of the bar to indicate the value
        ax.barh(position, filled_width, height=bar_width, color=color, edgecolor='black', align='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')  # Hide axes
        
        # Add label and value
        ax.text(-0.1, position, f"{label}:", fontsize=14, va='center', ha='right', color=self.FONT_COLOR, weight='bold')
        ax.text(1.1, position, f"{value:.1f}%", fontsize=14, va='center', ha='left', color=self.FONT_COLOR, weight='bold')


    def render(self, mode='human'):
        """
        Render the environment using Matplotlib, incorporating logos.
        """
        # Prepare data for plotting
        agent_ls_info = self.infos.get('agent_ls', {})
        agent_bat_info = self.infos.get('agent_bat', {})
        agent_dc_info = self.infos.get('agent_dc', {})
        common_info = self.infos.get('__common__', {})

        # Extract necessary data
        original_workload = agent_ls_info.get('ls_original_workload', 0) * 100  # Convert to percentage
        shifted_workload = agent_ls_info.get('ls_shifted_workload', 0) * 100
        temp = common_info.get('weather', 0)
        bat_soc = agent_bat_info.get('bat_SOC', 0) * 100  # Convert to percentage
        cooling_setpoint = agent_dc_info.get('dc_crac_setpoint', 0)
        energy_consumption = agent_bat_info.get('dc_total_power_kW', 0)
        carbon_footprint = agent_bat_info.get('bat_CO2_footprint', 0)
        water_usage = agent_bat_info.get('dc_water_usage', 0)
        carbon_intensity = agent_bat_info.get('bat_avg_CI', 320)  # Example default value
        
        day = agent_ls_info.get('day', 0)
        hour = agent_ls_info.get('hour', 0)

        # Create a figure and axes
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        # Display the background image
        ax.imshow(self.background_image, extent=[0, 1, 0, 1], aspect='auto')

        # Draw workload bars with improved positioning
        ax_workload = fig.add_axes([0.40, 0.85, 0.2, 0.05])
        self.draw_bar(ax_workload, 'Original Workload', original_workload, 0)
        ax_computed_workload = fig.add_axes([0.40, 0.80, 0.2, 0.05])
        self.draw_bar(ax_computed_workload, 'Computed Workload', shifted_workload, 0)
        
        # Draw a bar for the battery SoC
        ax_battery = fig.add_axes([0.40, 0.17, 0.2, 0.05])

        # Compute the color based on the battery SoC
        color = self.get_soc_color(bat_soc)

        # Draw the bar with the computed color
        self.draw_bar(ax_battery, 'Battery SoC', bat_soc, 0, color=color)


        # Overlay the dynamic text at appropriate positions
        # Adjust the positions (x, y) based on your background layout
        
        ax.text(0.675, 0.641, f'External Temp (°C)', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        ax.text(0.66, 0.55, f'{temp:.1f}', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)

        ax.text(0.59, 0.415,  f'Cooling Setpoint (°C)', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        ax.text(0.582, 0.346, f'{cooling_setpoint:.1f}', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        
        ax.text(0.280, 0.637, f'Energy Grid Carbon Intensity', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        ax.text(0.295, 0.500, f'{carbon_intensity/1000:.1f} gCO2/Wh', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)

        # ax.text(0.35, 0.327, f'Battery SoC', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        # ax.text(0.35, 0.265, f'{bat_soc:.1f} (%)', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        
        # Prin the day and hour at the top right corner
        ax.text(0.95, 0.95, f'Day: {day}', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        ax.text(0.95, 0.90, f'Hour: {hour}', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)

        # Display final metrics at the bottom
        metrics_text = (
            f'Energy Consumption: {energy_consumption:.2f} MWh\n'
            f'Carbon Footprint: {carbon_footprint:.2f} KgCO2\n'
            f'Water Usage: {water_usage:.2f} L'
        )
        
        ax.text(0.176, 0.085, f'Energy Consumption:', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        ax.text(0.176, 0.055, f'{energy_consumption/1000:.2f} MWh', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        
        ax.text(0.5, 0.085, f'Carbon Footprint:', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        ax.text(0.5, 0.055, f'{carbon_footprint/1000:.2f} KgCO2', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        
        ax.text(0.824, 0.085, f'Water Usage:', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)
        ax.text(0.824, 0.055, f'{water_usage:.2f} L', fontsize=14, ha='center', va='center', weight='bold', color=self.FONT_COLOR)        

        
        # Add a small plot for carbon intensity history
        N = len(self.carbon_intensity_history)  # Total number of timesteps
        timestep_duration_minutes  = 15  # Duration of each timestep in minutes
        timestep_duration_hours = timestep_duration_minutes / 60  # Convert to hours

        # max_minutes_ago = (N - 1) * timestep_duration  # Total time span in minutes
        
        # Generate 5 equally spaced indices
        indices = np.linspace(0, N - 1, num=5).astype(int)

        # Ensure indices are within valid range
        indices = np.clip(indices, 0, N - 1)
        
        # Time steps in minutes ago (from oldest to most recent)
        time_steps_minutes_ago = [(N - i - 1) * timestep_duration_hours  for i in range(N)]  # e.g., [60, 45, 30, 15, 0]

        # Select time steps for x-ticks
        xticks_to_show = [time_steps_minutes_ago[i] for i in indices]

        xlabels = [str(int(t)) if t != 0 else 'Now' for t in xticks_to_show]

        # Create the plot
        ax_history = fig.add_axes([0.1, 0.20, 0.15, 0.125])  # Adjust position and size
        ax_history.plot(time_steps_minutes_ago, np.array(self.carbon_intensity_history)/1000, color='tab:blue')

        # Select x-ticks every 4 timesteps
        ax_history.set_xticks(xticks_to_show)
        ax_history.set_xticklabels(xlabels, fontsize=8)

        # Invert x-axis so 'Now' is at the right
        ax_history.invert_xaxis()

        # Set labels and title
        ax_history.set_title('Carbon Intensity History', fontsize=10)
        ax_history.set_xlabel('Hours Ago', fontsize=8)
        ax_history.set_ylabel('CI (gCO₂/Wh)', fontsize=8)
        ax_history.tick_params(axis='both', which='major', labelsize=8)
        ax_history.grid(True)
        ax_history.set_facecolor((0.95, 0.95, 0.95, 0.5))  # Semi-transparent background
        
        # Add a small plot for the external weather history
        # Create the plot
        ax_weather = fig.add_axes([0.75, 0.4, 0.15, 0.125])
        ax_weather.plot(time_steps_minutes_ago, self.external_temperature_history, color='tab:red')
        
        # Select x-ticks every 4 timesteps
        ax_weather.set_xticks(xticks_to_show)
        ax_weather.set_xticklabels(xlabels, fontsize=8)

        # Invert x-axis so 'Now' is at the right
        ax_weather.invert_xaxis()
        
        # Set labels and title
        ax_weather.set_title('External Temp History', fontsize=10)
        ax_weather.set_xlabel('Hours Ago', fontsize=8)
        ax_weather.set_ylabel('Temp (°C)', fontsize=8)
        ax_weather.tick_params(axis='both', which='major', labelsize=8)
        ax_weather.grid(True)
        ax_weather.set_facecolor((0.95, 0.95, 0.95, 0.5))
            
        # Save the figure to a file
        if not hasattr(self, 'render_step'):
            self.render_step = 0
        else:
            self.render_step += 1

        filename = f'evaluation_render/render_{self.render_step:04d}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

