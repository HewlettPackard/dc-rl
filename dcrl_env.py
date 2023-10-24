import os
from typing import Optional, Tuple, Union
import csv

import gymnasium as gym
import numpy as np
import pandas as pd

from ray.rllib.env import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

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

        # Maximum battery capacity
        'max_bat_cap_Mw': 2,
        
        # weight of the individual reward (1=full individual, 0=full collaborative, default=0.8)
        'individual_reward_weight': 0.8,
        
        # flexible load ratio of the total workload
        'flexible_load': 0.5,
        
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

    def __init__(self, raw_config: Union[dict, EnvContext]):
        dict.__init__(self, self.DEFAULT_CONFIG.copy())

        # Override defaults with the passed config
        for key, val in raw_config.items():
            self[key] = val

        # Copy over RLLib's EnvContext parameters
        if isinstance(raw_config, EnvContext):
            self.worker_index = raw_config.worker_index
            self.num_workers = raw_config.num_workers
            self.recreated_worker = raw_config.recreated_worker
            self.vector_index = raw_config.vector_index


class DCRL(MultiAgentEnv):
    def __init__(self, env_config: Union[dict, EnvContext] = {}):
        '''
        Args:
            env_config (dict): Dictionary containing parameters as defined in 
                            EnvConfig above
        '''
        super().__init__()

        FUTURE_STEPS = 12*4
        
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

        # Assign month according to worker index, if available
        if hasattr(env_config, 'worker_index'):
            self.month = int((env_config.worker_index - 1) % 12)
        elif 'worker_index' in env_config.keys():
            self.month = int((env_config['worker_index'] - 1) % 12)
        else:
            self.month = env_config.get('month', 0)

        # self.month = 6
        
        self.evaluation_mode = env_config['evaluation']

        self._agent_ids = set(self.agents)

        ci_loc, wea_loc = obtain_paths(self.location)
        
        ls_reward_method = 'default_ls_reward' if not 'ls_reward' in env_config.keys() else env_config['ls_reward']
        self.ls_reward_method = reward_creator.get_reward_method(ls_reward_method)

        dc_reward_method =  'default_dc_reward' if not 'dc_reward' in env_config.keys() else env_config['dc_reward']
        self.dc_reward_method = reward_creator.get_reward_method(dc_reward_method)
        
        bat_reward_method = 'default_bat_reward' if not 'bat_reward' in env_config.keys() else env_config['bat_reward']
        self.bat_reward_method = reward_creator.get_reward_method(bat_reward_method)
        
        self.ls_env = make_ls_env(self.month, test_mode=self.evaluation_mode, future_steps=FUTURE_STEPS, n_vars_battery=0, flexible_workload_ratio=0.1)
        self.dc_env = make_dc_pyeplus_env(self.month+1, ci_loc, max_bat_cap_Mw=self.max_bat_cap_Mw, use_ls_cpu_load=True, future_steps=FUTURE_STEPS) 
        self.bat_env = make_bat_fwd_env(self.month, max_bat_cap_Mw=self.max_bat_cap_Mw, future_steps=FUTURE_STEPS)

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

        # Create the managers: date/hour/time manager, workload manager, weather manager, and CI manager.
        self.init_day = get_init_day(self.month)
        self.t_m = Time_Manager(self.init_day)
        self.workload_m = Workload_Manager(workload_filename=self.workload_file, flexible_workload_ratio=flexible_load, init_day=self.init_day)
        self.weather_m = Weather_Manager(init_day=self.init_day, location=wea_loc, filename=self.weather_file, future_steps=FUTURE_STEPS, evaluation_noise=False)
        self.ci_m = CI_Manager(init_day=self.init_day, location=ci_loc, filename=self.ci_file, future_steps=FUTURE_STEPS, evaluation_noise=False)

        # This actions_are_logits is True only for MADDPG, because RLLib defines MADDPG only for continuous actions.
        self.actions_are_logits = env_config.get("actions_are_logits", False)
        
        self.data_to_record = {
            'datetime': [],
            'energy_without_battery': [],
            'energy_with_battery': [],
            'carbon_footprint': [],
            'carbon_intensity': [],
            'energy_cost': [],
            'original_workload': [],
            'shifted_workload': [],
            'exterior_temperature': [],
            'interior_temperature': [],
            'hvac_setpoint': [],
            'battery_soc': [],
            'sum IT power': [],
            'sum fan power': []
        }
        
        self.tou = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.41, 7: 0.41,
                    8: 0.41, 9: 0.41, 10: 0.41, 11: 0.30, 12: 0.30, 13: 0.30, 14: 0.30, 
                    15: 0.30, 16: 0.27, 17: 0.27, 18: 0.27, 19: 0.27, 20: 0.27, 21: 0.27, 
                    22: 0.25, 23: 0.25
    }

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
        workload, day_workload = self.workload_m.reset() # Workload manager
        temp, future_norm_temperature = self.weather_m.reset() # Weather manager
        ci_i, ci_i_future = self.ci_m.reset() # CI manager. ci_i -> CI in the current timestep.
        
        # Set the external ambient temperature to data center environment
        self.dc_env.set_ambient_temp(temp)
        
        # Update the workload of the load shifting environment
        self.ls_env.update_workload(day_workload, workload)
        
        # Reset all the environments
        ls_s, self.ls_info = self.ls_env.reset()
        self.dc_state, self.dc_info = self.dc_env.reset()
        bat_s, self.bat_info = self.bat_env.reset()
         
        # Update the shared observation space
        batSoC = bat_s[1]
        
        # dc state -> [time (sine/cosine enconded), original dc observation, current workload, current normalized CI, battery SOC]
        self.dc_state = np.hstack((t_i, self.dc_state, workload, ci_i_future[0], future_norm_temperature))
        var_to_LS_energy = get_energy_variables(self.dc_state)
        
        # ls_state -> [time (sine/cosine enconded), original ls observation, current+future normalized CI, current workload, energy variables from DC, battery SoC]
        self.ls_state = np.hstack((t_i, ls_s, ci_i_future, workload, var_to_LS_energy))
        
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
        
        if len(self.data_to_record) > 0:
            # self.save_to_csv(f'ASHRAE2_data_{self.location}.csv')
            for key in self.data_to_record:
                self.data_to_record[key] = []  # Clear the lists after saving
                        
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
        temp, future_norm_temperature = self.weather_m.step()
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
        forced = False
        if hour > 23:
            if self.ls_info['ls_norm_load_left'] > 0:
                action = 1
                forced = True
        
        self.ls_state, _, self.ls_terminated, self.ls_truncated, self.ls_info = self.ls_env.step(action)
        
        self.ls_info['forced'] = forced

        # Now, the data center environment/agent.
        if "agent_dc" in self.agents:
            action = action_dict["agent_dc"]
        else:
            action = self.base_agents["agent_dc"].do_nothing_action()

        # Update the data center environment/agent.
        shifted_wkld = self.ls_info['ls_shifted_workload']
        
        # ToDo: Save on each time step, the different variables we want to plot.
        
        self.dc_env.set_shifted_wklds(shifted_wkld)
        self.dc_env.set_ambient_temp(temp)
        
        # Do a step in the data center environment
        # By default, the reward is ignored. The reward is calculated after the battery env step with the total energy usage.
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
        
        # Update the state of the bat state
        batSoC = self.bat_state[1]
        self.bat_state = np.hstack((t_i, self.bat_state, ci_i_future))
        
        # self.dc_reward = -1.0 * self.bat_info['bat_total_energy_with_battery_KWh'] / 1e3  # The raw reward of the DC is directly the total energy consumption in MWh.

        # Update the shared variables
        self.dc_state = np.hstack((t_i, self.dc_state, shifted_wkld, ci_i_future[0], future_norm_temperature))
        
        # We need to update the LS state with the DC energy variables and the final battery SoC.
        var_to_LS_energy = get_energy_variables(self.dc_state)
        self.ls_state = np.hstack((t_i, self.ls_state, ci_i_future, workload, var_to_LS_energy))

        # self.dc_state = np.random.normal(0, 2, size=len(self.ls_state))
        # params should be a dictionary with all of the info requiered plus other aditional information like the external temperature, the hour, the day of the year, etc.
        # Merge the self.bat_info, self.ls_info, self.dc_info in one dictionary called info_dict
        info_dict = {**self.bat_info, **self.ls_info, **self.dc_info}
        add_info = {"outside_temp": temp, "day": day, "hour": hour, "day_workload": day_workload, "norm_CI": ci_i_future[0], "forecast_CI": ci_i_future}
        reward_params = {**info_dict, **add_info}
        self.ls_reward, self.dc_reward, self.bat_reward = self.calculate_reward(reward_params)
        
        # If agent_ls is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_ls" in self.agents:
            obs['agent_ls'] = self.ls_state
            rew["agent_ls"] = self.indv_reward * self.ls_reward + self.collab_reward * self.bat_reward + self.collab_reward * self.dc_reward
            terminated["agent_ls"] = terminal
            info["agent_ls"] = self.ls_info

            # if len(self.ls_state) != self.
            
        # If agent_dc is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_dc" in self.agents:
            obs["agent_dc"] = self.dc_state
            rew["agent_dc"] = self.indv_reward * self.dc_reward + self.collab_reward * self.ls_reward + self.collab_reward * self.bat_reward
            terminated["agent_dc"] = terminal
            self.dc_info["bat_total_energy_with_battery_KWh"] = self.bat_info["bat_total_energy_with_battery_KWh"]
            self.dc_info["bat_CO2_footprint"] = self.bat_info["bat_CO2_footprint"]
            self.dc_info["ls_unasigned_day_load_left"] = self.ls_info["ls_unasigned_day_load_left"]
            info["agent_dc"] = self.dc_info
            
         # If agent_bat is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_bat" in self.agents:
            obs["agent_bat"] = self.bat_state
            rew["agent_bat"] = self.indv_reward * self.bat_reward + self.collab_reward * self.dc_reward + self.collab_reward * self.ls_reward
            terminated["agent_bat"] = terminal
            info["agent_bat"] = self.bat_info

        if terminal:
            terminated["__all__"] = True
            truncated["__all__"] = True
            for agent in self.agents:
                truncated[agent] = True
        
        
        # Save variables for dashboard:
        
        # Constructing datetime using pandas
        # start_of_year = pd.Timestamp('2023-01-01')
        # specific_date = start_of_year + pd.Timedelta(days=day - 1)  # Subtracting 1 because day starts from 1

        # Combining specific date with the hour to form the full datetime
        # dt = specific_date + pd.Timedelta(hours=hour)
        # datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')

        # self.data_to_record['datetime'].append(datetime_str) # Constructing datetime
        # self.data_to_record['energy_without_battery'].append(self.bat_info['bat_total_energy_without_battery_KWh'] / 1e3)
        # self.data_to_record['energy_with_battery'].append(self.bat_info['bat_total_energy_with_battery_KWh'] / 1e3)
        # self.data_to_record['carbon_footprint'].append(self.bat_info['bat_CO2_footprint'])
        # self.data_to_record['carbon_intensity'].append(self.bat_info['bat_avg_CI'])
        
        # tou_rate = self.tou[int((hour//1)%24)]
        # energy_cost = tou_rate * self.bat_info['bat_total_energy_with_battery_KWh']   # Adjust this if your energy key is different
        # self.data_to_record['energy_cost'].append(energy_cost)
        
        # self.data_to_record['original_workload'].append(self.ls_info['ls_original_workload'])
        # self.data_to_record['shifted_workload'].append(self.ls_info['ls_shifted_workload'])
        
        # self.data_to_record['exterior_temperature'].append(self.dc_info['dc_ext_temperature'])
        # self.data_to_record['interior_temperature'].append(self.dc_info['dc_int_temperature'])
        # self.data_to_record['hvac_setpoint'].append(self.dc_info['dc_crac_setpoint'])
        
        # self.data_to_record['battery_soc'].append(self.bat_info['bat_SOC'])

        # self.data_to_record['sum IT power'].append(self.dc_info['sum IT power'])
        # self.data_to_record['sum fan power'].append(self.dc_info['sum fan power'])

        # info['datetime'] = datetime_str
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
 
    def save_to_csv(self, filename):
        # Determine if we need to write headers
        write_header = not os.path.exists(filename)
        
        with open(filename, 'a', newline='') as csvfile:  # 'a' mode for appending
            writer = csv.writer(csvfile)
            
            # If file didn't exist before, write the headers
            if write_header:
                writer.writerow(['Datetime', 
                                 'Energy Without Battery',
                                 'Energy With Battery', 
                                 'Carbon Footprint', 
                                 'Carbon Intensity',
                                 'Energy Cost',
                                 'Original Workload',
                                 'Shifted Workload',
                                 'Exterior Temperature',
                                 'Interior Temperature',
                                 'HVAC Setpoint',
                                 'Battery SoC'])
                
            for i in range(len(self.data_to_record['datetime'])):
                writer.writerow([self.data_to_record['datetime'][i],
                                 self.data_to_record['energy_without_battery'][i],
                                 self.data_to_record['energy_with_battery'][i],
                                 self.data_to_record['carbon_footprint'][i],
                                 self.data_to_record['carbon_intensity'][i],
                                 self.data_to_record['energy_cost'][i],
                                 self.data_to_record['original_workload'][i],
                                 self.data_to_record['shifted_workload'][i],
                                 self.data_to_record['exterior_temperature'][i],
                                 self.data_to_record['interior_temperature'][i],
                                 self.data_to_record['hvac_setpoint'][i],
                                 self.data_to_record['battery_soc'][i]])

            # if write_header:
            #     writer.writerow(['Datetime', 
            #                      'Energy Without Battery',
            #                      'Energy With Battery', 
            #                      'Carbon Footprint', 
            #                      'Carbon Intensity',
            #                      'Energy Cost',
            #                      'Original Workload',
            #                      'Shifted Workload',
            #                      'Exterior Temperature',
            #                      'Interior Temperature',
            #                      'HVAC Setpoint',
            #                      'Battery SoC',
            #                      'sum IT power',
            #                      'sum fan power'])
                
            # for i in range(len(self.data_to_record['datetime'])):
            #     writer.writerow([self.data_to_record['datetime'][i],
            #                      self.data_to_record['energy_without_battery'][i],
            #                      self.data_to_record['energy_with_battery'][i],
            #                      self.data_to_record['carbon_footprint'][i],
            #                      self.data_to_record['carbon_intensity'][i],
            #                      self.data_to_record['energy_cost'][i],
            #                      self.data_to_record['original_workload'][i],
            #                      self.data_to_record['shifted_workload'][i],
            #                      self.data_to_record['exterior_temperature'][i],
            #                      self.data_to_record['interior_temperature'][i],
            #                      self.data_to_record['hvac_setpoint'][i],
            #                      self.data_to_record['battery_soc'][i],
            #                      self.data_to_record['sum IT power'][i],
            #                      self.data_to_record['sum fan power'][i]])
if __name__ == '__main__':

    env = DCRL()

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
            'agent_dc': env.dc_env.action_space.sample(),
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