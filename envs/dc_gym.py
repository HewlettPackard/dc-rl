from typing import Optional, Tuple
import numpy as np
import pandas as pd

import gymnasium as gym

import envs.datacenter as DataCenter
# import utils.dc_config as DC_Config
import utils.dc_config_reader as DC_Config
from utils import reward_creator

class dc_gymenv(gym.Env):
    
    def __init__(self, observation_variables : list[str],
                       observation_space : gym.spaces.Box,
                       action_variables: list[str],
                       action_space : gym.spaces.Discrete,
                       action_mapping: dict,
                       ranges : dict[str,list],
                       weather_ts: pd.DataFrame,  # this data frame should be time indexed for the code to work
                       ci_ts : pd.DataFrame,   # this data frame should be time indexed for the code to work
                       add_cpu_usage : bool,
                       time_delta : pd.Timedelta,  # eg pd.Timedelta('15m')
                       min_temp : float,
                       max_temp : float,
                       action_definition : dict,
                       cpu_usage_ts : pd.DataFrame = None,   # this data frame should be time indexed for the code to work
                       seed : int = 123,
                       episode_length_in_time : pd.Timedelta = None,  # can be 1 week in minutes eg pd.Timedelta('7days')
                       reward_method : str = 'default_dc_reward'
                       ):
        """Creates the data center environment

        Args:
            observation_variables (list[str]): The partial list of variables that will be evaluated inside this evironment.The actual
                                                gym space may include other variables like sine cosine of hours, day of year, cpu usage,
                                                carbon intensity and battery state of charge.
            observation_space (gym.spaces.Box): The gym observations space following gymnasium standard
            action_variables (list[str]): The list of action variables for the environment. It is used to create the info dict returned by
                                        the environment
            action_space (gym.spaces.Discrete): The gym action space following gymnasium standard
            action_mapping (dict): A mapping from agent discrete action choice to actual delta change in setpoint. The mapping is defined in
                                    utils.make_pyeplus_env.py
            ranges (dict[str,list]): The upper and lower bounds on the observation_variables
            weather_ts (pd.DataFrame): The weather dataframe for the given 1 month/ 30 days period
            time_delta (pd.Timedelta): The sampling frequency of the dataset for the environment
            max_temp (float): The maximum temperature allowed for the CRAC setpoint
            min_temp (float): The minimum temperature allowed for the CRAC setpoint
            action_definition (dict): A mapping of the action name to the default or initialized value. Specified in utils.make_pyeplus_env.py
            cpu_usage_ts (pd.DataFrame, optional): The cpu usage dataframe for the given 1 month/ 30 days period. Defaults to None.
            episode_length_in_time (pd.Timedelta, optional): The maximum length after which the done flag should be True. Defaults to None. 
                                                            Setting none causes done to be True after data set is exausted.
            reward_method (str, optional) : Default or custom reward function to be used for evaluating the reward
        """
        
        self.observation_variables = observation_variables
        self.observation_space = observation_space
        self.action_variables = action_variables
        self.action_space = action_space
        self.action_mapping = action_mapping
        self.ranges = ranges
        self.weather_ts = weather_ts
        self.cpu_usage_ts = cpu_usage_ts
        self.ci_ts = ci_ts
        self.seed = seed
        self.add_cpu_usage = add_cpu_usage
        self.ambient_temp = 20
        self.scale_obs = False
        self.obs_max = []
        self.obs_min = []
        
        self.ts_start_idx, self.ts_end_idx = self.weather_ts.index[0], self.weather_ts.index[-1]
        self.ts_idx = self.ts_start_idx
        self.ts_end = False
        self.time_delta = time_delta
        self.max_episode_length_in_time = episode_length_in_time
        self.curr_episode_length_in_time = pd.Timedelta('0m')
        self.reward_method = reward_creator.get_reward_method(reward_method=reward_method)
        
        # similar to reset
        self.dc = DataCenter.DataCenter_ITModel(num_racks=DC_Config.NUM_RACKS,
                                                rack_supply_approach_temp_list=DC_Config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                                                rack_CPU_config=DC_Config.RACK_CPU_CONFIG,
                                                max_W_per_rack=DC_Config.MAX_W_PER_RACK,
                                                DC_ITModel_config=DC_Config)
        
        # TODO: Initialize these default values outside the gym environment
        self.CRAC_Fan_load, self.CT_Cooling_load, self.CRAC_cooling_load, self.Compressor_load = 100, 1000, 1000, 100
        self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, self.rackwise_outlet_temp = [], [], []
        self.cpu_load = 0.5
        self.bat_SoC = 300*1e3  # all units are SI
        
        self.raw_curr_state = None
        self.raw_next_state = None
        self.raw_curr_stpt = action_definition['cooling setpoints']['initial_value']
        self.max_temp = max_temp
        self.min_temp = min_temp
        
        super().__init__()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=self.seed)
        
        # TODO: Initialize these default values outside the gym environment
        self.CRAC_Fan_load, self.CT_Cooling_load, self.CRAC_cooling_load, self.Compressor_load = 100, 1000, 1000, 100  
        self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, self.rackwise_outlet_temp = [], [], []
        
        if self.ts_end:
            self.ts_idx = self.ts_start_idx
            self.ts_end = False
        else:
            self.ts_idx += self.time_delta
        
        self.raw_curr_state = self.get_obs()
        
        if self.scale_obs:
            return self.normalize(self.raw_curr_state), {}  # TODO : Ask AGP what variable he needs
        
    def step(self, action):
        
        crac_setpoint_delta = self.action_mapping[action]
        self.raw_curr_stpt += crac_setpoint_delta
        self.raw_curr_stpt = max(min(self.raw_curr_stpt,self.max_temp), self.min_temp)
    
        ITE_load_pct_list = [self.cpu_load*100 for i in range(DC_Config.NUM_RACKS)]  # TODO *100
        
        self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, self.rackwise_outlet_temp = \
            self.dc.compute_datacenter_IT_load_outlet_temp(ITE_load_pct_list=ITE_load_pct_list, CRAC_setpoint=self.raw_curr_stpt)
            
        avg_CRAC_return_temp = DataCenter.calculate_avg_CRAC_return_temp(rack_return_approach_temp_list=DC_Config.RACK_RETURN_APPROACH_TEMP_LIST,
                                                                     rackwise_outlet_temp=self.rackwise_outlet_temp)
        data_center_total_ITE_Load = sum(self.rackwise_cpu_pwr)+sum(self.rackwise_itfan_pwr)
        
        #ambient_temp=self.weather_ts.loc[self.ts_idx, 'dry bulb temperature']
        self.CRAC_Fan_load, self.CT_Cooling_load, self.CRAC_Cooling_load, self.Compressor_load = DataCenter.calculate_HVAC_power(CRAC_setpoint=self.raw_curr_stpt,
                                                                         avg_CRAC_return_temp=avg_CRAC_return_temp,
                                                                         ambient_temp=self.ambient_temp,
                                                                         data_center_full_load=data_center_total_ITE_Load,
                                                                         DC_Config=DC_Config)
        
        # calculate reward
        # self.reward = - 1.0 * ((data_center_total_ITE_Load + self.CT_Cooling_load)-40000)/(160000-40000)  # TODO: random agents shows -40000 to -160000
        self.reward = self.reward_method(
            params = {
                'data_center_total_ITE_Load' : data_center_total_ITE_Load,
                'CT_Cooling_load' : self.CT_Cooling_load,
                'energy_lb' : 40000,
                'energy_ub' : 160000
                }
        )
        
        # calculate done
        self.ts_end = (self.ts_idx + self.time_delta) == self.ts_end_idx  # epsidoe ends if data ends

        if self.max_episode_length_in_time is not None:
            episode_end = (self.curr_episode_length_in_time >= self.max_episode_length_in_time)
        else:
            episode_end = False
            
        if not episode_end:
           self.curr_episode_length_in_time += self.time_delta
           
        done = episode_end | self.ts_end
        
        # calculate truncated
        truncated = False
        
        # move to next time index
        self.ts_idx += self.time_delta
        
        # calculate self.raw_next_state
        self.raw_next_state = self.get_obs()
        # add info dictionary  # TODO: Ask AGP what infos he needs
        self.info = {
            'IT POWER w' : data_center_total_ITE_Load,
            'HVAC POWER w' : self.CT_Cooling_load,
            'Total Power kW' : (data_center_total_ITE_Load + self.CT_Cooling_load)/1000,
            'crac_setpoint_delta' : crac_setpoint_delta,
            'ambient_temp' : self.ambient_temp,
            'cpu_load' : self.cpu_load,
            'raw_action' : crac_setpoint_delta,
            'setpoint' : self.raw_curr_stpt,
            '%HVAC/IT' : self.CT_Cooling_load/data_center_total_ITE_Load
        }
        
        # return processed/unprocessed state to agent
        if self.scale_obs:
            return self.normalize(self.raw_next_state), self.reward, done, truncated, self.info

    def NormalizeObservation(self,):
        self.scale_obs = True
        for obs_var in self.observation_variables:
            self.obs_min.append(self.ranges[obs_var][0])
            self.obs_max.append(self.ranges[obs_var][1])
        
        self.obs_min = np.array(self.obs_min)
        self.obs_max = np.array(self.obs_max)
        self.obs_delta = self.obs_max - self.obs_min

    def normalize(self,obs):
        return (obs-self.obs_min)/self.obs_delta

    def get_obs(self):
        zone_air_therm_cooling_stpt = 20  # in C, default for reset state
        if self.raw_curr_stpt is not None:
            zone_air_therm_cooling_stpt = self.raw_curr_stpt
        
        zone_air_temp = 20  # in C, default for reset state
        if self.rackwise_outlet_temp:
            zone_air_temp = sum(self.rackwise_outlet_temp)/len(self.rackwise_outlet_temp)

        # 'Facility Total HVAC Electricity Demand Rate(Whole Building)'  ie 'HVAC POWER'
        hvac_power = self.CT_Cooling_load

        # 'Facility Total Building Electricity Demand Rate(Whole Building)' ie 'IT POWER'
        it_power = sum(self.rackwise_cpu_pwr) + sum(self.rackwise_itfan_pwr)

        return [self.ambient_temp, zone_air_therm_cooling_stpt,zone_air_temp,hvac_power,it_power]

    def set_shifted_wklds(self, cpu_load):
        self.cpu_load = cpu_load
    
    def set_ambient_temp(self, ambient_temp):
        self.ambient_temp = ambient_temp
        
    def set_bat_SoC(self, bat_SoC):
        self.bat_SoC = bat_SoC
        
class dc_gymenv_standalone(dc_gymenv):
    
    def __init__(self, env_config):
        
        # adjust based on month
        month = env_config.worker_index
        start_pd_datetime = pd.to_datetime(f'{month:02d}/01/2022 00:00:00')
        end_pd_datetime = start_pd_datetime + pd.Timedelta('30 days')
        weather_ts, ci_ts, cpu_usage_ts = env_config['weather_ts'].loc[start_pd_datetime:end_pd_datetime,:], \
                                        env_config['ci_ts'].loc[start_pd_datetime:end_pd_datetime,:], \
                                        env_config['cpu_usage_ts'].loc[start_pd_datetime:end_pd_datetime,:]
        
        super().__init__(
                       env_config['observation_variables'],
                       observation_space=env_config['observation_space'],
                       action_variables=env_config['action_variables'],
                       action_space=env_config['action_space'],
                       action_mapping=env_config['action_mapping'],
                       ranges=env_config['ranges'],
                       weather_ts=weather_ts,
                       ci_ts=ci_ts,
                       add_cpu_usage=env_config['add_cpu_usage'],
                       time_delta=env_config['time_delta'],
                       min_temp=env_config['min_temp'],
                       max_temp=env_config['max_temp'],
                       action_definition=env_config['action_definition'],
                       cpu_usage_ts= None if env_config['use_ls_cpu_load'] else cpu_usage_ts,
                       episode_length_in_time=env_config['episode_length_in_time']
        )
        
        self.NormalizeObservation()