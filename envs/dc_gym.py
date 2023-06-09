from typing import Optional, Tuple
import numpy as np
import pandas as pd

import gymnasium as gym

import envs.datacenter as DataCenter
import utils.dc_config_reader as DC_Config
from utils import reward_creator

class dc_gymenv(gym.Env):
    
    def __init__(self, observation_variables : list[str],
                       observation_space : gym.spaces.Box,
                       action_variables: list[str],
                       action_space : gym.spaces.Discrete,
                       action_mapping: dict,
                       ranges : dict[str,list],  # this data frame should be time indexed for the code to work
                       add_cpu_usage : bool,
                       min_temp : float,
                       max_temp : float,
                       action_definition : dict,
                       seed : int = 123,
                       episode_length_in_time : pd.Timedelta = None,  # can be 1 week in minutes eg pd.Timedelta('7days')
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
            max_temp (float): The maximum temperature allowed for the CRAC setpoint
            min_temp (float): The minimum temperature allowed for the CRAC setpoint
            action_definition (dict): A mapping of the action name to the default or initialized value. Specified in utils.make_pyeplus_env.py
            episode_length_in_time (pd.Timedelta, optional): The maximum length after which the done flag should be True. Defaults to None. 
                                                            Setting none causes done to be True after data set is exausted.
        """
        
        self.observation_variables = observation_variables
        self.observation_space = observation_space
        self.action_variables = action_variables
        self.action_space = action_space
        self.action_mapping = action_mapping
        self.ranges = ranges
        self.seed = seed
        self.add_cpu_usage = add_cpu_usage
        self.ambient_temp = 20
        self.scale_obs = False
        self.obs_max = []
        self.obs_min = []
                
        # similar to reset
        self.dc = DataCenter.DataCenter_ITModel(num_racks=DC_Config.NUM_RACKS,
                                                rack_supply_approach_temp_list=DC_Config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                                                rack_CPU_config=DC_Config.RACK_CPU_CONFIG,
                                                max_W_per_rack=DC_Config.MAX_W_PER_RACK,
                                                DC_ITModel_config=DC_Config)
        
        self.CRAC_Fan_load, self.CT_Cooling_load, self.CRAC_cooling_load, self.Compressor_load = 100, 1000, 1000, 100
        self.CW_pump_load = 30
        self.CT_pump_load = 30
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

        """
        Reset `dc_gymenv` to initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Environment options.

        Returns:
            raw_curr_state (List[float]): Current state of the environmment
            {} (dict): A dictionary that containing additional information about the environment state
        """

        super().reset(seed=self.seed)

        self.CRAC_Fan_load, self.CT_Cooling_load, self.CRAC_cooling_load, self.Compressor_load = 100, 1000, 1000, 100  
        self.CW_pump_load = 30
        self.CT_pump_load = 30
        self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, self.rackwise_outlet_temp = [], [], []
        
        self.raw_curr_state = self.get_obs()
        
        if self.scale_obs:
            return self.normalize(self.raw_curr_state), {}  
        
    def step(self, action):

        """
        Makes an environment step in`dc_gymenv.

        Args:
            action_id (int): Action to take.

        Returns:
            observations (List[float]): Current state of the environmment
            reward (float): reward value.
            done (bool): A boolean value signaling the if the episode has ended.
            info (dict): A dictionary that containing additional information about the environment state
        """
        
        crac_setpoint_delta = self.action_mapping[action]
        self.raw_curr_stpt += crac_setpoint_delta
        self.raw_curr_stpt = max(min(self.raw_curr_stpt, self.max_temp), self.min_temp)
    
        ITE_load_pct_list = [self.cpu_load*100 for i in range(DC_Config.NUM_RACKS)] 
        
        self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, self.rackwise_outlet_temp = \
            self.dc.compute_datacenter_IT_load_outlet_temp(ITE_load_pct_list=ITE_load_pct_list, CRAC_setpoint=self.raw_curr_stpt)
            
        avg_CRAC_return_temp = DataCenter.calculate_avg_CRAC_return_temp(rack_return_approach_temp_list=DC_Config.RACK_RETURN_APPROACH_TEMP_LIST,
                                                                         rackwise_outlet_temp=self.rackwise_outlet_temp)
        
        data_center_total_ITE_Load = sum(self.rackwise_cpu_pwr) + sum(self.rackwise_itfan_pwr)
        
        self.CRAC_Fan_load, self.CT_Cooling_load, self.CRAC_Cooling_load, self.Compressor_load, self.CW_pump_load,self.CT_pump_load  = DataCenter.calculate_HVAC_power(CRAC_setpoint=self.raw_curr_stpt,
                                                                                                                                                                       avg_CRAC_return_temp=avg_CRAC_return_temp,
                                                                                                                                                                       ambient_temp=self.ambient_temp,
                                                                                                                                                                       data_center_full_load=data_center_total_ITE_Load,
                                                                                                                                                                       DC_Config=DC_Config)
        
        # calculate reward
        self.reward = 0
                
        # calculate self.raw_next_state
        self.raw_next_state = self.get_obs()
        # add info dictionary 
        self.info = {
            'dc_ITE_total_power_kW': data_center_total_ITE_Load / 1e3,
            'dc_HVAC_total_power_kW': self.CT_Cooling_load / 1e3,
            'dc_total_power_kW': (data_center_total_ITE_Load + self.CT_Cooling_load) / 1e3,
            'dc_power_lb_kW': 1000,
            'dc_power_ub_kW': 7000,
            'dc_crac_setpoint_delta': crac_setpoint_delta,
            'dc_crac_setpoint': self.raw_curr_stpt,
            'dc_cpu_workload_percent': self.cpu_load,
            'dc_int_temperature': np.mean(self.rackwise_outlet_temp),
            'dc_CW_pump_power_kW': self.CW_pump_load,
            'dc_CT_pump_power_kW': self.CT_pump_load,
        }
        

        #Done and truncated are managed by the main class, implement individual function if needed
        truncated = False
        done = False 
        # return processed/unprocessed state to agent
        if self.scale_obs:
            return self.normalize(self.raw_next_state), self.reward, done, truncated, self.info

    def NormalizeObservation(self,):
        """
        Obtains the value for normalizing the observation.
        """
        self.scale_obs = True
        for obs_var in self.observation_variables:
            self.obs_min.append(self.ranges[obs_var][0])
            self.obs_max.append(self.ranges[obs_var][1])
        
        self.obs_min = np.array(self.obs_min)
        self.obs_max = np.array(self.obs_max)
        self.obs_delta = self.obs_max - self.obs_min

    def normalize(self,obs):
        """
        Normalizes the observation.
        """
        return (obs-self.obs_min)/self.obs_delta

    def get_obs(self):
        """
        Returns the observation at the current time step.

        Returns:
            observation (List[float]): Current state of the environmment.
        """
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
        """
        Updates the current CPU workload.
        """
        self.cpu_load = cpu_load
    
    def set_ambient_temp(self, ambient_temp):
        """
        Updates the external temperature.
        """
        self.ambient_temp = ambient_temp
        
    def set_bat_SoC(self, bat_SoC):
        """
        Updates the battery state of charge.
        """
        self.bat_SoC = bat_SoC
        
class dc_gymenv_standalone(dc_gymenv):
    
    def __init__(self, env_config):
        
        super().__init__(
                       env_config['observation_variables'],
                       observation_space=env_config['observation_space'],
                       action_variables=env_config['action_variables'],
                       action_space=env_config['action_space'],
                       action_mapping=env_config['action_mapping'],
                       ranges=env_config['ranges'],
                       add_cpu_usage=env_config['add_cpu_usage'],
                       min_temp=env_config['min_temp'],
                       max_temp=env_config['max_temp'],
                       action_definition=env_config['action_definition'],
                       episode_length_in_time=env_config['episode_length_in_time']
        )
        
        self.NormalizeObservation()