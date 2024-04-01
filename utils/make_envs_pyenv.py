import os

import gymnasium as gym
import numpy as np
import pandas as pd

from envs.bat_env_fwd_view import BatteryEnvFwd as battery_env_fwd
from envs.carbon_lsv2 import CarbonLoadEnv
from envs.dc_gym import dc_gymenv
from utils.utils_cf import get_init_day

import envs.datacenter as DataCenter
import utils.dc_config_reader as DC_Config

import itertools

def make_ls_env(month,
                n_vars_energy : int = 4,
                n_vars_battery : int = 1,
                test_mode=False,):
    """Method to build the Load shifting environment

    Args:
        month (int): Month of the year in which the agent is training.
        n_vars_energy (int, optional): Number of variables from the Energy environment. Defaults to 4.
        n_vars_battery (int, optional): Number of variables from the Battery environment. Defaults to 1.

    Returns:
        CarbonLoadEnv: Load Shifting environment
    """
    
    return CarbonLoadEnv(n_vars_energy=n_vars_energy,
                         n_vars_battery=n_vars_battery,
                         test_mode=test_mode,
                         )

def make_bat_fwd_env(month,
                    max_bat_cap_Mw : float = 2.0,
                    charging_rate : float = 0.5,
                    ):
    """Method to build the Battery environment.

    Args:
        month (int): Month of the year in which the agent is training.
        max_bat_cap_Mw (float, optional): Max battery capacity. Defaults to 2.0.
        charging_rate (float, optional): Charging rate of the battery. Defaults to 0.5.
        reward_method (str, optional): Method used to calculate the rewards. Defaults to 'default_bat_reward'.

    Returns:
        battery_env_fwd: Batery environment.
    """
    init_day = get_init_day(month)
    env_config= {'n_fwd_steps':4,
                 'max_bat_cap':max_bat_cap_Mw,
                 'charging_rate':charging_rate,
                 'start_point':init_day,
                 'dcload_max': 2.5, 
                 'dcload_min': 0.4}
    bat_env = battery_env_fwd(env_config)
    return bat_env

def make_dc_pyeplus_env(month : int = 1,
                        location : str = 'NYIS',
                        weather_filename: str = 'USA_NY_New.York-Kennedy.epw',
                        workload_filename: str = 'Alibaba_CPU_Data_Hourly_1.csv',
                        max_bat_cap_Mw : float = 2.0,
                        add_cpu_usage : bool = True,
                        add_CI : bool = True,
                        episode_length_in_time : pd.Timedelta = None,
                        use_ls_cpu_load : bool = False,
                        num_sin_cos_vars : int = 4,
                        ):
    """Method that creates the data center environment with the timeline, location, proper data files, gym specifications and auxiliary methods

    Args:
        month (int, optional): The month of the year for which the Environment uses the weather and Carbon Intensity data. Defaults to 1.
        location (str, optional): The geographical location in a standard format for which Carbon Intensity files are accessed. Supported options are 
                                'NYIS', 'AZPS', 'BPAT'. Defaults to 'NYIS'.
        weather_filename (str, optional): Filename that stores the weather data. Files should be stored under ./data/Weather. Currently supports .epw file only.  Defaults to
                                        'USA_NY_New.York-Kennedy.epw'.
        workload_filename (str, optional): Filename that stores the default CPU workload data. Files should be stored under ./data/Workload. Defaults to
                                        'Alibaba_CPU_Data_Hourly_1.csv'.
        max_bat_cap_Mw (float, optional): The battery capacity in Megawatts for the installed battery. Defaults to 2.0.
        add_cpu_usage (bool, optional): Boolean Flag to indicate whether cpu usage is part of the environment statespace. Defaults to True.
        add_CI (bool, optional): Boolean Flag to indicate whether Carbon Intensity is part of the environment statespace. Defaults to True.
        episode_length_in_time (pd.Timedelta, optional): Length of an episode in terms of pandas time-delta object. Defaults to None.
        use_ls_cpu_load (bool, optional): Use the cpu workload value from a separate Load Shifting agent. This turns of reading default cpu data. Defaults to False.
        num_sin_cos_vars (int, optional): Number of sin and cosine variable that will be added externally from the centralized data source
    Returns:
        envs.dc_gym.dc_gymenv: The environment instantiated with the particular month.
    """
    carbon_intensity_filename : str = f'{location}_NG_&_avgCI.csv'
    observation_variables = []
    ############################################################################
    ######################### Standard Variables included as default ###########
    ############################################################################
    observation_variables += [
        'Site Outdoor Air Drybulb Temperature(Environment)',
        'Zone Thermostat Cooling Setpoint Temperature(West Zone)',
        'Zone Air Temperature(West Zone)',
        'Facility Total HVAC Electricity Demand Rate(Whole Building)',  # 'HVAC POWER'
        # TODO: Will add sum of IT POWER  and HVAC Power Here if AGP wants it
        'Facility Total Building Electricity Demand Rate(Whole Building)'  #  'IT POWER'
    ]
        
    observation_space = gym.spaces.Box(low=np.float32(-2.0*np.ones(len(observation_variables)+num_sin_cos_vars+int(3*float(add_cpu_usage)))),
                                       high=np.float32(5.0*np.ones(len(observation_variables)+num_sin_cos_vars+int(3*float(add_cpu_usage)))),
                                       )
    
    ################################################################################
    ########################## Action Variables ####################################
    ################################################################################
    
    action_variables = ['Cooling_Setpoint_RL']
    action_definition = {'cooling setpoints': {'name': 'Cooling_Setpoint_RL', 'initial_value': 18}}
    min_temp = 15.0
    max_temp = 21.6
    action_mapping = {
        0: (-5),
        1: (-2),
        2: (-1),
        3: (-0.5),
        4: (0),
        5: (0.5),
        6: (1),
        7: (2),
        8: (5)
    }
    action_space = gym.spaces.Discrete(len(action_mapping))
    
    
    ################################################################################
    ########################## Variable Ranges #####################################
    ################################################################################
    
    # Perform Cooling Tower Sizing
    # This step determines the potential maximum loading of the CT
    # setting a higher ambient temp here will cause the CT to consume less power for cooling water under normal ambient temperature
    # setting a lower value of min_CRAC_setpoint will cause the CT to consume more power for higher crac setpoints during normal use
    ctafr, ct_rated_load = DataCenter.chiller_sizing(DC_Config, min_CRAC_setpoint = 14.0, ambient_temp = 40.0)  # we assume 14 so that there is no oob error
    DC_Config.CT_REFRENCE_AIR_FLOW_RATE = ctafr
    DC_Config.CT_FAN_REF_P = ct_rated_load
    
    # Perform sizing of ITE power and ambient temperature
    # Find highest and lowest values of ITE power, rackwise outlet temperature
    dc = DataCenter.DataCenter_ITModel(num_racks=DC_Config.NUM_RACKS, rack_supply_approach_temp_list=DC_Config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                                    rack_CPU_config=DC_Config.RACK_CPU_CONFIG, max_W_per_rack=DC_Config.MAX_W_PER_RACK, DC_ITModel_config=DC_Config)
    raw_curr_stpt_list = range(16,23)
    cpu_load_list = range(0,110,10) # We assume same data center load for all servers; Here it will be max
    p = itertools.product(raw_curr_stpt_list,cpu_load_list)
    dc_ambient_temp_list = []
    total_ite_pwr = []
    for raw_curr_stpt, cpu_load in p:
        ITE_load_pct_list = [cpu_load for i in range(DC_Config.NUM_RACKS)] 
        rackwise_cpu_pwr, rackwise_itfan_pwr, rackwise_outlet_temp = \
            dc.compute_datacenter_IT_load_outlet_temp(ITE_load_pct_list=ITE_load_pct_list, CRAC_setpoint=raw_curr_stpt)
        total_ite_pwr.append(sum(rackwise_cpu_pwr) + sum(rackwise_itfan_pwr))
        dc_ambient_temp_list.append(sum(rackwise_outlet_temp)/len(rackwise_outlet_temp))   
    
    ranges = {
        'sinhour': [-1.0, 1.0], #0
        'coshour': [-1.0, 1.0], #1
        'sindayOTY':[-1.0, 1.0], #2
        'cosdayOTY':[-1.0, 1.0], #3
        'hour':[0.0, 23.0], #4
        'dayOTY':[1.0, 366.0], #5 
        
        'Site Outdoor Air Drybulb Temperature(Environment)': [-10.0, 40.0], #6
        'Zone Thermostat Cooling Setpoint Temperature(West Zone)': [10.0, 30.0],  # reasonable range for setpoint; can be updated based on need #7
        'Zone Air Temperature(West Zone)':[0.9*min(dc_ambient_temp_list),1.1*max(dc_ambient_temp_list)],
        'Facility Total HVAC Electricity Demand Rate(Whole Building)':  [0.0, 1.1*ct_rated_load],  # this is cooling tower power
        'Facility Total Electricity Demand Rate(Whole Building)': [1.0e5, 1.0e6],  # TODO: This is not a part of the observation variables right now
        'Facility Total Building Electricity Demand Rate(Whole Building)':[0.9*min(total_ite_pwr), 1.1*max(total_ite_pwr)],  # this is it power
        
        'cpuUsage':[0.0, 1.0],
        'carbonIntensity':[0.0, 1000.0],
        'batterySoC': [0.0, max_bat_cap_Mw*1e6]
        
    }
    
    ################################################################################
    ############################## Create the Environment ##########################
    ################################################################################
        
    dc_env = dc_gymenv(observation_variables=observation_variables,
                    observation_space=observation_space,
                    action_variables=action_variables,
                    action_space=action_space,
                    action_mapping=action_mapping,
                    ranges=ranges,
                    add_cpu_usage=add_cpu_usage,
                    min_temp=min_temp,
                    max_temp=max_temp,
                    action_definition=action_definition,
                    DC_Config = DC_Config,
                    episode_length_in_time=episode_length_in_time
                    )
    
    dc_env.NormalizeObservation()
    
    return dc_env
    
    
    