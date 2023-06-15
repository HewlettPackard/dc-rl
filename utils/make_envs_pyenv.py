import os
import numpy as np
import pandas as pd
import gymnasium as gym
from envs.dc_gym import dc_gymenv, dc_gymenv_standalone
import utils.data_processor as data_processor
from envs.carbon_ls import CarbonLoadEnv
from utils.utils_cf import get_init_day, Workload_Manager
from envs.bat_env_fwd_view import BatteryEnvFwd as battery_env_fwd
from utils.utils_cf import get_init_day, Workload_Manager, CI_Manager

def make_ls_env(month,
                reward_method,
                n_vars_energy : int = 4,
                n_vars_battery : int = 1):
    
    env_config = {'reward_method':reward_method}
    
    return CarbonLoadEnv(n_vars_energy=n_vars_energy,
                         n_vars_battery=n_vars_battery,
                         env_config=env_config)

def make_bat_fwd_env(month,
                    max_bat_cap_Mw : float = 2.0,
                    twenty_four_hr_episodes : bool = False,
                    charging_rate : float = 0.5,
                    reward_method : str = 'default_bat_reward'
                    ):

    init_day = get_init_day(month)
    env_config= {'n_fwd_steps':4,
                 'max_bat_cap':max_bat_cap_Mw,
                 'charging_rate':charging_rate,
                 '24hr_episodes':twenty_four_hr_episodes,
                 'start_point':init_day,
                 'dcload_max': 1.81, 
                 'dcload_min': 0.6,
                 'reward_method':reward_method}
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
                        standalone_pyeplus : bool = False,
                        num_sin_cos_vars : int = 4,
                        reward_method : str = 'default_dc_reward'
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
        standalone_pyeplus (bool, optional): Boolean Flag to indicate whether the data center environment is being tested in a standalone manner or not. Defaults to False.
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
    ranges = {
        'sinhour': [-1.0, 1.0], #0
        'coshour': [-1.0, 1.0], #1
        'sindayOTY':[-1.0, 1.0], #2
        'cosdayOTY':[-1.0, 1.0], #3
        'hour':[0.0, 23.0], #4
        'dayOTY':[1.0, 366.0], #5 
        
        'Site Outdoor Air Drybulb Temperature(Environment)': [-10.0, 40.0], #6
        'Zone Thermostat Cooling Setpoint Temperature(West Zone)': [10.0, 30.0],  # reasonable range for setpoint; can be updated based on need #7
        'Zone Air Temperature(West Zone)':[10, 35],
        'Facility Total HVAC Electricity Demand Rate(Whole Building)':  [0, 2.5e6],
        'Facility Total Electricity Demand Rate(Whole Building)': [1.0e5, 1.0e6],  # TODO: This is not a part of the observation variables right now
        'Facility Total Building Electricity Demand Rate(Whole Building)':[3.0e5, 5.0e6],
        
        'cpuUsage':[0.0, 1.0],
        'carbonIntensity':[0.0, 1000.0],
        'batterySoC': [0.0, max_bat_cap_Mw*1e6]
        
    }
    ################################################################################
    ############################## Time Series Data ################################
    ################################################################################
    weather_ts, ci_ts, cpu_usage_ts =  data_processor.dfs_creator(weather_filename = weather_filename,
                                                                  carbon_intensity_filename = carbon_intensity_filename,
                                                                  workload_filename = workload_filename,
                                                                  start_date_time = '01/01/2022 00:00:00',
                                                                  end_date_time = '01/01/2023 00:00:00',)
    
    ################################################################################
    ############################## Create the Environment ##########################
    ################################################################################
    
    if not standalone_pyeplus:
        
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
                        episode_length_in_time=episode_length_in_time,
                        reward_method=reward_method
                        )
        
        dc_env.NormalizeObservation()
        
        return dc_env
    # test in standalone mode
    else:
        env_config = {'observation_variables': observation_variables,
                        'observation_space':observation_space,
                        'action_variables':action_variables,
                        'action_space':action_space,
                        'action_mapping' : action_mapping,
                        'ranges': ranges,
                        'weather_ts':weather_ts,
                        'ci_ts':ci_ts,
                        'cpu_usage_ts' : cpu_usage_ts,
                        'add_cpu_usage':add_cpu_usage,
                        'time_delta':pd.Timedelta('15m'),
                        'min_temp':min_temp,
                        'max_temp':max_temp,
                        'action_definition':action_definition,
                        'use_ls_cpu_load' : use_ls_cpu_load,  # changed here
                        'episode_length_in_time': episode_length_in_time
        }
        
        return dc_gymenv_standalone, env_config
    
    
    