import random
import string

import gymnasium as gym
import numpy as np
import pandas as pd

import custom_sinergym
from custom_sinergym.utils.rewards import myLinearRewardwCost
from custom_sinergym.utils.wrappers import (NormalizeObservation,
                                            tinyMultiObsWrapper)

from envs.bat_env_fwd_view import BatteryEnvFwd as battery_env_fwd
from envs.carbon_ls import CarbonLoadEnv
from utils.managers import CI_Manager, Workload_Manager
from utils.utils_cf import get_init_day


def make_dc_env(month, location):
    """Method to build the energy model using EnergyPlus

    Args:
        month (int): Month of the year in which the agent is training.
        location ('string'): Location from we are taking the weather data from.

    Returns:
        datacenter_env: Energy environment
    """
    datacenter_env = 'Eplus-datacenter-mixed-continuous-v1'

    obs_variables = [
                    'Site Outdoor Air Drybulb Temperature(Environment)',
                    'Zone Thermostat Cooling Setpoint Temperature(West Zone)',
                    'Zone Air Temperature(West Zone)',
                    'Facility Total HVAC Electricity Demand Rate(Whole Building)',
                    'Facility Total Electricity Demand Rate(Whole Building)',
                    'Facility Total Building Electricity Demand Rate(Whole Building)'
                    ]

    add_cpu_usage = True
    add_sincos = True

    if add_sincos:
        date_var = 4
    else:
        date_var = 2

    if add_cpu_usage:
        cpu_var = 1
    else:
        cpu_var = 0

    add_vars = 2 

    new_observation_space = gym.spaces.Box(low=-5.0e9, high=5.0e9,
                                           shape=(date_var + len(obs_variables) + cpu_var + add_vars,),
                                           dtype=np.float32)


    new_action_definition={
        'cooling setpoints': {'name': 'Cooling_Setpoint_RL', 'initial_value': 18}
    }

    new_action_variables = [
        'Cooling_Setpoint_RL',
    ]

    min_temp = 16.0
    max_temp = 26.0

    new_action_mapping = {
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
    
    new_action_space = gym.spaces.discrete.Discrete(len(new_action_mapping))
    timestep = 4
    years = 1
    start_year = 1989
    start_date = pd.to_datetime(f'1989-{(month+1):02d}-01')
    end_date = start_date + pd.DateOffset(days=31)
    end_day = end_date.day
    end_month = end_date.month
    end_year = end_date.year

    extra_params={'timesteps_per_hour':timestep,
                'runperiod':(1,start_date.month,start_year, end_day,end_month,end_year)} #  (start_day, start_month, start_year, end_day, end_month, end_year)

    s = string.ascii_letters+string.digits
    random_seed = '/' + ''.join(random.sample(s, 16)) + '/'
    weather = 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw'
    if 'ny' in location.lower():
        weather = 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw'
    elif 'az' in location.lower():
        weather = 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'

    dc_env = gym.make(datacenter_env,
                observation_variables=obs_variables,
                observation_space=new_observation_space,
                action_variables=new_action_variables,
                action_mapping=new_action_mapping,
                action_definition=new_action_definition,
                action_space=new_action_space,
                config_params=extra_params,
                weather_file=weather,
                weather_variability=(4.0, 1.0, 0.01),
                reward=myLinearRewardwCost,
                delta_actions=True,
                seed=random_seed,
                temp_range=[min_temp, max_temp],
                add_sincos=add_sincos,
                external_cpu_scheme='custom',
                add_cpu_usage=add_cpu_usage)

    ranges = {'year': [start_year, start_year+years],
        'month': [6.0, 10.0],
        'sindayOTY': [1.0, 366.0],
        'cosdayOTY': [1.0, 366.0],
        'day': [1.0, 31.0],
        'hour': [0.0, 23.0],
        'sinhour': [0.0, 23.0],
        'coshour': [0.0, 23.0],
        'cosmonth': [1.0, 12.0],
        'day_of_the_month': [0.0, 76.0],
        'Site Outdoor Air Drybulb Temperature(Environment)': [-5.0, 20.0],
        'extTemp1': [-10.0, 10.0],
        'extTemp2': [-10.0, 10.0],
        'extTemp3': [-10.0, 10.0],
        'Site Outdoor Air Relative Humidity(Environment)': [19.0, 100.0],
        'Site Wind Speed(Environment)': [0.0, 11.8],
        'Site Wind Direction(Environment)': [0.0, 357.5],
        'Site Diffuse Solar Radiation Rate per Area(Environment)': [0.0, 566.0],
        'Site Direct Solar Radiation Rate per Area(Environment)': [0.0, 906.0],
        'Zone Thermostat Heating Setpoint Temperature(West Zone)': [1.0, 1.0],
        'Zone Thermostat Cooling Setpoint Temperature(West Zone)': [-5.0, 5.0],
        'Zone Air Temperature(West Zone)': [min_temp, max_temp],
        'Zone Air Relative Humidity(West Zone)': [23.335423, 72.85425],
        'Zone ITE CPU Electricity Rate(West Zone)': [10679.612, 34566.9],
        'Zone ITE Fan Electricity Rate(West Zone)': [6962.26, 25274.865],
        'Zone Thermostat Heating Setpoint Temperature(East Zone)': [1.0, 1.0],
        'Zone Thermostat Cooling Setpoint Temperature(East Zone)': [min_temp, max_temp],
        'Zone Air Temperature(East Zone)': [min_temp, max_temp],
        'Zone Air Relative Humidity(East Zone)': [23.709562, 74.624275],
        'Zone ITE CPU Electricity Rate(East Zone)': [11917.506, 38515.42],
        'Zone ITE Fan Electricity Rate(East Zone)': [7770.8535, 28150.297],
        'Facility Total HVAC Electricity Demand Rate(Whole Building)': [3.0e4, 1.1e8],
        'Facility Total Electricity Demand Rate(Whole Building)': [1.0e5, 1.0e9],
        'Facility Total Building Electricity Demand Rate(Whole Building)': [6.0e4, 9.0e8],
        'Power Utilization Effectiveness(EMS)': [1.0335548, 1.9118807],
        'cpuUsage': [0, 1],
        'carbonIntensity': [0, 1],
        'batterySoC': [0, 1]}
    dc_env = NormalizeObservation(dc_env, ranges, add_sincos=True)
    dc_env = tinyMultiObsWrapper(dc_env, n=3, add_sincos=True)
    return dc_env

def make_ls_env(month, test_mode=False, n_vars_energy=4, n_vars_battery=1):
    """Method to build the Load shifting environment

    Args:
        month (int): Month of the year in which the agent is training.
        n_vars_energy (int, optional): Number of variables from the Energy environment. Defaults to 4.
        n_vars_battery (int, optional): Number of variables from the Battery environment. Defaults to 1.
        test_mode (bool,optional): Use or not evaluation mode

    Returns:
        CarbonLoadEnv: Load Shifting environment
    """

    total_wkl = Workload_Manager().get_total_wkl()
    
    return CarbonLoadEnv(n_vars_energy=n_vars_energy, n_vars_battery=n_vars_battery, test_mode=test_mode)

def make_bat_fwd_env(month):

    """Method to build the Battery environment.

    Args:
        month (int): Month of the year in which the agent is training.

    Returns:
        battery_env_fwd: Batery environment.
    """

    init_day = get_init_day(month)
    env_config = {'n_fwd_steps':4, 'max_bat_cap':2, 'charging_rate':0.5, '24hr_episodes':True,
                'start_point':init_day, 'dcload_max': 1.2, 'dcload_min': 0.05, 
                }
    bat_env = battery_env_fwd(env_config)
    return bat_env
