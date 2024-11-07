#%%
import os
import random
import warnings
    
import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete, Tuple

from dcrl_env import DCRL

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
    )

#%%
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = {
    # GA config
    'config1' : {
        'location': 'GA',
        'cintensity_file': 'GA_NG_&_avgCI.csv',
        'weather_file': 'USA_GA_Phoenix-Sky.Harbor.epw',
        'workload_file': 'GoogleClusteData_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc3.json',
        'datacenter_capacity_mw' : 1.1,
        'timezone_shift': 8,
        'month': 7,
        'days_per_episode': 30,
        },

    # NY config
    'config2' : {
        'location': 'NY',
        'cintensity_file': 'NY_NG_&_avgCI.csv',
        'weather_file': 'USA_NY_New.York-LaGuardia.epw',
        'workload_file': 'GoogleClusteData_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc2.json',
        'datacenter_capacity_mw' : 1,
        'timezone_shift': 0,
        'month': 7,
        'days_per_episode': 30
        },

    # WA config
    'config3' : {
        'location': 'CA',
        'cintensity_file': 'CA_NG_&_avgCI.csv',
        'weather_file': 'USA_CA_San.Jose-Mineta.epw',
        'workload_file': 'GoogleClusteData_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 0.9,
        'timezone_shift': 16,
        'month': 7,
        'days_per_episode': 30
        },
    
    # List of active low-level agents
    'active_agents': ['agent_dc'],

    # config for loading trained low-level agents
    'low_level_actor_config': {
        'harl': {
            'algo' : 'happo',
            'env' : 'dcrl',
            'exp_name' : 'll_actor',
            'model_dir': f'{CURR_DIR}/seed-00001-2024-04-22-20-59-21/models',
            },
        'rllib': {
            'checkpoint_path': f'{CURR_DIR}/maddpg/checkpoint_000000/',
            'is_maddpg': True
        }
    },
}
#%%
# Init all datacenter environments
DC1 = DCRL(DEFAULT_CONFIG['config1'])
DC2 = DCRL(DEFAULT_CONFIG['config2'])
DC3 = DCRL(DEFAULT_CONFIG['config3'])

# Reset the environments
DC1.reset()
DC2.reset()
DC3.reset()
#%%
'''
I want to plot the CPU utilization of the datacenters in one day to show the behavior of the workload utilization across different locations with the time different. Use the method get_total_wkl from the workload manager to get the total workload of the datacenters in the simulated period
Also, I want to plot the Carbon intensity of the datacenters in one day to show the behavior of the carbon intensity across different locations with the time different. To do this, I will use the method get_total_ci of the CIManager.
'''
#%%
cpu_utilization = {'DC1': [], 'DC2': [], 'DC3': []}
original_cpu_utilization = {'DC1': [], 'DC2': [], 'DC3': []}
carbon_intensity = {'DC1': [], 'DC2': [], 'DC3': []}
weather = {'DC1': [], 'DC2': [], 'DC3': []}

cpu_utilization['DC1'].append(DC1.workload_m.get_total_wkl())
cpu_utilization['DC2'].append(DC2.workload_m.get_total_wkl())
cpu_utilization['DC3'].append(DC3.workload_m.get_total_wkl())

original_cpu_utilization['DC1'].append(DC1.workload_m.original_data)
original_cpu_utilization['DC2'].append(DC2.workload_m.original_data)
original_cpu_utilization['DC3'].append(DC3.workload_m.original_data)

carbon_intensity['DC1'].append(DC1.ci_m.get_total_ci())
carbon_intensity['DC2'].append(DC2.ci_m.get_total_ci())
carbon_intensity['DC3'].append(DC3.ci_m.get_total_ci())

weather['DC1'].append(DC1.weather_m.get_total_weather())
weather['DC2'].append(DC2.weather_m.get_total_weather())
weather['DC3'].append(DC3.weather_m.get_total_weather())

#%% Apply a rolling window to the data
window = 4
cpu_utilization['DC1'][0] = np.convolve(cpu_utilization['DC1'][0], np.ones(window), 'valid') / window
cpu_utilization['DC2'][0] = np.convolve(cpu_utilization['DC2'][0], np.ones(window), 'valid') / window
cpu_utilization['DC3'][0] = np.convolve(cpu_utilization['DC3'][0], np.ones(window), 'valid') / window

original_cpu_utilization['DC1'][0] = np.convolve(original_cpu_utilization['DC1'][0], np.ones(window), 'valid') / window
original_cpu_utilization['DC2'][0] = np.convolve(original_cpu_utilization['DC2'][0], np.ones(window), 'valid') / window
original_cpu_utilization['DC3'][0] = np.convolve(original_cpu_utilization['DC3'][0], np.ones(window), 'valid') / window

carbon_intensity['DC1'][0] = np.convolve(carbon_intensity['DC1'][0], np.ones(window), 'valid') / window
carbon_intensity['DC2'][0] = np.convolve(carbon_intensity['DC2'][0], np.ones(window), 'valid') / window
carbon_intensity['DC3'][0] = np.convolve(carbon_intensity['DC3'][0], np.ones(window), 'valid') / window

weather['DC1'][0] = np.convolve(weather['DC1'][0], np.ones(window), 'valid') / window
weather['DC2'][0] = np.convolve(weather['DC2'][0], np.ones(window), 'valid') / window
weather['DC3'][0] = np.convolve(weather['DC3'][0], np.ones(window), 'valid') / window
# %% Now let's plot the CPU utilization on the first 2 days. Each entry is 15 minutes. So, to represent 2 days, I need to plot from index 0 to index 4*24*2
import matplotlib.pyplot as plt
import pandas as pd
days = 28
timesteps = 4*24*days
init_day = 1
x = pd.date_range(start=f'2022-07-{init_day}', periods=timesteps, freq='15T')
plt.figure(figsize=(13, 5))
plt.plot(x, cpu_utilization['DC1'][0][init_day*4*24:(init_day+days)*4*24]*100, label='GA', linewidth=2)
plt.plot(x, cpu_utilization['DC2'][0][init_day*4*24:(init_day+days)*4*24]*100, label='NY', linewidth=2)
plt.plot(x, cpu_utilization['DC3'][0][init_day*4*24:(init_day+days)*4*24]*100, label='CA', linewidth=2)
plt.xlabel('Days') 
plt.ylabel('Data Center Utilization (%)')
plt.title('Utilization of Data Centers on July month')
plt.legend()
plt.grid(True, linestyle='--')
plt.xlim(x[0], x[-1])
plt.savefig('plots/GreenDCC_cpu_utilization.pdf', bbox_inches='tight')
plt.show()

#%%
x = pd.date_range(start=f'2022-07-{init_day}', periods=timesteps, freq='15T')
plt.figure(figsize=(15, 5))
plt.plot(x, original_cpu_utilization['DC1'][0][init_day*4*24:(init_day+days)*4*24], label='GA')
# plt.plot(x, original_cpu_utilization['DC2'][0][init_day*4*24:(init_day+days)*4*24], label='NY')
# plt.plot(x, original_cpu_utilization['DC3'][0][init_day*4*24:(init_day+days)*4*24], label='CA')
plt.xlabel('Time') 
plt.ylabel('Original CPU Utilization')
plt.title('CPU Utilization of Datacenters')
plt.legend()
plt.grid(True, linestyle='--')
plt.xlim(x[0], x[-1])
plt.show()



# %%

# %% Now the same but for the carbon intensity
plt.figure(figsize=(13, 5))
plt.plot(x, carbon_intensity['DC1'][0][init_day*4*24:(init_day+days)*4*24], label='GA', linewidth=2)
plt.plot(x, carbon_intensity['DC2'][0][init_day*4*24:(init_day+days)*4*24], label='NY', linewidth=2)
plt.plot(x, carbon_intensity['DC3'][0][init_day*4*24:(init_day+days)*4*24], label='CA', linewidth=2)
plt.xlabel('Days') 
plt.ylabel('Carbon Intensity (gCO2/kWh)')
plt.title('Carbon Intensity of Data Centers')
plt.legend()
plt.grid(True, linestyle='--')
plt.xlim(x[0], x[-1])
plt.savefig('plots/GreenDCC_carbon_intensity.pdf', bbox_inches='tight')
plt.show()


# %% Now the temperature
plt.figure(figsize=(13, 5))
plt.plot(x, weather['DC1'][0][init_day*4*24:(init_day+days)*4*24], label='GA', linewidth=2)
plt.plot(x, weather['DC2'][0][init_day*4*24:(init_day+days)*4*24], label='NY', linewidth=2)
plt.plot(x, weather['DC3'][0][init_day*4*24:(init_day+days)*4*24], label='CA', linewidth=2)
plt.xlabel('Days') 
plt.ylabel('External Temperature (°C)')
plt.title('External Temperature of Data Centers')
plt.legend()
plt.grid(True, linestyle='--')
plt.xlim(x[0], x[-1])
plt.savefig('plots/GreenDCC_temperature.pdf', bbox_inches='tight')
plt.show()



# %%
