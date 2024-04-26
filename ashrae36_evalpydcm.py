#%%
from dcrl_env import DCRL
from utils.trim_and_respond import trim_and_respond_ctrl
from tqdm import tqdm
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
#%%
# os.remove('ASHRAE2_data_ny.csv')
# will create a bunch of these for rollouts
# month_idxs = list(range(0,12))
month_idxs = [6]

monthly_co2 = []
monthly_energy = []
monthly_water = []
monthly_reward = []

actions = []
int_temps = []
setpoints = []
LOCATION = 'NY'
with tqdm(total=len(month_idxs), desc="Processing", unit="iteration") as pbar:
    for month_idx in month_idxs:
        
        if LOCATION == 'NY':
            env = DCRL(
                env_config={'worker_index' : month_idx,
                            'agents': ['agent_dc'],
                            # Datafiles
                            'location': 'ny',
                            'cintensity_file': 'NYIS_NG_&_avgCI.csv',
                            'weather_file': 'USA_NY_New.York-Kennedy.epw',
                            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv'}
                        )
        elif LOCATION == 'AZ':
            env = DCRL(
                env_config={'worker_index' : month_idx,
                            'agents': ['agent_dc'],
                            # Datafiles
                            'location': 'az',
                            'cintensity_file': 'AZPS_NG_&_avgCI.csv',
                            'weather_file': 'USA_AZ_Tucson-Davis-Monthan.epw',
                            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv'}
                        )
        else:
            env = DCRL(
                env_config={'worker_index' : month_idx,
                            'agents': ['agent_dc'],
                            # Datafiles
                            'location': 'wa',
                            'cintensity_file': 'BPAT_NG_&_avgCI.csv',
                            'weather_file': 'USA_WA_Port.Angeles-Fairchild.epw',
                            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv'}
                        )
        hvac_agent = trim_and_respond_ctrl()
        
        states, info = env.reset()
        action = {}
        truncated = False
        
        episodic_co2 = []
        episodic_energy = []
        episodic_water = []
        episodic_reward = []
        
        
        # an episode
        while not truncated:
            if info['agent_dc'] == {}:
                int_temp = 20
            else:
                int_temp = info['agent_dc']['dc_int_temperature']
                
            action['agent_dc'] = hvac_agent.action(int_temp)
            actions.append(action['agent_dc'])
            int_temps.append(int_temp)
            
            states, rew, terminated, truncated, info = env.step(action)
            # if action['agent_dc'] != 4:
            #     print(f'Action is different from 4: {action["agent_dc"]}')
            truncated = truncated['__all__']
            
            setpoints.append(info['__common__']['dc_crac_setpoint'])
            episodic_co2.append(info['__common__']['bat_CO2_footprint'])
            episodic_energy.append(info['__common__']['bat_total_energy_with_battery_KWh'])
            episodic_water.append(info['__common__']['dc_water_usage'])
            episodic_reward.append(rew["agent_dc"])
        
        monthly_co2.append(sum(episodic_co2)/len(episodic_co2))
        monthly_energy.append(sum(episodic_energy)/len(episodic_energy))
        monthly_water.append(sum(episodic_water)/len(episodic_water))
        monthly_reward.append(sum(episodic_reward))
        
        # Update the progress bar
        pbar.update(1)
        states, infos = env.reset()
print(f"Average bat_total_energy_with_battery_KWh {sum(monthly_energy)/len(monthly_energy):.2f}")  
print(f"Average monthly episode reward {sum(monthly_reward)/len(monthly_reward):.2f}")
print(f"Average episodic_co2 {sum(monthly_co2)/len(monthly_co2):.2f}")
print(f"Average water usage {sum(monthly_water)/len(monthly_water):.2f}")

#%%
# df = pd.read_csv('ASHRAE2_data_ny.csv')

# print(f"Std of HVAC setpoint: {df['HVAC Setpoint'].values.std():.2f}")

"""
--- NY ---
Average bat_total_energy_with_battery_KWh 51.82
Average monthly episode reward 322.64
Average episodic_co2 27063.79
Average water usage 826.36
---    ---

"""
#%%
# Plot the behavior or the agent
# Plot, for June month, the internal temperature, the setpoint, and the actions
import matplotlib.pyplot as plt

# Create the figure and the first Y-axis
fig, ax1 = plt.subplots()

sel_ini = 0
sel_end = sel_ini+1000

color = 'tab:red'
# ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature (°C)', color=color)
ax1.plot(int_temps[sel_ini:sel_end], color=color, label='Room Temperature')
ax1.plot(setpoints[sel_ini:sel_end], color='tab:orange', label='HVAC Setpoint', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Instantiate a second Y-axis that shares the same X-axis
ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Action', color=color)
ax2.plot(actions[sel_ini:sel_end], color=color, label='HVAC Action')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

# Show the plot
plt.title('HVAC Agent Behavior')
plt.show()

#%%