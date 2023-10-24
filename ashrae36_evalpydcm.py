from dcrl_env import DCRL
from utils.trim_and_respond import trim_and_respond_ctrl
from tqdm import tqdm
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# os.remove('ASHRAE2_data_ny.csv')
# will create a bunch of these for rollouts
month_idxs = list(range(0,12))

monthly_co2 = []
monthly_energy = []
monthly_loadleft = []
monthly_reward = []

LOCATION = 'WA'
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
                            'cintensity_file': 'WAAT_NG_&_avgCI.csv',
                            'weather_file': 'USA_WA_Port.Angeles-Fairchild.epw',
                            'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv'}
                        )
        hvac_agent = trim_and_respond_ctrl()
        
        states, info = env.reset()
        action = {}
        terminated = False
        
        episodic_co2 = []
        episodic_energy = []
        episodic_loadleft = []
        episodic_reward = []
        
        
        # an episode
        while not terminated:
            if info['agent_dc'] == {}:
                int_temp = 20
            else:
                int_temp = info['agent_dc']['dc_int_temperature']
                
            action['agent_dc'] = hvac_agent.action(int_temp)
            states, rew, terminated, truncated, info = env.step(action)
            # if action['agent_dc'] != 4:
            #     print(f'Action is different from 4: {action["agent_dc"]}')
            terminated = terminated['__all__']
            episodic_co2.append(info['agent_dc']['bat_CO2_footprint'])
            episodic_energy.append(info['agent_dc']['bat_total_energy_with_battery_KWh'])
            episodic_loadleft.append(info['agent_dc']['ls_unasigned_day_load_left'])
            episodic_reward.append(rew["agent_dc"])
        
        monthly_co2.append(sum(episodic_co2)/len(episodic_co2))
        monthly_energy.append(sum(episodic_energy)/len(episodic_energy))
        monthly_loadleft.append(sum(episodic_loadleft)/len(episodic_loadleft))
        monthly_reward.append(sum(episodic_reward))
        
        # Update the progress bar
        pbar.update(1)
        states, infos = env.reset()
print(f"Average bat_total_energy_with_battery_KWh {sum(monthly_energy)/len(monthly_energy):.2f}")  
print(f"Average monthly episode reward {sum(monthly_reward)/len(monthly_reward):.2f}")
print(f"Average episodic_co2 {sum(monthly_co2)/len(monthly_co2):.2f}")
print(f"Average monthly_loadleft {sum(monthly_loadleft)/len(monthly_loadleft):.2f}")

# df = pd.read_csv('ASHRAE2_data_ny.csv')

# print(f"Std of HVAC setpoint: {df['HVAC Setpoint'].values.std():.2f}")

"""
Average bat_total_energy_with_battery_KWh 28.52126949218589
Average monthly episode reward 0.14407044411870587
Average episodic_co2 15385.838502882043
Average monthly_loadleft 0.0

"""