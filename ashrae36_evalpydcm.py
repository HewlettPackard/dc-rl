from dcrl_env import DCRL
from utils.trim_and_respond import trim_and_respond_ctrl
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# will create a bunch of these for rollouts
month_idxs = list(range(0,12))

monthly_co2 = []
monthly_energy = []
monthly_loadleft = []
monthly_reward = []

with tqdm(total=len(month_idxs), desc="Processing", unit="iteration") as pbar:
    for month_idx in month_idxs:
        
        env = DCRL(
            env_config={'worker_index' : month_idx,
                        'agents': ['agent_dc'],
                        
                        # Datafiles
                        'location': 'ny',
                        'cintensity_file': 'NYIS_NG_&_avgCI.csv',
                        'weather_file': 'USA_NY_New.York-Kennedy.epw',
                        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv'}
                    )
        hvac_agent = trim_and_respond_ctrl()
        
        states, infos = env.reset()
        action = {}
        terminated = False
        
        episodic_co2 = []
        episodic_energy = []
        episodic_loadleft = []
        episodic_reward = []
        
        
        # an episode
        while not terminated:
            obs = states["agent_dc"]
            action['agent_dc'] = hvac_agent.action(obs)
            states, rew, terminated, truncated, info = env.step(action)
            terminated = terminated["__all__"]
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

print(f"Average bat_total_energy_with_battery_KWh {sum(monthly_energy)/len(monthly_energy)}")  
print(f"Average monthly episode reward {sum(monthly_reward)/len(monthly_reward)}")
print(f"Average episodic_co2 {sum(monthly_co2)/len(monthly_co2)}")
print(f"Average monthly_loadleft {sum(monthly_loadleft)/len(monthly_loadleft)}")

        


"""
Average bat_total_energy_with_battery_KWh 28.463686130437356
Average monthly episode reward 421.94614652498643
Average episodic_co2 14839.00239855321
Average monthly_loadleft 0.0

"""