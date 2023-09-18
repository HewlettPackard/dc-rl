#%%
from utils.make_envs_pyenv import make_dc_pyeplus_env
from utils.utils_cf import get_energy_variables, get_init_day, obtain_paths
from utils.make_envs import make_dc_env
import time
import numpy as np

#%%
# Datafiles
location = 'ny'
        
ci_loc, wea_loc = obtain_paths(location)
month = 0


dcrl_dc_env = make_dc_pyeplus_env(month+1, ci_loc, max_bat_cap_Mw=2.0, use_ls_cpu_load=True) 
eplus_dc_env = make_dc_env(month, location) 


#%%
# %timeit make_dc_pyeplus_env(month+1, ci_loc, max_bat_cap_Mw=2.0, use_ls_cpu_load=True)
# 
# %%
_ = dcrl_dc_env.reset()
_ = eplus_dc_env.reset()
# %%

# Measure the step time

action = np.int64(4)
num_steps = 1000

_ = dcrl_dc_env.reset()

start_time = time.time()
for _ in range(0, num_steps):
    dcrl_dc_env.step(action)
avg_step_time_dcrl = (time.time() - start_time) / num_steps

_ = eplus_dc_env.reset()
start_time = time.time()
for _ in range(0, num_steps):
    eplus_dc_env.step(action)
avg_step_time_eplus = (time.time() - start_time) / num_steps

print(f'DCRL avg step time: {avg_step_time_dcrl:.5f} seconds')
print(f'Eplus avg step time: {avg_step_time_eplus:.5f} seconds')
# %%
num_steps = 4*24*30*10

_ = dcrl_dc_env.reset()
times_dcrl = []
j = 0
for i in range(0, num_steps):
    action = np.int64(np.random.randint(0, 8))
    start_time = time.time()
    dcrl_dc_env.step(action)
    times_dcrl.append(time.time() - start_time)
    if j > 4*24*30:
        _ = dcrl_dc_env.reset()
        j = 0
    j += 1
        
_ = eplus_dc_env.reset()
times_eplus = []
j = 0
for i in range(0, num_steps):
    action = np.int64(np.random.randint(0, 8))
    start_time = time.time()
    eplus_dc_env.step(action)
    times_eplus.append(time.time() - start_time)
    if j > 4*24*30:
        _ = eplus_dc_env.reset()
        j = 0
    j += 1


times_dcrl = np.array(times_dcrl)
times_eplus = np.array(times_eplus)

print(f'DCRL avg step time: {times_dcrl.mean()*1e6:.2f} ± {times_dcrl.std()*1e6:.2f} µs')
print(f'Eplus avg step time: {times_eplus.mean()*1e6:.2f} ± {times_eplus.std()*1e6:.2f} µs')
# %% Simulation time of DCRL
num_sims = 100
days = 7
sim_times_dcrl = []
for k in range(0, num_sims):
    start_time = time.time()
    _ = dcrl_dc_env.reset()
    i = 0
    while i < 4*24*days:
        action = np.int64(np.random.randint(0, 8))
        dcrl_dc_env.step(action)
        i += 1
    sim_times_dcrl.append(time.time() - start_time)

sim_times_dcrl = np.array(sim_times_dcrl)
print(f'DCRL simulation times: {sim_times_dcrl.mean():.3f} s ± {sim_times_dcrl.std()*1e3:.3f} ms')
# %% Simulation time of EnergyPlys
num_sims = 7
days = 7
sim_times_eplus = []
for k in range(0, num_sims):
    start_time = time.time()
    _ = eplus_dc_env.reset()
    i = 0
    while i < 4*24*days:
        action = np.int64(np.random.randint(0, 8))
        eplus_dc_env.step(action)
        i += 1
    sim_times_eplus.append(time.time() - start_time)
    
sim_times_eplus = np.array(sim_times_eplus)
print(f'Eplus simulation times: {sim_times_eplus.mean():.3f} s ± {sim_times_eplus.std()*1e3:.3f} ms')

# %%
