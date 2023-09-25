#%%
from utils.make_envs_pyenv import make_dc_pyeplus_env
from utils.utils_cf import get_energy_variables, get_init_day, obtain_paths
from utils.make_envs import make_dc_env
import time
import numpy as np


# Datafiles
location = 'ny'
        
ci_loc, wea_loc = obtain_paths(location)
month = 0

dcrl_dc_env = make_dc_pyeplus_env(month+1, ci_loc, max_bat_cap_Mw=2.0, use_ls_cpu_load=True) 

num_sims = 3
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
# %%
