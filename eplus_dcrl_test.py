from utils.make_envs_pyenv import make_dc_pyeplus_env
from utils.utils_cf import get_energy_variables, get_init_day, obtain_paths
from utils.make_envs import make_dc_env
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

location = 'ny'

ci_loc, wea_loc = obtain_paths(location)
month = 0


dcrl_dc_env = make_dc_pyeplus_env(month+1, ci_loc, max_bat_cap_Mw=2.0, use_ls_cpu_load=True) 
eplus_dc_env = make_dc_env(month, location) 

# Measure the step time
action = np.int64(4)
num_steps = 1000
_ = dcrl_dc_env.reset()
start_time = time.time()
for _ in range(0, num_steps):
    dcrl_dc_env.step(action)
avg_step_time_dcrl = (time.time() - start_time) / num_steps
print("######################################################")
print(f'PyDCM avg step time: {avg_step_time_dcrl:.5f} seconds')
print("######################################################")

# Measure the step time
action = np.int64(4)
num_steps = 1000
_ = eplus_dc_env.reset()
start_time = time.time()
for _ in range(0, num_steps):
    eplus_dc_env.step(action)
avg_step_time_eplus = (time.time() - start_time) / num_steps
print("######################################################")
print(f'Eplus avg step time: {avg_step_time_eplus:.5f} seconds')
print("######################################################")


print("######################################################")
print(f'Energy Plus Implementation')
print("######################################################")
# Create a tqdm progress bar to wrap your loop and enable the timer
with tqdm(total=1000, desc="Processing", unit="iteration") as pbar:
    for _ in range(1000):
        dc_state, _, dc_terminated, dc_truncated, dc_info = eplus_dc_env.step(action)
        pbar.update(1)


print("######################################################")
print(f'PyDCM Implementation')
print("######################################################")
# Create a tqdm progress bar to wrap your loop and enable the timer
with tqdm(total=1000, desc="Processing", unit="iteration") as pbar:
    for _ in range(1000):
        dc_state, _, dc_terminated, dc_truncated, dc_info = dcrl_dc_env.step(action)
        pbar.update(1)
