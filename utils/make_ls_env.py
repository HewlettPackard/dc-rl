from envs.carbon_ls import CarbonLoadEnv
from utils.utils_cf import get_init_day, Workload_Manager

def make_ls_env(month,
                reward_method,
                n_vars_energy : int = 4,
                n_vars_battery : int = 1):

    total_wkl = Workload_Manager().get_total_wkl()
    
    env_config = {'reward_method':reward_method}
    
    return CarbonLoadEnv(n_vars_energy=n_vars_energy,
                         n_vars_battery=n_vars_battery,
                         env_config=env_config)
