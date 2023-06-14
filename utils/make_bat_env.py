from envs.bat_env_fwd_view import BatteryEnvFwd as battery_env_fwd
from utils.utils_cf import get_init_day, Workload_Manager, CI_Manager

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
