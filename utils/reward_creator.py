def default_ls_reward(params: dict) -> float:
    norm_load_left = params['norm_load_left']
    out_of_time = params['out_of_time']
    penalty = params['penalty']
    if out_of_time:
        reward = -norm_load_left*penalty
    else:
        reward = 0
    return reward

def default_dc_reward(params: dict) -> float:
    data_center_total_ITE_Load = params['data_center_total_ITE_Load']
    CT_Cooling_load = params['CT_Cooling_load']
    energy_lb,  energy_ub = params['energy_lb'], params['energy_ub']
    return - 1.0 * ((data_center_total_ITE_Load + CT_Cooling_load)-energy_lb)/(energy_ub-energy_lb)

def default_bat_reward(params: dict) -> float:
    total_energy_with_battery = params['total_energy_with_battery']
    norm_CI = params['norm_CI']
    a_t = params['a_t']
    dcload_min = params['dcload_min']
    dcload_max = params['dcload_max']
    battery_current_load = params['battery_current_load']
    max_bat_cap = params['max_bat_cap']
    
    norm_net_dc_load = (total_energy_with_battery / 1e3 - dcload_min) / (dcload_max - dcload_min)
    rew_footprint = -1.0 * norm_CI * norm_net_dc_load

    return rew_footprint

# Example custom reward method
def custom_agent_reward(params: dict) -> float:
    #read reward input parameters from dict object
    #custom reward calculations 
    custom_reward = 0.0 #update with custom reward shaping 
    return custom_reward

REWARD_METHOD_MAP = {
    'default_dc_reward' : default_dc_reward,
    'default_bat_reward': default_bat_reward,
    'default_ls_reward' : default_ls_reward,
    # Add custom reward methods here
    'custom_agent_reward' : custom_agent_reward,
}

def get_reward_method(reward_method : str = 'default_dc_reward'):
    assert reward_method in REWARD_METHOD_MAP.keys(), f"Specified Reward Method {reward_method} not in REWARD_METHOD_MAP"
    
    return REWARD_METHOD_MAP[reward_method]

