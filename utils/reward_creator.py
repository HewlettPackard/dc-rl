# File where the rewards are defined
import numpy as np

def default_ls_reward(params: dict) -> float:
    """
    Calculates a reward value based on normalized load shifting.

    Args:
        params (dict): Dictionary containing parameters:
            norm_load_left (float): Normalized load left.
            out_of_time (bool): Indicator (alarm) whether the agent is in the last hour of the day.
            penalty (float): Penalty value.

    Returns:
        float: Reward value.
    """
    # Energy part of the reward
    total_energy_with_battery = params['bat_total_energy_with_battery_KWh']
    norm_CI = params['norm_CI']
    dcload_min = params['bat_dcload_min']
    dcload_max = params['bat_dcload_max']
        
    # Calculate the reward associted to the energy consumption
    norm_net_dc_load = (total_energy_with_battery - dcload_min) / (dcload_max - dcload_min)
    footprint = -1.0 * norm_CI * norm_net_dc_load

    # Penalize the agent for each task that was dropped due to queue limit
    penalty_per_dropped_task = -10  # Define the penalty value per dropped task
    tasks_dropped = params['ls_tasks_dropped']
    penalty_dropped_tasks = tasks_dropped * penalty_per_dropped_task
    
    tasks_in_queue = params['ls_tasks_in_queue']
    current_step = params['ls_current_hour']
    penalty_tasks_queue = 0
    if current_step % (24*4) >= (23*4):   # Penalty for queued tasks at the end of the day
        factor_hour = (current_step % (24*4)) / 96 # min = 0.95833, max = 0.98953
        factor_hour = (factor_hour - 0.95833) / (0.98935 - 0.95833)
        penalty_tasks_queue = -1.0 * factor_hour * tasks_in_queue/10  # Penalty for each task left in the queue
    
    if current_step % (24*4) == 0:   # Penalty for queued tasks at the end of the day
        penalty_tasks_queue = -1.0 * tasks_in_queue/10 # Penalty for each task left in the queue
    
    current_CI = params['norm_CI']
    forecast_CI = np.mean(params['forecast_CI'])
    ls_action = params['ls_action'] #  Actions: 0 - Postpone (decrease utilization), 1 - Do Nothing, 2 - execute (Increase utilization)
    # Time-based incentive using tasks_in_queue
    postpone_bonus = 0
    if current_CI > forecast_CI and ls_action == 0: # Encourage postponing tasks when CI is high and perform action is to postpone
        postpone_bonus = 0.1  # Reward for postponing tasks during high CI periods
    
    # Encourage executing tasks when CI is low
    execute_bonus = 0
    if current_CI < forecast_CI and ls_action == 2: # Encourage executing tasks when CI is low and perform action is to execute
        execute_bonus = 0.1  # Reward for executing tasks during low CI periods
    
    reward = footprint + penalty_dropped_tasks + penalty_tasks_queue + postpone_bonus + execute_bonus
    return reward

# Initialize these values with extreme values
min_load = 1000 #  float('inf')
max_load = 2200 #  float('-inf')

# Initialize running mean and standard deviation
mean_load = 2300.0  # 1500.0
std_load = 430.0 #  300.0
load_count = 0

load_stats = []
water_stats = []

# def update_mean_std(total_load):
#     global mean_load, std_load, load_count
    
#     load_count += 1
    
#     # Update the mean using a running average formula
#     old_mean = mean_load
#     mean_load += (total_load - mean_load) / load_count
    
#     # Update the standard deviation using Welford's method
#     std_load += (total_load - old_mean) * (total_load - mean_load)
    
#     print(f"New mean: {mean_load}, New std: {np.sqrt(std_load / load_count)}")

target_load = 1440.0  # Set target load here

def normalize_reward_target_based(total_load):
    return -(total_load - target_load) / target_load

def normalize_reward_mean_std(total_load):
    # Update the running statistics
    # update_mean_std(total_load)
    # load_stats.append(total_load)

    if std_load == 0:
        return 0.0
    
    # Z-score normalization
    # z_score = (total_load - mean_load) / np.sqrt(std_load / load_count)
    # z_score = -1 * ((total_load - 1200) / 250)
    z_score = -1 * ((total_load - 1900) / 550)
    
    # Optionally scale to a desired range, such as [-1, 1]
    # normalized_reward = np.tanh(-z_score)  # Apply negative to align with minimizing load
    
    return z_score


def update_load_bounds(total_load):
    global min_load, max_load
    if total_load < min_load:
        min_load = total_load
        print(f"New min load: {min_load}")
    if total_load > max_load:
        max_load = total_load
        print(f"New max load: {max_load}")

# def normalize_reward(total_load):
#     if max_load == min_load:
#         # Avoid division by zero; this would happen if the load hasn't varied at all
#         return 0.0
#     # load_stats.append(total_load)
#     # Normalize total_load and negate it to encourage reduction
#     normalized_reward = -10 * ((total_load - min_load) / (max_load - min_load))
#     return normalized_reward

def dynamic_normalized_dc_reward(total_load):
    # update_load_bounds(total_load)
    # load_stats.append(total_load)
    return normalize_reward(total_load)

def default_dc_reward(params: dict) -> float:
    """
    Calculates a reward value based on the data center's total ITE Load and CT Cooling load.

    Args:
        params (dict): Dictionary containing parameters:
            data_center_total_ITE_Load (float): Total ITE Load of the data center.
            CT_Cooling_load (float): CT Cooling load of the data center.
            load_lb (float): Lower bound of the load (kW).
            load_ub (float): Upper bound of the load (kW).

    Returns:
        float: Reward value.
    """
    data_center_total_ITE_Load = params['dc_ITE_total_power_kW']
    cooling_load = params['dc_HVAC_total_power_kW']
    # load_lb,  load_ub = params['dc_power_lb_kW']*1.75, params['dc_power_ub_kW']*0.75
    
    total_load = data_center_total_ITE_Load + cooling_load
    # if total_load < load_lb:
    #     print(f"Warning: Total load {total_load} is below the lower bound {load_lb}")
    # elif total_load > load_ub:
    #     print(f"Warning: Total load {total_load} is above the upper bound {load_ub}")
        
    # reward = dynamic_normalized_dc_reward(total_load)
    reward = normalize_reward_mean_std(total_load)
    # reward = normalize_reward_target_based(total_load)
    # Penalize water consumption
    water_penalty = 0
    # if 'dc_water_usage' in params:
        # water_stats.append(params['dc_water_usage'])
        # water_penalty = -1.0 * params['dc_water_usage'] / 1000  # Penalty for each liter of water used
        # water_penalty = -1 * ((params['dc_water_usage'] - 850) / 200)
    
    reward += water_penalty
    return reward


def dc_reward_with_temp_stability(params: dict) -> float:
    """
    Calculates a reward value based on the data center's total ITE Load, CT Cooling load, 
    and CPU temperature stability.

    Args:
        params (dict): Dictionary containing parameters:
            data_center_total_ITE_Load (float): Total ITE Load of the data center.
            CT_Cooling_load (float): CT Cooling load of the data center.
            current_cpu_temp (float): Current CPU temperature.
            previous_cpu_temp (float): Previous CPU temperature 15 minutes ago.

    Returns:
        float: Reward value.
    """
    data_center_total_ITE_Load = params['dc_ITE_total_power_kW']
    cooling_load = params['dc_HVAC_total_power_kW']
    # load_lb,  load_ub = params['dc_power_lb_kW']*1.75, params['dc_power_ub_kW']*0.75
    
    total_load = data_center_total_ITE_Load + cooling_load
    
    base_reward = normalize_reward_mean_std(total_load)

    # Penalize temperature variations
    current_cpu_temp = params.get('dc_current_cpu_temp', 0.0)
    previous_cpu_temp = params.get('dc_previous_cpu_temp', 0.0)
    
    temp_rise = current_cpu_temp - previous_cpu_temp
    temp_penalty = 0
    
    if np.abs(temp_rise) > 10:  # Example threshold for penalizing temperature rise
        temp_penalty = -0.1 * (np.abs(temp_rise) - 10)  # Penalty increases with temperature rise
    
    return base_reward + temp_penalty



def default_bat_reward(params: dict) -> float:
    """
    Calculates a reward value based on the battery usage.

    Args:
        params (dict): Dictionary containing parameters:
            total_energy_with_battery (float): Total energy with battery.
            norm_CI (float): Normalized Carbon Intensity.
            dcload_min (float): Minimum DC load.
            dcload_max (float): Maximum DC load.

    Returns:
        float: Reward value.
    """
    total_energy_with_battery = params['bat_total_energy_with_battery_KWh']
    norm_CI = params['norm_CI']
    dcload_min = params['bat_dcload_min']
    dcload_max = params['bat_dcload_max']
    
    norm_net_dc_load = (total_energy_with_battery - dcload_min) / (dcload_max - dcload_min)
    rew_footprint = -1.0 * norm_CI * norm_net_dc_load #Added scalar to line up with dc reward

    return rew_footprint


def custom_agent_reward(params: dict) -> float:
    """
    A template for creating a custom agent reward function.

    Args:
        params (dict): Dictionary containing custom parameters for reward calculation.

    Returns:
        float: Custom reward value. Currently returns 0.0 as a placeholder.
    """
    # read reward input parameters from dict object
    # custom reward calculations 
    custom_reward = 0.0 # update with custom reward shaping 
    return custom_reward

# Example of ToU reward based on energy usage and price of electricity
# ToU reward is based on the ToU (Time of Use) of the agent, which is the amount of the energy time
# the agent spends on the grid times the price of the electricity.
# This example suppose that inside the params there are the following keys:
#   - 'energy_usage': the energy usage of the agent
#   - 'hour': the hour of the day
def tou_reward(params: dict) -> float:
    """
    Calculates a reward value based on the Time of Use (ToU) of energy.

    Args:
        params (dict): Dictionary containing parameters:
            energy_usage (float): The energy usage of the agent.
            hour (int): The current hour of the day (24-hour format).

    Returns:
        float: Reward value.
    """
    
    # ToU dict: {Hour, price}
    tou = {0: 0.25,
           1: 0.25,
           2: 0.25,
           3: 0.25,
           4: 0.25,
           5: 0.25,
           6: 0.41,
           7: 0.41,
           8: 0.41,
           9: 0.41,
           10: 0.41,
           11: 0.30,
           12: 0.30,
           13: 0.30,
           14: 0.30,
           15: 0.30,
           16: 0.27,
           17: 0.27,
           18: 0.27,
           19: 0.27,
           20: 0.27,
           21: 0.27,
           22: 0.25,
           23: 0.25,
           }
    
    # Obtain the price of electricity at the current hour
    current_price = tou[params['hour']]
    # Obtain the energy usage
    energy_usage = params['bat_total_energy_with_battery_KWh']
    
    # The reward is negative as the agent's objective is to minimize energy cost
    tou_reward = -1.0 * energy_usage * current_price

    return tou_reward


def renewable_energy_reward(params: dict) -> float:
    """
    Calculates a reward value based on the usage of renewable energy sources.

    Args:
        params (dict): Dictionary containing parameters:
            renewable_energy_ratio (float): Ratio of energy coming from renewable sources.
            total_energy_consumption (float): Total energy consumption of the data center.

    Returns:
        float: Reward value.
    """
    assert params.get('renewable_energy_ratio') is not None, 'renewable_energy_ratio is not defined. This parameter should be included using some external dataset and added to the reward_info dictionary'
    renewable_energy_ratio = params['renewable_energy_ratio'] # This parameter should be included using some external dataset
    total_energy_consumption = params['bat_total_energy_with_battery_KWh']
    factor = 1.0 # factor to scale the weight of the renewable energy usage

    # Reward = maximize renewable energy usage - minimize total energy consumption
    reward = factor * renewable_energy_ratio  -1.0 * total_energy_consumption
    return reward


def energy_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on energy efficiency.

    Args:
        params (dict): Dictionary containing parameters:
            ITE_load (float): The amount of energy spent on computation (useful work).
            total_energy_consumption (float): Total energy consumption of the data center.

    Returns:
        float: Reward value.
    """
    it_equipment_power = params['dc_ITE_total_power_kW']  
    total_power_consumption = params['dc_total_power_kW']  
    
    reward = it_equipment_power / total_power_consumption
    return reward


def energy_PUE_reward(params: dict) -> float:
    """
    Calculates a reward value based on Power Usage Effectiveness (PUE).

    Args:
        params (dict): Dictionary containing parameters:
            total_energy_consumption (float): Total energy consumption of the data center.
            it_equipment_energy (float): Energy consumed by the IT equipment.

    Returns:
        float: Reward value.
    """
    total_power_consumption = params['dc_total_power_kW']  
    it_equipment_power = params['dc_ITE_total_power_kW']  
    
    # Calculate PUE
    pue = total_power_consumption / it_equipment_power if it_equipment_power != 0 else float('inf')
    
    # We aim to get PUE as close to 1 as possible, hence we take the absolute difference between PUE and 1
    # We use a negative sign since RL seeks to maximize reward, but we want to minimize PUE
    reward = -abs(pue - 1)
    
    return reward


def temperature_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on the efficiency of cooling in the data center.

    Args:
        params (dict): Dictionary containing parameters:
            current_temperature (float): Current temperature in the data center.
            optimal_temperature_range (tuple): Tuple containing the minimum and maximum optimal temperatures for the data center.

    Returns:
        float: Reward value.
    """
    assert params.get('optimal_temperature_range') is not None, 'optimal_temperature_range is not defined. This parameter should be added to the reward_info dictionary'
    current_temperature = params['dc_int_temperature'] 
    optimal_temperature_range = params['optimal_temperature_range']
    min_temp, max_temp = optimal_temperature_range
    
    if min_temp <= current_temperature <= max_temp:
        reward = 1.0
    else:
        if current_temperature < min_temp:
            reward = -abs(current_temperature - min_temp)
        else:
            reward = -abs(current_temperature - max_temp)
    return reward

def water_usage_efficiency_reward(params: dict) -> float:
    """
    Calculates a reward value based on the efficiency of water usage in the data center.
    
    A lower value of water usage results in a higher reward, promoting sustainability
    and efficiency in water consumption.

    Args:
        params (dict): Dictionary containing parameters:
            dc_water_usage (float): The amount of water used by the data center in a given period.

    Returns:
        float: Reward value. The reward is higher for lower values of water usage, 
        promoting reduced water consumption.
    """
    dc_water_usage = params['dc_water_usage']
    
    # Calculate the reward. This is a simple inverse relationship; many other functions could be applied.
    # Adjust the scalar as needed to fit the scale of your rewards or to emphasize the importance of water savings.
    reward = -0.01 * dc_water_usage
    
    return reward

# Other reward methods can be added here.

REWARD_METHOD_MAP = {
    'default_dc_reward' : default_dc_reward,
    'default_bat_reward': default_bat_reward,
    'default_ls_reward' : default_ls_reward,
    # Add custom reward methods here
    'custom_agent_reward' : custom_agent_reward,
    'tou_reward' : tou_reward,
    'renewable_energy_reward' : renewable_energy_reward,
    'energy_efficiency_reward' : energy_efficiency_reward,
    'energy_PUE_reward' : energy_PUE_reward,
    'temperature_efficiency_reward' : temperature_efficiency_reward,
    'water_usage_efficiency_reward' : water_usage_efficiency_reward,
}

def get_reward_method(reward_method : str = 'default_dc_reward'):
    """
    Maps the string identifier to the reward function

    Args:
        reward_method (string): Identifier for the reward function.

    Returns:
        function: Reward function.
    """
    assert reward_method in REWARD_METHOD_MAP.keys(), f"Specified Reward Method {reward_method} not in REWARD_METHOD_MAP"
    
    return REWARD_METHOD_MAP[reward_method]

