#%%
import os
import sys
import copy  # Import copy to make a deep copy of metrics
import warnings 
import json
import numpy as np
warnings.filterwarnings('ignore')

import pandas as pd

import torch
import matplotlib.pyplot as plt
sys.path.insert(0,os.getcwd())  # or sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'heterogeneous_sustaindc')))
from harl.runners import RUNNER_REGISTRY
from harl.utils.trans_tools import _t2n

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'dc-rl')))
from utils.base_agents import BaseLoadShiftingAgent, BaseHVACAgent, BaseBatteryAgent
#%%
MODEL_PATH = 'trained_models'
SAVE_EVAL = "results"
ENV = 'sustaindc'
LOCATION = "az"
AGENT_TYPE = "haa2c"
RUN = "seed-00001-2024-06-04-20-41-56"
ACTIVE_AGENTS = ['agent_ls', 'agent_dc', 'agent_bat']
NUM_EVAL_EPISODES = 1

# Define paths and configurations as usual
path = '/lustre/guillant/sustaindc/results/sustaindc/az/happo/happo_liquid_dc_lr_001_16_09/seed-00001-2024-08-20-09-37-35'
with open(path + '/config.json', encoding='utf-8') as file:
    saved_config = json.load(file)

algo_args, env_args, main_args = saved_config['algo_args'], saved_config['env_args'], saved_config['main_args']

algo_args['train']['n_rollout_threads'] = 1
algo_args['eval']['n_eval_rollout_threads'] = 1
algo_args['train']['model_dir'] = path + '/models'
algo_args["logger"]["log_dir"] = SAVE_EVAL
algo_args["eval"]["eval_episodes"] = NUM_EVAL_EPISODES
algo_args["eval"]["dump_eval_metrcs"] = True

# Initialize the actors and environments with the chosen configurations
expt_runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)
baseline_actors = {
    "agent_ls": BaseLoadShiftingAgent(), 
    "agent_dc": BaseHVACAgent(), 
    "agent_bat": BaseBatteryAgent()
}

# Function to evaluate and store metrics
def run_evaluation(do_baseline=False):
    metrics = {
        'agent_1': [],
        'agent_2': [],
        'agent_3': []
    }
    
    expt_runner.prep_rollout()
    eval_episode = 0
    eval_obs, eval_share_obs, eval_available_actions = expt_runner.eval_envs.reset()
    
    eval_rnn_states = np.zeros(
        (
            expt_runner.algo_args["eval"]["n_eval_rollout_threads"],
            expt_runner.num_agents,
            expt_runner.recurrent_n,
            expt_runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_masks = np.ones(
        (expt_runner.algo_args["eval"]["n_eval_rollout_threads"], expt_runner.num_agents, 1),
        dtype=np.float32,
    )

    while True:
        eval_actions_collector = []
        for agent_id in range(expt_runner.num_agents):
            eval_actions, temp_rnn_state = expt_runner.actor[agent_id].act(
                eval_obs[:, agent_id],
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            if do_baseline:
                eval_actions = torch.tensor([[baseline_actors[ACTIVE_AGENTS[agent_id]].act()]])
            eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            eval_actions_collector.append(_t2n(eval_actions))

        eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = expt_runner.eval_envs.step(eval_actions)
        
        if expt_runner.dump_info:
            for i in range(expt_runner.algo_args["eval"]["n_eval_rollout_threads"]):
                # Check for keys specific to each agent
                if 'dc_ITE_total_power_kW' in eval_infos[i][0]:
                    metrics['agent_2'].append({
                        key: eval_infos[i][0].get(key, None) for key in [
                            'dc_ITE_total_power_kW', 'dc_HVAC_total_power_kW', 'dc_total_power_kW', 'dc_power_lb_kW', 
                            'dc_power_ub_kW', 'dc_crac_setpoint_delta', 'dc_crac_setpoint', 'dc_cpu_workload_fraction', 
                            'dc_int_temperature', 'dc_CW_pump_power_kW', 'dc_CT_pump_power_kW', 'dc_water_usage', 'dc_exterior_ambient_temp',
                            'outside_temp', 'day', 'hour', 'dc_average_server_temp', 'dc_average_pipe_temp', 'dc_heat_removed', 'dc_pump_power_kW', 
                        ]
                    })
                if 'ls_original_workload' in eval_infos[i][0]:
                    metrics['agent_1'].append({
                        key: eval_infos[i][0].get(key, None) for key in [
                            'ls_original_workload', 'ls_shifted_workload', 'ls_action', 'ls_norm_load_left',
                            'ls_unasigned_day_load_left', 'ls_penalty_flag', 'ls_tasks_in_queue',
                            'ls_tasks_dropped', 'ls_current_hour'
                        ]
                    })
                if 'bat_action' in eval_infos[i][0]:
                    metrics['agent_3'].append({
                        key: eval_infos[i][0].get(key, None) for key in [
                            'bat_action', 'bat_SOC', 'bat_CO2_footprint', 'bat_avg_CI', 'bat_total_energy_without_battery_KWh',
                            'bat_total_energy_with_battery_KWh', 'bat_max_bat_cap',
                            'bat_dcload_min', 'bat_dcload_max',
                        ]
                    })

        eval_dones_env = np.all(eval_dones, axis=1)

        eval_rnn_states[
            eval_dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (eval_dones_env == True).sum(),
                expt_runner.num_agents,
                expt_runner.recurrent_n,
                expt_runner.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        eval_masks = np.ones(
            (expt_runner.algo_args["eval"]["n_eval_rollout_threads"], expt_runner.num_agents, 1),
            dtype=np.float32,
        )
        eval_masks[eval_dones_env == True] = np.zeros(
            ((eval_dones_env == True).sum(), expt_runner.num_agents, 1), dtype=np.float32
        )

        for eval_i in range(expt_runner.algo_args["eval"]["n_eval_rollout_threads"]):
            if eval_dones_env[eval_i]:
                eval_episode += 1

        if eval_episode >= expt_runner.algo_args["eval"]["eval_episodes"]:
            break

    return metrics

# Run baseline
baseline_metrics = run_evaluation(do_baseline=True)

# Run trained algorithm
trained_metrics = run_evaluation(do_baseline=False)
#%
# Calculate total ITE energy consumption
trained_ite_energy = np.sum([metric['dc_ITE_total_power_kW'] for metric in trained_metrics['agent_2']])
baseline_ite_energy = np.sum([metric['dc_ITE_total_power_kW'] for metric in baseline_metrics['agent_2']])

# Calculate total HVAC energy consumption
trained_hvac_energy = np.sum([metric['dc_HVAC_total_power_kW'] for metric in trained_metrics['agent_2']])
baseline_hvac_energy = np.sum([metric['dc_HVAC_total_power_kW'] for metric in baseline_metrics['agent_2']])

# Calculate total carbon emissions
trained_carbon_emissions = np.sum([metric['bat_CO2_footprint'] for metric in trained_metrics['agent_3']])
baseline_carbon_emissions = np.sum([metric['bat_CO2_footprint'] for metric in baseline_metrics['agent_3']])

# Calculate total water usage
trained_water_usage = np.sum([metric['dc_water_usage'] for metric in trained_metrics['agent_2']])
baseline_water_usage = np.sum([metric['dc_water_usage'] for metric in baseline_metrics['agent_2']])

print(f"Summary of Comparison:")
print(f"ITE Energy Consumption: Trained: {trained_ite_energy/1e3:.3f} MWh, Baseline: {baseline_ite_energy/1e3:.3f} MWh, reduction (%): {100 * (baseline_ite_energy - trained_ite_energy) / baseline_ite_energy:.3f}")
print(f"HVAC Energy Consumption: Trained: {trained_hvac_energy/1e3:.3f} MWh, Baseline: {baseline_hvac_energy/1e3:.3f} MWh, reduction (%): {100 * (baseline_hvac_energy - trained_hvac_energy) / baseline_hvac_energy:.3f}")
print(f"Total Energy Consumption: Trained: {trained_ite_energy/1e3 + trained_hvac_energy/1e3:.3f} MWh, Baseline: {baseline_ite_energy/1e3 + baseline_hvac_energy/1e3:.3f} MWh, reduction (%): {100 * (baseline_ite_energy + baseline_hvac_energy - trained_ite_energy - trained_hvac_energy) / (baseline_ite_energy + baseline_hvac_energy):.3f}")
print(f"Carbon Emissions: Trained: {trained_carbon_emissions / 1e3:.3f} kgCO2, Baseline: {baseline_carbon_emissions / 1e3:.3f} kgCO2, reduction (%): {100 * (baseline_carbon_emissions - trained_carbon_emissions) / baseline_carbon_emissions:.3f}")
print(f"Water Usage: Trained: {trained_water_usage:.3f} liters, Baseline: {baseline_water_usage:.3f} liters, reduction (%): {100 * (baseline_water_usage - trained_water_usage) / baseline_water_usage:.3f}")


#%% Now plot the pump speed vs carbon intensity
# Assuming metrics['agent_2'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in trained_metrics['agent_2']]
carbon_intensities = [metric['bat_avg_CI'] for metric in trained_metrics['agent_3']]  # Replace with the correct key

# Define the number of points to plot
num_points = 96*10
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
carbon_intensities = carbon_intensities[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3  # Use a larger window size to smooth the data more
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_carbon_intensities = pd.Series(carbon_intensities).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(5, 3))  # Adjust height as necessary

# Plot smoothed carbon intensity
ax1.set_xlabel('Time (15-min intervals)')
ax1.set_ylabel('Carbon Intensity (gCO2/kWh)', color='tab:blue')
ax1.plot(smoothed_carbon_intensities, color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax2.plot(smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(0, len(smoothed_pump_speeds))

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_carbon_intensity.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%% Pump speed vs workload utilization
# Assuming metrics['agent_2'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in trained_metrics['agent_2']]
workload_utilizations = [metric['dc_cpu_workload_fraction'] for metric in trained_metrics['agent_2']]  # Replace with the correct key

# Define the number of points to plot
num_points = 96*10
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
workload_utilizations = workload_utilizations[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3  # Use a larger window size to smooth the data more
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_workload_utilizations = pd.Series(workload_utilizations).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(5, 3))  # Adjust height as necessary

# Plot smoothed workload utilization
ax1.set_xlabel('Time (15-min intervals)')
ax1.set_ylabel('Workload Utilization (%)', color='tab:blue')
ax1.plot(smoothed_workload_utilizations * 100, color='tab:blue', linewidth=2)  # Convert to percentage
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax2.plot(smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(0, len(smoothed_pump_speeds))

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_workload_utilization.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%% Pump speed vs workload utilization
# Assuming metrics['agent_2'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in trained_metrics['agent_2']]
outside_temps = [metric['outside_temp'] for metric in trained_metrics['agent_2']]  # Replace with the correct key

# Define the number of points to plot
num_points = 96*10
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
outside_temps = outside_temps[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3  # Use a larger window size to smooth the data more
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_outside_temps = pd.Series(outside_temps).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(5, 3))  # Adjust height as necessary

# Plot smoothed workload utilization
ax1.set_xlabel('Time (15-min intervals)')
ax1.set_ylabel('Outside Temp (°C)', color='tab:blue')
ax1.plot(smoothed_outside_temps, color='tab:blue', linewidth=2)  # Convert to percentage
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax2.plot(smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(0, len(smoothed_pump_speeds))

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_workload_utilization.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%% Now pump speed vs dc_water_usage
pump_speeds = [metric['dc_crac_setpoint'] for metric in trained_metrics['agent_2']]
water_usage = [metric['dc_water_usage'] for metric in trained_metrics['agent_2']]  # Replace with the correct key

# Define the number of points to plot
num_points = 96*10
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
outside_temps = outside_temps[init_point:init_point + num_points]

# Smooth the data using a rolling window
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_water_usage = pd.Series(water_usage).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(5, 3))  # Adjust height as necessary

# Plot smoothed water usage
ax1.set_xlabel('Time (15-min intervals)')
ax1.set_ylabel('Water Usage (l)', color='tab:blue')
ax1.plot(smoothed_water_usage, color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax2.plot(smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(0, len(smoothed_pump_speeds))

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_water_usage.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming metrics['agent_2'] and metrics['agent_3'] contain your data
outside_temps = [metric['outside_temp'] for metric in trained_metrics['agent_2']]
workload_utilizations = [metric['dc_cpu_workload_fraction'] for metric in trained_metrics['agent_2']]  # Replace with the correct key
carbon_intensities = [metric['bat_avg_CI'] for metric in trained_metrics['agent_3']]  # Replace with the correct key

# Define the number of points to plot
num_points = 96*10
init_point = 225

# Select the data to plot
outside_temps = outside_temps[init_point:init_point + num_points]
workload_utilizations = workload_utilizations[init_point:init_point + num_points]
carbon_intensities = carbon_intensities[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3
smoothed_outside_temps = pd.Series(outside_temps).rolling(window=window_size).mean().dropna()
smoothed_workload_utilizations = pd.Series(workload_utilizations).rolling(window=window_size).mean().dropna()
smoothed_carbon_intensities = pd.Series(carbon_intensities).rolling(window=window_size).mean().dropna()

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(5, 6.5), sharex=True)  # 3 subplots, 1 column

# Plot smoothed outside temperature
axs[0].set_ylabel('Outside Temp (°C)', color='tab:blue')
axs[0].plot(smoothed_outside_temps, color='tab:blue', linewidth=2)
axs[0].tick_params(axis='y', labelcolor='tab:blue')
axs[0].grid(linestyle='--')

# Plot smoothed workload utilization
axs[1].set_ylabel('Workload Utilization (%)', color='tab:green')
axs[1].plot(smoothed_workload_utilizations * 100, color='tab:green', linewidth=2)
axs[1].tick_params(axis='y', labelcolor='tab:green')
axs[1].grid(linestyle='--')

# Plot smoothed carbon intensity
axs[2].set_xlabel('Time (15-min intervals)')
axs[2].set_ylabel('Carbon Intensity (gCO2/kWh)', color='tab:red')
axs[2].plot(smoothed_carbon_intensities, color='tab:red', linewidth=2)
axs[2].tick_params(axis='y', labelcolor='tab:red')
axs[2].grid(linestyle='--')

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/operating_conditions.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming metrics['agent_2'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in trained_metrics['agent_2']]
average_server_temperatures = [metric['dc_average_server_temp'] for metric in trained_metrics['agent_2']]  # Convert from Kelvin to Celsius
average_pipe_temperatures = [metric['dc_average_pipe_temp'] for metric in trained_metrics['agent_2']]  # Convert from Kelvin to Celsius

# Define the supply temperature (fixed at 27°C)
supply_temperature = 27

# Define the number of points to plot
num_points = 96*30
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
average_server_temperatures = average_server_temperatures[init_point:init_point + num_points]
average_pipe_temperatures = average_pipe_temperatures[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_average_server_temperatures = pd.Series(average_server_temperatures).rolling(window=window_size).mean().dropna()
smoothed_average_pipe_temperatures = pd.Series(average_pipe_temperatures).rolling(window=window_size).mean().dropna()

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)  # 2 subplots, 1 column

# Plot smoothed pump speed (Action)
axs[0].set_ylabel('Pump Speed (l/s)', color='tab:red')
axs[0].plot(smoothed_pump_speeds, color='tab:red', linewidth=2)
axs[0].tick_params(axis='y', labelcolor='tab:red')
axs[0].grid(linestyle='--')

# Plot temperatures (States)
axs[1].set_xlabel('Time (15-min intervals)')
axs[1].set_ylabel('Temperature (°C)')
axs[1].plot([supply_temperature] * len(smoothed_pump_speeds), color='tab:blue', linestyle='--', linewidth=2, label='Supply Temperature (27°C)')
axs[1].plot(smoothed_average_server_temperatures, color='tab:green', linewidth=2, label='Average Server Temp')
axs[1].plot(smoothed_average_pipe_temperatures, color='tab:orange', linewidth=2, linestyle='-', label='Average Return Temp')
axs[1].tick_params(axis='y')
axs[1].grid(linestyle='--')

# Add legends
axs[1].legend(loc='upper center', bbox_to_anchor=(0.43, 1.35),
           ncol=2, fancybox=False, shadow=False)

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_temperatures.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming metrics['agent_1'] and metrics['agent_2'] contain your data
it_powers = [metric['dc_ITE_total_power_kW'] for metric in trained_metrics['agent_2']]
cooling_powers = [metric['dc_HVAC_total_power_kW'] for metric in trained_metrics['agent_2']]
carbon_footprints = [metric['bat_CO2_footprint'] for metric in trained_metrics['agent_3']]

# Define the number of points to plot
num_points = 96*10
init_point = 225

# Select the data to plot
it_powers = it_powers[init_point:init_point + num_points]
cooling_powers = cooling_powers[init_point:init_point + num_points]
carbon_footprints = carbon_footprints[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3
smoothed_it_powers = pd.Series(it_powers).rolling(window=window_size).mean().dropna()
smoothed_cooling_powers = pd.Series(cooling_powers).rolling(window=window_size).mean().dropna()
smoothed_carbon_footprints = pd.Series(carbon_footprints).rolling(window=window_size).mean().dropna()

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(5, 6.5), sharex=True)  # 3 subplots, 1 column

# Plot smoothed IT Power
axs[0].set_ylabel('IT Power (kW)', color='tab:blue')
axs[0].plot(smoothed_it_powers, color='tab:blue', linewidth=2)
axs[0].tick_params(axis='y', labelcolor='tab:blue')
axs[0].grid(linestyle='--')

# Plot smoothed Cooling Power
axs[1].set_ylabel('Cooling Power (kW)', color='tab:green')
axs[1].plot(smoothed_cooling_powers, color='tab:green', linewidth=2)
axs[1].tick_params(axis='y', labelcolor='tab:green')
axs[1].grid(linestyle='--')

# Plot smoothed Carbon Footprint
axs[2].set_xlabel('Time (15-min intervals)')
axs[2].set_ylabel('Carbon Footprint (kgCO2)', color='tab:red')
axs[2].plot(smoothed_carbon_footprints/1e3, color='tab:red', linewidth=2)
axs[2].tick_params(axis='y', labelcolor='tab:red')
axs[2].grid(linestyle='--')

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/effects_it_cooling_carbon.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming metrics['agent_1'] and metrics['agent_3'] contain your data
original_workloads = [metric['ls_original_workload'] for metric in trained_metrics['agent_1']]
shifted_workloads = [metric['ls_shifted_workload'] for metric in trained_metrics['agent_1']]
battery_soc = [metric['bat_SOC'] for metric in trained_metrics['agent_3']]

# Define the number of points to plot
num_points = 96 * 3  # Example for 3 days
init_point = 225

# Select the data to plot
original_workloads = original_workloads[init_point:init_point + num_points]
shifted_workloads = shifted_workloads[init_point:init_point + num_points]
battery_soc = battery_soc[init_point:init_point + num_points]

# Scale the data between 0 and 100, using the maximum and minimum values
battery_soc = (battery_soc - np.min(battery_soc)) / (np.max(battery_soc) - np.min(battery_soc))
 
# Smooth the data using a rolling window
window_size = 3
smoothed_original_workloads = pd.Series(original_workloads).rolling(window=window_size).mean().dropna()
smoothed_shifted_workloads = pd.Series(shifted_workloads).rolling(window=window_size).mean().dropna()
smoothed_battery_soc = pd.Series(battery_soc).rolling(window=window_size).mean().dropna()

# Create the figure with subplots
fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)  # 2 subplots, 1 column

# Top plot: Original and shifted workloads
axs[0].set_ylabel('Workload Utilization (%)')
axs[0].plot(smoothed_original_workloads*100, label='Original Workload', color='tab:blue', linewidth=2)
axs[0].plot(smoothed_shifted_workloads*100, label='Shifted Workload', color='tab:orange', linewidth=2)
axs[0].legend(loc='upper right')
axs[0].grid(linestyle='--')

# Bottom plot: Battery SOC
axs[1].set_xlabel('Time (15-min intervals)')
axs[1].set_ylabel('Battery SOC (%)')
axs[1].plot(smoothed_battery_soc * 100, label='Battery SOC', color='tab:green', linewidth=2)
axs[1].grid(linestyle='--')

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/workload_vs_battery_soc.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%% Now pump speed vs dc_water_usage


#%%
# Assuming metrics['agent_2'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in metrics['agent_2']]
return_temperatures = [metric['dc_average_pipe_temp'] for metric in metrics['agent_2']]  # Convert from Kelvin to Celsius
supply_temperature = 27  # Constant supply temperature

# Define the number of points to plot
num_points = 96*3
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
return_temperatures = return_temperatures[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3  # Use a larger window size to smooth the data more
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_return_temperatures = pd.Series(return_temperatures).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust height as necessary

# Plot smoothed return temperature
ax1.set_xlabel('Time (15-min intervals)')
ax1.set_ylabel('Temperature (°C)', color='tab:blue')
ax1.plot(smoothed_return_temperatures, color='tab:blue', linewidth=2, label='Return Temperature')
ax1.axhline(y=supply_temperature, color='tab:green', linestyle='--', linewidth=2, label='Supply Temperature')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax2.plot(smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.8, label='Pump Speed')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(0, len(smoothed_pump_speeds))

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_return_supply_temperature.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Assuming metrics['agent_2'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in metrics['agent_2']]
return_temperatures = [metric['dc_int_temperature'] - 273.15 for metric in metrics['agent_2']]  # Convert from Kelvin to Celsius
workload_utilizations = [metric['dc_cpu_workload_fraction'] * 100 for metric in metrics['agent_2']]  # Convert to percentage

# Define the number of points to plot
num_points = 96*3
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
return_temperatures = return_temperatures[init_point:init_point + num_points]
workload_utilizations = workload_utilizations[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3  # Use a rolling window to smooth the data more
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_return_temperatures = pd.Series(return_temperatures).rolling(window=window_size).mean().dropna()
smoothed_workload_utilizations = pd.Series(workload_utilizations).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 4))  # Adjust height and width as necessary

# Plot smoothed workload utilization on the left y-axis
ax1.set_xlabel('Time (15-min intervals)')
ax1.set_ylabel('Workload Utilization (%)', color='tab:blue')
ax1.plot(smoothed_workload_utilizations, color='tab:blue', linewidth=2, linestyle='-', label='Workload Utilization', alpha=0.9)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax2.plot(smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.9, label='Pump Speed')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Create a third y-axis for return temperature
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Position the third axis outward
ax3.set_ylabel('Return Temperature (°C)', color='tab:orange')
ax3.plot(smoothed_return_temperatures, color='tab:orange', linewidth=2, alpha=0.9, linestyle='--', label='Return Temperature')
ax3.tick_params(axis='y', labelcolor='tab:orange')

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(0, len(smoothed_pump_speeds))

# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_3, labels_3 = ax3.get_legend_handles_labels()

# Combine legends from all axes and place them at the top of the plot
ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='upper center', bbox_to_anchor=(0.5, 1.25),
           ncol=2, fancybox=False, shadow=False)

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_workload_vs_return_temperature.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Assuming metrics['agent_2'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in metrics['agent_2']]
cooling_power = [metric['dc_HVAC_total_power_kW'] for metric in metrics['agent_2']]
it_power = [metric['dc_ITE_total_power_kW'] for metric in metrics['agent_2']]

# Define the number of points to plot
num_points = 96*3
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
cooling_power = cooling_power[init_point:init_point + num_points]
it_power = it_power[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3  # Use a rolling window to smooth the data more
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_cooling_power = pd.Series(cooling_power).rolling(window=window_size).mean().dropna()
smoothed_it_power = pd.Series(it_power).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 4))  # Adjust height and width as necessary

# Plot smoothed pump speed on the left y-axis
ax1.set_xlabel('Time (15-min intervals)')
ax1.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax1.plot(smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.9, label='Pump Speed')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Create a second y-axis for power (cooling and IT) on the right
ax2 = ax1.twinx()
ax2.set_ylabel('Power (kW)', color='tab:blue')

# Plot smoothed cooling power
ax2.plot(smoothed_cooling_power, color='tab:blue', linewidth=2, alpha=0.9, label='Cooling Power')

# Plot smoothed IT power
ax2.plot(smoothed_it_power, color='tab:green', linewidth=2, alpha=0.9, label='IT Power')

# Adjust tick colors and labels for ax2
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(0, len(smoothed_pump_speeds))

# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

# Combine legends from both axes and place them at the top of the plot
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.25),
           ncol=3, fancybox=False, shadow=False)

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_cooling_vs_it_power.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Assuming metrics['agent_2'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in metrics['agent_2']]
supply_temperature = 27  # Constant supply temperature
average_server_temperatures = [metric['dc_average_server_temp'] for metric in metrics['agent_2']]  # Convert from Kelvin to Celsius
average_pipe_temperatures = [metric['dc_average_pipe_temp'] for metric in metrics['agent_2']]  # Convert from Kelvin to Celsius

# Define the number of points to plot
num_points = 96*3
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
average_server_temperatures = average_server_temperatures[init_point:init_point + num_points]
average_pipe_temperatures = average_pipe_temperatures[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 3  # Use a rolling window to smooth the data more
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_average_server_temperatures = pd.Series(average_server_temperatures).rolling(window=window_size).mean().dropna()
smoothed_average_pipe_temperatures = pd.Series(average_pipe_temperatures).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 4))  # Adjust height and width as necessary

# Plot smoothed inlet temperature on the left y-axis
ax1.set_xlabel('Time (15-min intervals)')
ax1.set_ylabel('Temperature (°C)', color='tab:blue')
ax1.axhline(y=supply_temperature, color='tab:blue', linestyle='-', linewidth=2, label='Supply Temperature')
ax1.plot(smoothed_average_server_temperatures, color='tab:green', linewidth=2, linestyle='-', alpha=0.9, label='Average Internal Server Temperature')
ax1.plot(smoothed_average_pipe_temperatures, color='tab:orange', linewidth=2, linestyle='--',  alpha=0.9,  label='Average Return Temperature')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed on the right
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax2.plot(smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.9, label='Pump Speed')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(0, len(smoothed_pump_speeds))

# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

# Combine legends from both axes and place them at the top of the plot
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.25),
           ncol=2, fancybox=False, shadow=False)

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/inlet_vs_server_vs_pipe_vs_pump_speed.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()


#%%
if expt_runner.dump_info:
    # Convert collected data to DataFrame and save as CSV
    expt_runner.dump_metrics_to_csv(metrics, eval_episode)
    print("Data saved to evaluation_data.csv.")

expt_runner.close()


# %%
