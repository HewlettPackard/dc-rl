#%%
import os
import sys
import copy  # Import copy to make a deep copy of metrics
import warnings 
import json
import numpy as np
warnings.filterwarnings('ignore')
from tabulate import tabulate
import matplotlib.dates as mdates

import pandas as pd

import torch
import matplotlib.pyplot as plt
sys.path.insert(0,os.getcwd())  # or sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'heterogeneous_sustaindc')))
from harl.runners import RUNNER_REGISTRY
from harl.utils.trans_tools import _t2n

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'dc-rl')))
from utils.base_agents import BaseLoadShiftingAgent, BaseHVACAgent, BaseBatteryAgent
from utils.rbc_agents import RBCLiquidAgent, RBCLoadShiftingAgent

#%%
# MODEL_PATH = 'trained_models'
# ENV = 'sustaindc'
# LOCATION = "az"
# AGENT_TYPE = "haa2c"
# RUN = "seed-00001-2024-06-04-20-41-56"
# ACTIVE_AGENTS = ['agent_ls', 'agent_dc', 'agent_bat']

baseline_actors = {
    "agent_ls": RBCLoadShiftingAgent(max_queue_length=0.8), 
    "agent_dc": RBCLiquidAgent(), 
    "agent_bat": BaseBatteryAgent()
}

# Function to evaluate and store metrics
def run_evaluation(do_baseline=False, eval_episodes=1, eval_type='random'):
    all_metrics = []
    SAVE_EVAL = "results"
    NUM_EVAL_EPISODES = 5

    # Define paths and configurations as usual
    # run = 'happo/happo_liquid_dc_64_16_4_2actions_4obs/seed-00001-2024-08-23-21-29-01'
    # run = 'happo/happo_liquid_dc_8_8_8_2actions_3obs/seed-00001-2024-08-23-21-28-19'
    # run = 'happo/happo_liquid_dc_64_64_64_2actions_4obs/seed-00001-2024-08-23-21-25-26' # Looks good, but bad performance -0.6%
    # run = 'happo/happo_liquid_dc_64_64_2actions_4obs/seed-00001-2024-08-23-18-53-06' # Looks good, 1.27% energy reduction
    # run = 'happo/happo_liquid_dc_16_16_2actions_4obs/seed-00001-2024-08-23-18-52-40' # Looks good, 0.89% energy reduction
    # run = 'happo/happo_liquid_dc_64_64_2actions_4obs_water/seed-00001-2024-08-26-15-07-24'
    # run = 'happo/happo_liquid_dc_64_64_2actions_4obs/seed-00001-2024-08-26-13-13-12'
    # run = 'happo/happo_liquid_dc_64_64_2actions_4obs/seed-00001-2024-08-26-13-14-45'
    # run = 'happo/happo_liquid_dc_64_64_2actions_4obs_water/seed-00001-2024-08-26-15-07-24'
    # run = 'happo/happo_liquid_dc_64_64_water_std/seed-00001-2024-08-26-16-51-09'
    # run = 'hatrpo/hatrpo_liquid_dc_256_256_2actions_4obs_000001/seed-00001-2024-08-26-21-13-20'
    # run = 'happo/happo_liquid_dc_256_256_2actions_4obs/seed-00004-2024-08-26-21-25-18' # Looks good, 1.51% energy reduction
    # run = 'happo/happo_liquid_dc_256_256_2actions_4obs/seed-00002-2024-08-26-21-24-52' # 1.19%
    # run = "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00003-2024-08-26-21-25-05" # 0.95%
    # run = "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00005-2024-08-26-21-25-32"
    # run = 'happo/happo_liquid_dc_64_64_2actions_4obs_2stk/seed-00002-2024-08-27-15-28-04' # 1.25$
    # run = 'happo/happo_liquid_dc_256_256_2actions_5obs_range_t_i_sigmoid_nonormalization_default_values/seed-00002-2024-08-30-04-32-21'
    # run = 'happo/happo_liquid_dc_recovering_old_data/seed-00002-2024-09-02-23-33-17'
    # run = 'happo/debug_ls_64_64/seed-00100-2024-09-09-23-19-56' # 0.663%
    # run = 'happo/debug_ls_2_debugmode_discrete_32_16_only_shiftable/seed-03219-2024-09-15-15-58-05' # 26.82% # Using the Sin CI
    # run = 'happo/random_run_0_20240915_175950_discrete_nopenalty/seed-06721-2024-09-15-17-59-51' # 24.04%
    # run = 'happo/debug_ls_7_debugmode_discrete_32_16_only_shiftable_1threads/seed-07572-2024-09-15-17-57-04' # 22.98%
    # run = 'happo/debug_ls_4_onlyhour_newreward/seed-05735-2024-09-16-21-52-16' # 8.47% Reduction
    # run = 'happo/debug_ls_5_new_obs_expanded_randomhour/seed-05415-2024-09-16-22-48-16' # 6.693% ± 0.028 Reduction
    # run = 'happo/debug_ls_9_new_reward_5timessqrtplus1_8threads_128eplen_8numminibatch_taskhistogram_weatherinfo_0000001actorentr/seed-02195-2024-09-18-23-28-55' # 
    run = 'happo/debug_ls_new_reward_5timessqrtplus1_8threads_128eplen_8numminibatch_taskhistogram_weatherinfo_001actorentr/seed-01321-2024-09-18-23-25-57' # Reduction (%): 8.840 ± 0.337
    
    path = f'/lustre/guillant/sustaindc/results/sustaindc/ca/{run}'
    with open(path + '/config.json', encoding='utf-8') as file:
        saved_config = json.load(file)

    algo_args, env_args, main_args = saved_config['algo_args'], saved_config['env_args'], saved_config['main_args']

    algo_args['train']['n_rollout_threads'] = 1
    algo_args['eval']['n_eval_rollout_threads'] = 1
    algo_args['train']['model_dir'] = path + '/models'
    algo_args["logger"]["log_dir"] = SAVE_EVAL
    algo_args["eval"]["eval_episodes"] = NUM_EVAL_EPISODES
    algo_args["eval"]["dump_eval_metrcs"] = True
    env_args['days_per_episode'] = 7
    env_args['location'] = 'ca'
    env_args['initialize_queue_at_reset'] = True
    algo_args['seed']['seed_specify'] = True
    algo_args['seed']['seed'] = 0

    # Initialize the actors and environments with the chosen configurations
    expt_runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)

    # Get the active agents from the environment arguments
    active_agents = expt_runner.env_args['agents']

    for run_i in range(eval_episodes):
        # Initialize metrics for only the active agents
        metrics = {agent: [] for agent in baseline_actors}
        metrics['global'] = []
        
        expt_runner.prep_rollout()
        eval_episode = 0
        # Set the seed for the environment
        expt_runner.eval_envs.envs[0].env.env.seed(run_i)
    
        # Now reset the environment
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
            for agent_id, agent_name in enumerate(active_agents):
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
                    # Extract the workload from the eval_infos is available
                    if agent_name == 'agent_ls':
                        if eval_type == 'random':
                            supply_temp = np.random.choice([0, 1, 2], p=[0.25, 0.25, 0.5])
                            eval_actions = torch.tensor([[supply_temp]])
                        elif eval_type == 'fixed':
                            supply_temp = 1
                            eval_actions = torch.tensor([[supply_temp]])
                        elif eval_type == 'following_ci':
                            workload = expt_runner.eval_envs.envs[0].env.env.workload_m.get_next_workload()
                            # For the RBC I need this: current_ci, ci_forecast, tasks_in_queue, oldest_task_age, queue_max_len
                            # obs = eval_obs[:, agent_id]
                            # hour_sin_cos = obs[:, 0:2][0][0]
                            # current_workload = obs[:, 2][0]
                            # queue_status = obs[:, 3][0]
                            # ci_future = obs[:, 4:12][0]
                            # current_ci = obs[:, 12][0]
                            # ci_past = obs[:, 13:17][0]
                            # next_workload = obs[:, 17][0]
                            # current_out_temperature = obs[:, 18][0]
                            # next_out_temperature = obs[:, 19][0]
                            # oldest_task_age = obs[:, 20][0]
                            # average_task_age = obs[:, 21][0]
                            # action = baseline_actors[agent_name].act(current_ci, ci_future, queue_status, oldest_task_age, 0.8)
                            action = baseline_actors[agent_name].act(eval_obs[:, agent_id][0])
                            eval_actions = torch.tensor([[action]]) #torch.tensor([[0.25]])
                    else:
                        eval_actions = torch.tensor([[baseline_actors[agent_name].act()]])
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
                        metrics['agent_dc'].append({
                            key: eval_infos[i][0].get(key, None) for key in [
                                'dc_ITE_total_power_kW', 'dc_HVAC_total_power_kW', 'dc_total_power_kW', 'dc_power_lb_kW', 
                                'dc_power_ub_kW', 'dc_crac_setpoint_delta', 'dc_crac_setpoint', 'dc_cpu_workload_fraction', 
                                'dc_int_temperature', 'dc_CW_pump_power_kW', 'dc_CT_pump_power_kW', 'dc_water_usage', 'dc_exterior_ambient_temp',
                                'outside_temp', 'day', 'hour', 'dc_average_server_temp', 'dc_average_pipe_temp', 'dc_heat_removed', 'dc_pump_power_kW', 
                                'dc_coo_m_flow_nominal', 'dc_coo_mov_flow_actual', 'dc_supply_liquid_temp', 'dc_return_liquid_temp'
                            ]
                        })
                    if 'ls_original_workload' in eval_infos[i][0]:
                        metrics['agent_ls'].append({
                            key: eval_infos[i][0].get(key, None) for key in [
                                'ls_original_workload', 'ls_shifted_workload', 'ls_action', 'ls_norm_load_left',
                                'ls_unasigned_day_load_left', 'ls_penalty_flag', 'ls_tasks_in_queue', 'ls_norm_tasks_in_queue',
                                'ls_tasks_dropped', 'ls_current_hour', 'ls_overdue_penalty', 'ls_computed_tasks',
                                'ls_oldest_task_age', 'ls_average_task_age',
                            ]
                        })
                    if 'bat_action' in eval_infos[i][0]:
                        metrics['agent_bat'].append({
                            key: eval_infos[i][0].get(key, None) for key in [
                                'bat_action', 'bat_SOC', 'bat_CO2_footprint', 'bat_avg_CI', 'bat_total_energy_without_battery_KWh',
                                'bat_total_energy_with_battery_KWh', 'bat_max_bat_cap',
                                'bat_dcload_min', 'bat_dcload_max',
                            ]
                        })
                    metrics['global'].append({'reward': eval_rewards[0][0][0]})

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
            
        all_metrics.append(metrics)
        
    return all_metrics

# Run baseline
num_runs = 1
# baseline_random_metrics_runs = run_evaluation(do_baseline=True, eval_type='random', eval_episodes=num_runs)
baseline_fixed_metrics_runs = run_evaluation(do_baseline=True, eval_type='fixed', eval_episodes=num_runs)
# baseline_metrics_runs = run_evaluation(do_baseline=True, eval_type='following_ci', eval_episodes=num_runs)

# Run trained algorithm
trained_metrics_runs = run_evaluation(do_baseline=False, eval_episodes=num_runs)

#%%
def calculate_total_computed_tasks(metrics_runs, run_type):
    total_computed_tasks_per_run = []
    for run_index, run_metrics in enumerate(metrics_runs):
        total_computed_tasks = 0
        for step_metrics in run_metrics['agent_ls']:
            ls_computed_tasks = step_metrics.get('ls_computed_tasks', 0)
            if ls_computed_tasks is not None:
                total_computed_tasks += ls_computed_tasks
        total_computed_tasks_per_run.append(total_computed_tasks)
        print(f"{run_type} Run {run_index + 1}: Total computed tasks: {total_computed_tasks}")
    return total_computed_tasks_per_run
    
#%
# Calculate total computed tasks for baseline runs
baseline_total_computed_tasks = calculate_total_computed_tasks(baseline_fixed_metrics_runs, "Baseline")

# Calculate total computed tasks for trained runs
trained_total_computed_tasks = calculate_total_computed_tasks(trained_metrics_runs, "Trained")

#%
#% Calculate average and standard deviation
# Calculate absolute values and reduction for each run
def calculate_values_and_reduction(trained_metrics_runs, baseline_metrics_runs, metric_name):
    trained_totals = []
    baseline_totals = []
    reductions = []

    for trained_metrics, baseline_metrics in zip(trained_metrics_runs, baseline_metrics_runs):
        if metric_name in trained_metrics['agent_dc'][0]:
            trained_values = [metric[metric_name] for metric in trained_metrics['agent_dc']]
            baseline_values = [metric[metric_name] for metric in baseline_metrics['agent_dc']]
        elif metric_name in trained_metrics['agent_bat'][0]:
            trained_values = [metric[metric_name] for metric in trained_metrics['agent_bat']]
            baseline_values = [metric[metric_name] for metric in baseline_metrics['agent_bat']]
        else:
            trained_values = [metric[metric_name] for metric in trained_metrics['agent_ls']]
            baseline_values = [metric[metric_name] for metric in baseline_metrics['agent_ls']]

        # Sum the values for the entire run
        trained_total = sum(trained_values)/1e6
        baseline_total = sum(baseline_values)/1e6

        # Calculate reduction percentage for this run
        reduction = 100 * (baseline_total - trained_total) / baseline_total
        reductions.append(reduction)
        trained_totals.append(trained_total)
        baseline_totals.append(baseline_total)

    # Calculate the mean and standard deviation of the reductions
    return {
        'trained_avg': np.mean(trained_totals),
        'trained_std': np.std(trained_totals),
        'baseline_avg': np.mean(baseline_totals),
        'baseline_std': np.std(baseline_totals),
        'reduction_avg': np.mean(reductions),
        'reduction_std': np.std(reductions)
    }

# compare the trained vs fixed baseline for energy and carbon emissions
energy_results_fixed = calculate_values_and_reduction(trained_metrics_runs, baseline_fixed_metrics_runs, 'bat_total_energy_without_battery_KWh')
co2_results_fixed = calculate_values_and_reduction(trained_metrics_runs, baseline_fixed_metrics_runs, 'bat_CO2_footprint')
water_results_fixed = calculate_values_and_reduction(trained_metrics_runs, baseline_fixed_metrics_runs, 'dc_water_usage')

print(f"Fixed Baseline - Total Energy: {energy_results_fixed['baseline_avg']:.3f} ± {energy_results_fixed['baseline_std']:.3f}")
print(f"Trained - Total Energy: {energy_results_fixed['trained_avg']:.3f} ± {energy_results_fixed['trained_std']:.3f}")
print(f"Reduction (%): {energy_results_fixed['reduction_avg']:.3f} ± {energy_results_fixed['reduction_std']:.3f}")

print(f"Fixed Baseline - Carbon Emissions: {co2_results_fixed['baseline_avg']:.3f} ± {co2_results_fixed['baseline_std']:.3f}")
print(f"Trained - Carbon Emissions: {co2_results_fixed['trained_avg']:.3f} ± {co2_results_fixed['trained_std']:.3f}")
print(f"Reduction (%): {co2_results_fixed['reduction_avg']:.3f} ± {co2_results_fixed['reduction_std']:.3f}")


# print(f"Total Energy Consumption: Reduction (%): {energy_reduction:.3f} ± {energy_std:.3f}")
# print(f"Carbon Emissions: Reduction (%): {co2_reduction:.3f} ± {co2_std:.3f}")
# print(f"Water Usage: Reduction (%): {water_reduction:.3f} ± {water_std:.3f}")


#%% Now Plot the original workload (ls_original_workload) vs the shifted workload (ls_shifted_workload) in one y-axis, and in the other y-axis the carbon intensity
trained_metrics = trained_metrics_runs[0]
original_workloads = [metric['ls_original_workload'] for metric in trained_metrics['agent_ls']]
shifted_workloads = [metric['ls_shifted_workload']  + metric['ls_original_workload']*0.4 for metric in trained_metrics['agent_ls']]
carbon_intensities = [metric['bat_avg_CI'] for metric in trained_metrics['agent_bat']]

# Extract also the day and the hour to plot the data
days = [metric['day'] for metric in trained_metrics['agent_dc']]
hours = [metric['hour'] for metric in trained_metrics['agent_dc']]

# Days enconde the number of days from January 1st, 2024
# Convert the days to a date

# Now include the hours in the days
time_intervals = [pd.to_datetime('2024-01-01') + pd.Timedelta(days=day) + pd.Timedelta(hours=hour) for day, hour in zip(days, hours)]

# Define the number of points to plot
num_points = 96*7
init_point = 0

# Select the data to plot
original_workloads = original_workloads[init_point:init_point + num_points]
shifted_workloads = shifted_workloads[init_point:init_point + num_points]
carbon_intensities = carbon_intensities[init_point:init_point + num_points]
time_intervals = time_intervals[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_original_workloads = pd.Series(original_workloads).rolling(window=window_size).mean().dropna()
smoothed_shifted_workloads = pd.Series(shifted_workloads).rolling(window=window_size).mean().dropna()
smoothed_carbon_intensities = pd.Series(carbon_intensities).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed original workload
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Workload (%)', color='tab:blue')
ax1.plot(time_intervals, smoothed_original_workloads*100, color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot smoothed shifted workload in the same plot
ax1.plot(time_intervals, smoothed_shifted_workloads*100, color='tab:red', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:red')

# Plot the carbon intensity in a second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Carbon Intensity (gCO2/kWh)', color='tab:green')
ax2.plot(time_intervals, smoothed_carbon_intensities, color='tab:green', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:green')

# Only show the x-axis labels for every 2 days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

# Show the grid
ax1.grid(linestyle='--')

ax1.set_ylim(0, 100)

#%% Now Plot the original workload (ls_original_workload) vs the shifted workload (ls_shifted_workload) in one y-axis, and in the other y-axis the ambient temperature
original_workloads = [metric['ls_original_workload'] for metric in trained_metrics['agent_ls']]
shifted_workloads = [metric['ls_shifted_workload'] + metric['ls_original_workload']*0.4 for metric in trained_metrics['agent_ls']]
ambient_temperatures = [metric['dc_exterior_ambient_temp'] for metric in trained_metrics['agent_dc']]

# Select the data to plot
original_workloads = original_workloads[init_point:init_point + num_points]
shifted_workloads = shifted_workloads[init_point:init_point + num_points]
ambient_temperatures = ambient_temperatures[init_point:init_point + num_points]
time_intervals = time_intervals[init_point:init_point + num_points]
# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_original_workloads = pd.Series(original_workloads).rolling(window=window_size).mean().dropna()
smoothed_shifted_workloads = pd.Series(shifted_workloads).rolling(window=window_size).mean().dropna()
smoothed_ambient_temperatures = pd.Series(ambient_temperatures).rolling(window=window_size).mean().dropna()
# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary
# Plot smoothed original workload
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Workload (%)', color='tab:blue')
ax1.plot(time_intervals, smoothed_original_workloads*100, color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')
# Plot smoothed shifted workload in the same plot
ax1.plot(time_intervals, smoothed_shifted_workloads*100, color='tab:red', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:red')
# Plot the ambient temperature in a second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Ambient Temperature (°C)', color='tab:green')
ax2.plot(time_intervals, smoothed_ambient_temperatures, color='tab:green', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:green')
# Only show the x-axis labels for every 2 days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
# Show the grid
ax1.grid(linestyle='--')
ax1.set_ylim

#%% Now plot the tasks in queue vs the shifted workload in two differents y axis
tasks_in_queue = [metric['ls_tasks_in_queue'] for metric in trained_metrics['agent_ls']]
shifted_workloads = [metric['ls_shifted_workload'] + metric['ls_original_workload']*0.4  for metric in trained_metrics['agent_ls']]
# Define the number of points to plot
# num_points = 96*7
# init_point = 0

# Select the data to plot
tasks_in_queue = tasks_in_queue[init_point:init_point + num_points]
shifted_workloads = shifted_workloads[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_tasks_in_queue = pd.Series(tasks_in_queue).rolling(window=window_size).mean().dropna()
smoothed_shifted_workloads = pd.Series(shifted_workloads).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed tasks in queue
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Tasks in Queue', color='tab:blue')
ax1.plot(time_intervals, smoothed_tasks_in_queue, color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for shifted workload
ax2 = ax1.twinx()
ax2.set_ylabel('Workload (%)', color='tab:red')
ax2.plot(time_intervals, smoothed_shifted_workloads*100, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 100)  # Adjust the limits to match the scale of the data
# Only show the x-axis labels for every 2 days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

# Show the grid
ax1.grid(linestyle='--')


# Customize the layout to ensure no parts are cut off
plt.tight_layout()


#%% Now Plot the ls_action vs the carbon intensity
trained_metrics = trained_metrics_runs[0]
ls_actions = [metric['ls_action'] for metric in trained_metrics['agent_ls']]
# ls_actions = np.clip(ls_actions, -1, 1)
carbon_intensities = [metric['bat_avg_CI'] for metric in trained_metrics['agent_bat']]
# Define the number of points to plot
# num_points = 96*7
# init_point = 0

# Select the data to plot
ls_actions = ls_actions[init_point:init_point + num_points]
carbon_intensities = carbon_intensities[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_ls_actions = pd.Series(ls_actions).rolling(window=window_size).mean().dropna()
smoothed_carbon_intensities = pd.Series(carbon_intensities).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed ls_action
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Load Shifting Action', color='tab:blue')
ax1.plot(time_intervals, smoothed_ls_actions, color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for carbon intensity
ax2 = ax1.twinx()
ax2.set_ylabel('Carbon Intensity (gCO2/kWh)', color='tab:red')
ax2.plot(time_intervals, smoothed_carbon_intensities, color='tab:red', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:red')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
# Show the grid
ax1.grid(linestyle='--')

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# %% Now plot the number of overdue tasks, the average tasks age and the maximum task age
overdue_tasks = [metric['ls_overdue_penalty'] for metric in trained_metrics['agent_ls']]
average_task_age = [metric['ls_average_task_age'] for metric in trained_metrics['agent_ls']]
oldest_task_age = [metric['ls_oldest_task_age'] for metric in trained_metrics['agent_ls']]

# Define the number of points to plot
# num_points = 96*7
# init_point = 0

# Select the data to plot
overdue_tasks = overdue_tasks[init_point:init_point + num_points]
average_task_age = average_task_age[init_point:init_point + num_points]
oldest_task_age = oldest_task_age[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_overdue_tasks = pd.Series(overdue_tasks).rolling(window=window_size).mean().dropna()
smoothed_average_task_age = pd.Series(average_task_age).rolling(window=window_size).mean().dropna()
smoothed_oldest_task_age = pd.Series(oldest_task_age).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed overdue tasks
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Overdue Tasks', color='tab:blue')
ax1.plot(time_intervals, smoothed_overdue_tasks, color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Only show the x-axis labels for every 2 days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

ax2 = ax1.twinx()
ax2.set_ylabel('Average Task Age (Days)', color='tab:red')
ax2.plot(time_intervals, smoothed_average_task_age, color='tab:red', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:red')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('Oldest Task Age (Days)', color='tab:green')
ax3.plot(time_intervals, smoothed_oldest_task_age, color='tab:green', linewidth=2)
ax3.tick_params(axis='y', labelcolor='tab:green')

# Show the grid
ax1.grid(linestyle='--')
#%% Pump speed vs workload utilization
# Assuming metrics['agent_dc'] contains your data
pump_speeds = [metric['dc_coo_mov_flow_actual'] for metric in trained_metrics['agent_dc']]
workload_utilizations = [metric['dc_cpu_workload_fraction'] for metric in trained_metrics['agent_dc']]  # Replace with the correct key

# Define the number of points to plot
num_points = 96*7
init_point = 1300

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
workload_utilizations = workload_utilizations[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_workload_utilizations = pd.Series(workload_utilizations).rolling(window=window_size).mean().dropna()

# Assuming num_points corresponds to 96*7 which represents a week's worth of 15-min intervals
time_intervals = pd.date_range(start="2024-08-01", periods=len(smoothed_pump_speeds), freq="15T")

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed workload utilization
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Workload Utilization (%)', color='tab:blue')
ax1.plot(time_intervals, smoothed_workload_utilizations * 100, color='tab:blue', linewidth=2)  # Convert to percentage
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax2.plot(time_intervals, smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Format the x-axis to show days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days

# Rotate the date labels for better readability
plt.xticks(rotation=45, ha='right')
# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(time_intervals[0], time_intervals[-1])

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_workload_utilization_xl.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%% Supply temperature vs workload utilization
# Assuming metrics['agent_dc'] contains your data
supply_temperatures = [metric['dc_supply_liquid_temp'] for metric in trained_metrics['agent_dc']]
workload_utilizations = [metric['dc_cpu_workload_fraction'] for metric in trained_metrics['agent_dc']]  # Replace with the correct key

# Define the number of points to plot
# num_points = 96*10
# init_point = 200

# Select the data to plot
supply_temperatures = supply_temperatures[init_point:init_point + num_points]
workload_utilizations = workload_utilizations[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_supply_temperatures = pd.Series(supply_temperatures).rolling(window=window_size).mean().dropna()
smoothed_workload_utilizations = pd.Series(workload_utilizations).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed workload utilization
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Workload Utilization (%)', color='tab:blue')
ax1.plot(time_intervals, smoothed_workload_utilizations * 100, color='tab:blue', linewidth=2)  # Convert to percentage
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for supply temperature
ax2 = ax1.twinx()
ax2.set_ylabel('Supply Temperature (°C)', color='tab:red')
ax2.plot(time_intervals, smoothed_supply_temperatures, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Format the x-axis to show days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days

# Rotate the date labels for better readability
plt.xticks(rotation=45, ha='right')
# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(time_intervals[0], time_intervals[-1])

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/supply_temp_vs_workload_utilization_xl.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%% Pump speed vs outdoor temperature
# Assuming metrics['agent_dc'] contains your data
pump_speeds = [metric['dc_coo_mov_flow_actual'] for metric in trained_metrics['agent_dc']]
outside_temps = [metric['outside_temp'] for metric in trained_metrics['agent_dc']]  # Replace with the correct key

# Define the number of points to plot
# num_points = 96*10
# init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
outside_temps = outside_temps[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_outside_temps = pd.Series(outside_temps).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed outdoor temperature
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Outside Temp (°C)', color='tab:blue')
ax1.plot(time_intervals, smoothed_outside_temps, color='tab:blue', linewidth=2)  # Convert to percentage
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='tab:red')
ax2.plot(time_intervals, smoothed_pump_speeds, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Format the x-axis to show days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days

# Rotate the date labels for better readability
plt.xticks(rotation=45, ha='right')
# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(time_intervals[0], time_intervals[-1])

# Customize the layout to ensure no parts are cut off
plt.tight_layout()


# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_outside_temp_xl.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%% Now the Supply temperature vs outdoor temperature
# Assuming metrics['agent_dc'] contains your data
supply_liquid_temp = [metric['dc_supply_liquid_temp'] for metric in trained_metrics['agent_dc']]
outside_temps = [metric['outside_temp'] for metric in trained_metrics['agent_dc']]  # Replace with the correct key

# Define the number of points to plot
# num_points = 96*10
# init_point = 225

# Select the data to plot
supply_liquid_temp = supply_liquid_temp[init_point:init_point + num_points]
outside_temps = outside_temps[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_supply_liquid_temp = pd.Series(supply_liquid_temp).rolling(window=window_size).mean().dropna()
smoothed_outside_temps = pd.Series(outside_temps).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed outdoor temperature
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Outside Temp (°C)', color='tab:blue')
ax1.plot(time_intervals, smoothed_outside_temps, color='tab:blue', linewidth=2)  # Convert to percentage
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for supply temperature
ax2 = ax1.twinx()
ax2.set_ylabel('Supply Temperature (°C)', color='tab:red')
ax2.plot(time_intervals, smoothed_supply_liquid_temp, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')

# Format the x-axis to show days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days

# Rotate the date labels for better readability
plt.xticks(rotation=45, ha='right')
# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(time_intervals[0], time_intervals[-1])

# Customize the layout to ensure no parts are cut off
plt.tight_layout()


# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/supply_temp_vs_outside_temp_xl.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()


#%% Now plot the pump speed vs carbon intensity
# Assuming metrics['agent_dc'] contains your data
pump_speeds = [metric['dc_coo_mov_flow_actual'] for metric in trained_metrics['agent_dc']]
carbon_intensities = [metric['bat_avg_CI'] for metric in trained_metrics['agent_bat']]  # Replace with the correct key

# Define the number of points to plot
num_points = 96*10
init_point = 225

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
carbon_intensities = carbon_intensities[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
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

#%% Now plot the supply temperature vs carbon intensity
# Assuming metrics['agent_dc'] contains your data
supply_liquid_temp = [metric['dc_supply_liquid_temp'] for metric in trained_metrics['agent_dc']]
carbon_intensities = [metric['bat_avg_CI'] for metric in trained_metrics['agent_bat']]  # Replace with the correct key

# Define the number of points to plot
num_points = 96*10
init_point = 225

# Select the data to plot
supply_liquid_temp = supply_liquid_temp[init_point:init_point + num_points]
carbon_intensities = carbon_intensities[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_supply_liquid_temp = pd.Series(supply_liquid_temp).rolling(window=window_size).mean().dropna()
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
ax2.set_ylabel('Supply Temperature (°C)', color='tab:red')
ax2.plot(smoothed_supply_liquid_temp, color='tab:red', linewidth=2, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(0, len(smoothed_supply_liquid_temp))

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/pump_speed_vs_carbon_intensity.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

#%% Now pump speed vs dc_water_usage
pump_speeds = [metric['dc_crac_setpoint'] for metric in trained_metrics['agent_dc']]
water_usage = [metric['dc_water_usage'] for metric in trained_metrics['agent_dc']]  # Replace with the correct key

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

# Assuming metrics['agent_dc'] and metrics['agent_bat'] contain your data
outside_temps = [metric['outside_temp'] for metric in trained_metrics['agent_dc']]
workload_utilizations = [metric['dc_cpu_workload_fraction'] for metric in trained_metrics['agent_dc']]  # Replace with the correct key
carbon_intensities = [metric['bat_avg_CI'] for metric in trained_metrics['agent_bat']]  # Replace with the correct key

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

# Assuming metrics['agent_dc'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in trained_metrics['agent_dc']]
average_server_temperatures = [metric['dc_average_server_temp'] for metric in trained_metrics['agent_dc']]  # Convert from Kelvin to Celsius
average_pipe_temperatures = [metric['dc_average_pipe_temp'] for metric in trained_metrics['agent_dc']]  # Convert from Kelvin to Celsius
outside_temps = [metric['outside_temp'] for metric in trained_metrics['agent_dc']]  # Convert from Kelvin to Celsius

# Supply temperature
supply_temperature = [metric['dc_supply_liquid_temp'] for metric in trained_metrics['agent_dc']]  # Convert from Kelvin to Celsius

# Define the number of points to plot
num_points = 96*3
init_point = 250

# Select the data to plot
pump_speeds = pump_speeds[init_point:init_point + num_points]
average_server_temperatures = average_server_temperatures[init_point:init_point + num_points]
average_pipe_temperatures = average_pipe_temperatures[init_point:init_point + num_points]
supply_temperature = supply_temperature[init_point:init_point + num_points]

time_intervals = pd.date_range(start="2024-08-01", periods=len(supply_temperature), freq="15T")

# Smooth the data using a rolling window
window_size = 1
smoothed_pump_speeds = pd.Series(pump_speeds).rolling(window=window_size).mean().dropna()
smoothed_average_server_temperatures = pd.Series(average_server_temperatures).rolling(window=window_size).mean().dropna()
smoothed_average_pipe_temperatures = pd.Series(average_pipe_temperatures).rolling(window=window_size).mean().dropna()
smoothed_supply_temperature = pd.Series(supply_temperature).rolling(window=window_size).mean().dropna()

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
axs[1].plot(smoothed_supply_temperature, color='tab:blue', linestyle='-', linewidth=2, label='Supply Temp')
axs[1].plot(smoothed_average_server_temperatures, color='tab:orange', linewidth=2, label='Average Server Temp')
axs[1].plot(smoothed_average_pipe_temperatures, color='tab:green', linewidth=2, linestyle='-', label='Average Return Temp')
# axs[1].plot(time_intervals, outside_temps[init_point:init_point + num_points], color='tab:red', linewidth=2, linestyle='-', label='Outside Temp')
axs[1].tick_params(axis='y')
axs[1].grid(linestyle='--')
# axs[1].set_xlim(time_intervals[0], time_intervals[-1])

# Add legends
axs[1].legend(loc='upper center', bbox_to_anchor=(0.49, 1.3), ncol=2, fancybox=False, shadow=False)

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
plt.savefig('media/pump_speed_vs_temperatures_IAAI_3days.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming metrics['agent_ls'] and metrics['agent_dc'] contain your data
it_powers = [metric['dc_ITE_total_power_kW'] for metric in trained_metrics['agent_dc']]
cooling_powers = [metric['dc_HVAC_total_power_kW'] for metric in trained_metrics['agent_dc']]
carbon_footprints = [metric['bat_CO2_footprint'] for metric in trained_metrics['agent_bat']]

# Define the number of points to plot
num_points = 96*1
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

# Assuming metrics['agent_ls'] and metrics['agent_bat'] contain your data
original_workloads = [metric['ls_original_workload'] for metric in trained_metrics['agent_ls']]
shifted_workloads = [metric['ls_shifted_workload'] for metric in trained_metrics['agent_ls']]
battery_soc = [metric['bat_SOC'] for metric in trained_metrics['agent_bat']]

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
# Assuming metrics['agent_dc'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in metrics['agent_dc']]
return_temperatures = [metric['dc_average_pipe_temp'] for metric in metrics['agent_dc']]  # Convert from Kelvin to Celsius
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

# Assuming metrics['agent_dc'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in metrics['agent_dc']]
return_temperatures = [metric['dc_int_temperature'] - 273.15 for metric in metrics['agent_dc']]  # Convert from Kelvin to Celsius
workload_utilizations = [metric['dc_cpu_workload_fraction'] * 100 for metric in metrics['agent_dc']]  # Convert to percentage

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

# Assuming metrics['agent_dc'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in metrics['agent_dc']]
cooling_power = [metric['dc_HVAC_total_power_kW'] for metric in metrics['agent_dc']]
it_power = [metric['dc_ITE_total_power_kW'] for metric in metrics['agent_dc']]

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

# Assuming metrics['agent_dc'] contains your data
pump_speeds = [metric['dc_crac_setpoint'] for metric in metrics['agent_dc']]
supply_temperature = 27  # Constant supply temperature
average_server_temperatures = [metric['dc_average_server_temp'] for metric in metrics['agent_dc']]  # Convert from Kelvin to Celsius
average_pipe_temperatures = [metric['dc_average_pipe_temp'] for metric in metrics['agent_dc']]  # Convert from Kelvin to Celsius

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








#%% Plot the baselines
import matplotlib.dates as mdates
baseline_random = baseline_random_metrics_runs[0]
baseline_fixed = baseline_fixed_metrics_runs[0]
baseline_following = baseline_metrics_runs[0]
trained_metrics = trained_metrics_runs[0]

pump_speeds_random = [metric['dc_crac_setpoint'] for metric in baseline_random['agent_dc']]
pump_speeds_fixed = [metric['dc_crac_setpoint'] for metric in baseline_fixed['agent_dc']]
pump_speeds_following = [metric['dc_crac_setpoint'] for metric in baseline_following['agent_dc']]
workload_utilizations = [metric['dc_cpu_workload_fraction'] for metric in trained_metrics['agent_dc']]  # Replace with the correct key

# Define the number of points to plot
num_points = 96*7
init_point = 1300

# Select the data to plot
pump_speeds_random = pump_speeds_random[init_point:init_point + num_points]
pump_speeds_fixed = pump_speeds_fixed[init_point:init_point + num_points]
pump_speeds_following = pump_speeds_following[init_point:init_point + num_points]
workload_utilizations = workload_utilizations[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_pump_speeds_random = pd.Series(pump_speeds_random).rolling(window=window_size).mean().dropna()
smoothed_pump_speeds_fixed = pd.Series(pump_speeds_fixed).rolling(window=window_size).mean().dropna()
smoothed_pump_speeds_following = pd.Series(pump_speeds_following).rolling(window=window_size).mean().dropna()
smoothed_workload_utilizations = pd.Series(workload_utilizations).rolling(window=window_size).mean().dropna()

# Assuming num_points corresponds to 96*7 which represents a week's worth of 15-min intervals
time_intervals = pd.date_range(start="2024-08-01", periods=len(smoothed_pump_speeds_random), freq="15T")

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed workload utilization
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Workload Utilization (%)', color='tab:blue')
ax1.plot(time_intervals, smoothed_workload_utilizations * 100, color='tab:blue', linewidth=2)  # Convert to percentage
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='k')
ax2.plot(time_intervals, smoothed_pump_speeds_random, color='tab:orange', linewidth=2, alpha=0.3, label='Random')
ax2.plot(time_intervals, smoothed_pump_speeds_fixed, color='tab:green', linewidth=2, alpha=0.7, label='ASHRAE W32')
ax2.plot(time_intervals, smoothed_pump_speeds_following, color='tab:red', linewidth=2, alpha=0.9, label='RBC', linestyle='--')
ax2.tick_params(axis='y', labelcolor='k')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Add the legend of ax2 with title 'Baseline'
ax2.legend(loc='best', title='Baseline')
# Format the x-axis to show days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days

# Rotate the date labels for better readability
plt.xticks(rotation=45, ha='right')
# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(time_intervals[0], time_intervals[-1])

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/baselines_pump_speed_vs_workload_utilization_XL.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()

# %% Now baselines of supply temperature vs workload utilization
pump_speeds_random = [metric['dc_supply_liquid_temp'] for metric in baseline_random['agent_dc']]
pump_speeds_fixed = [metric['dc_supply_liquid_temp'] for metric in baseline_fixed['agent_dc']]
pump_speeds_following = [metric['dc_supply_liquid_temp'] for metric in baseline_following['agent_dc']]
workload_utilizations = [metric['dc_cpu_workload_fraction'] for metric in trained_metrics['agent_dc']]

# Select the data to plot
pump_speeds_random = pump_speeds_random[init_point:init_point + num_points]
pump_speeds_fixed = pump_speeds_fixed[init_point:init_point + num_points]
pump_speeds_following = pump_speeds_following[init_point:init_point + num_points]
workload_utilizations = workload_utilizations[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_pump_speeds_random = pd.Series(pump_speeds_random).rolling(window=window_size).mean().dropna()
smoothed_pump_speeds_fixed = pd.Series(pump_speeds_fixed).rolling(window=window_size).mean().dropna()
smoothed_pump_speeds_following = pd.Series(pump_speeds_following).rolling(window=window_size).mean().dropna()
smoothed_workload_utilizations = pd.Series(workload_utilizations).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed workload utilization
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Workload Utilization (%)', color='tab:blue')
ax1.plot(time_intervals, smoothed_workload_utilizations * 100, color='tab:blue', linewidth=2)  # Convert to percentage
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Supply Temperature (°C)', color='k')
ax2.plot(time_intervals, smoothed_pump_speeds_random, color='tab:orange', linewidth=2, alpha=0.3, label='Random')
ax2.plot(time_intervals, smoothed_pump_speeds_fixed, color='tab:green', linewidth=2, alpha=0.7, label='ASHRAE W32')
ax2.plot(time_intervals, smoothed_pump_speeds_following, color='tab:red', linewidth=2, alpha=0.9, label='RBC', linestyle='--')
ax2.tick_params(axis='y', labelcolor='k')
# ax2.set_ylim(0.045, 0.25)  # Adjust the limits to match the scale of the data

# Add the legend of ax2 with title 'Baseline'
ax2.legend(loc='best', title='Baseline')
# Format the x-axis to show days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days

# Rotate the date labels for better readability
plt.xticks(rotation=45, ha='right')
# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(time_intervals[0], time_intervals[-1])

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/baselines_supply_temp_vs_workload_utilization_XL.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()
# %% Now Baselines of external temperature vs pump speed
pump_speeds_random = [metric['dc_crac_setpoint'] for metric in baseline_random['agent_dc']]
pump_speeds_fixed = [metric['dc_crac_setpoint'] for metric in baseline_fixed['agent_dc']]
pump_speeds_following = [metric['dc_crac_setpoint'] for metric in baseline_following['agent_dc']]
outside_temps = [metric['outside_temp'] for metric in trained_metrics['agent_dc']]

# Select the data to plot
pump_speeds_random = pump_speeds_random[init_point:init_point + num_points]
pump_speeds_fixed = pump_speeds_fixed[init_point:init_point + num_points]
pump_speeds_following = pump_speeds_following[init_point:init_point + num_points]
outside_temps = outside_temps[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_pump_speeds_random = pd.Series(pump_speeds_random).rolling(window=window_size).mean().dropna()
smoothed_pump_speeds_fixed = pd.Series(pump_speeds_fixed).rolling(window=window_size).mean().dropna()
smoothed_pump_speeds_following = pd.Series(pump_speeds_following).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed outside temperature
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Outside Temp (°C)', color='tab:blue')
ax1.plot(time_intervals, outside_temps, color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Pump Speed (l/s)', color='k')
ax2.plot(time_intervals, smoothed_pump_speeds_random, color='tab:orange', linewidth=2, alpha=0.3, label='Random')
ax2.plot(time_intervals, smoothed_pump_speeds_fixed, color='tab:green', linewidth=2, alpha=0.7, label='ASHRAE W32')
ax2.plot(time_intervals, smoothed_pump_speeds_following, color='tab:red', linewidth=2, alpha=0.9, label='ASHRAE W32', linestyle='--')
ax2.tick_params(axis='y', labelcolor='k')

# Add the legend of ax2 with title 'Baseline'
ax2.legend(loc='best', title='Baseline')

# Format the x-axis to show days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days

# Rotate the date labels for better readability
plt.xticks(rotation=45, ha='right')
# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(time_intervals[0], time_intervals[-1])

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/baselines_outside_temp_vs_pump_speed_XL.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()


# %% Now Baselines of external temperature vs supply temperature
pump_speeds_random = [metric['dc_supply_liquid_temp'] for metric in baseline_random['agent_dc']]
pump_speeds_fixed = [metric['dc_supply_liquid_temp'] for metric in baseline_fixed['agent_dc']]
pump_speeds_following = [metric['dc_supply_liquid_temp'] for metric in baseline_following['agent_dc']]
outside_temps = [metric['outside_temp'] for metric in trained_metrics['agent_dc']]
# Select the data to plot
pump_speeds_random = pump_speeds_random[init_point:init_point + num_points]
pump_speeds_fixed = pump_speeds_fixed[init_point:init_point + num_points]
pump_speeds_following = pump_speeds_following[init_point:init_point + num_points]
outside_temps = outside_temps[init_point:init_point + num_points]

# Smooth the data using a rolling window
window_size = 1  # Use a larger window size to smooth the data more
smoothed_pump_speeds_random = pd.Series(pump_speeds_random).rolling(window=window_size).mean().dropna()
smoothed_pump_speeds_fixed = pd.Series(pump_speeds_fixed).rolling(window=window_size).mean().dropna()
smoothed_pump_speeds_following = pd.Series(pump_speeds_following).rolling(window=window_size).mean().dropna()

# Create the plot
fig, ax1 = plt.subplots(figsize=(6, 3))  # Adjust height as necessary

# Plot smoothed outside temperature
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Outside Temp (°C)', color='tab:blue')
ax1.plot(time_intervals, outside_temps, color='tab:blue', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for pump speed
ax2 = ax1.twinx()
ax2.set_ylabel('Supply Temperature (°C)', color='k')
ax2.plot(time_intervals, smoothed_pump_speeds_random, color='tab:orange', linewidth=2, alpha=0.3, label='Random')
ax2.plot(time_intervals, smoothed_pump_speeds_fixed, color='tab:green', linewidth=2, alpha=0.7, label='ASHRAE W32')
ax2.plot(time_intervals, smoothed_pump_speeds_following, color='tab:red', linewidth=2, alpha=0.9, label='RBC', linestyle='--')
ax2.tick_params(axis='y', labelcolor='k')

# Add the legend of ax2 with title 'Baseline'
ax2.legend(loc='best', title='Baseline')

# Format the x-axis to show days
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days

# Rotate the date labels for better readability
plt.xticks(rotation=45, ha='right')
# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Add grid and limits
ax1.grid(linestyle='--')
plt.xlim(time_intervals[0], time_intervals[-1])

# Customize the layout to ensure no parts are cut off
plt.tight_layout()

# Save the figure as a PDF without cutting off any parts
# plt.savefig('media/baselines_outside_temp_vs_supply_temp_XL.pdf', format='pdf', bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()
# %%
