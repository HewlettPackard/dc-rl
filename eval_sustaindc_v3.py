import json
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import sys
import copy  # Import copy to make a deep copy of metrics
import warnings 
warnings.filterwarnings('ignore')

import pandas as pd

import matplotlib.pyplot as plt
sys.path.insert(0,os.getcwd())  # or sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'heterogeneous_sustaindc')))
from harl.runners import RUNNER_REGISTRY
from harl.utils.trans_tools import _t2n

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'dc-rl')))
from utils.base_agents import BaseLoadShiftingAgent, BaseHVACAgent, BaseBatteryAgent, RBCLiquidAgent

# Function to evaluate a single month for both baseline and trained models
def evaluate_month(run, month, num_runs):
    # Run baseline evaluation
    baseline_metrics = run_evaluation(run, month, do_baseline=True, eval_episodes=num_runs)
    
    # Run trained model evaluation
    trained_metrics = run_evaluation(run, month, do_baseline=False, eval_episodes=num_runs)
    
    return (month, baseline_metrics, trained_metrics)

def run_evaluation(run, month, do_baseline=False, eval_episodes=1):
    all_metrics = []
    
    baseline_actors = {
    "agent_ls": BaseLoadShiftingAgent(), 
    "agent_dc": RBCLiquidAgent(), 
    "agent_bat": BaseBatteryAgent()
    }

    
    # Define a unique directory for each run and month
    SAVE_EVAL = os.path.join("results", run, f"month_{month}", "baseline" if do_baseline else "trained")
    os.makedirs(SAVE_EVAL, exist_ok=True)  # Ensure the directory exists

    path = f'/lustre/guillant/sustaindc/results/sustaindc/az/{run}'
    with open(os.path.join(path, 'config.json'), encoding='utf-8') as file:
        saved_config = json.load(file)

    algo_args, env_args, main_args = saved_config['algo_args'], saved_config['env_args'], saved_config['main_args']

    # Update algorithm arguments
    algo_args['train']['n_rollout_threads'] = 1
    algo_args['eval']['n_eval_rollout_threads'] = 1
    algo_args['train']['model_dir'] = os.path.join(path, 'models')
    algo_args["logger"]["log_dir"] = SAVE_EVAL
    algo_args["eval"]["eval_episodes"] = 1  # Each episode handled in the loop
    algo_args["eval"]["dump_eval_metrcs"] = True

    # Set the month for evaluation
    env_args['month'] = month

    # Initialize the experiment runner
    expt_runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)

    # Get active agents
    active_agents = expt_runner.env_args['agents']

    for run_i in range(eval_episodes):
        # Initialize metrics for active agents
        metrics = {agent: [] for agent in baseline_actors}
        
        expt_runner.prep_rollout()
        eval_episode_count = 0
        
        # Set the seed for reproducibility
        expt_runner.eval_envs.envs[0].env.env.seed(run_i)
        
        # Reset the environment
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
                    # Handle baseline actions
                    if agent_name == 'agent_dc':
                        workload = expt_runner.eval_envs.envs[0].env.env.workload_m.get_next_workload()
                        eval_actions = torch.tensor([[baseline_actors[agent_name].act(workload)]])
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
                    info = eval_infos[i][0]
                    # Collect metrics for each agent
                    if 'dc_ITE_total_power_kW' in info:
                        metrics['agent_dc'].append({key: info.get(key) for key in [
                            'dc_ITE_total_power_kW', 'dc_HVAC_total_power_kW', 'dc_total_power_kW', 'dc_power_lb_kW', 
                            'dc_power_ub_kW', 'dc_crac_setpoint_delta', 'dc_crac_setpoint', 'dc_cpu_workload_fraction', 
                            'dc_int_temperature', 'dc_CW_pump_power_kW', 'dc_CT_pump_power_kW', 'dc_water_usage', 'dc_exterior_ambient_temp',
                            'outside_temp', 'day', 'hour', 'dc_average_server_temp', 'dc_average_pipe_temp', 'dc_heat_removed', 'dc_pump_power_kW', 
                            'dc_coo_m_flow_nominal', 'dc_coo_mov_flow_actual', 'dc_supply_liquid_temp', 'dc_return_liquid_temp'
                        ]})
                    if 'ls_original_workload' in info:
                        metrics['agent_ls'].append({key: info.get(key) for key in [
                            'ls_original_workload', 'ls_shifted_workload', 'ls_action', 'ls_norm_load_left',
                            'ls_unasigned_day_load_left', 'ls_penalty_flag', 'ls_tasks_in_queue',
                            'ls_tasks_dropped', 'ls_current_hour'
                        ]})
                    if 'bat_action' in info:
                        metrics['agent_bat'].append({key: info.get(key) for key in [
                            'bat_action', 'bat_SOC', 'bat_CO2_footprint', 'bat_avg_CI', 'bat_total_energy_without_battery_KWh',
                            'bat_total_energy_with_battery_KWh', 'bat_max_bat_cap',
                            'bat_dcload_min', 'bat_dcload_max',
                        ]})

            eval_dones_env = np.all(eval_dones, axis=1)

            # Reset RNN states and masks if done
            eval_rnn_states[eval_dones_env] = 0
            eval_masks = np.ones_like(eval_masks)
            eval_masks[eval_dones_env] = 0

            # Increment eval_episode_count
            eval_episode_count += np.sum(eval_dones_env)

            if eval_episode_count >= expt_runner.algo_args["eval"]["eval_episodes"]:
                break

        all_metrics.append(metrics)

    return all_metrics

def main():
    # Define the run path you want to evaluate
    run = 'happo/happo_liquid_dc_64_64_2actions_4obs/seed-00001-2024-08-23-18-53-06'
    
    # Define the months you want to evaluate (April to September)
    months = [4, 5, 6, 7, 8, 9]
    
    # Number of evaluation episodes per month
    num_runs = 10
    
    # Use ProcessPoolExecutor to run evaluations in parallel
    with ProcessPoolExecutor() as executor:
        # Submit all month evaluations to the executor
        futures = {executor.submit(evaluate_month, run, month, num_runs): month for month in months}
        
        # Collect results as they complete
        results = {}
        for future in as_completed(futures):
            month, baseline_metrics, trained_metrics = future.result()
            results[month] = (baseline_metrics, trained_metrics)
            print(f"Completed evaluation for month {month}")
    
    # Aggregate metrics from all months
    all_baseline_metrics = []
    all_trained_metrics = []
    
    for month in months:
        baseline, trained = results.get(month, ([], []))
        all_baseline_metrics.extend(baseline)
        all_trained_metrics.extend(trained)
    
    # Define a function to calculate mean and std for a given metric
    def calculate_avg_std(metric_name, metrics_runs):
        all_values = []
        for metrics in metrics_runs:
            # Check if metric exists in any agent's data
            for agent, data in metrics.items():
                for entry in data:
                    if metric_name in entry and entry[metric_name] is not None:
                        all_values.append(entry[metric_name])
        return np.mean(all_values), np.std(all_values)
    
    # Calculate averages and standard deviations for the metrics
    trained_avg_ite_energy, trained_std_ite_energy = calculate_avg_std('dc_ITE_total_power_kW', all_trained_metrics)
    baseline_avg_ite_energy, baseline_std_ite_energy = calculate_avg_std('dc_ITE_total_power_kW', all_baseline_metrics)

    trained_avg_hvac_energy, trained_std_hvac_energy = calculate_avg_std('dc_HVAC_total_power_kW', all_trained_metrics)
    baseline_avg_hvac_energy, baseline_std_hvac_energy = calculate_avg_std('dc_HVAC_total_power_kW', all_baseline_metrics)

    trained_avg_carbon_emissions, trained_std_carbon_emissions = calculate_avg_std('bat_CO2_footprint', all_trained_metrics)
    baseline_avg_carbon_emissions, baseline_std_carbon_emissions = calculate_avg_std('bat_CO2_footprint', all_baseline_metrics)

    trained_avg_water_usage, trained_std_water_usage = calculate_avg_std('dc_water_usage', all_trained_metrics)
    baseline_avg_water_usage, baseline_std_water_usage = calculate_avg_std('dc_water_usage', all_baseline_metrics)

    # Print summary
    print(f"Summary of Comparison Across {num_runs} Runs for Each Month:")
    print(f"ITE Energy Consumption: Reduction (%): {100 * (baseline_avg_ite_energy - trained_avg_ite_energy) / baseline_avg_ite_energy:.3f} ± {100 * trained_std_ite_energy / baseline_avg_ite_energy:.3f}")
    print(f"HVAC Energy Consumption: Reduction (%): {100 * (baseline_avg_hvac_energy - trained_avg_hvac_energy) / baseline_avg_hvac_energy:.3f} ± {100 * trained_std_hvac_energy / baseline_avg_hvac_energy:.3f}")
    print(f"Total Energy Consumption: Reduction (%): {100 * (baseline_avg_ite_energy + baseline_avg_hvac_energy - trained_avg_ite_energy - trained_avg_hvac_energy) / (baseline_avg_ite_energy + baseline_avg_hvac_energy):.3f} ± {100 * (trained_std_ite_energy + trained_std_hvac_energy) / (baseline_avg_ite_energy + baseline_avg_hvac_energy):.3f}")
    print(f"Carbon Emissions: Reduction (%): {100 * (baseline_avg_carbon_emissions - trained_avg_carbon_emissions) / baseline_avg_carbon_emissions:.3f} ± {100 * trained_std_carbon_emissions / baseline_avg_carbon_emissions:.3f}")
    print(f"Water Usage: Reduction (%): {100 * (baseline_avg_water_usage - trained_avg_water_usage) / baseline_avg_water_usage:.3f} ± {100 * trained_std_water_usage / baseline_avg_water_usage:.3f}")

if __name__ == "__main__":
    main()