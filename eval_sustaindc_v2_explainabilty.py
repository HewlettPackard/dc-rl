#%% Explainability
import os
import sys
import copy  # Import copy to make a deep copy of metrics
import warnings 
import json
import numpy as np
warnings.filterwarnings('ignore')
from tabulate import tabulate

import pandas as pd

import torch
import matplotlib.pyplot as plt
sys.path.insert(0,os.getcwd())  # or sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'heterogeneous_sustaindc')))
from harl.runners import RUNNER_REGISTRY
from harl.utils.trans_tools import _t2n

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'dc-rl')))
from utils.base_agents import BaseLoadShiftingAgent, BaseHVACAgent, BaseBatteryAgent
from utils.rbc_agents import RBCLiquidAgent
#%%
# MODEL_PATH = 'trained_models'
# ENV = 'sustaindc'
# LOCATION = "az"
# AGENT_TYPE = "haa2c"
# RUN = "seed-00001-2024-06-04-20-41-56"
# ACTIVE_AGENTS = ['agent_ls', 'agent_dc', 'agent_bat']

baseline_actors = {
    "agent_ls": BaseLoadShiftingAgent(), 
    "agent_dc": RBCLiquidAgent(), 
    "agent_bat": BaseBatteryAgent()
}

# Function to evaluate and store metrics
def run_evaluation(do_baseline=False, eval_episodes=1, eval_type='random'):
    all_metrics = []
    all_states = []
    all_actions = []
    SAVE_EVAL = "results"
    NUM_EVAL_EPISODES = eval_episodes

    run = 'happo/liquid_2_agents/seed-01606-2024-11-06-22-32-23'

    path = f'/lustre/guillant/dc-rl/results/sustaindc/ca/{run}'
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

    # Get the active agents from the environment arguments
    active_agents = expt_runner.env_args['agents']

    for run_i in range(eval_episodes):
        # Initialize metrics for only the active agents
        metrics = {agent: [] for agent in baseline_actors}
        episode_states = {agent: [] for agent in baseline_actors}
        episode_actions = {agent: [] for agent in baseline_actors}
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
                    if agent_name == 'agent_dc':
                        if eval_type == 'random':
                            pump_speed = np.random.uniform(0, 1)
                            supply_temp = np.random.uniform(0, 1)
                            eval_actions = torch.tensor([[pump_speed, supply_temp]])
                        elif eval_type == 'fixed':
                            pump_speed = 0.444  # 0.25 l/s
                            supply_temp = 0.567 # 32°C (W32 ASHRAE GUIDELINES)
                            eval_actions = torch.tensor([[pump_speed, supply_temp]])
                        elif eval_type == 'following_workload':
                            workload = expt_runner.eval_envs.envs[0].env.env.workload_m.get_next_workload()
                            eval_actions = torch.tensor([[baseline_actors[agent_name].act(workload)]]) #torch.tensor([[0.25]])
                    else:
                        eval_actions = torch.tensor([[baseline_actors[agent_name].act()]])
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))
                
                # Store the current state and action
                episode_states[agent_name].append(eval_obs[0][agent_id].tolist())
                episode_actions[agent_name].append(eval_actions[0].tolist())

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
                                'ls_tasks_dropped', 'ls_current_hour'
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
            
        # After the episode ends, append the episode data to all_states and all_actions
        all_states.append(episode_states)
        all_actions.append(episode_actions)
        all_metrics.append(metrics)
        
    return all_metrics, all_states, all_actions

# Run baseline
num_runs = 1
trained_metrics_runs, trained_states_runs, trained_actions_runs = run_evaluation(do_baseline=False, eval_episodes=num_runs)


#%% 
import numpy as np
import pandas as pd

# Assuming you are interested in 'agent_dc' (Data Center Cooling Agent)
agent_name = 'agent_dc'

# Collect states and actions across all runs
states_list = []
actions_list = []

for run_idx in range(num_runs):
    episode_states = trained_states_runs[run_idx][agent_name]
    episode_actions = trained_actions_runs[run_idx][agent_name]

    # Convert list of states and actions to numpy arrays
    states_array = np.array(episode_states)
    actions_array = np.array(episode_actions)

    states_list.append(states_array)
    actions_list.append(actions_array)

# Concatenate data from all runs
all_states = np.concatenate(states_list, axis=0)
all_actions = np.concatenate(actions_list, axis=0)

# Create a DataFrame for easier analysis
data = pd.DataFrame(all_states, columns=[f'state_{i}' for i in range(all_states.shape[1])])
actions_df = pd.DataFrame(all_actions, columns=[f'action_{i}' for i in range(all_actions.shape[1])])

# Combine states and actions into one DataFrame
data = pd.concat([data, actions_df], axis=1)

#%%
# Compute the correlation matrix
correlation_matrix = data.corr()

# Display correlations between states and actions
state_columns = [col for col in data.columns if 'state' in col]
action_columns = [col for col in data.columns if 'action' in col]

# Extract the correlations between states and actions
state_action_corr = correlation_matrix.loc[state_columns, action_columns]

print("Correlations between state variables and actions:")
print(state_action_corr)


#%%
import matplotlib.pyplot as plt

# Example: Analyze the relationship between 'state_0' and 'action_0'
state_var = 'state_0'
action_var = 'action_0'

plt.figure(figsize=(8, 6))
plt.scatter(data[state_var], data[action_var], alpha=0.5)
plt.xlabel(f'{state_var}')
plt.ylabel(f'{action_var}')
plt.title(f'Scatter Plot of {action_var} vs {state_var}')
plt.grid(True)
plt.show()


#%%
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor

# Fit a regression model to predict the action from the states
X = data[state_columns]
y = data[action_var]

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Plot the partial dependence for the state variable of interest
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(model, X, [state_var], ax=ax)
plt.title(f'Partial Dependence of {action_var} on {state_var}')
plt.show()

#%%
import shap

# Use a small sample for SHAP to speed up computation
sample_size = 1000
X_sample = X.sample(n=sample_size, random_state=42)
y_sample = y.loc[X_sample.index]

# Retrain the model on the sample
model.fit(X_sample, y_sample)

# Create a SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_sample)

#%%