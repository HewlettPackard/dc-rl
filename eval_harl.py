import os
import sys
import warnings 
import json
import numpy as np
warnings.filterwarnings('ignore') 

sys.path.insert(0,os.getcwd())  # or sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'heterogeneous_dcrl')))
from harl.runners import RUNNER_REGISTRY
from harl.utils.trans_tools import _t2n

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'dc-rl')))
from utils.base_agents import BaseLoadShiftingAgent, BaseHVACAgent, BaseBatteryAgent

MODEL_PATH = 'trained_models'
SAVE_EVAL = "results"
ENV = 'sustaindc'
LOCATION = "az"
AGENT_TYPE = "haa2c"
RUN = "seed-00001-2024-06-04-20-41-56"
ACTIVE_AGENTS = ['agent_ls', 'agent_dc', 'agent_bat']
NUM_EVAL_EPISODES = 1

# load trained algo and env configs
with open(os.path.join(MODEL_PATH, ENV, LOCATION, AGENT_TYPE, AGENT_TYPE+"3agents",RUN,'config.json'), encoding='utf-8') as file:
        saved_config = json.load(file)

# read the algo, env and main args
algo_args, env_args, main_args = saved_config['algo_args'], saved_config['env_args'], saved_config['main_args']

# update the algo_args with the new values
algo_args['train']['n_rollout_threads'] = 1
algo_args['eval']['n_eval_rollout_threads'] = 1
algo_args['train']['model_dir'] = os.path.join(MODEL_PATH, ENV, LOCATION, AGENT_TYPE, AGENT_TYPE+"3agents",RUN, 'models')
# specify the top level folder to save the results
algo_args["logger"]["log_dir"] = SAVE_EVAL
# adjust number of eval episodes
algo_args["eval"]["eval_episodes"] = NUM_EVAL_EPISODES
# save evaluation results
algo_args["eval"]["dump_eval_metrcs"] = True

# initialize the actors and environments with the chosen configurations
expt_runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)
baseline_actors = {
            "agent_ls": BaseLoadShiftingAgent(), 
            "agent_dc": BaseHVACAgent(), 
            "agent_bat": BaseBatteryAgent()
        }


# Run the evaluation and log results manually
"""Evaluate the model."""
expt_runner.prep_rollout()
eval_episode = 0


# Dictionary to store metrics for each agent across all episodes
metrics = {
    'agent_1': [],
    'agent_2': [],
    'agent_3': []
}

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
    
    # append the information from each environment into the variables previously created using
    # a for loop (for i in range(n_agents), for j in range(n_environments) -> eval_infos[j][i]... extract information)
    eval_data = (
        eval_obs,
        eval_share_obs,
        eval_rewards,
        eval_dones,
        eval_infos,
        eval_available_actions,
    )

    if expt_runner.dump_info:
        for i in range(expt_runner.algo_args["eval"]["n_eval_rollout_threads"]):
            for j in range(expt_runner.num_agents):
                # Example extraction and storing of metrics for agent 1
                if j == 0:  # Assuming agent_1 corresponds to index 0
                    metrics[f'agent_{j+1}'].append({
                        key: eval_infos[i][j].get(key, None) for key in [
                            'ls_original_workload', 'ls_shifted_workload', 'ls_action', 'ls_norm_load_left',
                            'ls_unasigned_day_load_left', 'ls_penalty_flag', 'ls_tasks_in_queue',
                            'ls_tasks_dropped', 'ls_current_hour'
                        ]
                    })
                elif j == 1:  # Assuming agent_1 corresponds to index 0
                    metrics[f'agent_{j+1}'].append({
                        key: eval_infos[i][j].get(key, None) for key in [
                            'dc_ITE_total_power_kW', 'dc_HVAC_total_power_kW', 'dc_total_power_kW', 'dc_power_lb_kW', 
                            'dc_power_ub_kW', 'dc_crac_setpoint_delta', 'dc_crac_setpoint', 'dc_cpu_workload_fraction', 
                            'dc_int_temperature', 'dc_CW_pump_power_kW', 'dc_CT_pump_power_kW', 'dc_water_usage', 'dc_exterior_ambient_temp',
                            'outside_temp', 'day', 'hour'
                        ]
                    })
                elif j == 2:
                    metrics[f'agent_{j+1}'].append({
                        key: eval_infos[i][j].get(key, None) for key in [
                            'bat_action', 'bat_SOC', 'bat_CO2_footprint', 'bat_avg_CI', 'bat_total_energy_without_battery_KWh',
                            'bat_total_energy_with_battery_KWh', 'bat_max_bat_cap',
                            'bat_dcload_min', 'bat_dcload_max',
                        ]
                    })
                else:
                    print(f'There is an error while saving the evaluation metrics')

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
    

if expt_runner.dump_info:
    # Convert collected data to DataFrame and save as CSV
    expt_runner.dump_metrics_to_csv(metrics, eval_episode)
    print("Data saved to evaluation_data.csv.")

expt_runner.close()

