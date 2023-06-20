import os
import itertools
import logging
from copy import deepcopy
from joblib import Parallel, delayed, wrap_non_picklable_objects
from ray.rllib.policy.policy import PolicySpec
import tqdm
import numpy as np
import pandas as pd
from ray.rllib.algorithms.ppo import PPO #Select same algorithm as used in training

from dcrl_env import DCRL
from train_ppo import CONFIG #Import config of the desired algorithm

CHECKPOINT = './results/test/PPO_DCRL_c2f2a_00000_0_2023-06-16_16-51-50/checkpoint_001215/' #PATH TO CHECKPOINT

NUM_DAYS = 30
NUM_STEPS_PER_HOUR = 4

action_dict_ashrae = { 
                    'agent_ls' : 0,
                    'agent_dc' : np.int64(4),
                    'agent_bat' : 2
                    }

dummy_env = CONFIG.env(CONFIG.env_config)
ls_env, dc_env, bat_env = dummy_env.ls_env, dummy_env.dc_env, dummy_env.bat_env 

CONFIG = CONFIG.multi_agent(
            policies={
                "agent_ls": PolicySpec(
                    None,
                    ls_env.observation_space,
                    ls_env.action_space,
                    config={"agent_id" : 0},
                ),
                "agent_dc": PolicySpec(
                    None,
                    dc_env.observation_space,
                    dc_env.action_space,
                    config={"agent_id" : 1},
                ),
                "agent_bat": PolicySpec(
                    None,
                    bat_env.observation_space,
                    bat_env.action_space,
                    config={"agent_id" : 2},
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )

def run(run_id):
    trainer = PPO(deepcopy(CONFIG)) #Change to desired algorithm
    trainer.restore(CHECKPOINT)
    
    time_step_co2 = []
    time_step_price = []
    time_step_energy = []

    # Cycle over months
    for month in tqdm.tqdm(range(12)):
        env = DCRL(env_config={'month': month, 'actions_are_logits': True})
        obs, _ = env.reset()
        
        for i in range(24*NUM_STEPS_PER_HOUR*NUM_DAYS):
            action_dict = {}
            for agent in ['agent_ls', 'agent_dc', 'agent_bat']:
                action = trainer.compute_single_action(obs[agent], policy_id=agent)
                action_dict[agent] = action
                
            obs, _, _, _, info = env.step(action_dict)
            time_step_co2.append(info['agent_bat']['bat_CO2_footprint'])
            time_step_energy.append(info['agent_bat']['bat_total_energy_with_battery_KWh'])
    
    df = pd.DataFrame()
    df['energy'] = time_step_energy
    df['co2'] = time_step_co2

    name = '_dc_rl_multiagent'
    df.to_csv(f'./raw_results_ny/{name}_{run_id}.csv')


if __name__ == '__main__':
    num_runs = 1
    Parallel(num_runs, verbose=100)(
                        delayed(run)(run_id) 
                        for run_id in range(num_runs)
                        )
