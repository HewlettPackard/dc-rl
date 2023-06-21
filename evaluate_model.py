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
from ray.rllib.algorithms.a2c import A2C
from maddpg import MADDPGConfigStable, MADDPGStable


from dcrl_env import DCRL
from dcrl_eplus_env import DCRLeplus
from train_ppo import CONFIG as config_ppo  #Import config of the desired algorithm
from train_maddpg import CONFIG as config_maddpg
from train_a2c import CONFIG as config_a2c

#Select the algorithm
ALGORITHM = 'MADDPG' #PPO, A2C or MADDPG
CHECKPOINT = './results/test/MADDPGStable_DCRL_0608c_00000_0_2023-06-16_16-53-42/checkpoint_008795/' #PATH TO CHECKPOINT

if ALGORITHM == 'PPO':
    config = config_ppo
    train_algorithm = PPO
elif ALGORITHM == 'A2C':
    config = config_a2c
    train_algorithm = A2C
else:
    config = config_maddpg
    train_algorithm = MADDPGStable

environment=DCRL if not os.getenv('EPLUS') else DCRLeplus

NUM_DAYS = 30
NUM_STEPS_PER_HOUR = 4

action_dict_ashrae = { 
                    'agent_ls' : 0,
                    'agent_dc' : np.int64(4),
                    'agent_bat' : 2
                    }

dummy_env = config.env(config.env_config)
ls_env, dc_env, bat_env = dummy_env.ls_env, dummy_env.dc_env, dummy_env.bat_env 

config = config.multi_agent(
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
    trainer = train_algorithm(deepcopy(config)) #Change to desired algorithm
    trainer.restore(CHECKPOINT)
    
    time_step_co2 = []
    time_step_price = []
    time_step_energy = []

    # Cycle over months
    for month in tqdm.tqdm(range(12)):
        env = environment(env_config={'month': month, 'actions_are_logits': True})
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

    name = '_dc_rl_multiagent_maddpg' #Name of the output file
    df.to_csv(f'./raw_results_ny/{name}_{run_id}.csv') #Folder in where to store results


if __name__ == '__main__':
    num_runs = 1
    Parallel(num_runs, verbose=100)(
                        delayed(run)(run_id) 
                        for run_id in range(num_runs)
                        )
