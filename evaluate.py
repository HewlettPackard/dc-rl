from collections import defaultdict
from joblib import Parallel, delayed
from tabulate import tabulate

from tqdm import tqdm
import pandas as pd
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.tune.analysis import ExperimentAnalysis

# Please update to desired checkpoint. Path to exact checkpoint is required
CHECKPOINT = './results/test/PPO_DCRL_4b751_00000_0_2023-06-19_06-22-09/checkpoint_002740'

def run(run_id: int, location: str, stats):
    trainer = Algorithm.from_checkpoint(CHECKPOINT)

    env_config = trainer.config.env_config
    env_config['location'] = location
    
    co2 = 0
    energy = 0

    # Cycle over months
    for month in tqdm(range(12)):
        env_config['month'] = month

        # Create environment
        env = trainer.config.env(env_config=env_config)
        obs, _ = env.reset()
        done = False
    
        while not done: 
            action_dict = {}

            for agent in ['agent_ls', 'agent_dc', 'agent_bat']:
                action = trainer.compute_single_action(obs[agent], policy_id=agent)
                action_dict[agent] = action
                
            obs, _, terminated, _, info = env.step(action_dict)

            co2 += info['agent_bat']['bat_CO2_footprint']
            energy += info['agent_bat']['bat_total_energy_with_battery_KWh']
            
            done = terminated['__all__']
            
    stats[location]['co2'].append(co2)
    stats[location]['energy'].append(energy)

if __name__ == '__main__':

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Total number of runs for each location
    num_runs = 2
    locations = ['wa', 'ny', 'az']
    
    # Shared variable to collect results
    stats = defaultdict(lambda: defaultdict(list))

    # Use "threading" as the backed so that shared variable is accesible
    Parallel(num_runs*3, verbose=100, backend="threading")(
        delayed(run)(run_id, location, stats) 
        for run_id in range(num_runs)
        for location in locations
        )

    # Average over the runs and pretty print the results
    data = []
    for location, stat in stats.items():
        co2 = np.mean(stat['co2']) / 10**6
        energy = np.mean(stat['energy']) / 10**6

        data.append([location, co2, energy])

    print(f"\n\n\n\nEvaluation finished. Average of {num_runs} runs:")
    print(tabulate(
        tabular_data=data, 
        headers=['Location', 'CO2(MT)', 'Energy(MW)'],
        floatfmt='.3f'
        )
    )