#%%
import os
import pickle
from tabulate import tabulate
import numpy as np
import json
import ray
from tqdm import tqdm

from utils.checkpoint_finder import get_best_checkpoint

@ray.remote
def evaluate_location(algo_class, config, CHECKPOINT, location, NUM_RUNS):
    co2_aux = []
    energy_aux = []
    print(f'Evaluating {location} from checkpoint {CHECKPOINT}')
    for _ in range(NUM_RUNS):
        config._is_frozen = False
        config['env_config']['location'] = location

        trainer = algo_class(config)
        trainer.restore(CHECKPOINT)
        results = trainer.evaluate()

        co2 = results['evaluation']['custom_metrics']['average_CO2_footprint_mean']
        energy = results['evaluation']['custom_metrics']['average_total_energy_with_battery_mean']

        co2_aux.append(co2)
        energy_aux.append(energy)
        trainer.stop()
    print(f'Finished evaluation with location {location} from checkpoint {CHECKPOINT}')
    return [location, f'{np.mean(co2_aux):.2f} ± {np.std(co2_aux):.2f}', f'{np.mean(energy_aux):.2f} ± {np.std(energy_aux):.2f}']

@ray.remote
def evaluate_checkpoint(checkp):
    print(f'Evaluating checkpoing {checkp}')
    CHECKPOINT = get_best_checkpoint(checkp)

    # Load checkpoint state
    with open(os.path.join(CHECKPOINT, 'algorithm_state.pkl'), 'rb') as f:
        state = pickle.load(f)

    algo_class = state['algorithm_class']

    # Set evaluation parameters
    config = state['config'].copy(copy_frozen=False)
    config['env_config']['evaluation'] = True
    config.evaluation_interval = 1
    config.evaluation_num_workers = 12
    config.evaluation_duration = 12
    config.evaluation_duration_unit = 'episodes'

    print(f'Created the agent for checkpoint {CHECKPOINT} ALGO CLASS: {algo_class}')
    # Parallel invocation of location-based evaluation
    data_list = ray.get([evaluate_location.remote(algo_class, config, CHECKPOINT, location, NUM_RUNS) for location in ['az', 'ny', 'wa']])

    return checkp, data_list

#%%
if __name__ == '__main__':
    paths = ['results/test/PPO_DCRL_2a2ba_00000_0_2023-10-12_09-55-36',
             'results/test/PPO_DCRL_2cc70_00000_0_2023-10-12_09-55-40',
             'results/test/PPO_DCRL_3db9b_00000_0_2023-10-12_09-56-09',
             'results/test/A2C_DCRL_8b0fa_00000_0_2023-10-12_09-51-09',
             'results/test/A2C_DCRL_a7bda_00000_0_2023-10-12_09-51-57',
             'results/test/A2C_DCRL_ae138_00000_0_2023-10-12_09-52-08',
             'results/test/MADDPGStable_DCRL_c1bbb_00000_0_2023-10-12_11-52-41',
             'results/test/MADDPGStable_DCRL_fc90b_00000_0_2023-10-12_11-54-19',
             'results/test/MADDPGStable_DCRL_100b5_00000_0_2023-10-12_11-54-52'
             ]
    NUM_RUNS = 1

    # Initialize Ray with a CPU limit
    ray.init(ignore_reinit_error=True, log_to_driver=True, local_mode=False, num_cpus=80) # for example, limit to 8 CPUs

    # Parallel invocation of remote function with progress bar
    results_list = ray.get([evaluate_checkpoint.remote(checkp) for checkp in tqdm(paths, desc="Evaluating checkpoints")])

    # Collate results
    results2 = {res[0]: res[1] for res in results_list}

    # Save results2 to disk
    with open('results2_AZ.json', 'w') as f:
        json.dump(results2, f, indent=4)
