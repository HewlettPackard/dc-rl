#%%
import os
import pickle
from tabulate import tabulate
import numpy as np
import json
import ray
from tqdm import tqdm

from utils.checkpoint_finder import get_best_checkpoint
import os
import pickle
import numpy as np
import ray

from tabulate import tabulate
from utils.checkpoint_finder import get_best_checkpoint

# Define the Ray remote function to evaluate a checkpoint
@ray.remote
def evaluate_checkpoint(checkp):
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

    data = []
    for location in ['az', 'ny', 'wa']:
        co2_aux = []
        energy_aux = []
        load_left_aux = []
        for _ in range(NUM_RUNS):
            print(f"Evaluating for {location}...\n")

            config._is_frozen = False
            config['env_config']['location'] = location

            trainer = algo_class(config)
            trainer.restore(CHECKPOINT)
            results = trainer.evaluate()

            co2 = results['evaluation']['custom_metrics']['average_CO2_footprint_mean']/1000
            energy = results['evaluation']['custom_metrics']['average_total_energy_with_battery_mean']
            load_left = results['evaluation']['custom_metrics']['load_left_mean']

            co2_aux.append(co2)
            energy_aux.append(energy)
            load_left_aux.append(load_left)

            trainer.stop()

        data.append([location, f'{np.mean(co2_aux):.2f} ± {np.std(co2_aux):.2f}', f'{np.mean(energy_aux):.2f} ± {np.std(energy_aux):.2f}', f'{np.mean(load_left_aux):.2f} ± {np.std(load_left_aux):.2f}'])

    return checkp, data


#%%
if __name__ == '__main__':
    algo = 'MADDPG'
    sel_path = 2
    # paths = ['results/test/PPO_DCRL_2a2ba_00000_0_2023-10-12_09-55-36',
    #          'results/test/PPO_DCRL_2cc70_00000_0_2023-10-12_09-55-40',
    #          'results/test/PPO_DCRL_3db9b_00000_0_2023-10-12_09-56-09',
    #          'results/test/A2C_DCRL_8b0fa_00000_0_2023-10-12_09-51-09',
    #          'results/test/A2C_DCRL_a7bda_00000_0_2023-10-12_09-51-57',
    #          'results/test/A2C_DCRL_ae138_00000_0_2023-10-12_09-52-08',
    #          'results/test/MADDPGStable_DCRL_c1bbb_00000_0_2023-10-12_11-52-41',
    #          'results/test/MADDPGStable_DCRL_fc90b_00000_0_2023-10-12_11-54-19',
    #          'results/test/MADDPGStable_DCRL_100b5_00000_0_2023-10-12_11-54-52'
    #          ]
    
    if algo == 'PPO':
        paths = [
            'results/test/PPO_DCRL_2a2ba_00000_0_2023-10-12_09-55-36',
            'results/test/PPO_DCRL_2cc70_00000_0_2023-10-12_09-55-40',
            'results/test/PPO_DCRL_3db9b_00000_0_2023-10-12_09-56-09'
            ]
        paths = [paths[sel_path]]
    elif algo == 'A2C':
        paths = [
            'results/test/A2C_DCRL_8b0fa_00000_0_2023-10-12_09-51-09',
            'results/test/A2C_DCRL_a7bda_00000_0_2023-10-12_09-51-57',
            'results/test/A2C_DCRL_ae138_00000_0_2023-10-12_09-52-08'
            ]
        paths = [paths[sel_path]]
    else:
        paths = [
            'results/test/MADDPGStable_DCRL_c1bbb_00000_0_2023-10-12_11-52-41',
            'results/test/MADDPGStable_DCRL_fc90b_00000_0_2023-10-12_11-54-19',
            'results/test/MADDPGStable_DCRL_100b5_00000_0_2023-10-12_11-54-52'
            ]
        paths = [paths[sel_path]]
    
    
    NUM_RUNS = 3

    # Initialize Ray with a CPU limit
    ray.init(ignore_reinit_error=True, log_to_driver=True, local_mode=False, num_cpus=80)

    # Parallel invocation of remote function with progress bar
    results_list = ray.get([evaluate_checkpoint.remote(checkp) for checkp in paths])

    # Collate results
    results2 = {res[0]: res[1] for res in results_list}

    # Save results2 to disk
    with open(f'results_MADDPG_{algo}_{sel_path}.json', 'w') as f:
        json.dump(results2, f, indent=4)
