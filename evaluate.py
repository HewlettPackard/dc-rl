import os
import pickle
from tabulate import tabulate
import numpy as np

import ray

from utils.checkpoint_finder import get_best_checkpoint


if __name__ == '__main__':
    
    paths = ['results/test/PPO_DCRL_2a2ba_00000_0_2023-10-12_09-55-36',
             'results/test/PPO_DCRL_2cc70_00000_0_2023-10-12_09-55-40',
             'results/test/PPO_DCRL_3db9b_00000_0_2023-10-12_09-56-09',
             'results/test/A2C_DCRL_8b0fa_00000_0_2023-10-12_09-51-09',
             'results/test/A2C_DCRL_a7bda_00000_0_2023-10-12_09-51-57',
             'results/test/A2C_DCRL_ae138_00000_0_2023-10-12_09-52-08',
             'results/test/MADDPGStable_DCRL_c1bbb_00000_0_2023-10-12_11-52-41',
             'results/test/MADDPGStable_DCRL_fc90b_00000_0_2023-10-12_11-54-19',
             'results/test/MADDPGStable_DCRL_100b5_00000_0_2023-10-12_11-54-52',
             ]
    
    NUM_RUNS = 1
    results2 = {}
    for checkp in paths:
        CHECKPOINT = get_best_checkpoint(checkp)

        # log_to_driver ensures the RolloutWorkers don't log to the terminal
        ray.init(ignore_reinit_error=True, log_to_driver=False, local_mode=False)
        
        # Load checkpoint state
        with open(os.path.join(CHECKPOINT, 'algorithm_state.pkl'), 'rb') as f:
            state = pickle.load(f)

        algo_class = state['algorithm_class']

        # Set evaluation parameters
        config = state['config'].copy(copy_frozen=False)
        config['env_config']['evaluation'] = True
        config.evaluation_interval = 1
        config.evaluation_num_workers = 12
        
        # Number of episodes per location. 
        # 1 episode is 1 month, so (12 months * num_runs)
        config.evaluation_duration = 12
        config.evaluation_duration_unit = 'episodes'

        data = []
        
        for location in ['az', 'ny', 'wa']:
            co2_aux = []
            energy_aux = []
            for _ in range(NUM_RUNS):
                print(f"Evaluating for {location}...\n")

                # Unfreeze the config
                config._is_frozen = False
                config['env_config']['location'] = location

                trainer = algo_class(config)

                trainer.restore(CHECKPOINT)
                results = trainer.evaluate()

                co2 = results['evaluation']['custom_metrics']['average_CO2_footprint_mean']
                energy = results['evaluation']['custom_metrics']['average_total_energy_with_battery_mean']
                
                co2_aux.append(co2)
                energy_aux.append(energy)
                
                print(f'Current results for {location}: \n\tCO2: {co2} \n\tEnergy: {energy}\n')

                # Release resources
                trainer.stop()

            data.append([location, f'{np.mean(co2_aux):.2f} ± {np.std(co2_aux):.2f}', f'{np.mean(energy_aux):.2f} ± {np.std(energy_aux):.2f}'])

        # print(f"\n\n\n\nEvaluation finished. Average of {NUM_RUNS} runs:")
        # print(tabulate(
        #     tabular_data=data, 
        #     headers=['Location', 'CO2(MT)', 'Energy(MW)'],
        #     floatfmt='.3f'
        #     )
        # )
        results2[checkp] = data

    print(results2)