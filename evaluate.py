import os
import pickle
from tabulate import tabulate

import ray

CHECKPOINT = './results/test/PPO_DCRL_4b751_00000_0_2023-06-19_06-22-09/checkpoint_002740'
NUM_RUNS = 4

if __name__ == '__main__':
    
    # log_to_driver ensures the RolloutWorkers don't log to the terminal
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Load checkpoint state
    with open(os.path.join(CHECKPOINT, 'algorithm_state.pkl'), 'rb') as f:
        state = pickle.load(f)

    algo_class = state['algorithm_class']

    config = state['config'].copy(copy_frozen=False)
    config.evaluation_interval = 1
    config.evaluation_num_workers = os.cpu_count() // 2
    
    # Number of episodes per location. 
    # 1 episode is 1 month, so (12 months * num_runs)
    config.evaluation_duration = 12*NUM_RUNS
    config.evaluation_duration_unit = 'episodes'

    data = []
    
    for location in ['az', 'wa', 'ny']:

        print(f"Evaluating for {location}...\n")

        # Unfreeze the config
        config._is_frozen = False
        config.evaluation_config = config.overrides(
            env_config={'location': location}
        )

        trainer = algo_class(config)
        trainer.restore(CHECKPOINT)

        results = trainer.evaluate()

        co2 = results['evaluation']['custom_metrics']['average_CO2_footprint_mean']
        energy = results['evaluation']['custom_metrics']['average_total_energy_with_battery_mean']

        data.append([location, co2, energy])

        # Release resources
        trainer.stop()

    print(f"\n\n\n\nEvaluation finished. Average of {NUM_RUNS} runs:")
    print(tabulate(
        tabular_data=data, 
        headers=['Location', 'CO2(MT)', 'Energy(MW)'],
        floatfmt='.3f'
        )
    )
