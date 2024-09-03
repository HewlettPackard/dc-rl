import numpy as np
from tqdm import tqdm

# Function to run a single evaluation loop
def run_evaluation_loop(trainer, rbc_baseline, agent_type, max_iterations=4*24*7, seed=123):
    # Initialize the environment
    env = HeirarchicalDCRL(DEFAULT_CONFIG)
    done = False
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0
    
    workload_DC1, workload_DC2, workload_DC3 = [], [], []
    energy_consumption_DC1, energy_consumption_DC2, energy_consumption_DC3 = [], [], []
    carbon_emissions_DC1, carbon_emissions_DC2, carbon_emissions_DC3 = [], [], []
    water_consumption_DC1, water_consumption_DC2, water_consumption_DC3 = [], [], []

    with tqdm(total=max_iterations, ncols=150) as pbar:
        while not done:
            if agent_type == 0:  # One-step RL
                actions = trainer.compute_single_action(obs, explore=False)
            elif agent_type == 1:  # One-step greedy
                hier_obs = env.get_original_observation()
                ci = [hier_obs[dc]['ci'] for dc in env.datacenters]
                sender_idx = np.argmax(ci)  # Data center with the highest carbon intensity
                receiver_idx = np.argmin(ci)  # Data center with the lowest carbon intensity
                actions = np.zeros(3)
                if sender_idx == 0 and receiver_idx == 1:
                    actions[0] = 1.0
                elif sender_idx == 0 and receiver_idx == 2:
                    actions[1] = 1.0
                elif sender_idx == 1 and receiver_idx == 0:
                    actions[0] = -1.0
                elif sender_idx == 1 and receiver_idx == 2:
                    actions[2] = 1.0
                elif sender_idx == 2 and receiver_idx == 0:
                    actions[1] = -1.0
                elif sender_idx == 2 and receiver_idx == 1:
                    actions[2] = -1.0
            elif agent_type == 2:  # Multi-step greedy
                actions = rbc_baseline.multi_step_greedy()
            elif agent_type == 3:  # Equal workload distribution
                actions = rbc_baseline.equal_workload_distribution()
            else:  # Do nothing
                actions = np.zeros(3)
            
            obs, reward, terminated, done, info = env.step(actions)
            total_reward += reward

            workload_DC1.append(env.low_level_infos['DC1']['agent_ls']['ls_original_workload'])
            workload_DC2.append(env.low_level_infos['DC2']['agent_ls']['ls_original_workload'])
            workload_DC3.append(env.low_level_infos['DC3']['agent_ls']['ls_original_workload'])

            energy_consumption_DC1.append(env.low_level_infos['DC1']['agent_bat']['bat_total_energy_without_battery_KWh'])
            energy_consumption_DC2.append(env.low_level_infos['DC2']['agent_bat']['bat_total_energy_without_battery_KWh'])
            energy_consumption_DC3.append(env.low_level_infos['DC3']['agent_bat']['bat_total_energy_without_battery_KWh'])

            carbon_emissions_DC1.append(env.low_level_infos['DC1']['agent_bat']['bat_CO2_footprint'])
            carbon_emissions_DC2.append(env.low_level_infos['DC2']['agent_bat']['bat_CO2_footprint'])
            carbon_emissions_DC3.append(env.low_level_infos['DC3']['agent_bat']['bat_CO2_footprint'])

            water_consumption_DC1.append(env.low_level_infos['DC1']['agent_dc']['dc_water_usage'])
            water_consumption_DC2.append(env.low_level_infos['DC2']['agent_dc']['dc_water_usage'])
            water_consumption_DC3.append(env.low_level_infos['DC3']['agent_dc']['dc_water_usage'])

            pbar.update(1)
    
    metrics = {
        'total_reward': total_reward,
        'workload_DC1': workload_DC1,
        'workload_DC2': workload_DC2,
        'workload_DC3': workload_DC3,
        'energy_consumption_DC1': energy_consumption_DC1,
        'energy_consumption_DC2': energy_consumption_DC2,
        'energy_consumption_DC3': energy_consumption_DC3,
        'carbon_emissions_DC1': carbon_emissions_DC1,
        'carbon_emissions_DC2': carbon_emissions_DC2,
        'carbon_emissions_DC3': carbon_emissions_DC3,
        'water_consumption_DC1': water_consumption_DC1,
        'water_consumption_DC2': water_consumption_DC2,
        'water_consumption_DC3': water_consumption_DC3,
    }

    return metrics

# Function to run multiple simulations and compute averages and standard deviations
def run_multiple_simulations(num_runs, trainer, rbc_baseline):
    all_metrics = []
    agent_types = [0, 1, 2, 3, 4]  # Different agents

    for agent_type in agent_types:
        agent_metrics = []
        for _ in range(num_runs):
            metrics = run_evaluation_loop(trainer, rbc_baseline, agent_type)
            agent_metrics.append(metrics)
        
        # Calculate average and std for each metric across all runs
        avg_metrics = {key: np.mean([m[key] for m in agent_metrics], axis=0) for key in agent_metrics[0]}
        std_metrics = {key: np.std([m[key] for m in agent_metrics], axis=0) for key in agent_metrics[0]}
        
        print(f"Agent Type {agent_type} - Averages:")
        for key in avg_metrics:
            print(f"{key}: {np.mean(avg_metrics[key]):.3f} ± {np.mean(std_metrics[key]):.3f}")
        
        all_metrics.append({'avg': avg_metrics, 'std': std_metrics})
    
    return all_metrics

# Run the simulations
num_runs = 10  # Number of simulations to average over
all_metrics = run_multiple_simulations(num_runs, trainer, rbc_baseline)
