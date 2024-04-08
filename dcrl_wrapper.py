
#%%
from dcrl_env import DCRL
from hierarchical_workload_optimizer import WorkloadOptimizer
import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm

config1 = {'location': 'az',
          'cintensity_file': 'AZPS_NG_&_avgCI.csv',
          'weather_file': 'USA_AZ_Tucson-Davis-Monthan.epw',
          'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
          'datacenter_capacity_mwh' : 2}


config2 = {'location': 'ny',
          'cintensity_file': 'NYIS_NG_&_avgCI.csv',
          'weather_file': 'USA_NY_New.York-Kennedy.epw',
          'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
          'datacenter_capacity_mwh' : 1}


config3 = {'location': 'wa',
          'cintensity_file': 'WAAT_NG_&_avgCI.csv',
          'weather_file': 'USA_WA_Port.Angeles-Fairchild.epw',
          'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
          'datacenter_capacity_mwh' : 0.5}

DC1 = DCRL(config1)
DC2 = DCRL(config2)
DC3 = DCRL(config3)

#%%
checkpoint_path = 'results/test/PPO_DCRL_25a24_00000_0_2024-04-05_20-57-43/checkpoint_000579'
lower_level_actor = Algorithm.from_checkpoint(checkpoint_path)

# Define the environments
environments = {
    'DC1': DC1,
    'DC2': DC2,
    'DC3': DC3,
}

# Instantiate the top-level agent
top_level_actor = WorkloadOptimizer(environments.keys())

episode_reward = 0

# Initialize dictionaries to hold the observations and infos for each environment
observations = {}
infos = {}

# Initialize metrics dictionaries
metrics = { env_id: {'bat_CO2_footprint': [],
                     'bat_total_energy_with_battery_KWh': [],
                     'ls_tasks_dropped': [],
                     'dc_water_usage': [],
                     } for env_id in environments
            }

# Reset environments and store initial observations and infos
for env_id, env in environments.items():
    obs, info = env.reset()
    observations[env_id] = obs
    infos[env_id] = info

# Track completion status for each environment
all_done = {env_id: False for env_id in environments}

while not all(all_done.values()):
    # Obtain variables of each DC: [DCi_capacity, DCi_workload, DCi_weather, DCi_CI for env_i in environments]
    hier_obs = {} 
    for env_id, env in environments.items():
        hier_obs[env_id] = env.get_hierarchical_variables()
    
    # Obtain the new workload of each DC
    adjusted_workloads, transfer_matrix = top_level_actor.compute_adjusted_workload(hier_obs)

    # Update workload for all DCs
    for env_id, adj_workload in adjusted_workloads.items():
        environments[env_id].set_hierarchical_workload(round(adj_workload, 6))
        
    # Individual actions
    actions = {env_id: {} for env_id in environments}

    # Compute actions for each agent in each environment
    for env_id, env_obs in observations.items():
        if all_done[env_id]:  # Skip if environment is done
            continue

        for agent_id, agent_obs in env_obs.items():
            policy_id = agent_id  # Customize policy ID if necessary
            action = lower_level_actor.compute_single_action(agent_obs, policy_id=policy_id)
            actions[env_id][agent_id] = action

    # Step through each environment with computed actions
    for env_id in environments:
        if all_done[env_id]:
            continue

        new_obs, rewards, terminated, truncated, info = environments[env_id].step(actions[env_id])
        observations[env_id] = new_obs
        all_done[env_id] = terminated['__all__'] or truncated['__all__']

        # Update metrics for each environment
        env_metrics = metrics[env_id]
        env_metrics['bat_CO2_footprint'].append(info['agent_bat']['bat_CO2_footprint'])
        env_metrics['bat_total_energy_with_battery_KWh'].append(info['agent_bat']['bat_total_energy_with_battery_KWh'])
        env_metrics['ls_tasks_dropped'].append(info['agent_ls']['ls_tasks_dropped'])
        env_metrics['dc_water_usage'].append(info['agent_dc']['dc_water_usage'])

# After simulation, calculate average metrics for each environment
average_metrics = {
    env_id: {metric: sum(values) / len(values) for metric, values in env_metrics.items()}
    for env_id, env_metrics in metrics.items()
}

# Print average metrics for each environment
for env_id, env_metrics in average_metrics.items():
    print(f"Average Metrics for {env_id} Sustainability:")
    for metric, value in env_metrics.items():
        print(f"        {metric}: {value:.2f}")
    print()  # Blank line for readability


# %%
