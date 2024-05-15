#%%
import sys
from tqdm import tqdm

import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from heirarchical_env import HeirarchicalDCRL, HeirarchicalDCRLWithHysterisis, HeirarchicalDCRLWithHysterisisMultistep, DEFAULT_CONFIG
from hierarchical_workload_optimizer import WorkloadOptimizer

#%
trainer_single = Algorithm.from_checkpoint('./results/SingleStep/PPO_HeirarchicalDCRLWithHysterisis_59fd7_00000_0_2024-05-14_18-39-53/checkpoint_000350')
trainer_multi = Algorithm.from_checkpoint('./results/MultiStep/PPO_HeirarchicalDCRLWithHysterisisMultistep_659f8_00000_0_2024-05-14_18-40-12/checkpoint_005145')
#%
env = HeirarchicalDCRLWithHysterisisMultistep(DEFAULT_CONFIG)
# obtain the locations from DEFAULT_CONFIG
dc_location_mapping = {
    'DC1': DEFAULT_CONFIG['config1']['location'].upper(),
    'DC2': DEFAULT_CONFIG['config2']['location'].upper(),
    'DC3': DEFAULT_CONFIG['config3']['location'].upper(),
}
greedy_optimizer = WorkloadOptimizer(env.datacenters.keys())
#%%
def compare_transfer_actions(actions1, actions2):
    """Compare transfer actions for equality on specific keys."""
    # Check if both actions have the same set of transfers
    if set(actions1.keys()) != set(actions2.keys()):
        return False

    # Iterate through each transfer action and compare
    for key in actions1:
        action1 = actions1[key]
        action2 = actions2[key]

        # Check the specific keys within each transfer action
        if (action1['receiver'] != action2['receiver'] or
            action1['sender'] != action2['sender'] or
            not np.array_equal(action1['workload_to_move'], action2['workload_to_move'])):
            return False

    return True

max_iterations = 4*24*30
results_all = []

# Initialize lists to store the 'current_workload' metric
workload_DC1 = [[], [], [], [], []]
workload_DC2 = [[], [], [], [], []]
workload_DC3 = [[], [], [], [], []]

# Other lists to store the 'carbon_emissions' metric
carbon_emissions_DC1 = [[], [], [], [], []]
carbon_emissions_DC2 = [[], [], [], [], []]
carbon_emissions_DC3 = [[], [], [], [], []]

# Other lists to store the 'external_temperature' metric
external_temperature_DC1 = [[], [], [], [], []]
external_temperature_DC2 = [[], [], [], [], []]
external_temperature_DC3 = [[], [], [], [], []]

# Another list to store the carbon intensity of each datacenter
carbon_intensity = []

# 5 Different agents (One-step RL, Multi-step RL, One-step Greedy, Multi-step Greedy, Do nothing)

for i in [0, 1, 2, 3, 4]:
    done = False
    obs, _ = env.reset(seed=123)    

    actions_list = []
    rewards_list = []
    total_reward = 0    
    
    with tqdm(total=max_iterations, ncols=150) as pbar:
        while not done:
            if i == 0:
                actions = trainer_single.compute_single_action(obs)
            elif i == 1:
                actions = trainer_multi.compute_single_action(obs)
            elif i == 2:
                # One-step greedy
                ci = [obs[dc][-1] for dc in env.datacenters]
                denorm_ci = [env.low_level_infos[dc_key]['agent_bat']['bat_avg_CI'] for dc_key in env.datacenters.keys()]
                carbon_intensity.append(denorm_ci)
                actions = {'receiver': np.argmin(ci), 'sender': np.argmax(ci), 'workload_to_move': np.array([1.])}
                actions = {'transfer_1': actions}
            elif i == 3:
                # Multi-step greedy
                # sort the ci index with repect of their values
                ci = [obs[dc][-1] for dc in env.datacenters]
                sorted_ci = np.argsort(ci)
                # First create the 'transfer_1' action with the transfer from the datacenter with the highest ci to the lowest ci
                # Then, create the 'transfer_2' action with the transfer from the datacenter with the second highest ci to the second lowest ci
                actions = {}
                for j in range(len(sorted_ci)-1):
                    actions[f'transfer_{j+1}'] = {'receiver': np.argmin(ci), 'sender': sorted_ci[-(j+1)], 'workload_to_move': np.array([1.])}
                    
                 # Check if multi-step greedy actions are different from trainer actions
                trainer_action = trainer_multi.compute_single_action(obs)
                # trainer_action['transfer_1']['workload_to_move'] = 0.23
                # Compare actions element by element
                # if not compare_transfer_actions(actions, trainer_action):
                #     print("WARNING: Multi-step greedy actions differ from trainer actions.")
                #     print("Trainer actions: ", trainer_action)
                #     print("Multi-step greedy actions: ", actions)
            else:
                # Do nothing
                actions = {'sender': 0, 'receiver': 0, 'workload_to_move': np.array([0.0])}
                actions = {'transfer_1': actions}

            
            obs, reward, terminated, done, info = env.step(actions)

            # Obtain the 'current_workload' metric for each datacenter using the low_level_infos -> agent_ls -> ls_original_workload
            workload_DC1[i].append(env.low_level_infos['DC1']['agent_ls']['ls_original_workload'])  
            workload_DC2[i].append(env.low_level_infos['DC2']['agent_ls']['ls_original_workload'])
            workload_DC3[i].append(env.low_level_infos['DC3']['agent_ls']['ls_original_workload'])
            
            # Obtain the 'carbon_emissions' metric for each datacenter using the low_level_infos -> agent_bat -> bat_CO2_footprint
            carbon_emissions_DC1[i].append(env.low_level_infos['DC1']['agent_bat']['bat_CO2_footprint'])
            carbon_emissions_DC2[i].append(env.low_level_infos['DC2']['agent_bat']['bat_CO2_footprint'])
            carbon_emissions_DC3[i].append(env.low_level_infos['DC3']['agent_bat']['bat_CO2_footprint'])

            # Obtain the 'external_temperature' metric for each datacenter using the low_level_infos -> agent_dc -> dc_exterior_ambient_temp
            external_temperature_DC1[i].append(env.low_level_infos['DC1']['agent_dc']['dc_exterior_ambient_temp'])
            external_temperature_DC2[i].append(env.low_level_infos['DC2']['agent_dc']['dc_exterior_ambient_temp'])
            external_temperature_DC3[i].append(env.low_level_infos['DC3']['agent_dc']['dc_exterior_ambient_temp'])
            
            total_reward += reward
    
            actions_list.append(actions['transfer_1'])
            rewards_list.append(reward)
            
            pbar.update(1)

    results_all.append((actions_list, rewards_list))
    print(f'Not computed workload: {env.not_computed_workload:.2f}')
    # pbar.close()

    print(total_reward)
#%%
# First of all, let's smooth the metrics before plotting.
# We can smooth the metrics using the moving average method.
# We will use a window of 1 hour (4 timestep) for the moving average.

win_size = 8
workload_DC1 = np.array(workload_DC1)
workload_DC2 = np.array(workload_DC2)
workload_DC3 = np.array(workload_DC3)

carbon_emissions_DC1 = np.array(carbon_emissions_DC1)
carbon_emissions_DC2 = np.array(carbon_emissions_DC2)
carbon_emissions_DC3 = np.array(carbon_emissions_DC3)

external_temperature_DC1 = np.array(external_temperature_DC1)
external_temperature_DC2 = np.array(external_temperature_DC2)
external_temperature_DC3 = np.array(external_temperature_DC3)

# Smooth the 'current_workload' metric, remeber that workload_DC1.shape=(num_controllers, time_steps).
# Use scipy.ndimage.filters with 1D filter to only smooth the time dimension.
from scipy.ndimage import uniform_filter1d
smoothed_workload_DC1 = uniform_filter1d(workload_DC1, size=win_size, axis=1)
smoothed_workload_DC2 = uniform_filter1d(workload_DC2, size=win_size, axis=1)
smoothed_workload_DC3 = uniform_filter1d(workload_DC3, size=win_size, axis=1)

# Smooth the 'carbon_emissions' metric
smoothed_carbon_emissions_DC1 = uniform_filter1d(carbon_emissions_DC1, size=win_size, axis=1)
smoothed_carbon_emissions_DC2 = uniform_filter1d(carbon_emissions_DC2, size=win_size, axis=1)
smoothed_carbon_emissions_DC3 = uniform_filter1d(carbon_emissions_DC3, size=win_size, axis=1)

# Smooth the 'external_temperature' metric
smoothed_external_temperature_DC1 = uniform_filter1d(external_temperature_DC1, size=win_size, axis=1)
smoothed_external_temperature_DC2 = uniform_filter1d(external_temperature_DC2, size=win_size, axis=1)
smoothed_external_temperature_DC3 = uniform_filter1d(external_temperature_DC3, size=win_size, axis=1)

# Smooth the 'carbon_intensity' metric
smoothed_carbon_intensity = uniform_filter1d(carbon_intensity, size=win_size, axis=0)
#%%
import matplotlib.pyplot as plt
# Plot the 'current_workload' metric
controllers = ['One-step RL', 'Multi-step RL', 'One-step Greedy', 'Multi-step Greedy', 'Do nothing']
for i in range(len(controllers)):
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_workload_DC1[i][:4*24*7]*100, label=dc_location_mapping['DC1'], linestyle='--', linewidth=2, alpha=1)
    plt.plot(smoothed_workload_DC2[i][:4*24*7]*100, label=dc_location_mapping['DC2'], linestyle='-.', linewidth=2, alpha=0.9)
    plt.plot(smoothed_workload_DC3[i][:4*24*7]*100, label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=0.7)
    plt.title(f'Current Workload for {controllers[i]} Controller')
    plt.xlabel('Time Step')
    plt.ylabel('Current Workload (%)')
    plt.legend()
    plt.grid('on', linestyle='--', alpha=0.5)
    plt.ylim(0, 101)
    plt.show()
#%% Plot the 'carbon_emissions' metric

for i in range(len(controllers)):
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_carbon_emissions_DC1[i][:4*24*7]/1e6, label=dc_location_mapping['DC1'], linestyle='--', linewidth=2, alpha=1)
    plt.plot(smoothed_carbon_emissions_DC2[i][:4*24*7]/1e6, label=dc_location_mapping['DC2'], linestyle='-.', linewidth=2, alpha=0.9)
    plt.plot(smoothed_carbon_emissions_DC3[i][:4*24*7]/1e6, label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=0.7)
    plt.title(f'Carbon Emissions for {controllers[i]} Controller')
    plt.xlabel('Time Step')
    plt.ylabel('Carbon Emissions (MgCO2)')
    plt.legend()
    plt.grid('on', linestyle='--', alpha=0.5)
    plt.ylim(0.2, 1)
    plt.show()

#%% Let's plot the carbon intensity for each datacenter
# First, adapt the carbon_intensity list to be a numpy array. On each time step, the carbon intensity of each datacenter is stored in a list
# We need to convert this list to a numpy array to plot it np.array(carbon_intensity).shape = (time_steps, num_datacenters)
# Only plot the first week (:4*24*7)

plt.figure(figsize=(10, 6))
plt.plot(carbon_intensity[1:4*24*7], linestyle='-', linewidth=2, alpha=1)
plt.title('Carbon Intensity for Each Datacenter')
plt.xlabel('Time Step')
plt.ylabel('Carbon Intensity (gCO2/kWh)')
plt.legend(dc_location_mapping.values())
plt.grid('on', linestyle='--', alpha=0.5)
plt.show()

#%% 
'''
Now I want to plot the sum of carbon emissions for each controller on each timestep on the 3 locations (DC1, DC2, and DC3)
So, we can compare the sum of carbon emissions on each timestep for each controller.
The sum of carbon emissions is calculated as the sum of carbon emissions for each datacenter on each timestep.
For example, for the first controller (RL), the sum of carbon emissions on each timestep is calculated as:
sum_carbon_emissions = carbon_emissions_DC1[0] + carbon_emissions_DC2[0] + carbon_emissions_DC3[0]
'''
sum_carbon_emissions = []
for i in range(len(controllers)):
    sum_carbon_emissions.append(np.array(smoothed_carbon_emissions_DC1[i]) + np.array(smoothed_carbon_emissions_DC2[i]) + np.array(smoothed_carbon_emissions_DC3[i]))

# Plot the sum of carbon emissions for each controller on the same figure with different colors
plt.figure(figsize=(10, 6))
linestyles = ['--', '-.', '-', '--', '-.']
for i in range(len(controllers)):
    plt.plot(sum_carbon_emissions[i][:4*24*7]/1e6, label=controllers[i], linestyle=linestyles[i], linewidth=2, alpha=0.9)
    
plt.title('Sum of Carbon Emissions for Each Controller')
plt.xlabel('Time Step')
plt.ylabel('Sum of Carbon Emissions (MgCO2)')
plt.legend()
plt.grid('on', linestyle='--', alpha=0.5)
plt.show()

# %% Plot to represent the external temperature for each datacenter
i = 0
plt.figure(figsize=(10, 6))
plt.plot(external_temperature_DC1[i][:4*24*7], label=dc_location_mapping['DC1'], linestyle='-', linewidth=2, alpha=1)
plt.plot(external_temperature_DC2[i][:4*24*7], label=dc_location_mapping['DC2'], linestyle='-', linewidth=2, alpha=1)
plt.plot(external_temperature_DC3[i][:4*24*7], label=dc_location_mapping['DC3'], linestyle='-', linewidth=2, alpha=1)
plt.title(f'External Temperature on Each Controller')
plt.xlabel('Time Step')
plt.ylabel('External Temperature (°C)')
plt.legend()
plt.grid('on', linestyle='--', alpha=0.5)
plt.show()

# %%
