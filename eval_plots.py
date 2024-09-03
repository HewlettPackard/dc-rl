
#%%

import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars(logdir, scalar_name):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    if scalar_name in event_acc.Tags()["scalars"]:
        return event_acc.Scalars(scalar_name)
    else:
        return None

def process_runs(base_dir, run_paths, scalar_name):
    all_steps = []
    all_values = []

    for run_path in run_paths:
        full_path = os.path.join(base_dir, run_path, "logs")
        scalars = extract_scalars(full_path, scalar_name)
        
        if scalars:
            steps = np.array([s.step for s in scalars])
            values = np.array([s.value for s in scalars])
            all_steps.append(steps)
            all_values.append(values)

    # Align all runs to the same steps (assuming the steps are aligned)
    min_length = min(len(steps) for steps in all_steps)
    all_steps = [steps[:min_length] for steps in all_steps]
    all_values = [values[:min_length] for values in all_values]
    
    return np.array(all_steps[0]), np.array(all_values)

def plot_avg_std_with_baselines(base_dir, run_paths, scalar_name, baseline_rewards):
    steps, all_values = process_runs(base_dir, run_paths, scalar_name)
    
    avg_values = np.mean(all_values, axis=0)
    std_values = np.std(all_values, axis=0)

    plt.figure(figsize=(3.5, 2.75))
    plt.plot(steps, avg_values, label="RL-LC", color='tab:blue')
    plt.fill_between(steps, avg_values - std_values, avg_values + std_values, color='tab:blue', alpha=0.2)

    # Plot baselines
    plt.axhline(y=baseline_rewards['Random'], color='tab:orange', linestyle='--', label='Random')
    plt.axhline(y=baseline_rewards['Fixed 32°C'], color='tab:green', linestyle='--', label='ASHRAE W32')
    plt.axhline(y=baseline_rewards['Workload Following'], color='tab:red', linestyle='--', label='RBC')

    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--')
    plt.xlim([0, 0.7e8])
    
    plt.tight_layout()
    
    # Save the figure as pdf
    plt.savefig('media/our_vs_baselines_avg_reward.pdf', format='pdf')
    plt.show()

# Define the base directory where your runs are stored
base_dir = "results/sustaindc/az"

# List of run directories relative to the base directory
run_paths = [
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00002-2024-08-26-21-24-52",
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00003-2024-08-26-21-25-05",
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00004-2024-08-26-21-25-18",
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00005-2024-08-26-21-25-32",
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00006-2024-08-26-21-25-42",
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00007-2024-08-26-21-25-52",
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00007-2024-08-26-21-26-33",
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00008-2024-08-26-21-26-46",
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00009-2024-08-26-21-26-54",
    "happo/happo_liquid_dc_256_256_2actions_4obs/seed-00010-2024-08-26-21-27-03"
]

# Name of the scalar you want to plot from TensorBoard logs
scalar_name = "train/average_step_rewards"

# Define baseline rewards
baseline_rewards = {
    'Random': 981,
    'Fixed 32°C': 1108,
    'Workload Following': 1173
}

# Plot the metrics with baselines
plot_avg_std_with_baselines(base_dir, run_paths, scalar_name, baseline_rewards)


#%%
'''
Random Baseline - Total Energy: 1191564.035 ± 179740.681
Trained - Total Energy: 1065007.748 ± 163863.505
Reduction (%): 10.663 ± 0.503
Random Baseline - Carbon Emissions: 1443914844.491 ± 224798730.435
Trained - Carbon Emissions: 1290210996.366 ± 204877177.732
Reduction (%): 10.689 ± 0.508
Fixed Baseline - Total Energy: 1082812.538 ± 167166.507
Trained - Total Energy: 1065007.748 ± 163863.505
Reduction (%): 1.636 ± 0.308
Fixed Baseline - Carbon Emissions: 1311465989.272 ± 208986198.131
Trained - Carbon Emissions: 1290210996.366 ± 204877177.732
Reduction (%): 1.612 ± 0.312
Following Workload Baseline - Total Energy: 1082724.003 ± 167165.904
Trained - Total Energy: 1065007.748 ± 163863.505
Reduction (%): 1.628 ± 0.309
Following Workload Baseline - Carbon Emissions: 1311353780.045 ± 208985542.855
Trained - Carbon Emissions: 1290210996.366 ± 204877177.732
Reduction (%): 1.603 ± 0.314
'''

#%%
import matplotlib.pyplot as plt
import numpy as np

colors = {
    'Random': 'tab:orange',
    'ASHRAE W32': 'tab:green',
    'RBC': 'tab:red',
    'RL-LC': 'tab:blue'
}


# Data
labels = ['Random', 'ASHRAE W32', 'RBC', 'RL-LC']
energy_means = np.array([1191564.035, 1102812.538, 1082724.003, 1065007.748])/1e6
energy_stds = np.array([179740.681, 167166.507, 167165.904, 163863.505])/1e7
carbon_means = np.array([1443914844.491, 1331465989.272, 1311353780.045, 1290210996.366])/1e9
carbon_stds = np.array([224798730.435, 208986198.131, 208985542.855, 204877177.732])/1e10

# Create figure and axes
fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.75))

# Plot Energy Consumption
x = np.arange(len(labels))  # the label locations
width = 0.66  # the width of the bars

ax[0].bar(x, energy_means, width, yerr=energy_stds, capsize=5, 
          label='Energy Consumption', color=[colors[label] for label in labels])

ax[0].set_xlabel('Baselines vs RL-LC')
ax[0].set_ylabel('Avg Energy (MWh)')
# ax[0].set_title('Comparison of Total Energy Consumption')
ax[0].set_xticks(x)
# the size of the x-axis labels should be 2
ax[0].set_xticklabels(labels, fontsize=9)
# ax[0].legend()
ax[0].grid(True, linestyle='--', alpha=0.7)
ax[0].set_ylim([0.95, 1.25])

# Plot Carbon Emissions
ax[1].bar(x, carbon_means, width, yerr=carbon_stds, capsize=5, 
          label='Carbon Emissions', color=[colors[label] for label in labels])

ax[1].set_xlabel('Baselines vs RL-LC')
ax[1].set_ylabel('Avg Carbon Emiss. (Tonnes CO2)')
# ax[1].set_title('Comparison of Carbon Emissions')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels, fontsize=9)
# ax[1].legend()
ax[1].grid(True, linestyle='--', alpha=0.7)
ax[1].set_ylim([1.2, 1.5])

# Show the plot
plt.tight_layout()
# Save the figure as pdf
plt.savefig('media/energy_carbon_comparison.pdf', format='pdf')

plt.show()

# %%
