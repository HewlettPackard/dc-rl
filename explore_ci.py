#%%
import os
import pandas as pd

import matplotlib.pyplot as plt

#%%
folder_path = '/lustre/guillant/dc-rl/data/CarbonIntensity'

# Get all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Read each CSV file and obtain the avg_CI column and save it along with the location name in a dictionary
values = {}

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    values[file[:2]] = df['avg_CI']

#%%
# Now plot the values and the legend should be the key of the values dictionary
# I want to plot only the month 7.
# Knowing that the values on the csv are 1 hour apart, I can get the index of the first day of the month and the last day of the month
# and then plot only the values between those indexes
# I can use the index to get the values between those indexes
import numpy as np
selected_month = 7
init_index = selected_month * 30 * 24
end_index = (selected_month + 1) * 30 * 24

x_range = np.arange(init_index, end_index)/24
plt.figure(figsize=(10, 5))
for key, value in values.items():
    if key in ['IL', 'TX', 'NY', 'VA', 'GA', 'WA', 'AZ', 'CA']:
        # plt.plot(value[init_index:end_index]**3/200000, label=key, linestyle='-', linewidth=2, alpha=0.9)
        plt.plot(x_range, value[init_index:end_index], label=key, linestyle='-', linewidth=2, alpha=0.9)

plt.ylabel('Carbon Intensity (gCO2/kWh)', fontsize=16)
plt.xlabel('Day', fontsize=16)
# plt.xlim(init_index/24, end_index/24)
plt.title('Average Daily Carbon Intensity in Different Locations in July', fontsize=18)
plt.grid('on', linestyle='--', alpha=0.5)

plt.tick_params(axis='x', labelsize=12, rotation=45)  # Set the font size of xticks
plt.tick_params(axis='y', labelsize=12)  # Set the font size of yticks
plt.legend(fontsize=11.5, ncols=8)
plt.xlim(210, 240)
plt.ylim(-1)

plt.savefig('plots/GreenDCC_ci_all_locations.pdf', bbox_inches='tight')
plt.show()

#%%