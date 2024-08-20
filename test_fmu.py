
#%%
import pyfmi
import numpy as np
import matplotlib.pyplot as plt

import os
# if not os.environ['DYMOLA_RUNTIME_LICENSE']:
    # os.environ["DYMOLA_RUNTIME_LICENSE"] = "C:/Users/guillant/Downloads/00620bb3a060_21556_0000_1.LIC"
#%%
def get_percentage_value(time_series, current_time):
    # Sort the time series just in case it's not sorted
    # time_series.sort()

    # Total duration of the time series
    total_duration = time_series[-1][0]

    # Handle cyclic rotation by calculating the equivalent time within the series
    time_in_cycle = current_time % total_duration

    # Find the appropriate value for the current time
    for i, (time_point, value) in enumerate(time_series):
        if time_in_cycle < time_point:
            return time_series[i-1][1]

    # If current time is exactly at the last point, return the last value
    return time_series[-1][1]

# Example time series
time_series = [(0.0, 10), (900.0, 60), (1800.0, 20), (2700.0, 90), (3600.0, 10)]
# Testing the function
test_times = [0, 500, 1500, 2500, 3500, 4500, 5500]  # Example current times for testing

for current_time in test_times:
    value = get_percentage_value(time_series, current_time)
    print(f"At time {current_time}s, the percentage value is {value}%")
#%%
# Print the current working directory
print(os.getcwd())

# Print the files in the current working directory
print(os.listdir())
#%%
path2fmu = "envs/SLC_MIMO.fmu"

fmu_path = os.path.join(os.getcwd(), path2fmu)
print(f"FMU Path: {fmu_path}")
print(f"File exists: {os.path.exists(fmu_path)}")

# Load the FMU from the path + current working directory
fmu = pyfmi.load_fmu(fmu_path, kind='CS')
# fmu = pyfmi.load_fmu(path2fmu, kind='CS')

# Set simulation parameters
start_time = 0.0
stop_time = 100000
step_size = 15
num_steps = int((stop_time - start_time) / step_size)

# Initialize the model
fmu.setup_experiment(start_time=start_time, stop_time=stop_time)
fmu.initialize()

# Dictionary to store results
results = {}

# Variables to collect
variables = ['serverblock1.heatCapacitor.T', 'serverblock2.heatCapacitor.T', 'serverblock3.heatCapacitor.T','coo.Q_flow',\
             'serverblock1.convection.Q_flow','serverblock2.convection.Q_flow','serverblock3.convection.Q_flow']

for variable_name in variables:
    results[variable_name]=[]
results['time']=[]


input_var_names = ['processer_utilization', 
                   'stPT', 
                   'm_flow_in',  # Prescribed mass flow rate [kg/s]
                   ]
output_var_names = ['mov.P',  # Electrical power consumed [W]
                    'mov.m_flow_actual',  # Actual mass flow rate [kg/s]
                    'tempin.T', 
                    'tempout.T',
                    'tempatmixer.T',
                    'pipe3.sta_b.T',
                    'pipe3.sta_a.T',
                    'coo.m_flow_nominal',
                    ]
for input_var in input_var_names+output_var_names:
    results[input_var] = []
# Example time series
time_series = [(0.0, 10), (900.0, 30), (1800.0, 60), (2700.0, 100), (3600.0, 10)]

#%%
# pregenerate vectorized utilization data
current_time = start_time
prev = 0
input_ts = []
for step in range(num_steps):
    # Utilization, Temperature Setpoint, Pump flowrate
    new = 0.5*np.random.random()
    input_ts.append([get_percentage_value(time_series, current_time),
                     293.15,
                     new
                    ])
    prev = new
    current_time += step_size
#%%
# Run the simulation step-by-step
current_time = start_time
for step,input_ts_list in zip(range(num_steps), input_ts):

    # # set value of a variable
    # for input_var in input_var_names:
    #     if input_var=='processer_utilization':
    #         fmu.set(input_var,get_percentage_value(time_series, current_time))
    #     elif input_var=='stPT':
    #         fmu.set(input_var,np.random.normal(293.15,3.0))
    #     else:
    #         fmu.set(input_var,np.random.normal(1000.0,50.0))
    #     results[input_var].append(fmu.get(input_var))
    # vectorized setting
    fmu.set(input_var_names, input_ts_list)
    # fmu.set('coo.m_flow_nominal', 0.5)

    # Perform a simulation step
    fmu.do_step(current_t=current_time, step_size=step_size)

    # Collect results for the desired variables
    step_results = {var: fmu.get(var) for var in variables+input_var_names+output_var_names}
    print(f'Simulated time: {current_time}')
    # Store the results in the dictionary with the current time as the key
    results['time'].append(current_time)
    for var_name in variables+input_var_names+output_var_names:
        results[var_name].append(step_results[var_name])

    # Update the current time
    current_time += step_size

# Terminate the FMU
fmu.terminate()

#%%
# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'], np.array(results['serverblock1.heatCapacitor.T'])-273.15, 'r', label="serverblock1.heatCapacitor.T")
ax1.plot(results['time'], np.array(results['serverblock2.heatCapacitor.T'])-273.15, 'b', label="serverblock2.heatCapacitor.T")
ax1.plot(results['time'], np.array(results['serverblock3.heatCapacitor.T'])-273.15, 'g', label="serverblock3.heatCapacitor.T")
ax1.set_ylabel('deg C', color='k')
ax1.tick_params('y', colors='k')
ax1.set_xlabel('Time')
plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show() 
#%%
# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'], np.array(results['processer_utilization']), 'r', label='processer_utilization')
ax1.set_ylabel('percentage', color='k')
ax1.tick_params('y', colors='k')
ax1.set_xlabel('Time')
plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()
#%%
# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'], np.array(results['stPT']), 'r', label='stPT')
ax1.set_ylabel('stPT', color='r')
ax1.tick_params('y', colors='r')
ax1.set_xlabel('Time')

# Create a second Y axis
ax2 = ax1.twinx()
# Plot the second array with the second Y axis
ax2.plot(results['time'], np.array(results['m_flow_in']), 'b', label='m_flow_in',alpha=0.6)
ax2.set_ylabel('m_flow_in', color='b')
ax2.tick_params('y', colors='b')

plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()
#%%
# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'], np.array(results['serverblock1.convection.Q_flow']), 'r', label='serverblock1.convection.Q_flow')
ax1.plot(results['time'], np.array(results['serverblock2.convection.Q_flow']), 'g', label='serverblock2.convection.Q_flow')
ax1.plot(results['time'], np.array(results['serverblock3.convection.Q_flow']), 'b', label='serverblock3.convection.Q_flow')
ax1.set_ylabel('Q_flow', color='k')
ax1.tick_params('y', colors='k')
ax1.set_xlabel('Time')

plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()
#%%
# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'][:200000], np.array(results['coo.Q_flow'])[:200000], 'r', label='coo.Q_flow')
ax1.set_ylabel('coo.Q_flow', color='k')
ax1.tick_params('y', colors='k')
ax1.set_xlabel('Time')
plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()
#%% Selected flow rate vs actual flow rate
# 'm_flow_in' vs 'mov.m_flow_actual'

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'], np.array(results['m_flow_in']), 'tab:blue', label='m_flow_in')
ax1.plot(results['time'], np.array(results['mov.m_flow_actual']), 'tab:red', label='mov.m_flow_actual')

ax1.set_xlabel('Time')
plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()
#%% Also plot the Pump power

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'], np.array(results['mov.m_flow_actual']), 'tab:blue', label='mov.m_flow_actual')
ax1.set_ylabel('Actual flow', color='tab:blue')
ax1.tick_params('y', colors='tab:blue')
ax1.set_xlabel('Time')

# Create a second Y axis
ax2 = ax1.twinx()
# Plot the second array with the second Y axis
ax2.plot(results['time'], np.array(results['mov.P']), 'tab:red', label='mov.P',alpha=0.99)
ax2.set_ylabel('mov.P', color='tab:red')
ax2.tick_params('y', colors='tab:red')

plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()

#%% Now, plot the CPU utilization vs the output temperature of the water

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'], np.array(results['processer_utilization']), 'tab:blue', label='processer_utilization')
ax1.set_ylabel('CPU Util', color='tab:blue')
ax1.tick_params('y', colors='tab:blue')
ax1.set_xlabel('Time')

# Create a second Y axis
ax2 = ax1.twinx()
# Plot the second array with the second Y axis
ax2.plot(results['time'], np.array(results['tempatmixer.T'])- 272.15, 'tab:red', label='tempatmixer.T',alpha=0.9)
ax2.set_ylabel('tempatmixer.T', color='tab:red')
ax2.tick_params('y', colors='tab:red')

plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()
#%% Now, plot the intel temperature and the outlet temperature of the water in pipe3

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'], np.array(results['pipe3.sta_a.T']) - 272.15, 'tab:blue', label='pipe3.sta_a.T')
ax1.plot(results['time'], np.array(results['pipe3.sta_b.T'])- 272.15, 'tab:red', label='pipe3.sta_b.T')

ax1.set_xlabel('Time')
plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()

#%% Now flow the coo.flow_nominal
# Create figure and axes
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot the first array with the first Y axis
ax1.plot(results['time'], np.array(results['coo.m_flow_nominal']), 'tab:blue', label='coo.flow_nominal')

ax1.set_xlabel('Time')
plt.legend()
plt.grid()
plt.tight_layout()
# Show the plot
plt.show()




# %%
