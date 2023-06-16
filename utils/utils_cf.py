import os
import numpy as np
import pandas as pd

import os

file_path = os.path.abspath(__file__)
PATH = os.path.split(os.path.dirname(file_path))[0]


def obtain_paths(location):
    if "ny" in location.lower():
        return ['NYIS', 'USA_NY_New.York-Kennedy.epw']
    elif "az" in location.lower():
        return ['AZPS', 'USA_AZ_Tucson-Davis-Monthan.epw']
    elif "wa" in location.lower():
        return ['WAAT', 'USA_WA_Port.Angeles-Fairchild.epw']
    else:
        raise ValueError("Location not found")

def get_energy_variables(state):
    energy_vars = np.hstack((state[4:7],(state[7]+state[8])/2))
    return energy_vars

class CoherentNoise:
    def __init__(self, base, weight, desired_std_dev=0.1, scale=1):
        self.base = base
        self.weight = weight
        self.desired_std_dev = desired_std_dev
        self.scale = scale

    def generate(self, n_steps):
        steps = np.random.normal(loc=0, scale=self.scale, size=n_steps)
        random_walk = np.cumsum(self.weight * steps)
        random_walk_scaled = self.base + (random_walk / np.std(random_walk)) * self.desired_std_dev
        return random_walk_scaled


# Function to get the initial index of the day of a given month from a time-stamped dataset
def get_init_day(start_month=0):
    assert 0 <= start_month <= 11, "start_month should be between 0 and 11 (inclusive, 0-based, 0=January, 11=December)."

    # Read the CSV file and parse dates from the 'timestamp' column
    df = pd.read_csv(PATH+'/data/CarbonIntensity/NYIS_NG_&_avgCI.csv', parse_dates=['timestamp'], usecols=['timestamp'])
    
    # Extract the month from each timestamp and add it as a new column to the DataFrame
    df['month'] = pd.DatetimeIndex(df['timestamp']).month
    
    # Find the first day of the specified start month
    init_day = df[df['month'] == start_month+1].index[0]
    
    # Return the day number (0-based)
    return int(init_day/24)


# Function to normalize a value v given a minimum and a maximum
def normalize(v, min_v, max_v):
    return (v - min_v)/(max_v - min_v)

def standarize(v):
    return (v - np.mean(v))/np.std(v)

# Function to generate cosine and sine values for a given hour and day
def sc_obs(current_hour, current_day):
    # Normalize and round the current hour and day
    two_pi = np.pi * 2

    norm_hour = round(current_hour/24, 3) * two_pi
    norm_day = round((current_day)/365, 3) * two_pi
    
    # Calculate cosine and sine values for the current hour and day
    cos_hour = np.cos(norm_hour)*0.5 + 0.5
    sin_hour = np.sin(norm_hour)*0.5 + 0.5
    cos_day = np.cos(norm_day)*0.5 + 0.5
    sin_day = np.sin(norm_day)*0.5 + 0.5
    
    return [cos_hour, sin_hour, cos_day, sin_day]


class Time_Manager():
    def __init__(self, init_day=0, days_per_episode=30):
        self.init_day = init_day
        self.timestep_per_hour = 4
        self.days_per_episode = days_per_episode

    def reset(self):
        self.day = self.init_day
        self.hour = 0
        return sc_obs(self.hour, self.day)
        
    def step(self):
        if self.hour >= 24:
            self.hour=0
            self.day += 1
        self.hour += 1/self.timestep_per_hour
        return self.day, self.hour, sc_obs(self.hour, self.day), self.isterminal()
    
    def isterminal(self):
        done = False
        if self.day > self.init_day+self.days_per_episode:
            done = True
        return done




# Class to manage CPU workload data
class Workload_Manager():
    def __init__(self, filename='', init_day=0, future_steps=4, weight=0.001, desired_std_dev=0.025, flexible_workload_ratio=0.1):
        assert 0 <= flexible_workload_ratio <= 1, "flexible_workload_ratio should be between 0 and 1 (inclusive)."

        # Load CPU data from a CSV file
        # One year data=24*365=8760
        if filename == '':
            cpu_data_list = pd.read_csv(PATH+'/data/Workload/Alibaba_CPU_Data_Hourly_1.csv')['cpu_load'].values[:8760]
        else:
            cpu_data_list = pd.read_csv(PATH+f'/data/Workload/{filename}')['cpu_load'].values[:8760]

        assert len(cpu_data_list) == 8760, "The number of data points in the workload data is not one year data=24*365=8760."

        cpu_data_list = cpu_data_list.astype(float)
        self.time_step = 0
        self.future_steps = future_steps
        self.timestep_per_hour = 4
        self.time_steps_day = self.timestep_per_hour*24
        self.flexible_workload_ratio = flexible_workload_ratio
        self.init_day = init_day

        # Interpolate the CPU data to increase the number of data points
        x = range(0, len(cpu_data_list))
        xcpu_new = np.linspace(0, len(cpu_data_list), len(cpu_data_list)*self.timestep_per_hour)  
        self.cpu_smooth = np.interp(xcpu_new, x, cpu_data_list)
        
        # Save a copy of the original data
        self.original_data = self.cpu_smooth.copy()
                
        # Initialize CoherentNoise process
        self.coherent_noise = CoherentNoise(base=self.original_data[0], weight=weight, desired_std_dev=desired_std_dev)

    # Function to return all workload data
    def get_total_wkl(self):
        return np.array(self.cpu_smooth)

    # Function to reset the time step and return the workload at the first time step
    def reset(self):
        self.time_step = self.init_day*self.time_steps_day
        self.init_time_step = self.time_step
        
        # Add noise to the workload data using the CoherentNoise 
        self.cpu_smooth = self.original_data + self.coherent_noise.generate(len(self.original_data))

        self.cpu_smooth = np.clip(self.cpu_smooth, 0, 1)
        self.cpu_smooth = self.cpu_smooth * (1-self.flexible_workload_ratio)
        self.storage_load = np.sum(self.cpu_smooth[self.time_step:self.time_step+self.time_steps_day]*self.flexible_workload_ratio)

        return self.cpu_smooth[self.time_step], self.storage_load
        
    # Function to advance the time step and return the workload at the new time step
    def step(self):
        self.time_step += 1
        data_load = 0
        if self.time_step % self.time_steps_day == 0 and self.time_step != self.init_time_step:
            self.storage_load = np.sum(self.cpu_smooth[self.time_step:self.time_step+self.time_steps_day]*self.flexible_workload_ratio)
            data_load = self.storage_load
        
        # If it tries to read further, restart from the inital day
        if self.time_step >= len(self.cpu_smooth):
            self.time_step = self.init_time_step
        # assert self.time_step < len(self.cpu_smooth), f'Episode length: {self.time_step} is longer than the provide cpu_smooth: {len(self.cpu_smooth)}'
        return self.cpu_smooth[self.time_step], data_load
    
# Class to manage carbon intensity data
class CI_Manager():
    def __init__(self, filename='', location='NYIS', init_day=0, future_steps=4, weight=0.1, desired_std_dev=5):
        # Load carbon intensity data from a CSV file
        # One year data=24*365=8760
        if filename == '':
            carbon_data_list = pd.read_csv(PATH+f"/data/CarbonIntensity/{location}_NG_&_avgCI.csv")['avg_CI'].values[:8760]
        else:
            carbon_data_list = pd.read_csv(PATH+f"/data/CarbonIntensity/{filename}")['avg_CI'].values[:8760]

        assert len(carbon_data_list) == 8760, "The number of data points in the carbon intensity data is not one year data=24*365=8760."
        
        carbon_data_list = carbon_data_list.astype(float)
        self.init_day = init_day
        
        self.timestep_per_hour = 4
        self.time_steps_day = self.timestep_per_hour*24
        
        x = range(0, len(carbon_data_list))
        xcarbon_new = np.linspace(0, len(carbon_data_list), len(carbon_data_list)*self.timestep_per_hour)
        
        # Interpolate the carbon data to increase the number of data points
        self.carbon_smooth = np.interp(xcarbon_new, x, carbon_data_list)
        
        # Save a copy of the original data
        self.original_data = self.carbon_smooth.copy()
        
        self.time_step = 0
        
        # Initialize CoherentNoise process
        self.coherent_noise = CoherentNoise(base=self.original_data[0], weight=weight, desired_std_dev=desired_std_dev)
        
        self.future_steps = future_steps
        

    # Function to return all carbon intensity data
    def get_total_ci(self):
        return self.carbon_smooth

    def reset(self):
        self.time_step = self.init_day*self.time_steps_day
        
        # Add noise to the carbon data using the CoherentNoise
        self.carbon_smooth = self.original_data + self.coherent_noise.generate(len(self.original_data))
        
        self.carbon_smooth = np.clip(self.carbon_smooth, 0, None)

        self.min_ci = min(self.carbon_smooth)
        self.max_ci = max(self.carbon_smooth)
        # self.norm_carbon = normalize(self.carbon_smooth, self.min_ci, self.max_ci)
        self.norm_carbon = standarize(self.carbon_smooth)
        self.norm_carbon = (np.clip(self.norm_carbon, -1, 1) + 1) * 0.5

        return self.carbon_smooth[self.time_step], self.norm_carbon[self.time_step:self.time_step+self.future_steps]
    
    # Function to advance the time step and return the carbon intensity at the new time step
    def step(self):
        self.time_step +=1
        
        # If it tries to read further, restart from the initial index
        if self.time_step >= len(self.carbon_smooth):
            self.time_step = self.init_day*self.time_steps_day
            
        # assert self.time_step < len(self.carbon_smooth), 'Eposide length is longer than the provide CI_data'
        if self.time_step + self.future_steps > len(self.carbon_smooth):
            data = self.norm_carbon[self.time_step]*np.ones(shape=(self.future_steps))
        else:
            data = self.norm_carbon[self.time_step:self.time_step+self.future_steps]

        return self.carbon_smooth[self.time_step], data

# Class to manage weather data
# Where to obtain other weather files:
# https://climate.onebuilding.org/
class Weather_Manager():
    def __init__(self, filename='', location='NY', init_day=0, weight=0.01, desired_std_dev=0.5, temp_column=6):
        # Load weather data from a CSV file
        if filename == '':
            temperature_data = pd.read_csv(PATH+f'/data/Weather/{location}', skiprows=8, header=None).values[:,temp_column]
        else:
            temperature_data = pd.read_csv(PATH+f'/data/Weather/{filename}', skiprows=8, header=None).values[:,temp_column]
        
        temperature_data = temperature_data.astype(float)
        self.init_day = init_day
        # One year data=24*365=8760
        x = range(0, len(temperature_data))
        xtemperature_new = np.linspace(0, len(temperature_data), len(temperature_data)*4)
        
        self.min_temp = -10
        self.max_temp = 40
        
        # Interpolate the carbon data to increase the number of data points
        self.temperature_data = np.interp(xtemperature_new, x, temperature_data)
        self.norm_temp_data = normalize(self.temperature_data, self.min_temp, self.max_temp)

        self.time_step = 0
        
        # Save a copy of the original data
        self.original_data = self.temperature_data.copy()
        
        # Initialize CoherentNoise process
        self.coherent_noise = CoherentNoise(base=self.original_data[0], weight=weight, desired_std_dev=desired_std_dev)
                
        self.timestep_per_hour = 4
        self.time_steps_day = self.timestep_per_hour*24

    # Function to return all weather data
    def get_total_weather(self):
        return self.temperature_data

    # Function to reset the time step and return the weather at the first time step
    def reset(self):
        self.time_step = self.init_day*self.time_steps_day
        
        # Add noise to the temperature data using the CoherentNoise
        self.temperature_data = self.original_data + self.coherent_noise.generate(len(self.original_data))
        
        self.temperature_data = np.clip(self.temperature_data, self.min_temp, self.max_temp)
        self.norm_temp_data = normalize(self.temperature_data, self.min_temp, self.max_temp)

        return self.temperature_data[self.time_step], self.norm_temp_data[self.time_step]
    
    # Function to advance the time step and return the weather at the new time step
    def step(self):
        self.time_step += 1
        
        # If it tries to read further, restart from the initial index
        if self.time_step >= len(self.temperature_data):
            self.time_step = self.init_day*self.time_steps_day
            
        # assert self.time_step < len(self.temperature_data), 'Episode length is longer than the provide Temperature_data'
        return self.temperature_data[self.time_step], self.norm_temp_data[self.time_step]

