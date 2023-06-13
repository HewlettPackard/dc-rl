import os
import numpy as np
import pandas as pd

# Get the parent directory of the current working directory
PARENT_ABS_PATH = os.getcwd()  # os.path.split(os.getcwd())[0] only for jupyter notebook

# Author : AGP
class OUProcess:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2, dt=1e-2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x = np.ones(self.size) * self.mu

    def __call__(self):
        dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x = self.x + dx
        return self.x
    
def dfs_creator(weather_filename :str,
                carbon_intensity_filename : str,
                workload_filename: str,
                start_date_time : str = '01/01/2022 00:00:00',
                end_date_time : str = '01/01/2023 00:00:00',
                weather_data_freq : str = '1H',
                carbon_intensity_data_freq : str = '1H',
                workload_data_freq : str = '1H',
                sampling_frequency : str = '15min'):
    """_summary_

    Args:
        weather_filename (str): weather file
        carbon_intensity_filename (str): carbon intensity file
        workload_filename (str): cpu workload file; has data in fraction of full load
        start_date_time (_type_, optional): dataset start date. Defaults to '01/01/2022 00:00:00'.
        end_date_time (_type_, optional): dataset end date. Defaults to '01/01/2023 00:00:00'.
        weather_data_freq (str, optional): freq of weather data in weather_filename. Defaults to '1H'.
        carbon_intensity_freq (str, optional): freq ofcarbon_intensity data in carbon_intensity_filename. Defaults to '1H'.
        workload_intensity_freq (str, optional): freq of workload data in workload_filename. Defaults to '1H'.
        sampling_frequency (str, optional): Desired sampling frequency. Defaults to '15min'.
    """
    ############## weather dataframe #######################################
    weather_time_idx = pd.date_range(start = pd.to_datetime(start_date_time),
                                    end =pd.to_datetime(end_date_time),
                                    freq = weather_data_freq,
                                    inclusive='left')
    weather_series = pd.read_csv(PARENT_ABS_PATH + f'/data/Weather/{weather_filename}', skiprows=8, header=None).iloc[:,[6]]
    weather_series.columns = ['dry bulb temperature']
    weather_series.index = weather_time_idx
    weather_series = weather_series.resample('15min').interpolate(method='linear')
    
    ############## carbon intensity #########################################
    carbon_intensity_time_idx = pd.date_range(start = pd.to_datetime(start_date_time),
                                                end =pd.to_datetime(end_date_time),
                                                freq = carbon_intensity_data_freq,
                                                inclusive='left')
    carbon_intensity_series = pd.read_csv(PARENT_ABS_PATH + f'/data/CarbonIntensity/{carbon_intensity_filename}').loc[:,['avg_CI']]
    carbon_intensity_series.index = carbon_intensity_time_idx
    carbon_intensity_series = carbon_intensity_series.resample('15min').interpolate(method='linear')
    
    ############## cpu workload #############################################
    workload_time_idx = pd.date_range(start = pd.to_datetime(start_date_time),
                                    end =pd.to_datetime(end_date_time),
                                    freq = workload_data_freq,
                                    inclusive='left')
    workload_series = pd.read_csv(PARENT_ABS_PATH + f'/data/Workload/{workload_filename}').loc[:,['cpu_load']].iloc[:8760,:]
    workload_series.columns = ['cpu usage']
    workload_series.index = workload_time_idx
    workload_series = workload_series.resample('15min').interpolate(method='linear')
    
                    
    return weather_series, carbon_intensity_series, workload_series
    
    
    
    
    
    
    