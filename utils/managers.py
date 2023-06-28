import os
import numpy as np
import pandas as pd

file_path = os.path.abspath(__file__)
PATH = os.path.split(os.path.dirname(file_path))[0]

class CoherentNoise:
    """Class to add coherent noise to the data.

        Args:
            base (List[float]): Base data
            weight (float): Weight of the noise to be added
            desired_std_dev (float, optional): Desired standard deviation. Defaults to 0.1.
            scale (int, optional): Scale. Defaults to 1.
    """
    def __init__(self, base, weight, desired_std_dev=0.1, scale=1):
        """Initialize CoherentNoise class

        Args:
            base (List[float]): Base data
            weight (float): Weight of the noise to be added
            desired_std_dev (float, optional): Desired standard deviation. Defaults to 0.1.
            scale (int, optional): Scale. Defaults to 1.
        """
        self.base = base
        self.weight = weight
        self.desired_std_dev = desired_std_dev
        self.scale = scale

    def generate(self, n_steps):
        """
        Generate coherent noise 

        Args:
            n_steps (int): Length of the data to generate.

        Returns:
            numpy.ndarray: Array of generated coherent noise.
        """
        steps = np.random.normal(loc=0, scale=self.scale, size=n_steps)
        random_walk = np.cumsum(self.weight * steps)
        random_walk_scaled = self.base + (random_walk / np.std(random_walk)) * self.desired_std_dev
        return random_walk_scaled


# Function to normalize a value v given a minimum and a maximum
def normalize(v, min_v, max_v):
    """Function to normalize values

    Args:
        v (float): Value to be normalized
        min_v (float): Lower limit
        max_v (float): Upper limit

    Returns:
        float: Normalized value
    """
    return (v - min_v)/(max_v - min_v)

def standarize(v):
    """Function to standarize a list of values

    Args:
        v (float): Values to be normalized

    Returns:
        float: Normalized values
    """
    return (v - np.mean(v))/np.std(v)

# Function to generate cosine and sine values for a given hour and day
def sc_obs(current_hour, current_day):
    """Generate sine and cosine of the hour and day

    Args:
        current_hour (int): Current hour of the day
        current_day (int): Current day of the year

    Returns:
        List[float]: Sine and cosine of the hour and day
    """
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
    """Class to manage the time dimenssion over an episode

        Args:
            init_day (int, optional): Day to start from. Defaults to 0.
            days_per_episode (int, optional): Number of days that an episode would last. Defaults to 30.
    """
    def __init__(self, init_day=0, days_per_episode=30):
        """Class to manage the time dimenssion over an episode

        Args:
            init_day (int, optional): Day to start from. Defaults to 0.
            days_per_episode (int, optional): Number of days that an episode would last. Defaults to 30.
        """
        self.init_day = init_day
        self.timestep_per_hour = 4
        self.days_per_episode = days_per_episode

    def reset(self):
        """Reset time manager to initial day

        Returns:
            List[float]: Hour and day in sine and cosine form
        """
        self.day = self.init_day
        self.hour = 0
        return sc_obs(self.hour, self.day)
        
    def step(self):
        """Step function for the time maneger

        Returns:
            List[float]: Current hour and day in sine and cosine form.
            bool: Signal if the episode has reach the end.
        """
        if self.hour >= 24:
            self.hour=0
            self.day += 1
        self.hour += 1/self.timestep_per_hour
        return self.day, self.hour, sc_obs(self.hour, self.day), self.isterminal()
    
    def isterminal(self):
        """Function to identify terminal state

        Returns:
            bool: Signals if a state is terminal or not
        """
        done = False
        if self.day > self.init_day+self.days_per_episode:
            done = True
        return done


# Class to manage CPU workload data
class Workload_Manager():
    """Manager of the DC workload

        Args:
            workload_filename (str, optional): Filename of the CPU data. Defaults to ''. Should be a .csv file containing the CPU hourly normalized workload data between 0 and 1. Should contains 'cpu_load' column.
            init_day (int, optional): Initial day of the episode. Defaults to 0.
            future_steps (int, optional): Number of steps of the workload forecast. Defaults to 4.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.001.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 0.025.
            flexible_workload_ratio (float, optional): Ratio of the flexible workload amount. Defaults to 0.1.
    """
    def __init__(self, workload_filename='', init_day=0, future_steps=4, weight=0.001, desired_std_dev=0.025, flexible_workload_ratio=0.1):
        """Manager of the DC workload

        Args:
            workload_filename (str, optional): Filename of the CPU data. Defaults to ''. Should be a .csv file containing the CPU hourly normalized workload data between 0 and 1. Should contains 'cpu_load' column.
            init_day (int, optional): Initial day of the episode. Defaults to 0.
            future_steps (int, optional): Number of steps of the workload forecast. Defaults to 4.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.001.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 0.025.
            flexible_workload_ratio (float, optional): Ratio of the flexible workload amount. Defaults to 0.1.
        """
        assert 0 <= flexible_workload_ratio <= 1, "flexible_workload_ratio should be between 0 and 1 (inclusive)."

        # Load CPU data from a CSV file
        # One year data=24*365=8760
        if workload_filename == '':
            cpu_data_list = pd.read_csv(PATH+'/data/Workload/Alibaba_CPU_Data_Hourly_1.csv')['cpu_load'].values[:8760]
        else:
            cpu_data_list = pd.read_csv(PATH+f'/data/Workload/{workload_filename}')['cpu_load'].values[:8760]

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
        """Get current workload

        Returns:
            List[float]: CPU data
        """
        return np.array(self.cpu_smooth)

    # Function to reset the time step and return the workload at the first time step
    def reset(self):
        """Reset Workload_Manager

        Returns:
            float: CPU workload at current time step
            float: Amount of daily flexible workload
        """
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
        """Step function for the Workload_Manager

        Returns:
            float: CPU workload at current time step
            float: Amount of daily flexible workload
        """
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
    """Manager of the carbon intensity data

        Args:
            filename (str, optional): Filename of the CPU data. Defaults to ' '.
            location (str, optional): Location identifier. Defaults to 'NYIS'.
            init_day (int, optional): Initial day of the episode. Defaults to 0.
            future_steps (int, optional): Number of steps of the CI forecast. Defaults to 4.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.001.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 0.025.
    """
    def __init__(self, filename='', location='NYIS', init_day=0, future_steps=4, weight=0.1, desired_std_dev=5):
        """Manager of the carbon intesity data

        Args:
            filename (str, optional): Filename of the CPU data. Defaults to ''.
            location (str, optional): Location identifier. Defaults to 'NYIS'.
            init_day (int, optional): Initial day of the episode. Defaults to 0.
            future_steps (int, optional): Number of steps of the CI forecast. Defaults to 4.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.001.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 0.025.
        """
        # Load carbon intensity data from a CSV file
        # One year data=24*365=8760
        if not location == '':
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
        """Function to obtain the total carbon intensity

        Returns:
            List[float]: Total carbon intesity
        """
        return self.carbon_smooth

    def reset(self):
        """Reset CI_Manager

        Returns:
            float: Carbon intensity at current time step
            float: Normalized carbon intensity at current time step and it's forecast
        """
        self.time_step = self.init_day*self.time_steps_day
        
        # Add noise to the carbon data using the CoherentNoise
        self.carbon_smooth = self.original_data + self.coherent_noise.generate(len(self.original_data))
        
        self.carbon_smooth = np.clip(self.carbon_smooth, 0, None)

        self.min_ci = min(self.carbon_smooth)
        self.max_ci = max(self.carbon_smooth)
        self.norm_carbon = normalize(self.carbon_smooth, self.min_ci, self.max_ci)
        # self.norm_carbon = standarize(self.carbon_smooth)
        # self.norm_carbon = (np.clip(self.norm_carbon, -1, 1) + 1) * 0.5

        return self.carbon_smooth[self.time_step], self.norm_carbon[self.time_step:self.time_step+self.future_steps]
    
    # Function to advance the time step and return the carbon intensity at the new time step
    def step(self):
        """Step CI_Manager

        Returns:
            float: Carbon intensity at current time step
            float: Normalized carbon intensity at current time step and it's forecast
        """
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
    """Manager of the weather data.
       Where to obtain other weather files:
       https://climate.onebuilding.org/

        Args:
            filename (str, optional): Filename of the weather data. Defaults to ''.
            location (str, optional): Location identifier. Defaults to 'NY'.
            init_day (int, optional): Initial day of the year. Defaults to 0.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.001.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 0.025.
            temp_column (int, optional): Columng that contains the temperature data. Defaults to 6.
    """
    def __init__(self, filename='', location='NY', init_day=0, weight=0.01, desired_std_dev=0.5, temp_column=6):
        """Manager of the weather data.

        Args:
            filename (str, optional): Filename of the weather data. Defaults to ''.
            location (str, optional): Location identifier. Defaults to 'NY'.
            init_day (int, optional): Initial day of the year. Defaults to 0.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.001.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 0.025.
            temp_column (int, optional): Columng that contains the temperature data. Defaults to 6.
        """
        # Load weather data from a CSV file

        if not location == '':
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
        """Obtain the weather data in a List form

        Returns:
            List[form]: Total temperature data
        """
        return self.temperature_data

    # Function to reset the time step and return the weather at the first time step
    def reset(self):
        """Reset Weather_Manager

        Returns:
            float: Temperature a current step
            float: Normalized temperature a current step
        """
        self.time_step = self.init_day*self.time_steps_day
        
        # Add noise to the temperature data using the CoherentNoise
        self.temperature_data = self.original_data + self.coherent_noise.generate(len(self.original_data))
        
        self.temperature_data = np.clip(self.temperature_data, self.min_temp, self.max_temp)
        self.norm_temp_data = normalize(self.temperature_data, self.min_temp, self.max_temp)

        return self.temperature_data[self.time_step], self.norm_temp_data[self.time_step]
    
    # Function to advance the time step and return the weather at the new time step
    def step(self):
        """Step on the Weather_Manager

        Returns:
            float: Temperature a current step
            float: Normalized temperature a current step
        """
        self.time_step += 1
        
        # If it tries to read further, restart from the initial index
        if self.time_step >= len(self.temperature_data):
            self.time_step = self.init_day*self.time_steps_day
            
        # assert self.time_step < len(self.temperature_data), 'Episode length is longer than the provide Temperature_data'
        return self.temperature_data[self.time_step], self.norm_temp_data[self.time_step]