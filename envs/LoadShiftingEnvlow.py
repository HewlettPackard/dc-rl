import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import pandas as pd
import math 


class DataCenterEnv(gym.Env):
    """Custom Environment that simulates load shifting in a data center with forecasted carbon intensity."""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, forecast_hours=1, relative_ci=False, include_past_ci=False, queue_max_len=500):
        super(DataCenterEnv, self).__init__()
        
        self.forecast_intervals = forecast_hours
        self.relative_ci = relative_ci  # Flag to control CI representation
        self.include_past_ci = include_past_ci
        
        # Constants
        self.num_intervals_per_day = 24*4  # 15-minute intervals in a day
        self.simulated_days = 30
        
        self.min_carbon_intensity = 0  # Maximum simulated carbon intensity
        self.max_carbon_intensity = 1  # Maximum simulated carbon intensity
        
        self.min_dc_energy = 100
        self.max_dc_energy = 1000
        
        # Define action and observation space
        # Actions: 0 - Decrease, 1 - Do Nothing, 2 - Increase utilization
        self.action_space = spaces.Discrete(3)
        
        # Dynamically adjust observation space based on forecast_hours
        num_features_forecasted = 1 + self.forecast_intervals  # Current + forecasted CI + past_CI
        total_features = 4 + num_features_forecasted + 3  # [Sin(h), Cos(h), Sin(day), Cos(day)], CI (N), [DC energy, workload, queue status]
        
        # Using a large value for low and high
        # Note: Ensure these values do not interfere with the learning process by being too restrictive or too liberal
        low = np.array([-1, -1, -1, -1] + [0]*num_features_forecasted + [0, 0, 0])

        high = np.array([1, 1, 1, 1] + [1]*num_features_forecasted + [1, 1, 1])
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float16)
        
        # Environment state
        self.current_step = 0
        
        # Initialize the queue to manage individual delayed tasks
        self.task_queue = []  # A list to hold individual tasks
        self.queue_max_len = queue_max_len
        
        # Generate a year's worth of simulated data for carbon intensity and workload
        self.simulate_external_data()


    def simulate_external_data(self):
        self.carbon_intensity_data = self._generate_random_carbon_intensity(self.num_intervals_per_day)
        self.min_carbon_intensity = np.min(self.carbon_intensity_data)
        self.max_carbon_intensity = np.max(self.carbon_intensity_data)

        self.carbon_intensity_data = (self.carbon_intensity_data - self.min_carbon_intensity) / (self.max_carbon_intensity - self.min_carbon_intensity)
        # self.workload_data = np.random.uniform(low=0.5, high=1, size=self.simulated_days*self.num_intervals_per_day)
        self.workload_data = self.generate_workload_data()

    def generate_workload_data(self):
        days = self.simulated_days
        intervals_per_day = self.num_intervals_per_day
        total_intervals = days * intervals_per_day

        # Daily cycle - higher during day, lower at night
        daily_cycle = (np.sin(np.linspace(0, 2 * np.pi, intervals_per_day)) + 1) / np.random.uniform(1.5, 3)
        # Repeat the daily cycle for each day
        daily_pattern = np.tile(daily_cycle, days)

        # Weekly cycle - simulate higher workload on weekdays vs weekends
        weekly_cycle = np.array([1 if i % 7 < 5 else np.random.uniform(0.75, 1.5) for i in range(days)])  # 1 for weekdays, 0.7 for weekends
        # Repeat and tile the weekly pattern to match the daily intervals
        weekly_pattern = np.repeat(weekly_cycle, intervals_per_day)

        # Combine daily and weekly patterns
        combined_pattern = daily_pattern * weekly_pattern

        # Add growth trend (optional) - gradually increase workload over time to simulate growth
        growth_trend = np.linspace(np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2), total_intervals)  # Starting at 90% going up to 110% of the load

        # Add random noise for variability
        noise = np.random.uniform(-0.2, 0.2, total_intervals)  # Adjust the range as needed

        # Combine everything and ensure the workload is within realistic bounds (e.g., 0.5 to 1.0)
        workload_data = np.clip(combined_pattern * growth_trend + noise, 0.0, 1.0)
        return workload_data
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
        # Additionally, setting the global NumPy random seed
        np.random.seed(seed)
        
        return [seed]


    def reset(self, seed=None, options=None):
        # Optionally set a new seed for randomness control
        super().reset(seed=seed)  # This is crucial for gym version >= 26 compatibility
        
        
        # Regenerate external data for a new episode
        self.simulate_external_data()
        
        # Clear the task queue
        self.task_queue = []
        
        # Reset current utilization to some initial state, if necessary
        # For simplicity, we can start with 0 or an average expected utilization
        self.current_utilization = 0.0
        
        # Optionally, if you want to randomize the starting point of each episode (for variability)
        # you can uncomment the following line:
        # self.current_step = np.random.randint(0, len(self.workload_data) - (24*4)*7)  # For example, start at a random day
        
        # Reset the current step to the beginning of the dataset
        self.current_step = 0
        
        # Generate the initial observation by calling _get_observation
        initial_observation = self._get_observation()
        
        info = {'current_step': self.current_step
                }
        return initial_observation, info
    
    
    def step(self, action):
        non_shiftable_tasks_percentage = 0.8
        shiftable_tasks_percentage = 1 - non_shiftable_tasks_percentage

        assert self.action_space.contains(action), f"{action} is an invalid action"
        
        non_shiftable_tasks = int(math.ceil(self.workload_data[self.current_step] * non_shiftable_tasks_percentage * 100))
        shiftable_tasks = int(math.floor(self.workload_data[self.current_step] * shiftable_tasks_percentage * 100))
        tasks_dropped = 0  # Track the number of dropped tasks


        if action == 0:  # Use only the non_shiftable_tasks.
            # Attempt to queue shiftable tasks, tracking any that are dropped
            timestamp = self.current_step  # Current timestamp
            for _ in range(shiftable_tasks):
                if len(self.task_queue) < self.queue_max_len:  # Check if adding the task would exceed the queue limit
                    self.task_queue.append({'timestamp': timestamp, 'utilization': 1})
                else:
                    tasks_dropped += 1  # Increment dropped tasks count
            self.current_utilization = non_shiftable_tasks / 100
            
        # if action == 1, do nothing.
        elif action == 1:
            self.current_utilization = (non_shiftable_tasks + shiftable_tasks) / 100
        
        elif action == 2:  # Attempt to process as many tasks from the queue as possible or the max utilization (100%)
            # Determine the number of tasks that can be processed, considering total utilization limits
            tasks_to_process = min(len(self.task_queue), 100 - (non_shiftable_tasks + shiftable_tasks))
            for _ in range(tasks_to_process):
                if self.task_queue:
                    self.task_queue.pop(0)  # Remove a task from the queue for processing
            self.current_utilization = (non_shiftable_tasks + shiftable_tasks + tasks_to_process) / 100
    
        # Calculate energy consumed
        agent_energy_consumed = self._calculate_DC_energy()
        
        # Calculate carbon emissions
        current_carbon_intensity = self.carbon_intensity_data[self.current_step]
        agent_carbon_emissions = agent_energy_consumed * current_carbon_intensity
        
        # Update reward: negative for emissions, potentially penalize leftover tasks in the queue
        reward = -agent_carbon_emissions
        
        # Penalize the agent for each task that was dropped due to queue limit
        penalty_per_dropped_task = -10  # Define the penalty value per dropped task
        reward += tasks_dropped * penalty_per_dropped_task
    
        tasks_in_queue = len(self.task_queue)
        if self.current_step % (24*4) >= (23*4):   # Penalty for queued tasks at the end of the day
            factor_hour = (self.current_step % (24*4)) / 96 # min = 0.95833, max = 0.98953
            factor_hour = (factor_hour - 0.95833) / (0.98935 - 0.95833)
            reward -= factor_hour * tasks_in_queue/10  # Penalty for each task left in the queue
        
        if self.current_step % (24*4) == 0:   # Penalty for queued tasks at the end of the day
            reward -= len(self.task_queue)/10 # Penalty for each task left in the queue
            self.task_queue = []
        
        # Prepare for next step
        self.current_step += 1
        done = self.current_step >= len(self.workload_data) - 1  # Check if simulation is over
        
        # Construct observation
        observation = self._get_observation()  # Ensure you implement this method to get the current state
        
        # Additional info, could include things like current utilization, energy consumed, etc.
        info = {
            'current_utilization': self.current_utilization,
            'carbon_intensity': current_carbon_intensity,
            'energy_consumed': agent_energy_consumed,
            'carbon_emissions': agent_carbon_emissions,
            'tasks_in_queue': tasks_in_queue, 
            'current_step': self.current_step,
            'tasks_dropped': tasks_dropped
        }
        
        truncated = done
        
        return observation, reward, done, truncated, info
    
    def _get_observation(self):
        # [Sin(h), Cos(h), Sin(day), Cos(day)], CI (N), [DC energy, workload, queue status]
        
        # Calculate sine and cosine of the time of day
        current_interval = self.current_step % (24*4)  # Assuming 4 intervals per hour
        radians_per_interval = 2 * np.pi / (24*4)
        sin_time_of_day = np.sin(current_interval * radians_per_interval)
        cos_time_of_day = np.cos(current_interval * radians_per_interval)
        
        # Calculate current day of the week (assuming self.current_step starts from 0 at the beginning of the simulated period)
        # Assuming 0 = Monday, 6 = Sunday
        current_day_of_week = (self.current_step // (24*4)) % 7
        radians_per_day = 2 * np.pi / 7
        sin_day_of_week = np.sin(current_day_of_week * radians_per_day)
        cos_day_of_week = np.cos(current_day_of_week * radians_per_day)
    
        # Current carbon intensity
        current_CI = self.carbon_intensity_data[self.current_step]
        
        step = 4 # One sample every hour
            
        # Forecasted carbon intensity (example: next 4 intervals, adjust based on your setup)
        future_ci = self.carbon_intensity_data[self.current_step:self.current_step + self.forecast_intervals*4:step]
        # Ensure forecasted_CI has the desired length, fill with last value if necessary
        future_ci = np.pad(future_ci, (0, max(0, self.forecast_intervals - len(future_ci))), 'edge')

        if self.relative_ci:
            # Make forecasted CI relative to the current CI
            # future_ci = future_ci - current_CI
            future_ci = np.diff(future_ci, prepend=current_CI)

        # Data center energy consumption for the current step
        energy_consumption = self._calculate_DC_energy()
        
        # Current workload (normalized)
        current_workload = self.workload_data[self.current_step]
        
        # Queue status - length of the task queue
        queue_length = len(self.task_queue)
        
        # Combine all parts into a single observation array, including day of week encoding
        observation = np.array([sin_time_of_day, cos_time_of_day, sin_day_of_week, cos_day_of_week, current_CI] + 
                            list(future_ci) + [energy_consumption, current_workload, queue_length/self.queue_max_len], dtype=np.float16)
    
        
        if not self.observation_space.shape == observation.shape:
            print('There is something wrong')
            # raise ValueError(f"Observation shape {observation.shape} does not match expected shape {self.observation_space.shape}")
            
        flattened_observation = observation.flatten()
        return flattened_observation


    def _calculate_DC_energy(self):
        # Assert that current utilization is within the bounds [0, 1]
        assert 0 <= self.current_utilization <= 1, "Current utilization must be between 0 and 1"

        dc_energy = self.min_dc_energy + (self.max_dc_energy - self.min_dc_energy) * self.current_utilization
        return (dc_energy - self.min_dc_energy) / (self.max_dc_energy - self.min_dc_energy)
    
    
    def render(self, mode='human', close=False):
        # Rendering logic (optional)
        pass
    
    def _generate_random_carbon_intensity(self, intervals_per_day):
        days = self.simulated_days
        total_intervals = days * intervals_per_day

        # Sinusoidal function parameters
        A = 50  # Amplitude
        B = (2 * np.pi) / intervals_per_day  # Complete one cycle in a day
        C_8pm = 80  # Minimum at 8:00 PM, which is the 80th interval
        D = 150  # Midpoint of the carbon intensity values

        A_daily = 10  # Smaller amplitude for intra-day fluctuations
        B_daily = (2 * np.pi) / (intervals_per_day / 4)  # 4 cycles per day
        
        # Weekly patterns: Cycle over 7 days
        A_weekly = 20  # Amplitude for weekly fluctuations
        B_weekly = (2 * np.pi) / (intervals_per_day * 7)  # 1 cycle over 7 days
        
        # Seasonal fluctuations: Cycle over a year
        A_seasonal = 30  # Amplitude for seasonal fluctuations
        B_seasonal = (2 * np.pi) / (intervals_per_day * days)  # 1 cycle over a year
        
        # Time array for one year
        time_of_year = np.arange(intervals_per_day * days)
        
        # Composite carbon intensity function incorporating all fluctuations
        carbon_intensity_composite = (A * np.sin(B * (time_of_year % intervals_per_day - C_8pm)) + 
                                      A_daily * np.sin(B_daily * (time_of_year % intervals_per_day)) + 
                                      A_weekly * np.sin(B_weekly * time_of_year) + 
                                      A_seasonal * np.sin(B_seasonal * time_of_year) + 
                                      D)
        amplitude_medium = 10  # Noise amplitude
        noise = np.random.uniform(-amplitude_medium, amplitude_medium, size=total_intervals)
        carbon_intensity_with_noise = carbon_intensity_composite + noise
        return carbon_intensity_with_noise

