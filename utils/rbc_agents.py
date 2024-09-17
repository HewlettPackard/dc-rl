import numpy as np

class RBCBatteryAgent:
    def __init__(self, look_ahead=3, smooth_window=1, max_soc=0.9, min_soc=0.2):
        """
        Initialize the rule-based battery controller agent.
        In the battery environment, the actions are: {0: 'charge', 1: 'discharge', 2: 'idle'}

        Args:
            look_ahead (int): Number of steps to look ahead in the forecast.
            smooth_window (int): Window size for smoothing the forecast.
            max_soc (float): Maximum state of charge of the battery.
            min_soc (float): Minimum state of charge of the battery.
        """
        self.look_ahead = look_ahead
        self.smooth_window = smooth_window
        self.max_soc = max_soc
        self.min_soc = min_soc
        

    def act(self, carbon_intensity_values, current_soc):
        """
        Determine the action for the battery based on the carbon intensity forecast.

        Args:
            carbon_intensity_values (list): Forecasted carbon intensity values.
            current_soc (float): Current state of charge of the battery.

        Returns:
            int: Action to be taken (0: 'charge', 1: 'discharge', 2: 'idle').
        """
        # Calculate the smoothed carbon intensity forecasted using numpy convolve function
        window = self.smooth_window
        smoothed_carbon_intensity = np.convolve(carbon_intensity_values, np.ones(window), 'valid') / window
        
        # Get the current carbon intensity
        current_carbon_intensity = carbon_intensity_values[0]
        
        # Get the smoothed carbon intensity forecasted
        smoothed_carbon_intensity_forecasted = smoothed_carbon_intensity[self.look_ahead]
        
        # If the smoothed carbon intensity forecasted is higher than the current carbon intensity, charge the battery
        if smoothed_carbon_intensity_forecasted > current_carbon_intensity:
            return 0
        # If the smoothed carbon intensity forecasted is lower than the current carbon intensity, discharge the battery
        else:
            return 1
        
        # # Calculate the smoothed carbon intensity forecast using numpy convolve function
        # window = self.smooth_window
        # smoothed_carbon_intensity = np.convolve(carbon_intensity_values, np.ones(window), 'valid') / window

        # # Get the current carbon intensity
        # current_carbon_intensity = carbon_intensity_values[0]

        # # Get the smoothed carbon intensity forecasted
        # if len(smoothed_carbon_intensity) > self.look_ahead:
        #     smoothed_carbon_intensity_forecasted = smoothed_carbon_intensity[self.look_ahead]
        # else:
        #     smoothed_carbon_intensity_forecasted = smoothed_carbon_intensity[-1]

        # # Decision making based on smoothed carbon intensity and current SoC
        # if smoothed_carbon_intensity_forecasted > current_carbon_intensity:
        #     # Charge the battery if not already full
        #     if current_soc < self.max_soc:
        #         return 0
        #     else:
        #         return 2  # Idle if the battery is full
        # elif smoothed_carbon_intensity_forecasted < current_carbon_intensity:
        #     # Discharge the battery if not already empty
        #     if current_soc > self.min_soc:
        #         return 1
        #     else:
        #         return 2  # Idle if the battery is empty
        # else:
        #     return 2  # Idle if the smoothed carbon intensity forecasted is equal to the current carbon intensity


class RBCLiquidAgent:
    def __init__(self, min_pump_speed=0.05, max_pump_speed=0.5):
        self.min_pump_speed = min_pump_speed
        self.max_pump_speed = max_pump_speed
        
        print(f"Warning: for the Liquid Cooling Agent, supply temperature is set to 32°C, and pump speed is set to 0.25 l/s.")
        

    def act(self, workload=None):
        """
        Adjusts the pump speed based on the workload.
        
        Args:
            workload (float): The current workload utilization, expected to be between 0 and 1.
        
        Returns:
            action (float): The adjusted pump speed.
        """
        
        if workload is None:
            # Return the baseline action:
            pump_speed = 0.444  # 0.25 l/s
            supply_temp = 0.567 # 32°C (W32 ASHRAE GUIDELINES)
            return [pump_speed, supply_temp]
        
        else:
            # Ensure workload is within bounds
            workload = max(0, min(workload, 1))
            
            # Linear mapping from workload to pump speed
            pump_speed = self.min_pump_speed + workload * (self.max_pump_speed - self.min_pump_speed)
            
            return pump_speed

# class RBCLoadShiftingAgent:
#     def __init__(self, ci_threshold_high=0.7, ci_threshold_low=0.3, max_queue_length=0.8, max_age_threshold=0.8):
#         """
#         Initialize the rule-based controller for load shifting.
        
#         Args:
#             ci_threshold_high (float): Threshold above which tasks will be deferred.
#             ci_threshold_low (float): Threshold below which tasks will be processed.
#             max_queue_length (float): Maximum length of the deferral task queue (percentage).
#             max_age_threshold (float): Maximum age (in hours) a task can stay in the queue before it must be processed.
#         """
#         self.ci_threshold_high = ci_threshold_high
#         self.ci_threshold_low = ci_threshold_low
#         self.max_queue_length = max_queue_length
#         self.max_age_threshold = max_age_threshold

#     def act(self, current_ci, ci_forecast, tasks_in_queue, oldest_task_age, queue_max_len):
#         """
#         Decide whether to defer or process tasks based on carbon intensity and task queue status.
        
#         Args:
#             current_ci (float): The current carbon intensity.
#             ci_forecast (list[float]): The forecasted carbon intensity for the next hours.
#             tasks_in_queue (int): The current number of tasks in the deferral queue.
#             oldest_task_age (float): The age of the oldest task in the queue.
#             queue_max_len (int): Maximum length of the task queue.

#         Returns:
#             int: Action to take (0: Do nothing, -1: Defer tasks, 1: Process tasks).
#         """
#         # Normalize task queue length
#         queue_status = tasks_in_queue

#         # 1. If carbon intensity will decrease, defer tasks
#         # If in the next 4 hours, the current carbon intensity is higher than the forecasted, the value in ci_forecast will be negative
#         if all(ci_forecast[2] < 0) and oldest_task_age < self.max_age_threshold:
#             return 0  # Defer tasks

#         # 2. If carbon intensity will increase, process tasks from the DTQ
#         if all(ci_forecast[2] > 0) and queue_status > 0:
#             return 2  # Process tasks

#         # 3. Force processing of overdue tasks or if the queue is almost full
#         if oldest_task_age > self.max_age_threshold:
#             return 2  # Process tasks to avoid backlogs

#         # Default action: Do nothing
#         return 1

import numpy as np

class RBCLoadShiftingAgent:
    def __init__(self, max_queue_length, max_task_age=0.8, tolerance=0.05, trend_smoothing_window=3):
        """
        Initialize the rule-based load controller agent.
        
        Args:
            max_queue_length (int): Maximum number of tasks in the queue.
            max_task_age (float): Maximum age (in hours) a task can stay in the queue.
            tolerance (float): Small tolerance for deciding when carbon intensity is "similar" to current.
            trend_smoothing_window (int): Size of the window for smoothing the trend in carbon intensity.
        """
        self.max_queue_length = max_queue_length
        self.max_task_age = max_task_age
        self.tolerance = tolerance
        self.trend_smoothing_window = trend_smoothing_window
    
    def calculate_trend(self, ci_values):
        """
        Calculate the trend of carbon intensity using a smoothed difference between consecutive steps.

        Args:
            ci_values (list): Relative future (or past) carbon intensities.

        Returns:
            trend (float): The smoothed trend (positive means rising, negative means falling).
        """
        # Calculate differences between consecutive carbon intensity values
        differences = np.diff(ci_values)
        
        # Smooth the trend using a moving average (window size defined)
        smoothed_trend = np.convolve(differences, np.ones(self.trend_smoothing_window), 'valid') / self.trend_smoothing_window
        
        # Return the last value in the smoothed trend to indicate current trend direction
        return np.mean(smoothed_trend)
    
    def identify_trend_behavior(self, ci_past, ci_future):
        """
        Determine the carbon intensity trend based on past and future CI.

        Args:
            ci_past (list): Relative past carbon intensities.
            ci_future (list): Relative future carbon intensities.

        Returns:
            str: 'downward', 'upward', or 'neutral' based on the trend.
        """
        # Calculate trend for both past and future
        trend_past = self.calculate_trend(ci_past)
        trend_future = self.calculate_trend(ci_future)
        
        # If both past and future trends are negative, we're in a downward trend
        if trend_past < -self.tolerance and trend_future < -self.tolerance:
            return 'downward'
        
        # If both past and future trends are positive, we're in an upward trend
        if trend_past > self.tolerance and trend_future > self.tolerance:
            return 'upward'
        
        return 'neutral'
    
    def identify_peak_or_valley(self, ci_past, ci_future):
        """
        Identify whether the current time is in a peak or valley based on past and future carbon intensity trends.

        Args:
            ci_past (list): Relative past carbon intensities.
            ci_future (list): Relative future carbon intensities.

        Returns:
            str: 'peak', 'valley', or 'neutral' based on the analysis of the carbon intensity trends.
        """
        # Identify a peak: Past was lower, future will be lower (relative to the current CI)
        if ci_past[-1] < -self.tolerance and ci_future[0] > self.tolerance:
            return 'peak'
        
        # Identify a valley: Past was higher, future will be higher (relative to the current CI)
        if ci_past[-1] > self.tolerance and ci_future[0] < -self.tolerance:
            return 'valley'
        
        return 'neutral'
    
    def act(self, observation):
        """
        Determine the action based on the current observation.
        
        Args:
            observation (array): Observation vector as described in the environment.
        
        Returns:
            int: Action to be taken (0: defer tasks, 1: process tasks, 2: do nothing).
        """
        # Extract relevant parts of the observation                           
        current_ci = observation[12]  # Current carbon intensity
        ci_future = observation[4:12]  # Future carbon intensities (relative to current)
        ci_past = observation[13:17]  # Past carbon intensities (relative to current)
        oldest_task_age = observation[20]  # Age of the oldest task in queue
        queue_status = observation[3]  # Fraction of the queue being used (0 to 1)

        # Identify the trend behavior (upward, downward, or neutral)
        trend_behavior = self.identify_trend_behavior(ci_past, ci_future)

        # Identify peak or valley
        peak_or_valley = self.identify_peak_or_valley(ci_past, ci_future)

        # Rule 1: Defer tasks if we're in a peak and the trend is downward
        if peak_or_valley == 'peak' and trend_behavior == 'downward':
            return 0  # Defer tasks
        
        # Rule 2: Process tasks if we're in a valley or the trend is rising
        if peak_or_valley == 'valley' or trend_behavior == 'upward':
            return 2  # Process tasks

        # Rule 3: Force processing if the queue is almost full or tasks are too old
        if queue_status >= 0.9 or oldest_task_age > self.max_task_age:
            return 2  # Process tasks to prevent overflow
        
        # Default action: Do nothing if no clear valley or peak is detected
        return 1  # Idle (do nothing)


