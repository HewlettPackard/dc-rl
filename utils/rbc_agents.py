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

