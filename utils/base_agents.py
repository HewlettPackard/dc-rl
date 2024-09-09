import numpy as np

class BaseLoadShiftingAgent:
    """
    Base class for load shifting agents.

    Args:
        parameters (dict) :  Dictionary containing the agent parameters.
    """
    def __init__(self, parameters=None):
        """
        Args:
            parameters (dict) :  Dictionary containing the agent parameters.
        """
        self.parameters = parameters
        self.do_nothing_action_value = [1]
        # Add a warning message to inform the user to check if the do nothing action is the '1' action.
        print(f"Warning: Please check if the do nothing action for Load Shifting Agent is the '{self.do_nothing_action_value}' action.")
        
    def do_nothing_action(self):
        """
        Return the do nothing action.
        
        Returns:
            action (int): The action (do nothing) to be taken.
        """
        return self.do_nothing_action_value
    
    def act(self):
        """
        Return the do nothing action regardless of the input parameters.

        Returns:
            action (int): The action (do nothing) to be taken.
        """
        return self.do_nothing_action()
    
class BaseHVACAgent:
    """
    Base class for HVAC agents.

    Parameters
    ----------
    parameters : dict
        Dictionary containing the agent parameters.
    """
    def __init__(self, parameters=None):
        """

        Parameters
        ----------
        parameters : dict
            Dictionary containing the agent parameters.
        """
        self.parameters = parameters
        self.do_nothing_action_value = np.int64(1)
        # Add a warning message to inform the user to check if the do nothing action is the '1' action.
        print(f"Warning: Please check if the do nothing action for HVAC is the '{self.do_nothing_action_value}' action.")
        
    def do_nothing_action(self):
        """
        Return the do nothing action.
        
        Returns:
            action (int): The action (do nothing) to be taken.
        """
        return self.do_nothing_action_value

    def act(self):
        """
        Return the do nothing action regardless of the input parameters.

        Returns:
            action (int): The action (do nothing) to be taken.
        """
        return self.do_nothing_action()
    
class BaseBatteryAgent:
    """
    Base class for battery agents.

    Args:
        parameters (dict) :  Dictionary containing the agent parameters.
    """
    def __init__(self, parameters=None):
        """
        Args:
            parameters (dict) :  Dictionary containing the agent parameters.
        """
        self.parameters = parameters
        self.do_nothing_action_value = 2
        # Add a warning message to inform the user to check if the do nothing action is the '1' action.
        print(f"Warning: Please check if the do nothing action for Battery is the '{self.do_nothing_action_value}' action.")
        
    def do_nothing_action(self):
        """
        Return the do nothing action.
        
        Returns:
            action (int): The action (do nothing) to be taken.
        """
        return self.do_nothing_action_value
    
    def act(self, *args, **kwargs):
        """
        Return the do nothing action regardless of the input parameters.

        Args:
            *args: Arbitrary positional arguments.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            action (int): The action (do nothing) to be taken.
        """
        return self.do_nothing_action()

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