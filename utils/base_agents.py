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
    def do_nothing_action(self):
        """
        Return the do nothing action.
        
        Returns:
            action (int): The action (do nothing) to be taken.
        """
        return 1
    
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
    def do_nothing_action(self):
        """
        Return the do nothing action.
        
        Returns:
            action (int): The action (do nothing) to be taken.
        """
        return np.int64(1)

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
    def do_nothing_action(self):
        """
        Return the do nothing action.
        
        Returns:
            action (int): The action (do nothing) to be taken.
        """
        return 2
