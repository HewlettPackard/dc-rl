import numpy as np

class BaseLoadShiftingAgent:
    def __init__(self, parameters=None):
        self.parameters = parameters
    def do_nothing_action(self):
        return 1
    
class BaseHVACAgent:
    def __init__(self, parameters=None):
        self.parameters = parameters
    def do_nothing_action(self):
        return np.int64(4)

class BaseBatteryAgent:
    def __init__(self, parameters=None):
        self.parameters = parameters
    def do_nothing_action(self):
        return 2
