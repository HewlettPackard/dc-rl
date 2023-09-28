"""
https://mepacademy.com/top-6-hvac-control-strategies-to-save-energy/#:~:text=Using%20Trim%20and%20Respond%20control,setpoint%20to%2055.2%C2%B0F.
Based on #6 Supply Air Temperature Reset
"""

import numpy as np

class trim_and_respond_ctrl():
    
    def __init__(self,
                TandR_monitor_idx = 6,
                TandR_monitor : str = "avg_room_temp",
                TandR_monitor_limit : float = 25):
        
        assert (TandR_monitor == "avg_room_temp") | (TandR_monitor == "crac_return_temp"), f"invalid TandR_monitor monitor string : {TandR_monitor}"
        self.TandR_monitor = TandR_monitor
        self.TandR_monitor_limit = TandR_monitor_limit
        self.TandR_monitor_idx = TandR_monitor_idx
        
        self.response_duration_counter = 0
        self.response_duration_limit = 4  # assuming 1 hour if sampling interval is 15mins
        
    def set_limit(self,x):
        self.TandR_monitor_limit = x

    def action(self, obs):
        assert type(obs) == np.ndarray, "invalid obs type" 
        curr_val = obs[self.TandR_monitor_idx]

        if self.TandR_monitor_limit >= curr_val:
            if self.response_duration_counter>self.response_duration_limit:
                self.response_duration_counter += 1
                return 5  # maps to +0.5F  # trim
            else:
                return 4  # maps to +-0F 
        else:
            return 3  # maps to -0.5F

            
        
        
