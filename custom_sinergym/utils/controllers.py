"""Implementation of basic controllers."""
import math
from datetime import datetime
from typing import Any, List, Sequence

import numpy as np


class RandomController(object):

    def __init__(self, env: Any):
        """Random agent. It selects available actions randomly.

        Args:
            env (Any): Simulation environment.
        """
        self.env = env

    def act(self) -> Sequence[Any]:
        """Selects a random action from the environment's action_space.

        Returns:
            Sequence[Any]: Action chosen.
        """
        action = self.env.action_space.sample()
        return action

class myRandomController(object):

    def __init__(self, env: Any):
        """Random agent. It selects available actions randomly.

        Args:
            env (Any): Simulation environment.
        """
        self.env = env

    def predict(self, obs, deterministic=False) -> Sequence[Any]:
        """Selects a random action from the environment's action_space.

        Returns:
            Sequence[Any]: Action chosen.
        """
        action = self.env.action_space.sample()
        return np.int64(action), None
    
class FixedController(object):

    def __init__(self, env: Any):
        """Random agent. It selects available actions randomly.

        Args:
            env (Any): Simulation environment.
        """
        self.env = env

    def predict(self, obs, deterministic=False) -> Sequence[Any]:
        """Selects a random action from the environment's action_space.

        Returns:
            Sequence[Any]: Action chosen.
        """
        action = np.int64(4)
        return action, None

class RBC5Zone(object):

    def __init__(self, env: Any) -> None:
        """Agent based on static rules for controlling 5ZoneAutoDXVAV setpoints.
        Based on ASHRAE Standard 55-2004: Thermal Environmental Conditions for Human Occupancy.

        Args:
            env (Any): Simulation environment
        """

        self.env = env

        self.variables = env.variables

        self.setpoints_summer = (26, 29.0)
        self.setpoints_winter = (20.0, 23.5)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.variables['observation'], observation))
        year = int(obs_dict['year'])
        month = int(obs_dict['month'])
        day = int(obs_dict['day'])

        summer_start_date = datetime(year, 6, 1)
        summer_final_date = datetime(year, 9, 30)

        current_dt = datetime(year, month, day)

        # Get season comfort range
        if current_dt >= summer_start_date and current_dt <= summer_final_date:  # pragma: no cover
            season_range = self.setpoints_summer
        else:  # pragma: no cover
            season_range = self.setpoints_winter

        return (season_range[0], season_range[1])

def reverse_hours(sin_hour, cos_hour):
    # Inverse of the sin_hours
    hour_sin_1 = round(np.arcsin(sin_hour), 6)
    hour_sin_2 = round(np.pi - np.arcsin(sin_hour), 6)
    
    # Inverse of the cos_hours
    hour_cos_1 = round(np.arccos(cos_hour), 6)
    hour_cos_2 = round(2*np.pi - np.arcsin(cos_hour), 6)

    # Find the hour that are in both sin and coside
    if hour_sin_1 == hour_cos_1:
        real_hour = hour_cos_1
    else:
        real_hour = hour_cos_2
    
    if hour_sin_2 == -hour_cos_2:
        real_hour == -hour_cos_2
    return round(real_hour*23)

class myRBC(object):

    def __init__(self, env: Any, ranges: None):
        """Random agent. It selects available actions randomly.

        Args:
            env (Any): Simulation environment.
        """
        self.env = env
        self.ranges_norm = ranges
        
    def denormalize_obs(self, norm_obs, values):
        min_val, max_val = values
        # denormalize_obs = obs * (max_value - min_value) + min_value
        obs = (norm_obs / 2 + 0.5) * (max_val - min_val) + min_val

        return obs
    
    def predict(self, obs, deterministic=False) -> Sequence[Any]:
        """Selects a random action from the environment's action_space.

        Returns:
            Sequence[Any]: Action chosen.
        """
        sin_hour = obs[2]
        cos_hour = obs[3]
        
        # Inverse of the sin_hours and cos_hour
        curr_hour = reverse_hours(sin_hour, cos_hour)

        # calculate the original hour value
        # curr_hour = math.atan2(sin_hour, cos_hour) * (24 / (2 * math.pi))
        # curr_hour = self.denormalize_obs(obs[2], self.ranges_norm['hour'])
        rel_curr_setpoint = self.denormalize_obs(obs[5], self.ranges_norm['Zone Thermostat Cooling Setpoint Temperature(West Zone)'])
        curr_int_temp = self.denormalize_obs(obs[6], self.ranges_norm['Zone Air Temperature(West Zone)'])
        curr_setpoint = curr_int_temp + rel_curr_setpoint
        
        # Avg elec cost: 15.22
        if np.round(curr_hour) <= 6:
            setpoint_obj = 20.0
        elif np.round(curr_hour) <= 8:
            setpoint_obj = 19.0
        elif np.round(curr_hour) <= 18:
            setpoint_obj = 18.0
        elif np.round(curr_hour) <= 24:
            setpoint_obj = 19.0
        else:
            setpoint_obj = 19.0
        # Avg elec cost: 15.46
        # if np.round(curr_hour) <= 8:
        #     setpoint_obj = 20.0
        # elif np.round(curr_hour) <= 12:
        #     setpoint_obj = 19.0
        # elif np.round(curr_hour) <= 16:
        #     setpoint_obj = 18.0
        # elif np.round(curr_hour) <= 20:
        #     setpoint_obj = 19.0
        # elif np.round(curr_hour) <= 24:
        #     setpoint_obj = 20.0
            
        # if curr_setpoint > setpoint_obj then act increasing the setpoint
        if curr_setpoint >= setpoint_obj:
            action = 2 #: (-1),
        elif curr_setpoint <= setpoint_obj:
            action = 6 #: (1),
        else:
            action = 4 #: (0),
        return np.int64(action), None


class myRBCDatacenter(object):
    def __init__(self, env: Any, ranges: None) -> None:
        """Agent based on static rules for controlling 2ZoneDataCenterHVAC setpoints.
        Follows the ASHRAE recommended temperature ranges for data centers described in ASHRAE TC9.9 (2016).
        Args:
            env (Any): Simulation environment
        """

        self.env = env
        self.variables = env.variables

        # ASHRAE recommended temperature range = [18, 27] Celsius
        self.range_datacenter = (16, 18)
        self.ranges_norm = ranges

    def denormalize_obs(self, norm_obs, values):
        min_val, max_val = values
        # denormalize_obs = obs * (max_value - min_value) + min_value
        obs = (norm_obs / 2 + 0.5) * (max_val - min_val) + min_val

        return obs

    def predict(self, obs, deterministic=False) -> Sequence[Any]:
        """Select action based on indoor temperature.
        Args:
            observation (List[Any]): Perceived observation.
        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.variables['observation'], obs))

        west_temp_index = 2 + 4
        setpoint_index = 1 + 4

        mean_temp = self.denormalize_obs(obs[west_temp_index], self.ranges_norm['Zone Air Temperature(West Zone)'])
        rel_cool_setpoint = self.denormalize_obs(obs[setpoint_index], self.ranges_norm['Zone Thermostat Cooling Setpoint Temperature(West Zone)'])
        new_cool_setpoint = rel_cool_setpoint + mean_temp

        # new_cool_setpoint = current_cool_setpoint

        # New behaviour:
        
        # Determine the recommended temperature range
        if mean_temp < 16.0:# increase cooling setpoint +1
            new_cool_setpoint = 5 #: (+0.5),
        elif mean_temp > 16.0:  # decrease cooling setpoint -1
            new_cool_setpoint = 1 #: (-2),
        else:
            new_cool_setpoint = 4 #: (0),
            
        # # Determine the recommended temperature range
        # if mean_temp < self.range_datacenter[0] + 0.25 and new_cool_setpoint < self.range_datacenter[1]:# increase cooling setpoint +1
        #     new_cool_setpoint = 6 #: (+1),
        # elif mean_temp > self.range_datacenter[1] - 0.25 and new_cool_setpoint > self.range_datacenter[0]:  # decrease cooling setpoint -1
        #     new_cool_setpoint = 2 #: (-1),
        # else:
        #     new_cool_setpoint = 4 #: (0),
                
        # Original behaviour:  227.20 kWh
        # if mean_temp < self.range_datacenter[0]:  # increase cooling setpoint +1
        #     new_cool_setpoint = 6 #: (1),
        # elif mean_temp > self.range_datacenter[1]:  # decrease cooling setpoint -1
        #     new_cool_setpoint = 2 #: (-1),
        # else:
        #     new_cool_setpoint = 4 #: (0),
        return np.int64(new_cool_setpoint), None
        
class RBCDatacenter(object):

    def __init__(self, env: Any) -> None:
        """Agent based on static rules for controlling 2ZoneDataCenterHVAC setpoints.
        Follows the ASHRAE recommended temperature ranges for data centers described in ASHRAE TC9.9 (2016).
        Args:
            env (Any): Simulation environment
        """

        self.env = env
        self.variables = env.variables

        # ASHRAE recommended temperature range = [18, 27] Celsius
        self.range_datacenter = (18, 27)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.
        Args:
            observation (List[Any]): Perceived observation.
        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.variables['observation'], observation))

        # Mean temp in datacenter zones
        mean_temp = np.mean([obs_dict['Zone Air Temperature(West Zone)'],
                             obs_dict['Zone Air Temperature(East Zone)']])

        current_heat_setpoint = obs_dict[
            'Zone Thermostat Heating Setpoint Temperature(West Zone)']
        current_cool_setpoint = obs_dict[
            'Zone Thermostat Cooling Setpoint Temperature(West Zone)']

        new_heat_setpoint = current_heat_setpoint
        new_cool_setpoint = current_cool_setpoint

        if mean_temp < self.range_datacenter[0]:  # pragma: no cover
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint + 1
        elif mean_temp > self.range_datacenter[1]:  # pragma: no cover
            new_cool_setpoint = current_cool_setpoint - 1
            new_heat_setpoint = current_heat_setpoint - 1

        return (
            new_heat_setpoint,
            new_cool_setpoint)
