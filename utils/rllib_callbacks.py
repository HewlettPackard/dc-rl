from typing import Dict

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
"""
CustomCallbacks class that extends the DefaultCallbacks class and overrides its methods to customize the behavior of the callbacks during the RL training process.
"""

class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        """
        Method that is called at the beginning of each episode in the training process. It sets some user_data variables to be used later on.

        Args:
            worker: the worker object that is being used in the training process.
            base_env: the base environment that is being used in the training process.
            policies: the policies that are being used in the training process.
            episode: the episode object that is being processed.
            env_index: the index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        episode.user_data["net_energy_sum"] = 0
        episode.user_data["CO2_footprint_sum"] = 0
        
        episode.user_data["step_count"] = 0
        episode.user_data["instantaneous_net_energy"] = []
        episode.user_data["load_left"] = 0
    
    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs) -> None:
        """
        Method that is called at each step of each episode in the training process. It updates some user_data variables to be used later on.

        Args:
            worker: the worker object that is being used in the training process.
            base_env: the base environment that is being used in the training process.
            episode: the episode object that is being processed.
            env_index: the index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        net_energy = base_env.envs[0].bat_info["bat_total_energy_with_battery_KWh"]
        CO2_footprint = base_env.envs[0].bat_info["bat_CO2_footprint"]
        load_left = base_env.envs[0].ls_info["ls_unasigned_day_load_left"]
        episode.user_data["instantaneous_net_energy"].append(net_energy)
        
        episode.user_data["net_energy_sum"] += net_energy
        episode.user_data["CO2_footprint_sum"] += CO2_footprint
        episode.user_data["load_left"] += load_left
        
        episode.user_data["step_count"] += 1
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        """
        Method that is called at the end of each episode in the training process. It calculates some metrics based on the updated user_data variables.

        Args:
            worker: the worker object that is being used in the training process.
            base_env: the base environment that is being used in the training process.
            policies: the policies that are being used in the training process.
            episode: the episode object that is being processed.
            env_index: the index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        if episode.user_data["step_count"] > 0:
            average_net_energy = episode.user_data["net_energy_sum"] / episode.user_data["step_count"]
            average_CO2_footprint = episode.user_data["CO2_footprint_sum"] / episode.user_data["step_count"]
            total_load_left = episode.user_data["load_left"]
        else:
            average_net_energy = 0
            average_CO2_footprint = 0
            average_bat_actions = 0
            average_ls_actions = 0
            average_dc_actions = 0
            total_load_left = 0
        
        episode.custom_metrics["average_total_energy_with_battery"] = average_net_energy
        episode.custom_metrics["average_CO2_footprint"] = average_CO2_footprint
        episode.custom_metrics["load_left"] = total_load_left