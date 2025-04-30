from typing import Dict

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomCallbacks(DefaultCallbacks):
    """
    CustomCallbacks class that extends the DefaultCallbacks class and overrides its methods to customize the
    behavior of the callbacks during the RL training process.
    """

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        """
        Method that is called at the beginning of each episode in the training process. It sets some user_data
        variables to be used later on.

        Args:
            worker (Worker): The worker object that is being used in the training process.
            base_env (BaseEnv): The base environment that is being used in the training process.
            policies (Dict[str, Policy]): The policies that are being used in the training process.
            episode (MultiAgentEpisode): The episode object that is being processed.
            env_index (int): The index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        episode.user_data["net_energy_sum"] = 0
        episode.user_data["CO2_footprint_sum"] = 0
        episode.user_data["CO2_saved_footprint_sum"] = 0
        
        episode.user_data["step_count"] = 0
        episode.user_data["instantaneous_net_energy"] = []
        episode.user_data["load_left"] = 0
        episode.user_data["ls_tasks_in_queue"] = 0
        episode.user_data["ls_tasks_dropped"] = 0

        episode.user_data["water_usage"] = 0
        episode.user_data["reduced_water_usage"] = 0
        episode.user_data["water_savings"] = 0
    
    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs) -> None:
        """
        Method that is called at each step of each episode in the training process. It updates some user_data
        variables to be used later on.

        Args:
            worker (Worker): The worker object that is being used in the training process.
            base_env (BaseEnv): The base environment that is being used in the training process.
            episode (MultiAgentEpisode): The episode object that is being processed.
            env_index (int): The index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        # print(f'Logging env base with month: {base_env.envs[0].month}')
        # print(f'Logging env worker with month: {worker.env.month}')

        net_energy = base_env.envs[0].bat_info["bat_total_energy_with_battery_KWh"]
        CO2_footprint = base_env.envs[0].bat_info["bat_CO2_footprint"]
        CO2_saved_footprint = base_env.envs[0].bat_info["bat_CO2_saved_footprint"]

        load_left = base_env.envs[0].ls_info["ls_unasigned_day_load_left"]
        
        tasks_in_queue = base_env.envs[0].ls_info["ls_tasks_in_queue"]
        tasks_dropped = base_env.envs[0].ls_info["ls_tasks_dropped"]

        water_usage = base_env.envs[0].dc_info["dc_water_usage"]
        reduced_water_usage = base_env.envs[0].dc_info["dc_reduced_water_usage"]
        water_savings = base_env.envs[0].dc_info["dc_water_savings"]
        
        episode.user_data["instantaneous_net_energy"].append(net_energy)
        
        episode.user_data["net_energy_sum"] += net_energy
        episode.user_data["CO2_footprint_sum"] += CO2_footprint
        episode.user_data["CO2_saved_footprint_sum"] += CO2_footprint

        episode.user_data["load_left"] += load_left
        episode.user_data["ls_tasks_in_queue"] += tasks_in_queue
        episode.user_data["ls_tasks_dropped"] += tasks_dropped

        episode.user_data["water_usage"] += water_usage
        episode.user_data["reduced_water_usage"] += reduced_water_usage
        episode.user_data["water_savings"] += water_savings

        episode.user_data["step_count"] += 1
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        """
        Method that is called at the end of each episode in the training process. It calculates some metrics based
        on the updated user_data variables.

        Args:
            worker (Worker): The worker object that is being used in the training process.
            base_env (BaseEnv): The base environment that is being used in the training process.
            policies (Dict[str, Policy]): The policies that are being used in the training process.
            episode (MultiAgentEpisode): The episode object that is being processed.
            env_index (int): The index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        if episode.user_data["step_count"] > 0:
            average_net_energy = episode.user_data["net_energy_sum"] / episode.user_data["step_count"]
            average_CO2_footprint = episode.user_data["CO2_footprint_sum"] / episode.user_data["step_count"]
            average_CO2_saved_footprint = episode.user_data["CO2_saved_footprint_sum"] / episode.user_data["step_count"]
            total_load_left = episode.user_data["load_left"]
            total_tasks_in_queue = episode.user_data["ls_tasks_in_queue"]
            total_tasks_dropped = episode.user_data["ls_tasks_dropped"]

            total_water_usage = episode.user_data["water_usage"]
            total_reduced_water_usage = episode.user_data["reduced_water_usage"]
            total_water_savings = episode.user_data["water_savings"]

        else:
            average_net_energy = 0
            average_CO2_footprint = 0
            average_CO2_saved_footprint = 0
            average_bat_actions = 0
            average_ls_actions = 0
            average_dc_actions = 0
            total_load_left = 0
            total_tasks_in_queue = 0
            total_tasks_dropped = 0
            total_water_usage = 0
            total_reduced_water_usage = 0
            total_water_savings = 0

        
        episode.custom_metrics["average_total_energy_with_battery"] = average_net_energy
        episode.custom_metrics["average_CO2_footprint"] = average_CO2_footprint
        episode.custom_metrics["average_CO2_saved_footprint"] = average_CO2_saved_footprint
        episode.custom_metrics["load_left"] = total_load_left
        episode.custom_metrics["total_tasks_in_queue"] = total_tasks_in_queue
        episode.custom_metrics["total_tasks_dropped"] = total_tasks_dropped

        episode.custom_metrics["total_water_usage"] = total_water_usage
        episode.custom_metrics["total_reduced_water_usage"] = total_reduced_water_usage
        episode.custom_metrics["total_water_savings"] = total_water_savings