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
        
        episode.user_data["step_count"] = 0
        episode.user_data["instantaneous_net_energy"] = []
        episode.user_data["load_left"] = 0
    
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
        net_energy = base_env.envs[0].bat_info["bat_total_energy_with_battery_KWh"]
        CO2_footprint = base_env.envs[0].bat_info["bat_CO2_footprint"]
        load_left = base_env.envs[0].ls_info["ls_unasigned_day_load_left"]
        episode.user_data["instantaneous_net_energy"].append(net_energy)
        
        episode.user_data["net_energy_sum"] += net_energy
        episode.user_data["CO2_footprint_sum"] += CO2_footprint
        episode.user_data["load_left"] += load_left
        
        episode.user_data["step_count"] += 1

        dc_total_power_kW = base_env.envs[0].dc_info["dc_total_power_kW"]
        if 'dc_total_power_values' not in episode.user_data:
            episode.user_data['dc_total_power_values'] = []

        episode.user_data['dc_total_power_values'].append(dc_total_power_kW)



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
        
        dc_total_power_values = episode.user_data.get('dc_total_power_values', [])

        if dc_total_power_values:
            min_value = min(dc_total_power_values)
            max_value = max(dc_total_power_values)
            episode.custom_metrics["min_dc_total_power_kW"] = min_value
            episode.custom_metrics["max_dc_total_power_kW"] = max_value
        
    # def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
    #     # Compute min/max from postprocessed_batch['total_energy_with_battery']
    #     if agent_id == 'agent_dc':
    #         dc_total_power_values = [info['dc_total_power_kW'] for info in postprocessed_batch['infos'] if 'dc_total_power_kW' in info]

    #         min_value = np.min(dc_total_power_values)
    #         max_value = np.max(dc_total_power_values)

    #         # Store in custom metrics
    #         worker.custom_metrics[f'min_total_energy_{worker.worker_index}'] = min_value
    #         worker.custom_metrics[f'max_total_energy_{worker.worker_index}'] = max_value
