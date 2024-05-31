from typing import Dict

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import EnvType, PolicyID


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
        episode.user_data["ls_tasks_in_queue"] = 0
        episode.user_data["ls_tasks_dropped"] = 0

        episode.user_data["water_usage"] = 0
    
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
        
        tasks_in_queue = base_env.envs[0].ls_info["ls_tasks_in_queue"]
        tasks_dropped = base_env.envs[0].ls_info["ls_tasks_dropped"]

        water_usage = base_env.envs[0].dc_info["dc_water_usage"]
        
        episode.user_data["instantaneous_net_energy"].append(net_energy)
        
        episode.user_data["net_energy_sum"] += net_energy
        episode.user_data["CO2_footprint_sum"] += CO2_footprint
        episode.user_data["load_left"] += load_left
        episode.user_data["ls_tasks_in_queue"] += tasks_in_queue
        episode.user_data["ls_tasks_dropped"] += tasks_dropped

        episode.user_data["water_usage"] += water_usage

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
            total_load_left = episode.user_data["load_left"]
            total_tasks_in_queue = episode.user_data["ls_tasks_in_queue"]
            total_tasks_dropped = episode.user_data["ls_tasks_dropped"]

            total_water_usage = episode.user_data["water_usage"]

        else:
            average_net_energy = 0
            average_CO2_footprint = 0
            average_bat_actions = 0
            average_ls_actions = 0
            average_dc_actions = 0
            total_load_left = 0
            total_tasks_in_queue = 0
            total_tasks_dropped = 0
            total_water_usage = 0
        
        episode.custom_metrics["average_total_energy_with_battery"] = average_net_energy
        episode.custom_metrics["average_CO2_footprint"] = average_CO2_footprint
        episode.custom_metrics["load_left"] = total_load_left
        episode.custom_metrics["total_tasks_in_queue"] = total_tasks_in_queue
        episode.custom_metrics["total_tasks_dropped"] = total_tasks_dropped

        episode.custom_metrics["total_water_usage"] = total_water_usage
        
class HierarchicalDCRL_Callback(DefaultCallbacks):
    """
    Callback to log Hierarchical DCRL specific values
    """

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy] | None = None, episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:

        episode.custom_metrics["runningstats/mu1"] = base_env.vector_env.envs[0].stats1.mu
        episode.custom_metrics["runningstats/sigma_1"] = base_env.vector_env.envs[0].stats1.stddev
        episode.custom_metrics["runningstats/mu2"] = base_env.vector_env.envs[0].stats2.mu
        episode.custom_metrics["runningstats/sigma_2"] = base_env.vector_env.envs[0].stats2.stddev
        episode.custom_metrics["runningstats/cfp_reward"] = base_env.vector_env.envs[0].cfp_reward
        episode.custom_metrics["runningstats/workload_violation_rwd"] = base_env.vector_env.envs[0].workload_violation_rwd
        episode.custom_metrics["runningstats/combined_reward"] = base_env.vector_env.envs[0].combined_reward
        episode.custom_metrics["runningstats/hysterisis_cost"] = base_env.vector_env.envs[0].cost_of_moving_mw
        ax1,ax2,ax3 = base_env.vector_env.envs[0].action_choice
        episode.custom_metrics["runningstats/ax1"] = ax1
        episode.custom_metrics["runningstats/ax2"] = ax2
        episode.custom_metrics["runningstats/ax3"] = ax3

class CustomMetricsCallback(DefaultCallbacks):

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        if hasattr(base_env, 'vector_env'):
            metrics = base_env.vector_env.envs[0].metrics            
        else:
            metrics = base_env.envs[0].metrics
            
        cfp = 0
        for dc in metrics:
            cfp += sum(metrics[dc]['bat_CO2_footprint']) / 1e6

        episode.custom_metrics['custom_metrics/CFP']=  cfp