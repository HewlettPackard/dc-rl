from typing import Dict, Optional, Union

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


def f2c(t: float) -> float:
    """
    Converts temperature in Fahrenheit to Celsius using the formula (5/9)*(t-23).

    Args:
        t (float): Temperature in Fahrenheit.

    Returns:
        float: Temperature in Celsius.
    """
    return 5*(t-23)/9


class pyeplus_callback(DefaultCallbacks):
    """
    Custom callbacks class that extends the DefaultCallbacks class.

    Defines callback methods that are triggered at various points of the training process.
    """

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """
        Method that is called at the beginning of each episode in the training process.

        Initializes some user_data variables to be used later on.

        Args:
            worker (Worker): The worker object that is being used in the training process.
            base_env (BaseEnv): The base environment that is being used in the training process.
            policies (Dict[str, Policy]): The policies that are being used in the training process.
            episode (MultiAgentEpisode): The episode object that is being processed.
            env_index (int): The index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        episode.user_data['Total Power kW'] = 0
        episode.user_data['crac_setpoint_delta'] = []
        episode.user_data["step_count"] = 0
    
    
    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        """
        Method that is called at each step of each episode in the training process.

        Updates some user_data variables to be used later on.

        Args:
            worker (Worker): The worker object that is being used in the training process.
            base_env (BaseEnv): The base environment that is being used in the training process.
            episode (MultiAgentEpisode): The episode object that is being processed.
            env_index (int): The index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        Total_Power_kw = base_env.get_sub_environments()[0].info['Total Power kW']
        crac_setpoint_delta = base_env.get_sub_environments()[0].info['crac_setpoint_delta']
        
        episode.user_data['Total Power kW'] += Total_Power_kw
        episode.user_data["crac_setpoint_delta"].append(crac_setpoint_delta)
        episode.user_data["step_count"] += 1
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """
        Method that is called at the end of each episode in the training process.

        Calculates some metrics based on the user_data variables updated during the episode.

        Args:
            worker (Worker): The worker object that is being used in the training process.
            base_env (BaseEnv): The base environment that is being used in the training process.
            policies (Dict[str, Policy]): The policies that are being used in the training process.
            episode (MultiAgentEpisode): The episode object that is being processed.
            env_index (int): The index of the environment within the worker task.
            **kwargs: additional arguments that can be passed.
        """
        if episode.user_data["step_count"] > 0:
            average_net_energy = 0.25*episode.user_data["Total Power kW"] / episode.user_data["step_count"]
            average_dc_actions = np.sum(episode.user_data["crac_setpoint_delta"]) / episode.user_data["step_count"]
        else:
            average_net_energy = 0
            average_dc_actions = 0
        
        episode.custom_metrics["avg_power_per_episode_kW"] = average_net_energy
        episode.custom_metrics["avg_crac_stpt_delta_per_episode"] = average_dc_actions
            
def idx_to_source_sink_mapper(nodes):
    """
    Given the number of ordered nodes, it creates a dense graph and populates a
    dictionary where the key is an index that maps to a tuple of the source and sink node ids with source value always being less than sink value
    Note: The source and sink nodes are the 0 indexed nodes in the graph 
    eg 
    idx :   0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    source: 0, 0, 0, 0, 1, 1, 1, 2, 2, 3 : start iteration with this until i = 0 to N-2 ie range(N-1)
    ctr :   0, 1, 2, 3, 0, 1, 2, 0, 1, 0 : start iteration with this until ctr = 0 to [N-(i+2)] ie range(N-(i+1)) for each i from above
    sink:   1, 2, 3, 4, 2, 3, 4, 3, 4, 4 : get this by adding ctr to source+1 ie sink = source+ctr+1
    """
    
    source_sink_idx = {}
    
    for source_idx in range(nodes-1):
        offset = int(nodes*source_idx - source_idx*(source_idx+1)/2)
        for counter in range(nodes-(source_idx+1)):
            
            source_sink_idx[offset+counter] = (source_idx, source_idx+counter+1)
            
    return source_sink_idx