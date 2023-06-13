from typing import Dict, Optional, Union
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

def f2c(t):
    return 5*(t-23)/9

class pyeplus_callback(DefaultCallbacks):
    
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        
        episode.user_data['Total Power kW'] = 0
        episode.user_data['crac_setpoint_delta'] = []
        episode.user_data["step_count"] = 0
        
        
    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        Total_Power_kw = base_env.get_sub_environments()[0].info['Total Power kW']
        crac_setpoint_delta = base_env.get_sub_environments()[0].info['crac_setpoint_delta']
        
        episode.user_data['Total Power kW'] += Total_Power_kw
        episode.user_data["crac_setpoint_delta"].append(crac_setpoint_delta)
        episode.user_data["step_count"] += 1
        
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        
        if episode.user_data["step_count"] > 0:
            average_net_energy = 0.25*episode.user_data["Total Power kW"] / episode.user_data["step_count"]
            average_dc_actions = np.sum(episode.user_data["crac_setpoint_delta"]) / episode.user_data["step_count"]
        else:
            average_net_energy = 0
            average_dc_actions = 0

        episode.custom_metrics["avg_power_per_episode_kW"] = average_net_energy
        episode.custom_metrics["avg_crac_stpt_delta_per_episode"] = average_dc_actions

            