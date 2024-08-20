import torch
import gymnasium
from gymnasium import spaces
import numpy as np
import supersuit as ss

from harl.envs.sustaindc.sustaindc_ptzoo import SustainDCPettingZooEnv
from pettingzoo.utils.conversions import parallel_wrapper_fn

class HARLSustainDCEnv:
    def __init__(self, env_args):
        """
        Initialize the HARLSustainDCEnv class.

        Args:
            env_args (dict): Environment arguments for SustainDCPettingZooEnv.
        """
        self.env_args = env_args
        self.env = SustainDCPettingZooEnv(self.env_args)
        self.n_agents = len(self.env.env.agents)
        self.cur_step = 0

        # Pad action and observation spaces to have the same shape (To use MAPPO 
        # and the algorithms that require the same shape)
        # self.env = ss.pad_action_space_v0(ss.pad_observations_v0(self.env))
        self.env = ss.pad_action_space_v0(ss.pad_observations_v0(self.env))

        # self.env = ss.frame_stack_v1(self.env, 2)

        self._seed = 0
        self.agents = self.env.possible_agents

        self.share_observation_space = self.unwrap(self.env.unwrapped.share_observation_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)

        self.discrete = True

    def reset(self):
        """
        Reset the environment.

        Returns:
            tuple: Observation, shared observation, and available actions.
        """
        self._seed += 1
        self.cur_step = 0
        obs = self.env.reset(seed=self._seed)

        # Extract the keys from obs in the same order
        agents = list(obs.keys())
        obs = self.unwrap(obs)

        states = tuple(o for o in obs)
        if self.env_args['nonoverlapping_shared_obs_space']:
            concat_states = []
            infos = self.env.unwrapped.env.infos

            # Common information
            time = infos['__common__']['time']
            ci = infos['__common__']['ci_future']
            concat_states.extend(time)
            concat_states.extend(ci)

            # Info from ls_env
            envs_infos = infos['__common__']['states']
            ls_env_info = envs_infos['agent_ls']
            concat_states.extend(ls_env_info[4:6])  # [Current workload and queue status]

            # Info from dc_env
            dc_env_info = envs_infos['agent_dc']
            concat_states.extend(dc_env_info[4:-1])  # [ambient_temp, zone_air_therm_cooling_stpt, zone_air_temp, hvac_power, it_power, next_workload]

            # Info from bat_env
            bat_env_info = envs_infos['agent_bat']
            concat_states.extend(bat_env_info[5].reshape(1,))  # battery_soc

            states = np.array(concat_states, dtype=np.float16)
        else:
            states = np.concatenate(states, axis=None)
        
        s_obs = self.repeat(states)
        avail_actions = self.get_avail_actions()
        return obs, s_obs, avail_actions

    def step(self, actions):
        """
        Take a step in the environment.

        Args:
            actions (list): Actions to be taken by the agents.

        Returns:
            tuple: Observation, shared observation, rewards, dones, info, and available actions.
        """
        actions = self.wrap(actions.flatten())
        obs, rew, term, trunc, info = self.env.step(actions)

        # Extract the keys from obs in the same order
        agents = list(obs.keys())
        obs = self.unwrap(obs)

        states = tuple(o for o in obs)
        if self.env_args['nonoverlapping_shared_obs_space']:
            concat_states = []
            infos = self.env.unwrapped.env.infos

            # Common information
            time = infos['__common__']['time']
            ci = infos['__common__']['ci_future']
            concat_states.extend(time)
            concat_states.extend(ci)

            # Info from ls_env
            envs_infos = infos['__common__']['states']
            ls_env_info = envs_infos['agent_ls']
            concat_states.extend(ls_env_info[4:6])  # [Current workload and queue status]

            # Info from dc_env
            dc_env_info = envs_infos['agent_dc']
            concat_states.extend(dc_env_info[4:-1])  # [ambient_temp, zone_air_therm_cooling_stpt, zone_air_temp, hvac_power, it_power, next_workload]

            # Info from bat_env
            bat_env_info = envs_infos['agent_bat']
            concat_states.extend(bat_env_info[5].reshape(1,))  # battery_soc

            states = np.array(concat_states, dtype=np.float16)
        else:
            states = np.concatenate(states, axis=None)

        s_obs = self.repeat(states)
        rewards = [[rew[agent]] for agent in self.agents]
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}

        return (
            obs,
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def seed(self, seed):
        """
        Set the random seed for the environment.

        Args:
            seed (int): The seed value.
        """
        self._seed = seed

    def close(self):
        """Close the environment."""
        pass

    def wrap(self, l):
        """
        Convert a list to a dictionary with agent names from the base environment.

        Args:
            l (list): List to be converted.

        Returns:
            dict: Dictionary with agent names as keys.
        """
        return {agent: l[i] for i, agent in enumerate(self.agents)}

    def unwrap(self, d):
        """
        Convert a dictionary to a list with agent names from the base environment.

        Args:
            d (dict): Dictionary to be converted.

        Returns:
            list: List of values from the dictionary.
        """
        return [d[agent] for agent in self.agents]

    def repeat(self, a):
        """
        Repeat an array for the number of agents.

        Args:
            a (array): Array to be repeated.

        Returns:
            list: List of repeated arrays.
        """
        return [a for _ in range(self.n_agents)]

    def get_avail_actions(self):
        """
        Get the available actions for all agents.

        Returns:
            list: List of available actions for each agent.
        """
        if self.discrete:
            return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n_agents)]
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """
        Get the available actions for a specific agent.

        Args:
            agent_id (int): ID of the agent.

        Returns:
            list: List of available actions.
        """
        return [1] * self.action_space[agent_id].n