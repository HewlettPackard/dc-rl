import datetime
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
        self.max_cycles = 25
        self.cur_step = 0

        # Pad action and observation spaces to have the same shape
        self.env = ss.pad_action_space_v0(self.env)
        self.env = ss.pad_observations_v0(self.env)

        self._seed = 0
        self.agents = self.env.possible_agents

        self.share_observation_space = self.unwrap(self.env.unwrapped.share_observation_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)

        self.discrete = True

        self.is_render = env_args.get("is_render", False)
        if self.is_render:
            self.experiment_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.render_episode = 0

    def _create_shared_observation(self, obs):
        """
        Create a shared observation (s_obs) based on the environment state.

        Args:
            obs (dict): Observations from the environment.

        Returns:
            tuple: Shared observation array.
        """
        states = tuple(o for o in obs)
        if self.env_args.get('nonoverlapping_shared_obs_space', False):
            concat_states = []
            infos = self.env.unwrapped.env.infos

            # # Common information
            # concat_states.extend(infos['__common__']['time'])
            # concat_states.extend(infos['__common__']['ci_future'])

            # # Info from ls_env
            # ls_env_info = infos['agent_ls']
            # current_workload = infos['__common__']['workload']
            # queue_status = ls_env_info['ls_norm_tasks_in_queue']
            # concat_states.extend([current_workload, queue_status])  # Current workload and queue status

            # # Info from dc_env
            # dc_env_info = infos['agent_dc']
            # ambient_temp = dc_env_info['dc_exterior_ambient_temp']
            # concat_states.extend([ambient_temp])  # Ambient temp

            # # Info from bat_env
            # bat_env_info = infos['agent_bat']
            # bat_soc = bat_env_info['bat_SOC']
            # concat_states.extend([bat_soc])  # Battery SOC
            
            # The whole information available without repeating the same information
            concat_states.extend(states[0]) # ls_state
            concat_states.extend([states[1][11], states[1][13]]) # dc_state (next_workload and next_exterior_temp)
            concat_states.extend([states[2][-1]]) # bat_state (SOC)
            

            states = np.array(concat_states, dtype=np.float32)
        else:
            states = np.concatenate(states, axis=None)
        
        return self.repeat(states)

    def reset(self):
        """
        Reset the environment.

        Returns:
            tuple: Observation, shared observation, and available actions.
        """
        self.render_episode += 1
        self._seed += 1
        self.cur_step = 0
        obs = self.env.reset(seed=self._seed)
        obs = self.unwrap(obs)
        
        s_obs = self._create_shared_observation(obs)
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
        obs = self.unwrap(obs)
        
        s_obs = self._create_shared_observation(obs)
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

    def render(self):
        """
        Render the environment.
        """
        self.env.render()
