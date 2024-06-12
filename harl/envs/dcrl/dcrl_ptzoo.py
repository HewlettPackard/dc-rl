from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

class DCRLPettingZooEnv(ParallelEnv):
    def __init__(self, env_config):
        super().__init__()
        if env_config['partial_obs']:
            if env_config['month'] == 1:
                print("\n using partially observable states \n")
            from sustaindc_env import DCRL  # pylint: disable=import-error,import-outside-toplevel
        else:
            if env_config['month'] == 1:
                print("\n using fully observable states \n")
            from dcrl_env_harl import DCRL  # pylint: disable=import-error,import-outside-toplevel
        self.env = DCRL(env_config)  # Your existing DCRL environment
        self.possible_agents = self.env.agents  # List of agents
        self.agents = self.env.agents

        # Define observation and action spaces
        self.observation_spaces = {agent: space for agent, space in zip(self.possible_agents, self.env.observation_space)}
        self.action_spaces = {agent: space for agent, space in zip(self.possible_agents, self.env.action_space)}
        
        if env_config['nonoverlapping_shared_obs_space']:
            # ls_state[0:10]->10 variables; dc_state[4:9]->5 variables & bat_state[5]->1 variables
            self.share_observation_space = {agent: spaces.Box(low=0.0, high=1.0, shape=(20,), dtype=np.float32) for agent in self.possible_agents}
        else:  
            # Find the maximum dimension of observation space
            max_obs_dim = max(space.shape[0] for space in self.observation_spaces.values())

            # Calculate the total dimension for the shared observation space
            total_shared_dim = max_obs_dim * len(self.possible_agents)

            # Create a shared observation space for each agent
            self.share_observation_space = {
                agent: spaces.Box(low=np.float32(0), high=np.float32(1), shape=(total_shared_dim,), dtype=np.float32)
                for agent in self.possible_agents
            }
        
        self.metadata = {'render.modes': []}  # If no rendering is supported

    def reset(self, seed=None, options=None):
        """
        Reset the environment and return initial observations for all agents.
        """
        if seed is not None:
            np.random.seed(seed)  # Example of setting seed, adjust based on your environment's requirements
        
        # initial_observation should be a dictionary with agent names as keys and their observations as values
        initial_observations_with_info = self.env.reset()
        # return {agent: obs for agent, obs in zip(self.possible_agents, initial_observations)}
        return initial_observations_with_info

    def step(self, actions):
        """
        Take actions for all agents, process the environment's reaction, and return the next set of observations, rewards, etc.
        """
        obs, rewards, dones, truncateds, infos = self.env.step(actions)
        # obs = {agent: o for agent, o in zip(self.possible_agents, obs)}
        # rewards = {agent: rew for agent, rew in zip(self.possible_agents, rewards)}
        # dones = {agent: done for agent, done in zip(self.possible_agents, dones)}
        # infos = {agent: info for agent, info in zip(self.possible_agents, infos)}
        
        # obs = {agent: obs[agent] for agent in self.possible_agents}
        # rewards = {agent: rewards[agent] for agent in self.possible_agents}
        # dones = {agent: dones[agent] for agent in self.possible_agents}
        # infos = {agent: infos[agent] for agent in self.possible_agents}
        
        return (
            {agent: obs[agent] for agent in self.possible_agents},
            {agent: rewards[agent] for agent in self.possible_agents},
            {agent: dones[agent] for agent in self.possible_agents},
            {agent: truncateds[agent] for agent in self.possible_agents},
            {agent: infos[agent] for agent in self.possible_agents}
        )

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
