import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

class SustainDCPettingZooEnv(ParallelEnv):
    def __init__(self, env_config):
        """
        Initialize the SustainDCPettingZooEnv class.

        Args:
            env_config (dict): Configuration for the SustainDC environment.
        """
        super().__init__()
        if env_config['partial_obs']:
            if env_config['month'] == 1:
                print(f"\n using partially observable states and month: {env_config['month']}\n")
            from sustaindc_env import SustainDC  # pylint: disable=import-error,import-outside-toplevel
        else:
            raise NotImplementedError("Fully observable states are no longer supported. Please set 'partial_obs' to True.")
        
        self.env = SustainDC(env_config)  # Your existing SustainDC environment
        self.possible_agents = self.env.agents  # List of agents
        self.agents = self.env.agents

        # Define observation and action spaces
        self.observation_spaces = {agent: space for agent, space in zip(self.possible_agents, self.env.observation_space)}
        self.action_spaces = {agent: space for agent, space in zip(self.possible_agents, self.env.action_space)}
        
        if env_config['nonoverlapping_shared_obs_space']:
            # ls_state[0:10]->10 variables; dc_state[4:9]->5 variables & bat_state[5]->1 variable
            self.share_observation_space = {agent: spaces.Box(low=-2.0, high=2.0, shape=(9,), dtype=np.float32) for agent in self.possible_agents}
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

        Args:
            seed (int, optional): Seed for random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            dict: Initial observations for all agents.
        """
        if seed is not None:
            np.random.seed(seed)  # Example of setting seed, adjust based on your environment's requirements
        
        # initial_observation should be a dictionary with agent names as keys and their observations as values
        initial_observations_with_info = self.env.reset()
        return initial_observations_with_info

    def step(self, actions):
        """
        Take actions for all agents, process the environment's reaction, and return the next set of observations, rewards, etc.

        Args:
            actions (dict): Actions to be taken by the agents.

        Returns:
            tuple: Observations, rewards, done flags, truncated flags, and info for all agents.
        """
        obs, rewards, dones, truncateds, infos = self.env.step(actions)
        
        return (
            {agent: obs[agent] for agent in self.possible_agents},
            {agent: rewards[agent] for agent in self.possible_agents},
            {agent: dones[agent] for agent in self.possible_agents},
            {agent: truncateds[agent] for agent in self.possible_agents},
            {agent: infos[agent] for agent in self.possible_agents}
        )

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode (str): Mode of rendering.

        Returns:
            Rendered output.
        """
        return self.env.render(mode=mode)

    def close(self):
        """
        Close the environment.
        """
        self.env.close()
