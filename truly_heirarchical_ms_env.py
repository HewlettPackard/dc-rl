from tqdm import tqdm
from ray.rllib.env import MultiAgentEnv
from heirarchical_env import HeirarchicalDCRL, DEFAULT_CONFIG
from tensorboardX import SummaryWriter

class TrulyHeirarchicalDCRL(HeirarchicalDCRL, MultiAgentEnv):

    def __init__(self, config):
        HeirarchicalDCRL.__init__(self, config)
        MultiAgentEnv.__init__(self)
        self.writer = SummaryWriter("logs")
        self.global_step = 0

    def step(self, actions: dict):
        if "high_level_policy" in actions:
            return self._high_level_step(actions["high_level_policy"])
        else:
            return self._low_level_step(actions)

    def _high_level_step(self, action):
        overassigned_workload = self.safety_enforcement(action)
        obs = {}
        rewards = {}
        self.steps_left_lower = 5
        for dc in self.datacenter_ids:
            obs[dc + '_ls_policy'] = self.low_level_observations[dc]['agent_ls']
            rewards[dc + '_ls_policy'] = 0
        done = truncated = {"__all__": False}
        self.cumulative_reward = 0
        return obs, rewards, done, truncated, {}

    def _low_level_step(self, actions):
        self.steps_left_lower -= 1
        low_level_actions = {dc: {'agent_ls': actions[dc + '_ls_policy']} for dc in self.datacenter_ids}
        done = self.low_level_step(low_level_actions)
        obs = {}
        rewards = {}
        # Low-level obs and rewards
        for dc in self.datacenter_ids:
            obs[dc + '_ls_policy'] = self.low_level_observations[dc]['agent_ls']
            rewards[dc + '_ls_policy'] = self.low_level_rewards[dc]['agent_ls']
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        self.cumulative_reward += self.calc_reward()
        if done or self.steps_left_lower == 0:
            self.heir_obs = {}
            for dc in self.datacenter_ids:
                self.heir_obs[dc] = self.get_dc_variables(dc)

            terminated["__all__"] = done
            truncated["__all__"] = False
            obs['high_level_policy'] = self.heir_obs
            rewards['high_level_policy'] = self.cumulative_reward
            
            if done:
                totalfp = 0
                for dc in self.datacenter_ids:
                    totalfp += sum(self.metrics[dc]['bat_CO2_footprint'])
                print("Total CO2 footprint: ", totalfp)

                # Log the scalar totalfp to TensorBoard
                self.writer.add_scalar("Total CO2 footprint", totalfp, self.global_step)
                self.writer.flush()
                self.global_step += 1  # Increment the step counter

        return obs, rewards, terminated, truncated, {}  

    
    def reset(self, seed=None, options=None):
        super().reset(seed)
        obs = {}
        obs['high_level_policy'] = self.heir_obs

        return obs, {}
    
if __name__ == '__main__':
    env = TrulyHeirarchicalDCRL(DEFAULT_CONFIG)

    done = False
    obs, _ = env.reset()
    
    with tqdm(total=env._max_episode_steps) as pbar:
        while not done:
            actions = {}
            actions['high_level_policy'] = env.action_space.sample()
            for dc in env.datacenter_ids:
                actions[dc + '_ls_policy'] = env.datacenters['DC1'].ls_env.action_space.sample()

            obs, rewards, terminated, truncated, info = env.step(actions)
            done = truncated['__all__']

            pbar.update(1)