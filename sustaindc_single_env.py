"""Single-agent view of SustainDC: one agent observes everything and controls all three actions.

The agent sees the de-duplicated joint observation (see utils/joint_obs.py) and outputs one joint action
for (load shifting, cooling, battery). Must be created from the repo root (data paths are relative).
"""
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sustaindc_env import SustainDC
from utils.joint_obs import build_joint_obs, joint_obs_dim

AGENTS = ('agent_ls', 'agent_dc', 'agent_bat')
N_SUB_ACTIONS = (3, 3, 3)  # ls: defer/do-nothing/process, dc: setpoint -1/0/+1, bat: charge/discharge/idle


def decode_joint_action(action, action_type='multidiscrete'):
    """Map a joint action (MultiDiscrete vector or Discrete(27) index) to SustainDC's {agent: int} action dict."""
    if action_type == 'discrete':
        sub = np.unravel_index(int(np.asarray(action).reshape(())), N_SUB_ACTIONS)
    else:
        sub = np.asarray(action).reshape(-1)
    return {agent: int(a) for agent, a in zip(AGENTS, sub)}


class SustainDCSingleAgentEnv(gym.Env):
    """Gymnasium wrapper around SustainDC with a joint observation and a joint action.

    Extra config keys (besides the usual SustainDC ones):
        action_type: "multidiscrete" (default) -> MultiDiscrete([3, 3, 3]), one categorical head per sub-action.
                     "discrete" -> Discrete(27), one flat categorical over all combinations (needed for DQN).
        reward_type: "agent_ls" (default) -> rew['agent_ls'] after collaborative mixing, i.e. the signal the
                     HARL shared critic trains on. "ls_raw" -> the unmixed load-shifting reward
                     (footprint + task penalties).
        include_dc_setpoint: passed to SustainDC, which appends the normalized current CRAC setpoint to the cooling
                     observation; the joint observation then includes it as its last element (30 dims instead of 29).
                     Default False; the training yaml turns it on for both this env and MARL.
        include_pv_obs: passed to SustainDC, which appends 5 PV features to every agent's observation; the joint
                     observation then ends with them (+5 dims). Default False.
    """

    metadata = {'render_modes': []}

    def __init__(self, env_config=None):
        super().__init__()
        env_config = dict(env_config or {})
        self.action_type = env_config.pop('action_type', 'multidiscrete')
        self.reward_type = env_config.pop('reward_type', 'agent_ls')
        self.include_dc_setpoint = bool(env_config.get('include_dc_setpoint', False))
        self.include_pv_obs = bool(env_config.get('include_pv_obs', False))
        env_config['agents'] = list(AGENTS)

        if self.action_type not in ('multidiscrete', 'discrete'):
            raise ValueError(f"action_type must be 'multidiscrete' or 'discrete', got {self.action_type!r}")
        if self.reward_type not in ('agent_ls', 'ls_raw'):
            raise ValueError(f"reward_type must be 'agent_ls' or 'ls_raw', got {self.reward_type!r}")

        self.env = SustainDC(env_config)

        obs_dim = joint_obs_dim(self.include_dc_setpoint, self.include_pv_obs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        if self.action_type == 'multidiscrete':
            self.action_space = spaces.MultiDiscrete(N_SUB_ACTIONS)
        else:
            self.action_space = spaces.Discrete(int(np.prod(N_SUB_ACTIONS)))

    def decode_action(self, action):
        """Map a joint action to SustainDC's {agent: int} action dict."""
        return decode_joint_action(action, self.action_type)

    def _obs(self):
        return build_joint_obs(self.env.ls_state, self.env.dc_state, self.env.bat_state, self.include_dc_setpoint,
                               self.include_pv_obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # SustainDC samples the start day/hour with `random` and uses np.random internally.
            random.seed(seed)
            np.random.seed(seed)
        self.env.reset()
        return self._obs(), {}

    def step(self, action):
        obs, rew, _, truncs, info = self.env.step(self.decode_action(action))
        reward = float(rew['agent_ls'] if self.reward_type == 'agent_ls' else self.env.ls_reward)
        truncated = bool(truncs.get('__all__', False) or truncs['agent_ls'])
        # info['agent_ls'] already merges ls/dc/bat infos and the reward params.
        return self._obs(), reward, False, truncated, info['agent_ls']

    def close(self):
        pass
