import os
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from utils import make_envs
from utils.base_agents import (BaseBatteryAgent, BaseHVACAgent,
                               BaseLoadShiftingAgent)
from utils.utils_cf import (CI_Manager, Time_Manager, Weather_Manager,
                            Workload_Manager, get_init_day, obtain_paths)
    
class DCRLeplus(MultiAgentEnv):
    def __init__(self, env_config=None):
        # create agent ids
        self.agents = env_config['agents']
        self.location = env_config['location']
        self.ci_file = env_config['cintensity_file']
        self.weather_file = env_config['weather_file']
        self.indv_reward = env_config['individual_reward_weight']
        self.collab_reward = (1 - self.indv_reward) / 2
        
        self.flexible_load = env_config['flexible_load']
        
        ci_loc, wea_loc = obtain_paths(self.location)
        
        self._agent_ids = set(self.agents)

        self.terminateds = set()
        self.truncateds = set()
        
        ls_reward_method = 'default_ls_reward' if not 'ls_reward' in env_config.keys() else env_config['ls_reward']
        dc_reward_method = 'default_dc_reward' if not 'dc_reward' in env_config.keys() else env_config['dc_reward']
        bat_reward_method = 'default_bat_reward' if not 'bat_reward' in env_config.keys() else env_config['bat_reward']

        # Assign month according to worker index, if available
        if hasattr(env_config, 'worker_index'):
            month = int((env_config.worker_index - 1) % 12)
        else:
            month = env_config.get('month', 0)

        self.ls_env = make_envs.make_ls_env(month, self.location, reward_method=ls_reward_method)
        self.dc_env = make_envs.make_dc_env(month, self.location, reward_method=dc_reward_method) 
        self.bat_env = make_envs.make_bat_fwd_env(month, self.location, reward_method=bat_reward_method)

        self._obs_space_in_preferred_format = True
        
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})
        self.base_agents = {}
        flexible_load = 0
        if "agent_ls" in self.agents:
            self.observation_space["agent_ls"] = self.ls_env.observation_space
            self.action_space["agent_ls"] = self.ls_env.action_space
            flexible_load = self.flexible_load
        else:
            self.base_agents["agent_ls"] = BaseLoadShiftingAgent()
            
        if "agent_dc" in self.agents:
            self.observation_space["agent_dc"] = self.dc_env.observation_space
            self.action_space["agent_dc"] = self.dc_env.action_space
        else:
            self.base_agents["agent_dc"] = BaseHVACAgent()
            
        if "agent_bat" in self.agents:
            self.observation_space["agent_bat"] = self.bat_env.observation_space
            self.action_space["agent_bat"] = self.bat_env.action_space
        else:
            self.base_agents["agent_bat"] = BaseBatteryAgent()
            
        self._action_space_in_preferred_format = True

        self.init_day = get_init_day(month)
        self.t_m = Time_Manager(init_day=self.init_day)
        
        self.workload_m = Workload_Manager(init_day=self.init_day, flexible_workload_ratio=flexible_load)
        self.ci_m = CI_Manager(init_day=self.init_day, location=ci_loc, filename=self.ci_file)

        self.actions_are_logits = env_config.get("actions_are_logits", False)
        self.which_agent = None

        super().__init__()

    def reset(self, *, seed=None, options=None):
        self.terminateds = set()
        self.truncateds = set()

        self.ls_terminated = False
        self.dc_terminated = False
        self.bat_terminated = False
        self.ls_truncated = False
        self.dc_truncated = False
        self.bat_truncated = False
        self.ls_reward = 0
        self.dc_reward = 0
        self.bat_reward = 0

        workload, day_workload = self.workload_m.reset()
        self.ls_env.update_workload(day_workload, workload)
        ls_s, self.ls_info = self.ls_env.reset()
        self.dc_state, self.dc_info = self.dc_env.reset()
        bat_s, self.bat_info = self.bat_env.reset()
        
        ci_i, ci_if = self.ci_m.reset()
        t_i = self.t_m.reset()
        
        self.ls_state = np.hstack((t_i, ls_s, ci_if, workload))
        var_to_LS_energy = [self.dc_state[i] for i in [4, 5, 6, 8]]
        batSoC = bat_s[1]
        self.ls_state = np.hstack((self.ls_state, var_to_LS_energy, batSoC))
        self.bat_state = np.hstack((t_i, bat_s, ci_if))

        states = {}
        infos = {}
        if "agent_ls" in self.agents:
            states["agent_ls"] = self.ls_state
            infos["agent_ls"] = self.ls_info
            
        if "agent_dc" in self.agents:
            states["agent_dc"] = self.dc_state
            infos["agent_dc"] = self.dc_info
            
        if "agent_bat" in self.agents:
            states["agent_bat"] = self.bat_state
            infos["agent_bat"] = self.bat_info
            
        return states, infos

    def step(self, action_dict: MultiAgentDict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        terminated["__all__"] = False
        truncated["__all__"] = False

        workload,day_workload = self.workload_m.step()
        ci_i, ci_if = self.ci_m.step()
        t_i = self.t_m.step()

        if self.actions_are_logits:
            for k, v in action_dict.items():
                if isinstance(v, np.ndarray):
                    action_dict[k] = np.random.choice(np.arange(len(v)), p=v)
        
        if "agent_ls" in self.agents:
            action = action_dict["agent_ls"]
        else:
            action = self.base_agents["agent_ls"].do_nothing_action()
            
        i = "agent_ls"
        self.ls_env.update_workload(day_workload, workload)
        self.ls_state, self.ls_penalties, self.ls_terminated, self.ls_truncated, self.ls_info = self.ls_env.step(action)
        self.ls_state = np.hstack((t_i, self.ls_state, ci_if, workload))
    
        rew_i = self.ls_penalties
        terminated_i = self.ls_terminated
        truncated_i = self.ls_truncated
        info_i = self.ls_info

        if "agent_ls" in self.agents:
            rew["agent_ls"] = rew_i
            terminated["agent_ls"] = terminated_i
            truncated["agent_ls"] = truncated_i
            info["agent_ls"] = info_i

        if "agent_dc" in self.agents:
            action = action_dict["agent_dc"]

        else:
            action = self.base_agents["agent_dc"].do_nothing_action()

            
        i = "agent_dc"
        shifted_wkld = self.ls_info['load']
        self.dc_env.set_shifted_wklds(shifted_wkld)
        ci = self.ls_state[6]
        batSoC = self.bat_state[1]
        self.dc_state[-3:] = [shifted_wkld, ci, batSoC]

        self.dc_state,_, self.dc_terminated, self.dc_truncated, self.dc_info = self.dc_env.step(action)
        obs_i =  self.dc_state 
        rew_i =  0
        terminated_i = self.dc_terminated
        truncated_i = self.dc_truncated
        info_i = self.dc_info

        if "agent_dc" in self.agents:
            obs["agent_dc"] = obs_i
            rew["agent_dc"] = rew_i
            terminated["agent_dc"] = terminated_i
            truncated["agent_dc"] = truncated_i
            info["agent_dc"] = info_i

        if "agent_bat" in self.agents:
            action = action_dict["agent_bat"]
        else:
            action = self.base_agents["agent_bat"].do_nothing_action()
            
        i = "agent_bat"
        self.bat_env.set_dcload(self.dc_info['DC_load']/1e3)
        self.bat_state = self.bat_env.update_state()
        self.bat_env.update_ci(ci_i, ci_if[0])
        self.bat_state, self.bat_reward, self.bat_terminated, self.bat_truncated, self.bat_info = self.bat_env.step(action)
        batSoC = self.bat_state[1]
        self.bat_state = np.hstack((t_i, self.bat_state, ci_if))
        obs_i =  self.bat_state 
        rew_i = self.bat_reward
        terminated_i = self.bat_terminated
        truncated_i = self.bat_truncated
        info_i = self.bat_info

        self.dc_reward = -1*self.bat_info['total_energy_with_battery']/1e3 

        var_to_LS_energy = [self.dc_state[i] for i in [4, 5, 6, 8]]
        
        self.ls_state = np.hstack((self.ls_state, var_to_LS_energy, batSoC))
        if "agent_ls" in self.agents:
            obs['agent_ls'] = self.ls_state
        
        if "agent_bat" in self.agents:
            obs["agent_bat"] = obs_i
            rew["agent_bat"] = rew_i
            terminated["agent_bat"] = terminated_i
            truncated["agent_bat"] = truncated_i
            info["agent_bat"] = info_i

        if "agent_bat" in self.agents:
            rew["agent_bat"] = self.indv_reward * rew["agent_bat"] + self.collab_reward * self.dc_reward + self.collab_reward * self.ls_penalties
            terminated["agent_bat"] = self.dc_terminated

        if "agent_ls" in self.agents:
            rew["agent_ls"] = self.indv_reward * (self.ls_penalties + self.bat_reward) + self.collab_reward * self.dc_reward
            terminated["agent_ls"] = self.dc_terminated

        if "agent_dc" in self.agents:
            obs["agent_dc"][-1] = batSoC
            rew["agent_dc"] = self.indv_reward * self.dc_reward + self.collab_reward * self.ls_penalties + self.collab_reward * self.bat_reward


        if self.dc_terminated:
            terminated["__all__"] = True
            truncated["__all__"] = True
            for agent in self.agents:
                truncated[agent] = True
                
        return obs, rew, terminated, truncated, info