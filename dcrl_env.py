import os
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from utils.make_ls_env import make_ls_env
from utils.make_pyeplus_env import make_dc_pyeplus_env
from utils.make_bat_env import make_bat_fwd_env


from utils.utils_cf import get_init_day, Time_Manager, Workload_Manager, CI_Manager, Weather_Manager,  obtain_paths, get_energy_variables
from utils.base_agents import BaseBatteryAgent, BaseLoadShiftingAgent, BaseHVACAgent

class DCRL(MultiAgentEnv):
    def __init__(self, env_config=None):
        # create agent ids
        self.agents = env_config['agents']
        self.location = env_config['location']
        self.ci_file = env_config['cintensity_file']
        self.weather_file = env_config['weather_file']
        self.max_bat_cap_Mw = env_config['max_bat_cap_Mw']
        
        # Assign month according to worker index, if available
        if hasattr(env_config, 'worker_index'):
            month = int((env_config.worker_index - 1) % 12)
        else:
            month = env_config.get('month', 0)

        self._agent_ids = set(self.agents)

        self.terminateds = set()
        self.truncateds = set()
        
        ci_loc, wea_loc = obtain_paths(self.location)
        
        self.ls_env = make_ls_env(month)
        self.dc_env = make_dc_pyeplus_env(month + 1, ci_loc, max_bat_cap_Mw=self.max_bat_cap_Mw, use_ls_cpu_load=True) 
        self.bat_env = make_bat_fwd_env(month, max_bat_cap_Mw = self.max_bat_cap_Mw)

        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})
        self.base_agents = {}
        flexible_load = 0
        
        if "agent_ls" in self.agents:
            self.observation_space["agent_ls"] = self.ls_env.observation_space
            self.action_space["agent_ls"] = self.ls_env.action_space
            flexible_load = 0.1
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


        self.init_day = get_init_day(month)
        self.workload_m = Workload_Manager(flexible_workload_ratio=flexible_load,init_day=self.init_day)
        self.ci_m = CI_Manager(init_day=self.init_day, location=ci_loc, filename=self.ci_file)
        self.weather_m = Weather_Manager(init_day=self.init_day, location=wea_loc, filename=self.weather_file)
        self.t_m = Time_Manager(self.init_day)

        self.actions_are_logits = env_config.get("actions_are_logits", False)
        self.which_agent = None

        super().__init__()

    def reset(self, *, seed=None, options=None):
        
        self.terminateds = set()
        self.truncateds = set()
        self.max_consumption = 0
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
        temp,n_temp = self.weather_m.reset()
        self.dc_env.set_ambient_temp(temp)
        self.ls_env.update_workload(day_workload, workload)
        ls_s, self.ls_info = self.ls_env.reset()
        self.dc_state, self.dc_info = self.dc_env.reset()
        bat_s, self.bat_info = self.bat_env.reset()
        

        ci_i,ci_if = self.ci_m.reset()
        t_i = self.t_m.reset()
        
        batSoC = bat_s[1]
        self.ls_state = np.hstack((t_i, ls_s, ci_if, workload))
        self.dc_state = np.hstack((t_i,self.dc_state,workload,ci_if[0],batSoC))
        var_to_LS_energy = get_energy_variables(self.dc_state)  # AN modified index
        
        self.ls_state = np.hstack((self.ls_state,var_to_LS_energy,batSoC))
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

        # print(f'Location: {self.location}')

        workload,day_workload = self.workload_m.step()
        ci_i, ci_if = self.ci_m.step()
        t_i = self.t_m.step()
        temp,n_temp = self.weather_m.step()

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
        self.ls_state, self.ls_penalties, self.ls_terminated, self.ls_truncated, self.ls_info  = self.ls_env.step(action)
        self.ls_state = np.hstack((t_i, self.ls_state, ci_if, workload))
        #print(self.ls_state)
    
        rew_i =  self.ls_penalties
        #print('REWARD LS', rew_i)
        terminated_i = self.ls_terminated
        truncated_i = self.ls_truncated
        info_i = self.ls_info

        if "agent_ls" in self.agents:
            rew["agent_ls"] = rew_i
            terminated["agent_ls"] = terminated_i
            truncated["agent_ls"] = truncated_i
            info["agent_ls"] = info_i

        # print(f"Deciding the action for agent_dc")
        if "agent_dc" in self.agents:
            action = action_dict["agent_dc"]
            # print('Is inside self.agents: ', action)

        else:
            action = self.base_agents["agent_dc"].do_nothing_action()
            # print('NO is inside self.agents, so the action is: ', action)
            # print(f'Inside self.agents: {self.agents}')

        wkld = self.ls_info['load']
        self.dc_env.set_shifted_wklds(wkld)
        ci = self.ls_state[6]
        batSoC = self.bat_state[1]
        self.dc_env.set_ambient_temp(temp)
        #self.dc_state[-3:] = [wkld, ci, batSoC]
        
        self.dc_state,_, self.dc_terminated, self.dc_truncated, self.dc_info = self.dc_env.step(action)#self._do_dc_env_step(action_dict)
        self.dc_state = np.hstack((t_i,self.dc_state,workload,ci_if[0],batSoC))

        if self.dc_info['IT POWER w'] > self.max_consumption:
            self.max_consumption = self.dc_info['IT POWER w']
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
        self.bat_env.set_dcload(self.dc_info['Total Power kW']/1e3)
        self.bat_state = self.bat_env.update_state()
        self.bat_env.update_ci(ci_i,ci_if[0])
        self.bat_state, self.bat_reward, self.bat_terminated, self.bat_truncated, self.bat_info = self.bat_env.step(action)
        batSoC = self.bat_state[1]
        self.bat_state = np.hstack((t_i,self.bat_state,ci_if))
        obs_i =  self.bat_state 
        rew_i = self.bat_reward
        terminated_i = self.bat_terminated
        truncated_i = self.bat_truncated
        info_i = self.bat_info
        
        self.dc_reward = -1*self.bat_info['total_energy_with_battery']/1e3 

        var_to_LS_energy = get_energy_variables(self.dc_state)  # AN modified index
        
        self.ls_state = np.hstack((self.ls_state,var_to_LS_energy,batSoC))
        if "agent_ls" in self.agents:
            obs['agent_ls'] = self.ls_state
        #self.ls_state = self.ls_env.update_state()
        
        if "agent_bat" in self.agents:
            obs["agent_bat"] = obs_i
            rew["agent_bat"] = rew_i
            terminated["agent_bat"] = terminated_i
            truncated["agent_bat"] = truncated_i
            info["agent_bat"] = info_i
            rew["agent_bat"] = rew["agent_bat"] + 0.1*self.dc_reward + 0.1*self.ls_penalties
            terminated["agent_bat"] = self.dc_terminated

        if "agent_ls" in self.agents:
            rew["agent_ls"] = rew["agent_ls"] + self.bat_reward + 0.1*self.dc_reward
            terminated["agent_ls"] = self.dc_terminated

        
        if "agent_dc" in self.agents:
            obs["agent_dc"][-1] = batSoC
            rew["agent_dc"] = self.dc_reward + 0.1*self.ls_penalties + 0.1*self.bat_reward

        if self.dc_terminated:
            terminated["__all__"] = True
            truncated["__all__"] = True
        
        return obs, rew, terminated, truncated, info


if __name__ == '__main__':

    env = DCRL(
        env_config={
            # Agents active
            'agents': ['agent_ls', 'agent_dc', 'agent_bat'],

            # Datafiles
            'location': 'ny',
            'cintensity_file': 'NYIS_NG_&_avgCI.csv',
            'weather_file': 'USA_NY_New.York-Kennedy.epw',

            # Battery capacity
            'max_bat_cap_Mw': 2,
            
            'reward_method': 'default_dc_reward'
            }
    )

    done = False
    env.reset()
    step = 0
    reward_ls = reward_dc = reward_bat = 0
    
    while not done:
        action_dict = {
            'agent_ls': env.ls_env.action_space.sample(),
            'agent_dc': env.dc_env.action_space.sample(),
            'agent_bat': env.bat_env.action_space.sample()
        }

        _, rewards, terminated, _, _ = env.step(action_dict)
        done = terminated['__all__']
        
        reward_ls += rewards['agent_ls']
        reward_dc += rewards['agent_dc']
        reward_bat += rewards['agent_bat']

        print(f"Step {step}, rewards = ", rewards)
        step += 1
    
    print("End of episode.")
    print("Load shifting agent reward = ", reward_ls)
    print("DC agent reward = ", reward_dc)
    print("Battery agent reward = ", reward_bat)