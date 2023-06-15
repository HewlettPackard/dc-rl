import os
from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from utils.make_envs_pyenv import make_dc_pyeplus_env, make_bat_fwd_env, make_ls_env
from utils.utils_cf import get_init_day, Time_Manager, Workload_Manager, CI_Manager, Weather_Manager, obtain_paths, get_energy_variables
from utils.base_agents import BaseBatteryAgent, BaseLoadShiftingAgent, BaseHVACAgent

class DCRL(MultiAgentEnv):
    def __init__(self, env_config=None):
        '''
        Args:
            env_config (dict): Dictionary containing parameters:
                agents: list of agent name (agent_ls, agent_dc, agent_bat)
                location: location of the environment for paths
                cintensity_file: path to the carbon intensity (CI) file
                weather_file: path to the weather file
                max_bat_cap_Mw: maximum battery capacity
                individual_reward_weight: weight of the individual reward (1=full individual, 0=full collaborative, default=0.8)
                flexible_load: flexible load ratio of the total workload (default = 0.1)
                ls_reward: method to calculate the load shifting reward
                dc_reward: method to calculate the dc reward
                bat_reward: method to calculate the battery reward
                worker_index: index of the worker. This parameter is added by RLLib.
        '''
        # create agent ids
        self.agents = env_config['agents']
        self.location = env_config['location']
        self.ci_file = env_config['cintensity_file']
        self.weather_file = env_config['weather_file']
        self.max_bat_cap_Mw = env_config['max_bat_cap_Mw']
        self.indv_reward = env_config['individual_reward_weight']
        self.collab_reward = (1 - self.indv_reward) / 2
        
        self.flexible_load = env_config['flexible_load']

        # Assign month according to worker index, if available
        if hasattr(env_config, 'worker_index'):
            month = int((env_config.worker_index - 1) % 12)
        else:
            month = env_config.get('month', 0)

        self._agent_ids = set(self.agents)

        # self.terminateds = set()
        # self.truncateds = set()
        
        ci_loc, wea_loc = obtain_paths(self.location)
        
        ls_reward_method = 'default_ls_reward' if not 'ls_reward' in env_config.keys() else env_config['ls_reward']
        dc_reward_method = 'default_dc_reward' if not 'dc_reward' in env_config.keys() else env_config['dc_reward']
        bat_reward_method = 'default_bat_reward' if not 'bat_reward' in env_config.keys() else env_config['bat_reward']
        
        self.ls_env = make_ls_env(month, reward_method=ls_reward_method)
        self.dc_env = make_dc_pyeplus_env(month+1, ci_loc, max_bat_cap_Mw=self.max_bat_cap_Mw, use_ls_cpu_load=True, reward_method=dc_reward_method) 
        self.bat_env = make_bat_fwd_env(month, max_bat_cap_Mw=self.max_bat_cap_Mw, reward_method=bat_reward_method)

        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
        
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


        self.init_day = get_init_day(month)
        self.workload_m = Workload_Manager(flexible_workload_ratio=flexible_load, init_day=self.init_day)
        self.ci_m = CI_Manager(init_day=self.init_day, location=ci_loc, filename=self.ci_file)
        self.weather_m = Weather_Manager(init_day=self.init_day, location=wea_loc, filename=self.weather_file)
        self.t_m = Time_Manager(self.init_day)

        # This actions_are_logits is True only for MADDPG, because RLLib defines MADDPG only for continuous actions.
        self.actions_are_logits = env_config.get("actions_are_logits", False)
        # self.which_agent = None

        super().__init__()

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Environment options.

        Returns:
            states (dict): Dictionary of states.
            infos (dict): Dictionary of infos.
        """
        # self.terminateds = set()
        # self.truncateds = set()
        # self.max_consumption = 0
        
        self.ls_terminated = False
        self.dc_terminated = False
        self.bat_terminated = False
        self.ls_truncated = False
        self.dc_truncated = False
        self.bat_truncated = False
        self.ls_reward = 0
        self.dc_reward = 0
        self.bat_reward = 0

        # Reset the managers
        t_i = self.t_m.reset() # Time manager
        workload, day_workload = self.workload_m.reset() # Workload manager
        temp, norm_temp = self.weather_m.reset() # Weather manager
        ci_i, ci_i_future = self.ci_m.reset() # CI manager. ci_i -> CI in the current timestep.
        
        # Set the external ambient temperature to data center environment
        self.dc_env.set_ambient_temp(temp)
        
        # Update the workload of the load shifting environment
        self.ls_env.update_workload(day_workload, workload)
        
        # Reset all the environments
        ls_s, self.ls_info = self.ls_env.reset()
        self.dc_state, self.dc_info = self.dc_env.reset()
        bat_s, self.bat_info = self.bat_env.reset()
         
        # Update the shared observation space
        batSoC = bat_s[1]
        
        # dc state -> [time (sine/cosine enconded), original dc observation, current workload, current normalized CI, battery SOC]
        self.dc_state = np.hstack((t_i, self.dc_state, workload, ci_i_future[0], batSoC))
        var_to_LS_energy = get_energy_variables(self.dc_state)
        
        # ls_state -> [time (sine/cosine enconded), original ls observation, current+future normalized CI, current workload, energy variables from DC, battery SoC]
        self.ls_state = np.hstack((t_i, ls_s, ci_i_future, workload, var_to_LS_energy, batSoC))
        
        # bat_state -> [time (sine/cosine enconded), battery SoC, current+future normalized CI]
        self.bat_state = np.hstack((t_i, bat_s, ci_i_future))

        states = {}
        infos = {}
        # Update self.agents considering the agents defined in the environment config.
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
        """
        Step the environment.

        Args:
            action_dict (MultiAgentDict): Dictionary of actions of each agent defined in self.agents.
  
        Returns:
            obs (dict): Dictionary of observations/states.
            rews (dict): Dictionary of rewards.
            terminated (dict): Dictionary of terminated flags.
            truncated (dict): Dictionary of truncated flags.
            infos (dict): Dictionary of infos.
        """
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        terminated["__all__"] = False
        truncated["__all__"] = False

        # Step in the managers
        t_i, terminal = self.t_m.step()
        workload, day_workload = self.workload_m.step()
        temp, norm_temp = self.weather_m.step()
        ci_i, ci_i_future = self.ci_m.step()

        # Transform the actions if the algorithm uses continuous action space. Like RLLib with MADDPG.
        if self.actions_are_logits:
            for k, v in action_dict.items():
                if isinstance(v, np.ndarray):
                    action_dict[k] = np.random.choice(np.arange(len(v)), p=v)
        
        # Extract the action from the action dictionary.
        # If the agent is declared, use the action from the action dictionary.
        # If the agent is not declared, use the default action (do nothing) of the base agent.
        if "agent_ls" in self.agents:
            action = action_dict["agent_ls"]
        else:
            action = self.base_agents["agent_ls"].do_nothing_action()
            
        # Now, update the load shifting environment/agent first.
        # i = "agent_ls"
        self.ls_env.update_workload(day_workload, workload)
        
        # Do a step
        self.ls_state, self.ls_penalties, self.ls_terminated, self.ls_truncated, self.ls_info = self.ls_env.step(action)

        # Now, the data center environment/agent.
        if "agent_dc" in self.agents:
            action = action_dict["agent_dc"]
        else:
            action = self.base_agents["agent_dc"].do_nothing_action()

        # Update the data center environment/agent.
        shifted_wkld = self.ls_info['load']
        self.dc_env.set_shifted_wklds(shifted_wkld)
        
        batSoC = self.bat_state[1]
        self.dc_env.set_ambient_temp(temp)
        
        # Do a step in the data center environment
        # By default, the reward is ignored. The reward is calculated after the battery env step with the total energy usage.
        self.dc_state, _, self.dc_terminated, self.dc_truncated, self.dc_info = self.dc_env.step(action)

        # Finally, the battery environment/agent.
        if "agent_bat" in self.agents:
            action = action_dict["agent_bat"]
        else:
            action = self.base_agents["agent_bat"].do_nothing_action()
            
        # The battery environment/agent is updated.
        self.bat_env.set_dcload(self.dc_info['Total Power kW'] / 1e3) # The DC load is updated with the total power in MW.
        self.bat_state = self.bat_env.update_state() # The state is updated with DC load
        self.bat_env.update_ci(ci_i, ci_i_future[0]) # Update the CI with the current CI, and the normalized current CI.
        
        # Make a step in the environment
        self.bat_state, self.bat_reward, self.bat_terminated, self.bat_truncated, self.bat_info = self.bat_env.step(action)
        
        # Update the state of the bat state
        batSoC = self.bat_state[1]
        self.bat_state = np.hstack((t_i, self.bat_state, ci_i_future))
        
        self.dc_reward = -1.0 * self.bat_info['total_energy_with_battery'] / 1e3  # The raw reward of the DC is directly the total energy consumption in MWh.

        # Update the shared variables
        self.dc_state = np.hstack((t_i, self.dc_state, shifted_wkld, ci_i_future[0], batSoC))
        
        # We need to update the LS state with the DC energy variables and the final battery SoC.
        var_to_LS_energy = get_energy_variables(self.dc_state)
        self.ls_state = np.hstack((t_i, self.ls_state, ci_i_future, workload, var_to_LS_energy, batSoC))
        
        # If agent_ls is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_ls" in self.agents:
            obs['agent_ls'] = self.ls_state
            # For the reward, we need to add to the load_shifting penalities, the battery reward, because include the CI reward.
            rew["agent_ls"] = self.indv_reward * (self.ls_penalties + self.bat_reward) + self.collab_reward * self.dc_reward
            terminated["agent_ls"] = terminal
            info["agent_ls"] = self.ls_info
        
        # If agent_dc is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_dc" in self.agents:
            obs["agent_dc"] = self.dc_state
            rew["agent_dc"] = self.indv_reward * self.dc_reward + self.collab_reward * self.ls_penalties + self.collab_reward * self.bat_reward
            terminated["agent_dc"] = terminal
            info["agent_dc"] = self.dc_info
            
         # If agent_bat is included in the agents list, then update the observation, reward, terminated, truncated, and info dictionaries. 
        if "agent_bat" in self.agents:
            obs["agent_bat"] = self.bat_state
            rew["agent_bat"] = self.indv_reward * self.bat_reward + self.collab_reward * self.dc_reward + self.collab_reward * self.ls_penalties
            terminated["agent_bat"] = terminal
            info["agent_bat"] = self.bat_info

        if terminal:
            terminated["__all__"] = True
            truncated["__all__"] = True
            for agent in self.agents:
                truncated[agent] = True
                
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
            
            # Collaborative weight in the reward
            'individual_reward_weight': 0.8,
            
            # Flexible load ratio
            'flexible_load': 0.1,
            
            # Specify reward methods
            'ls_reward': 'default_ls_reward',
            'dc_reward': 'default_dc_reward',
            'bat_reward': 'default_bat_reward'
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