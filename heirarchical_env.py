import os
import random
import warnings
    
import torch
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_tf

from harl.runners import RUNNER_REGISTRY
from harl.utils.configs_tools import get_defaults_yaml_args
from dcrl_env import DCRL
from hierarchical_workload_optimizer import WorkloadOptimizer
from utils.base_agents import *

# Boilerplate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf1, *_ = try_import_tf()
tf1.disable_v2_behavior()

warnings.filterwarnings(
    action="ignore",
    category=UserWarning
    )

DEFAULT_CONFIG = {
    # AZ config
    'config1' : {
        'location': 'az',
        'cintensity_file': 'AZPS_NG_&_avgCI.csv',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc3.json',
        'datacenter_capacity_mw' : 1.1,
        'timezone_shift': 8,
        'month': 0,
        'days_per_episode': 30,
        },

    # NY config
    'config2' : {
        'location': 'ny',
        'cintensity_file': 'NYIS_NG_&_avgCI.csv',
        'weather_file': 'USA_NY_New.York-Kennedy.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc2.json',
        'datacenter_capacity_mw' : 1,
        'timezone_shift': 0,
        'month': 0,
        'days_per_episode': 30
        },

    # WA config
    'config3' : {
        'location': 'wa',
        'cintensity_file': 'BPAT_NG_&_avgCI.csv',
        'weather_file': 'USA_WA_Port.Angeles-Fairchild.epw',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv',
        'dc_config_file': 'dc_config_dc1.json',
        'datacenter_capacity_mw' : 0.9,
        'timezone_shift': 16,
        'month': 0,
        'days_per_episode': 30
        },
    
    # List of active low-level agents
    'active_agents': ['agent_dc'],

    # config for loading trained low-level agents
    'low_level_actor_config': {
        'harl': {
            'algo' : 'happo',
            'env' : 'dcrl',
            'exp_name' : 'll_actor',
            'model_dir': '/lustre/gundechv/dc-rl/seed-00001-2024-04-22-20-59-21/models',
            },
        'rllib': {
            'checkpoint_path': '/lustre/gundechv/dc-rl/maddpg/checkpoint_000000/',
        }
    },
}


class LowLevelActorBase:

    def __init__(self, config: Dict = {}):
        
        self.do_nothing_actors = {
            "agent_ls": BaseLoadShiftingAgent(), 
            "agent_dc": BaseHVACAgent(), 
            "agent_bat": BaseBatteryAgent()
        }
    
    def compute_actions(self, observations: Dict, **kwargs) -> Dict:
        actions = {env_id: {} for env_id in observations}

        for env_id, obs in observations.items():
            for agent_id in obs:
                actions[env_id][agent_id] = self.do_nothing_actors[agent_id].do_nothing_action()

        return actions

class LowLevelActorHARL(LowLevelActorBase):

    def __init__(self, config, active_agents: list = []):
        super().__init__(config)

        config = config['harl']
        algo_args, env_args = get_defaults_yaml_args(config["algo"], config["env"])
        algo_args['train']['n_rollout_threads'] = 1
        algo_args['eval']['n_eval_rollout_threads'] = 1
        algo_args['train']['model_dir'] = config['model_dir']
    
        self.ll_actors = RUNNER_REGISTRY[config["algo"]](config, algo_args, env_args)
        self.active_agents = active_agents
        
    def compute_actions(self, observations, **kwargs):
        actions = {}
        
        eval_rnn_states = np.zeros((1, 1, 1), dtype=np.float32)
        eval_masks = np.ones((1, 1), dtype=np.float32)
        
        expected_length = self.ll_actors.actor[0].obs_space.shape[0]

        for agent_idx, (agent_id, agent_obs) in enumerate(observations.items()):
            if agent_id in self.active_agents:
                additional_length = expected_length - len(agent_obs)
                
                # Create an array of 1's with the required additional length
                ones_to_add = np.ones(additional_length, dtype=agent_obs.dtype)

                # Concatenate the current array with the array of 1's
                agent_obs = np.concatenate((agent_obs, ones_to_add))

                # eval_rnn_states and eval_masks is only being used on RNN
                # Obtain the action of each actor
                # TODO: Make sure that we are asking the correct actor for their agent_id and agent_idx
                action, _ = self.ll_actors.actor[agent_idx].act(agent_obs, eval_rnn_states, eval_masks, deterministic=True)
        
                actions[agent_id] = action.numpy()[0]
            else:
                actions[agent_id] = self.do_nothing_actors[agent_id].do_nothing_action()

        return actions

class LowLevelActorRLLIB(LowLevelActorBase):

    def __init__(self, config, active_agents: list = []):
        super().__init__(config)

        self.active_agents = active_agents

        self.policies = {}
        for agent_id in self.active_agents:
            with tf1.Session().as_default():
                self.policies[agent_id] = Policy.from_checkpoint(
                    f'{config["rllib"]["checkpoint_path"]}/policies/{agent_id}'
                    )

        # self.actor = Algorithm.from_checkpoint(config['checkpoint_path'])
        
    def compute_actions(self, observations, **kwargs):
        actions = {}
        
        for agent_id, obs in observations.items():
            if agent_id in self.active_agents:
                actions[agent_id] = self.policies[agent_id].compute_single_action(obs)[0]
                # actions[agent_id] = self.actor.compute_single_action(obs, policy_id=agent_id)
            else:
                actions[agent_id] = self.do_nothing_actors[agent_id].do_nothing_action()

        return actions
class HeirarchicalDCRL(gym.Env):

    def __init__(self, config):

        # Init all datacenter environments
        DC1 = DCRL(config['config1'])
        DC2 = DCRL(config['config2'])
        DC3 = DCRL(config['config3'])

        self.datacenters = {
            'DC1': DC1,
            'DC2': DC2,
            'DC3': DC3,
        }

        # Load trained lower level agent
        self.lower_level_actor = LowLevelActorHARL(
            config['low_level_actor_config'],
            config['active_agents']
            )
        
        # self.lower_level_actor = LowLevelActorRLLIB(
        #     config['low_level_actor_config'], 
        #     config['active_agents']
        #     )
        
        
        # if 'MADDPG' in config['low_level_actor']['rllib']['checkpoint_path']:
        #     for env in self.datacenters.values():
        #         env.actions_are_logits = True

        self.low_level_observations = {}

        # Define observation and action space
        self.observation_space = Dict({dc: Box(0, 10000, [4]) for dc in self.datacenters})
        self.action_space = MultiDiscrete([3, 3])

    def reset(self, seed=None, options=None):

        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        tf1.random.set_random_seed(0)

        self.low_level_observations = {}
        self.heir_obs = {}
        infos = {}

        # Reset environments and store initial observations and infos
        for env_id, env in self.datacenters.items():
            obs, info = env.reset()
            self.low_level_observations[env_id] = obs
            self.heir_obs[env_id] = np.array(env.get_hierarchical_variables())
            infos[env_id] = info
        
        # Initialize metrics
        self.metrics = {
            env_id: {
                'bat_CO2_footprint': [],
                'bat_total_energy_with_battery_KWh': [],
                'ls_tasks_dropped': [],
                'dc_water_usage': [],
            }
            for env_id in self.datacenters
        }   

        self.all_done = {env_id: False for env_id in self.datacenters}
        
        return self.heir_obs, infos
        
    def calc_reward(self):
        reward = 0
        for dc in self.rewards:
            reward += self.infos[dc]['agent_bat']['bat_CO2_footprint']
        return reward / 1e6

    def compute_adjusted_workloads(self, actions):
        
        datacenters = list(self.datacenters.keys())
        sender, receiver = [datacenters[i] for i in actions]

        s_capacity, s_workload, *_ = self.heir_obs[sender]
        r_capacity, r_workload, *_ = self.heir_obs[receiver]

        # Convert percentage workload to mwh
        s_mwh = s_capacity * s_workload
        r_mwh = r_capacity * r_workload

        # Calculate the amount to move
        mwh_to_move = min(s_mwh, r_capacity - r_mwh)
        s_mwh -= mwh_to_move
        r_mwh += mwh_to_move

        # Convert back to percentage workload
        s_workload  = s_mwh / s_capacity
        r_workload = r_mwh / r_capacity

        return {sender: s_workload, receiver: r_workload}
    
    def step(self, actions):
        
        # Shift workloads between datacenters according to 
        # the actions provided by the agent. This will return a dict with 
        # recommend workloads for all DCs
        if isinstance(actions, np.ndarray):
            actions = self.compute_adjusted_workloads(actions)

        # Set workload for all DCs accordingly
        for env_id, adj_workload in actions.items():
            if isinstance(adj_workload, np.ndarray):
                adj_workload = adj_workload[0]

            self.datacenters[env_id].set_hierarchical_workload(round(adj_workload, 6))

        # Compute actions for each agent in each environment
        low_level_actions = {}
        
        for env_id, env_obs in self.low_level_observations.items():
            # Skip if environment is done
            if self.all_done[env_id]:
                continue

            low_level_actions[env_id] = self.lower_level_actor.compute_actions(env_obs)

        # Step through each environment with computed low_level_actions
        self.low_level_infos = {}
        for env_id in self.datacenters:
            if self.all_done[env_id]:
                continue

            new_obs, rewards, terminated, truncated, info = self.datacenters[env_id].step(low_level_actions[env_id])
            self.low_level_observations[env_id] = new_obs
            self.all_done[env_id] = terminated['__all__'] or truncated['__all__']

            self.low_level_infos[env_id] = info

            # Update metrics for each environment
            env_metrics = self.metrics[env_id]
            env_metrics['bat_CO2_footprint'].append(info['agent_bat']['bat_CO2_footprint'])
            env_metrics['bat_total_energy_with_battery_KWh'].append(info['agent_bat']['bat_total_energy_with_battery_KWh'])
            env_metrics['ls_tasks_dropped'].append(info['agent_ls']['ls_tasks_dropped'])
            env_metrics['dc_water_usage'].append(info['agent_dc']['dc_water_usage'])

        done = any(self.all_done.values())

        # Get observations for the next step
        if not done:
            self.heir_obs = {}
            for env_id, env in self.environments.items():
                self.heir_obs[env_id] = np.array(env.get_hierarchical_variables())

        return self.heir_obs, self.calc_reward(), False, done, {}

if __name__ == '__main__':
    env = HeirarchicalDCRL(DEFAULT_CONFIG)
    done = False
    obs, _ = env.reset()
    greedy_optimizer = WorkloadOptimizer(env.environments.keys())
    
    max_iterations = 4*24*30
    # Antonio: Keep in mind that each environment is set to have days_per_episode=30. You can modify this parameter to simulate the whole year
    with tqdm(total=max_iterations) as pbar:
        while not done:
    
            # Random actions
            # actions = {key: val[0] for key, val in env.action_space.sample().items()}

            # Do nothing 
            actions = {dc: state[1] for dc, state in obs.items()}

            # Rule-based
            # actions, _ = greedy_optimizer.compute_adjusted_workload(obs)


            obs, reward, terminated, truncated, info = env.step(actions)
    obs, _ = env.reset(seed=0)
    total_reward = 0
    
    greedy_optimizer = WorkloadOptimizer(env.datacenters.keys())
            # Update the progress bar
            pbar.update(1)

    # After simulation, calculate average metrics for each environment
    average_metrics = {
        env_id: {metric: sum(values) / len(values) for metric, values in env_metrics.items()}
        for env_id, env_metrics in env.metrics.items()
    }
            # actions = env.action_space.sample()

            # Do nothing
            # actions = {dc: state[1] for dc, state in obs.items()}

            # One-step greedy
            # ci = [obs[dc][-1] for dc in env.datacenters]
            # actions = np.array([np.argmax(ci), np.argmin(ci)])
            
            # Greedy
            actions, _ = greedy_optimizer.compute_adjusted_workload(obs)
        print()  # Blank line for readability

    total_metrics = {}
            total_reward += reward
    for metric in env_metrics:
        total_metrics[metric] = 0.0
        for env_id in average_metrics:
            total_metrics[metric] += average_metrics[env_id][metric]

        print(f'{metric}: {total_metrics[metric]:,.2f}')        print(f"Average Metrics for {env.datacenters[env_id].location}:")
            print(f"\t{metric}: {value:,.2f}")
    # Sum metrics across datacenters
    print("Summed metrics across all DC:")
        print(f'\t{metric}: {total_metrics[metric]:,.2f}')

    print("Total reward = ", total_reward)