import gymnasium as gym
import numpy as np
from utils import reward_creator

class CarbonLoadEnv(gym.Env):
    def __init__(
        self,
        env_config = {},
        future=True,
        future_steps=4,
        flexible_workload_ratio=0.1,
        n_vars_energy=0,
        n_vars_battery=1,
        training_days=366,
        test_mode=False,
        ):
        """Creates load shifting envrionemnt

        Args:
            env_config (dict, optional): Customizable environment confing. Defaults to {}.
            future (bool, optional): To include CI forecast to the observation. Defaults to True.
            future_steps (int, optional): Number of time steps in the future in the forecast. Defaults to 4.
            flexible_workload_ratio (float, optional): Percentage of flexible workload. Defaults to 0.1.
            n_vars_energy (int, optional): Additional number of energy variables. Defaults to 0.
            n_vars_battery (int, optional): Additional number of variables from the battery. Defaults to 1.
            training_days (int, optional): Maximun number of training days in a episode. Defaults to 366.
            test_mode (bool, optional): Used for evaluation of the model. Defaults to False.
        """
        assert flexible_workload_ratio < 0.9, "flexible_workload_ratio should be lower than 0.9"
        self.action_space = gym.spaces.Discrete(2)
        if future:
            self.observation_space = gym.spaces.Box(
                low=-5e1,
                high=5e1,
                shape=(7 + future_steps + n_vars_energy + n_vars_battery,),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-5e1,
                high=5e1,
                shape=(7 + n_vars_energy + n_vars_battery,),
                dtype=np.float32,
            )

        self.reward_method = reward_creator.get_reward_method(env_config['reward_method'] 
                                                            if 'reward_method' in env_config.keys() 
                                                            else 'default_ls_reward')
        self.max_step = training_days * 24 * 4
        self.global_total_steps = 0
        self.test_mode = test_mode
        self.time_steps_day = 96
        self.load_to_assign = 3 * flexible_workload_ratio
        self.day_workload = 0
        self.workload = 0

    def reset(self):
        self.global_total_steps = 0
        self.storage_load = self.day_workload
        self.day_storage_load = self.storage_load
        done = False
        self.current_hour = 0.0
        self.day = 0
        alarm = 0
        norm_load_left = 1
        state = np.asarray(np.hstack(([alarm, norm_load_left])), dtype=np.float32)
        info = {"load": self.workload, "action": -1, "info_load_left": 0}
        return state, info

    def step(self, action):
        self.current_hour += 0.25
        if self.day_workload > 0:
            self.storage_load = self.day_workload
            self.day_storage_load = self.day_workload
        if self.current_hour >= 24:
            self.current_hour = 0
        alarm = 0
        if self.current_hour >= 23:
            alarm = 1
        done = False
        self.global_total_steps += 1
        if self.test_mode:
            residue = (self.time_steps_day - (self.global_total_steps % self.time_steps_day)) * self.load_to_assign
        else:
            residue = 1e9
        delta = 0
        if (action == 0 and self.storage_load > 0) or self.storage_load + self.load_to_assign > residue:
            delta = self.load_to_assign
            delta = min(self.storage_load, delta)
            total_wkl = delta+self.workload
            if total_wkl > 1:
                delta -= (total_wkl-1)
            self.storage_load -= delta
            self.workload += delta
        if self.global_total_steps >= self.max_step:
            done = True
    
        norm_load_left = round(self.storage_load / (self.day_storage_load + 1e-9), 3)
        info_load_left = 0
        out_of_time = False
        if self.current_hour >= 23.75:
            if self.storage_load > 0:
                out_of_time = True
                info_load_left = self.storage_load
        reward = self.reward_method(params={'norm_load_left':norm_load_left,
                                            'out_of_time':out_of_time,
                                            'penalty': 1e3})
        info = {"load": self.workload, 
                "action": action, 
                "info_load_left": info_load_left,
                "out_of_time": out_of_time}
        state = np.asarray(np.hstack(([alarm, norm_load_left])), dtype=np.float32)
        truncated = False
        return state, reward, done, truncated, info      
        
    def update_workload(self, day_workload, workload):
        self.day_workload = day_workload
        self.workload = workload