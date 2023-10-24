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
            test_mode (bool, optional): Used for evaluation of the model. Defaults to False.
        """
        assert flexible_workload_ratio < 0.9, "flexible_workload_ratio should be lower than 0.9"
        self.action_space = gym.spaces.Discrete(2)
        if future:
            self.observation_space = gym.spaces.Box(
                low=-10,
                high=21,
                shape=(7 + future_steps + n_vars_energy + n_vars_battery,),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-10,
                high=20,
                shape=(7 + n_vars_energy + n_vars_battery + 1,),
                dtype=np.float32,
            )


        self.global_total_steps = 0
        self.test_mode = test_mode
        self.time_steps_day = 24*4
        self.load_to_assign = 3 * flexible_workload_ratio
        self.day_workload = 0
        self.workload = 0

    def reset(self):
        """
        Reset `CarbonLoadEnv` to initial state.

        Returns:
            observations (List[float]): Current state of the environmment
            info (dict): A dictionary that containing additional information about the environment state
        """
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
        """
        Makes an environment step in`CarbonLoadEnv.

        Args:
            action (int): Action to take.

        Returns:
            observations (List[float]): Current state of the environmment
            reward (float): reward value.
            done (bool): A boolean value signaling the if the episode has ended.
            info (dict): A dictionary that containing additional information about the environment state
        """
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
        original_workload = self.workload
        if (action == 0 and self.storage_load > 0) or self.storage_load + self.load_to_assign > residue:
            delta = self.load_to_assign
            delta = min(self.storage_load, delta)
            total_wkl = delta+self.workload
            if total_wkl > 1:
                delta -= (total_wkl-1)
            self.storage_load -= delta
            self.workload += delta
    
        norm_load_left = round(self.storage_load / (self.day_storage_load + 1e-9), 3)
        info_load_left = 0
        out_of_time = False
        if self.current_hour >= 23.75:
            if self.storage_load > 0:
                out_of_time = True
                info_load_left = self.storage_load
        reward = 0 
        
        info = {"ls_original_workload": original_workload,
                "ls_shifted_workload": self.workload, 
                "ls_action": action, 
                "ls_norm_load_left": norm_load_left,
                "ls_unasigned_day_load_left": info_load_left,
                "ls_penalty_flag": out_of_time}
        
        state = np.asarray(np.hstack(([alarm, norm_load_left])), dtype=np.float32)

        #Done and truncated are managed by the main class, implement individual function if needed
        truncated = False
        done = False
        
        return state, reward, done, truncated, info 
        
    def update_workload(self, day_workload, workload):
        """
        Makes an environment step in`BatteryEnvFwd.

        Args:
            day_workload (float): Total amout of daily flexible workload.
            workload (float): Workload assigned at the current time step
        """
        self.day_workload = day_workload
        self.workload = workload