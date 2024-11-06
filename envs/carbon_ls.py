import gymnasium as gym
from gymnasium import spaces

import numpy as np
from utils import reward_creator
import math

class CarbonLoadEnv(gym.Env):
    def __init__(
        self,
        env_config = {},
        future=True,
        n_vars_ci=4,
        flexible_workload_ratio=0.2,
        n_vars_energy=0,
        n_vars_battery=1,
        test_mode=False,
        queue_max_len=500,
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
        self.flexible_workload_ratio = flexible_workload_ratio
        
        # Actions: 0 - Decrease, 1 - Do Nothing, 2 - Increase utilization
        self.action_space = spaces.Discrete(3)
        
        # State: [Sin(h), Cos(h), Sin(day_of_year), Cos(day_of_year), self.ls_state, ci_i_future (n_vars_ci), var_to_LS_energy (n_vars_energy), batSoC (n_vars_battery)], 
        # self.ls_state = [current_workload, queue status]
        if future:
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(6 + n_vars_ci + n_vars_energy + n_vars_battery,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(6 + n_vars_energy + n_vars_battery,),
                dtype=np.float32,
            )


        self.global_total_steps = 0
        self.test_mode = test_mode
        self.time_steps_day = 96
        # self.load_to_assign = 3 * flexible_workload_ratio
        # self.day_workload = 0
        self.workload = 0
        
        # Initialize the queue to manage individual delayed tasks
        self.tasks_queue = []  # A list to hold individual tasks
        self.queue_max_len = queue_max_len

    def reset(self, *, seed=None, options=None):
        """
        Reset `CarbonLoadEnv` to initial state.

        Returns:
            observations (List[float]): Current state of the environmment
            info (dict): A dictionary that containing additional information about the environment state
        """
        self.global_total_steps = 0
        
        # Clear the task queue
        self.tasks_queue = []
        
        done = False
        self.current_hour = 0.0
        self.day = 0
        
        # Queue status - length of the task queue
        current_workload = self.workload
        queue_length = 0
        
        state = np.asarray(np.hstack(([current_workload, queue_length/self.queue_max_len])), dtype=np.float32)
        
        info = {"ls_original_workload": self.workload,
                "ls_shifted_workload": self.workload,
                "action": -1,
                "info_load_left": 0,
                "ls_tasks_dropped": 0,
                "ls_tasks_in_queue": 0}
        
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

        shiftable_tasks_percentage = self.flexible_workload_ratio
        non_shiftable_tasks_percentage = 1 - shiftable_tasks_percentage

        assert self.action_space.contains(action), f"{action} is an invalid action"
        
        non_shiftable_tasks = int(math.ceil(self.workload * non_shiftable_tasks_percentage * 100))
        shiftable_tasks     = int(math.floor(self.workload * shiftable_tasks_percentage * 100))
        tasks_dropped = 0  # Track the number of dropped tasks
        
        if action == 0:  # Use only the non_shiftable_tasks.
            # Attempt to queue shiftable tasks, tracking any that are dropped
            timestamp = self.current_hour  # Current timestamp
            for _ in range(shiftable_tasks):
                if len(self.tasks_queue) < self.queue_max_len:  # Check if adding the task would exceed the queue limit
                    self.tasks_queue.append({'timestamp': timestamp, 'utilization': 1})
                else:
                    tasks_dropped += 1  # Increment dropped tasks count
            self.current_utilization = non_shiftable_tasks / 100
            
        # if action == 1, do nothing.
        elif action == 1:
            self.current_utilization = (non_shiftable_tasks + shiftable_tasks) / 100
        
        elif action == 2:  # Attempt to process as many tasks from the queue as possible or the max utilization (100%)
            # Determine the number of tasks that can be processed, considering total utilization limits
            tasks_to_process = min(len(self.tasks_queue), 100 - (non_shiftable_tasks + shiftable_tasks))
            for _ in range(tasks_to_process):
                if self.tasks_queue:
                    self.tasks_queue.pop(0)  # Remove a task from the queue for processing
            self.current_utilization = (non_shiftable_tasks + shiftable_tasks + tasks_to_process) / 100
            
        done = False
        self.global_total_steps += 1
        
        original_workload = self.workload

        if self.current_hour % (24*4) == 0:   # Penalty for queued tasks at the end of the day
            self.tasks_queue = []
            print(f'Checked that the tasks_queue is cleaned every 24 hours at {self.current_hour}')
        
        
        if self.current_hour >= 24:
            self.current_hour = 0
            
        reward = 0 
        
        tasks_in_queue = len(self.tasks_queue)
        
        current_workload = self.current_utilization
        
        info = {"ls_original_workload": original_workload,
                "ls_shifted_workload": current_workload, 
                "ls_action": action, 
                "ls_norm_load_left": 0,
                "ls_unasigned_day_load_left": 0,
                "ls_penalty_flag": 0,
                'ls_tasks_in_queue': tasks_in_queue, 
                'ls_tasks_dropped': tasks_dropped,
                'ls_current_hour': self.current_hour}


        #Done and truncated are managed by the main class, implement individual function if needed
        truncated = False
        done = False
        
        state = np.asarray(np.hstack(([current_workload, tasks_in_queue/self.queue_max_len])), dtype=np.float32)
        
        return state, reward, done, truncated, info 
        
    def update_workload(self, workload):
        """
        Makes an environment step in`BatteryEnvFwd.

        Args:
            workload (float): Workload assigned at the current time step
        """
        self.workload = workload