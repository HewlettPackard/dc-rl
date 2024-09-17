import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from collections import deque


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
        
        self.shiftable_tasks_percentage = self.flexible_workload_ratio
        self.non_shiftable_tasks_percentage = 1 - self.flexible_workload_ratio
        
        # Define a discrete action space with 3 actions:
        # 0: Defer all shiftable tasks, 1: Do nothing, 2: Process all deferred tasks
        self.action_space = spaces.Discrete(3)
        
        
        # State: [Sin(h), Cos(h), Sin(day_of_year), Cos(day_of_year), self.ls_state, ci_i_future (n_vars_ci), var_to_LS_energy (n_vars_energy), batSoC (n_vars_battery)], 
        # self.ls_state = [current_workload, queue status]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(20,), dtype=np.float32)


        self.global_total_steps = 0
        self.test_mode = test_mode
        self.time_steps_day = 96
        # self.load_to_assign = 3 * flexible_workload_ratio
        # self.day_workload = 0
        self.workload = 0
        
        # Initialize the queue to manage individual delayed tasks
        # self.tasks_queue = []  # A list to hold individual tasks

        self.queue_max_len = queue_max_len
        self.tasks_queue = deque(maxlen=self.queue_max_len)

    def reset(self, *, seed=None, options=None):
        """
        Reset `CarbonLoadEnv` to initial state.

        Returns:
            observations (List[float]): Current state of the environmment
            info (dict): A dictionary that containing additional information about the environment state
        """
        self.global_total_steps = 0
        self.tasks_queue.clear()
        
        # Initialize the current day and hour
        self.current_day = 0
        self.current_hour = 0
        
        # Queue status - length of the task queue
        current_workload = self.workload
        queue_length = 0
        
        # Initial task ages will be zero at reset
        oldest_task_age = 0.0
        average_task_age = 0.0
        
        state = np.asarray(np.hstack(([current_workload,
                                       queue_length/self.queue_max_len,
                                       oldest_task_age,
                                       average_task_age])), dtype=np.float32)
        
    
        info = {"load": self.workload,
                "action": 0,
                "info_load_left": 0,
                'ls_queue_max_len': self.queue_max_len,
                "ls_tasks_dropped": 0,
                "ls_tasks_in_queue": 0, 
                "ls_norm_tasks_in_queue": 0,
                'ls_tasks_processed': 0,
                'ls_enforced': False,
                'ls_oldest_task_age': oldest_task_age,
                'ls_average_task_age': average_task_age,
                'ls_overdue_penalty': 0}
        
        return state, info


    def has_computational_room(self, workload_rest_day=0):
        """
        Check if there is enough computational room to process the tasks in the queue considering future workloads.

        Returns:
            bool: True if there is enough room to process the tasks in the queue, False otherwise.
        """
        
        # Initialize the available capacity at the current timestep
        total_available_capacity = 1.0 - self.workload
        
        # Add the available capacity for the future timesteps in the day
        total_available_capacity += np.sum(1.0 - workload_rest_day)
        
        # Estimate total tasks that need to be processed in the queue
        tasks_in_queue = len(self.tasks_queue)

        # Check if there is enough room to process the queued tasks
        if tasks_in_queue <= total_available_capacity * 100:  # Convert capacity to task space
            return True
        else:
            return False

    def step(self, action, workload_rest_day=0):
        """
        Makes an environment step in `CarbonLoadEnv`.

        Args:
            action (float): Continuous action between -1 and 1.
                            -1: Defer all shiftable tasks.
                             1: Process tasks from the DTQ to maximize utilization.
                             Values between -1 and 0 defer a fraction of tasks, and values between 0 and 1 process a fraction of tasks in the DTQ.

        Returns:
            state (List[float]): Current state of the environment.
            reward (float): Reward value.
            done (bool): A boolean signaling if the episode has ended.
            info (dict): A dictionary containing additional information about the environment state.
        """
        
        # self.current_hour += 0.25
        enforced = 0

        non_shiftable_tasks = int(math.ceil(self.workload * self.non_shiftable_tasks_percentage * 100))
        shiftable_tasks     = int(math.floor(self.workload * self.shiftable_tasks_percentage * 100))
        tasks_dropped = 0  # Track the number of dropped tasks
        actual_tasks_to_process = 0  # Track the number of processed tasks
        # action_value = -1.0 
        # print(f'Action value: {action_value}')
        # Step 1: Identify tasks older than 24 hours
        # overdue_tasks = []
        # for task in self.tasks_queue:
        #     # Calculate task age based on both day and hour
        #     task_age = (self.current_day - task['day']) * 24 + (self.current_hour - task['hour'])
        #     if task_age > 24:
        #         overdue_tasks.append(task)
        # Retrieve overdue tasks
        overdue_tasks = [task for task in self.tasks_queue if (self.current_day - task['day']) * 24 + (self.current_hour - task['hour']) > 24]
        overdue_penalty = len(overdue_tasks)
        
        # Calculate initial available capacity (before accounting for overdue tasks)
        available_capacity = 90 - (non_shiftable_tasks + shiftable_tasks) # Limit the available capacity to 90% to avoid overloading the system

        # Step 2: If there is available capacity, add overdue tasks to non-shiftable tasks
        overdue_tasks_to_process = 0
        overdue_task_count = 0
        if available_capacity > 0 and len(overdue_tasks) > 0:
            overdue_task_count = len(overdue_tasks)
            tasks_that_can_be_processed = min(overdue_task_count, available_capacity)
            # non_shiftable_tasks += tasks_that_can_be_processed
            overdue_task_count = tasks_that_can_be_processed
            # print(f'Overdue tasks: {overdue_task_count}, tasks that can be processed: {tasks_that_can_be_processed}, available capacity: {available_capacity}, non_shiftable_tasks: {non_shiftable_tasks}')
            # Remove those overdue tasks that can be processed
            overdue_tasks_to_process = tasks_that_can_be_processed
            for task in overdue_tasks[:tasks_that_can_be_processed]:
                self.tasks_queue.remove(task)
        
        # Update available capacity after adding overdue tasks
        available_capacity = 90 - (non_shiftable_tasks + shiftable_tasks + overdue_tasks_to_process) # Limit the available capacity to 90% to avoid overloading the system
        action = action[0]
        if action == 0:  # Defer tasks
            tasks_to_defer = shiftable_tasks

            # Attempt to queue deferred tasks
            tasks_to_add = min(tasks_to_defer, self.queue_max_len - len(self.tasks_queue))
            tasks_dropped += tasks_to_defer - tasks_to_add

            # Vectorized: Add multiple tasks at once
            self.tasks_queue.extend([{'day': self.current_day, 'hour': self.current_hour, 'utilization': 1}] * tasks_to_add)

            # Update utilization
            self.current_utilization = (overdue_tasks_to_process + (shiftable_tasks - tasks_to_add)) / 100
        
        elif action == 2:  # Process tasks from the queue
            # available_capacity = 100 - (non_shiftable_tasks + shiftable_tasks)  # Remaining capacity
            if available_capacity >= 1:  # At least 1 task can be processed
                tasks_to_process = available_capacity
                actual_tasks_to_process = min(len(self.tasks_queue), tasks_to_process)  # Limit by queue length

                # if we can process all of the tasks in the queue, make it faster with clear() instead of with a for loop
                if actual_tasks_to_process == len(self.tasks_queue):
                    self.tasks_queue.clear()
                else:
                    # Vectorized: Pop multiple tasks at once
                    for _ in range(actual_tasks_to_process):
                        self.tasks_queue.popleft()

                # Update utilization to include processed DTQ tasks
                self.current_utilization = (shiftable_tasks + actual_tasks_to_process + overdue_tasks_to_process) / 100
                
            else:
                self.current_utilization = (shiftable_tasks + overdue_tasks_to_process) / 100

        else:  # action == 1, Do nothing
            # self.current_utilization = (non_shiftable_tasks + shiftable_tasks) / 100
            self.current_utilization = (shiftable_tasks + overdue_tasks_to_process) / 100

        self.global_total_steps += 1
        
        original_workload = self.workload
        tasks_in_queue = len(self.tasks_queue)

            
        reward = 0 
        
        # Calculate the age of each task in the queue
        if len(self.tasks_queue) > 0:
            task_ages = [(self.current_day - task['day']) * 24 + (self.current_hour - task['hour']) for task in self.tasks_queue]
            oldest_task_age = max(task_ages)
            average_task_age = sum(task_ages) / len(task_ages)
        else:
            oldest_task_age = 0.0
            average_task_age = 0.0
            
        info = {"ls_original_workload": original_workload,
                "ls_shifted_workload": self.current_utilization, 
                "ls_action": action, 
                "ls_norm_load_left": 0,
                "ls_unasigned_day_load_left": 0,
                "ls_penalty_flag": 0,
                'ls_queue_max_len': self.queue_max_len,
                'ls_tasks_in_queue': tasks_in_queue, 
                'ls_norm_tasks_in_queue': tasks_in_queue/self.queue_max_len,
                'ls_tasks_dropped': tasks_dropped,
                'ls_current_hour': self.current_hour,
                'ls_tasks_processed': actual_tasks_to_process,
                'ls_enforced': enforced,
                'ls_oldest_task_age': oldest_task_age/24,
                'ls_average_task_age': average_task_age/24,
                'ls_overdue_penalty': overdue_penalty,
                'ls_computed_tasks': int(self.current_utilization*100)}



        #Done and truncated are managed by the main class, implement individual function if needed
        truncated = False
        done = False
        
        if self.current_utilization > 1 or self.current_utilization < 0:
            print('WARNING, the utilization is out of bounds')
        state = np.asarray(np.hstack(([self.current_utilization, tasks_in_queue/self.queue_max_len, oldest_task_age/24, average_task_age/24])), dtype=np.float32)
        
        return state, reward, done, truncated, info 
        
    def update_workload(self, workload):
        """
        Makes an environment step in`BatteryEnvFwd.

        Args:
            workload (float): Workload assigned at the current time step
        """
        if workload < 0 or workload > 1:
            print('WARNING, the workload is out of bounds')
            # Raise an error if the workload is out of bounds
            raise ValueError("The workload should be between 0 and 1")
        self.workload = workload
    
    def update_current_date(self, current_day, current_hour):
        """
        Update the current hour in the environment.

        Args:
            current_hour (float): Current hour in the environment.
        """
        self.current_day = current_day
        self.current_hour = current_hour