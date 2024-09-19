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
        initialize_queue_at_reset=False
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
        
        # Define a single continuous action space: [-1, 1]
        # -1: Defer all shiftable tasks
        # 0: Do nothing
        # 1: Process all DTQ tasks
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        
        # State: [Sin(h), Cos(h), Sin(day_of_year), Cos(day_of_year), self.ls_state, ci_i_future (n_vars_ci), var_to_LS_energy (n_vars_energy), batSoC (n_vars_battery)], 
        # self.ls_state = [current_workload, queue status]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(26,), dtype=np.float32)


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
        self.initialize_queue_at_reset = initialize_queue_at_reset

    # Calculate task age histogram
    def get_task_age_histogram(self, tasks_queue, current_day, current_hour):
        age_bins = [0, 6, 12, 18, 24, np.inf]  # Age bins in hours
        task_ages = [
            (current_day - task['day']) * 24 + (current_hour - task['hour'])
            for task in self.tasks_queue
        ]
        histogram, _ = np.histogram(task_ages, bins=age_bins)
        normalized_histogram = histogram / max(len(self.tasks_queue), 1)  # Avoid division by zero
        
        normalized_histogram[-1] = 1 if normalized_histogram[-1] > 0 else 0
        return normalized_histogram  # Returns an array of proportions
    
    def reset(self, *, seed=None, options=None):
        """
        Reset `CarbonLoadEnv` to initial state.

        Returns:
            observations (List[float]): Current state of the environmment
            info (dict): A dictionary that containing additional information about the environment state
        """
        self.global_total_steps = 0
        self.tasks_queue.clear()
        
        if self.initialize_queue_at_reset:
            # Initialize the task queue with tasks of varying ages
            initial_queue_length = np.random.randint(1, self.queue_max_len // 4)

            # Generate random task ages between 0 and 24 hours
            max_task_age = 24  # Maximum age in hours
            task_ages = np.random.random_integers(0, max_task_age*4, initial_queue_length)/4
            
            # Generate task ages using an exponential distribution
            max_task_age = 24  # Maximum age in hours
            # Set the rate parameter (lambda) for the exponential distribution
            lambda_param = 1.0 / 4.0  # Mean age of 6 hours (adjust as needed)
            task_ages = np.round(np.random.exponential(scale=1.0 / lambda_param, size=initial_queue_length) * 4)/4

            # Cap the task ages at max_task_age
            task_ages = np.clip(task_ages, 0, max_task_age)

            
            # Sort the task ages in descending order
            task_ages = np.sort(task_ages)[::-1]

            for age in task_ages:
                # Compute the day and hour when the task was added
                task_day = self.current_day
                task_hour = self.current_hour - age

                # Adjust day and hour if task_hour is negative
                while task_hour < 0:
                    task_hour += 24
                    task_day -= 1  # Task was added on a previous day

                # Ensure task_day is non-negative
                if task_day < 0:
                    task_day = 0
                    task_hour = 0  # Reset to the earliest possible time

                # Create the task with its timestamp
                task = {'day': task_day, 'hour': task_hour, 'utilization': 1}
                self.tasks_queue.append(task)
        else:
            # Start with an empty task queue
            pass

        # Calculate queue_length, oldest_task_age, average_task_age
        tasks_in_queue = len(self.tasks_queue)
        if tasks_in_queue > 0:
            task_ages = [
                (self.current_day - task['day']) * 24 + (self.current_hour - task['hour'])
                for task in self.tasks_queue
            ]
            oldest_task_age = max(task_ages)
            average_task_age = sum(task_ages) / len(task_ages)
        else:
            oldest_task_age = 0.0
            average_task_age = 0.0

        task_age_histogram = self.get_task_age_histogram(self.tasks_queue, self.current_day, self.current_hour)
    
        # Compute state
        current_workload = self.workload  # Ensure self.workload is set appropriately
        state = np.asarray(np.hstack(([current_workload,
                                tasks_in_queue/self.queue_max_len,
                                oldest_task_age/24,
                                average_task_age/24,
                                task_age_histogram])), dtype=np.float32)
        
    
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
                'ls_overdue_penalty': 0,
                'ls_task_age_histogram': task_age_histogram,}
        
        
        return state, info


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
        
        enforced = False

        non_shiftable_tasks = int(math.ceil(self.workload * self.non_shiftable_tasks_percentage * 100))
        shiftable_tasks     = int(math.floor(self.workload * self.shiftable_tasks_percentage * 100))
        tasks_dropped = 0  # Track the number of dropped tasks
        actual_tasks_processed = 0  # Track the number of processed tasks
        action_value = np.clip(action[0], -1.0, 1.0)  # Clip the action to [-1, 1]  # Single continuous action
        
        # Handle overdue tasks
        overdue_tasks = [task for task in self.tasks_queue if (self.current_day - task['day']) * 24 + (self.current_hour - task['hour']) > 24]
        overdue_penalty = len(overdue_tasks)

        # Calculate initial available capacity
        available_capacity = 90 - (non_shiftable_tasks + shiftable_tasks)  # Limit to 90% capacity

        # Process overdue tasks if there's capacity
        overdue_tasks_to_process = 0
        if available_capacity > 0 and len(overdue_tasks) > 0:
            overdue_task_count = len(overdue_tasks)
            tasks_that_can_be_processed = min(overdue_task_count, available_capacity)
            overdue_tasks_to_process = tasks_that_can_be_processed

            for task in overdue_tasks[:tasks_that_can_be_processed]:
                self.tasks_queue.remove(task)

        
        # Update available capacity after processing overdue tasks
        available_capacity = 90 - (non_shiftable_tasks + shiftable_tasks + overdue_tasks_to_process)

        
        if action_value < 0:  # Defer a fraction of shiftable tasks
            fraction_to_defer = abs(action_value)  # Convert to positive fraction between 0 and 1
            tasks_to_defer = int(shiftable_tasks * fraction_to_defer)

            # Attempt to queue deferred tasks
            available_queue_space = self.queue_max_len - len(self.tasks_queue)
            tasks_to_add = min(tasks_to_defer, available_queue_space)
            tasks_dropped += tasks_to_defer - tasks_to_add

            # Add deferred tasks to the queue
            self.tasks_queue.extend(
                [{'day': self.current_day, 'hour': self.current_hour, 'utilization': 1}] * tasks_to_add
            )

            # Update utilization
            self.current_utilization = ((shiftable_tasks - tasks_to_add) + overdue_tasks_to_process) / 100

        
        elif action_value > 0:  # Process a fraction of tasks from the queue
            fraction_to_process = action_value  # Fraction between 0 and 1
            if available_capacity > 0:
                max_tasks_can_process = min(len(self.tasks_queue), available_capacity)
                tasks_to_process = int(max_tasks_can_process * fraction_to_process)

                # if we can process all of the tasks in the queue, make it faster with clear() instead of with a for loop
                if tasks_to_process == len(self.tasks_queue):
                    self.tasks_queue.clear()
                else:
                    # Vectorized: Pop multiple tasks at once
                    for _ in range(tasks_to_process):
                        self.tasks_queue.popleft()

                # # Process tasks from the queue
                # for _ in range(tasks_to_process):
                #     self.tasks_queue.popleft()
                actual_tasks_processed = tasks_to_process

                # Update utilization
                self.current_utilization = (shiftable_tasks + overdue_tasks_to_process + actual_tasks_processed) / 100
            else:
                self.current_utilization = (shiftable_tasks + overdue_tasks_to_process) / 100

        else:  # action_value == 0, Do nothing
            self.current_utilization = (shiftable_tasks + overdue_tasks_to_process) / 100

        # if not self.initialize_queue_at_reset: # That means that we are on eval mode
            # self.current_utilization += non_shiftable_tasks / 100
            
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
        
        task_age_histogram = self.get_task_age_histogram(self.tasks_queue, self.current_day, self.current_hour)

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
                'ls_tasks_processed': actual_tasks_processed,
                'ls_enforced': enforced,
                'ls_oldest_task_age': oldest_task_age/24,
                'ls_average_task_age': average_task_age/24,
                'ls_overdue_penalty': overdue_penalty,
                'ls_computed_tasks': int(self.current_utilization*100),
                'ls_task_age_histogram': task_age_histogram,}


        #Done and truncated are managed by the main class, implement individual function if needed
        truncated = False
        done = False
        
        if self.current_utilization > 1 or self.current_utilization < 0:
            print('WARNING, the utilization is out of bounds')
        state = np.asarray(np.hstack(([self.current_utilization,
                                       tasks_in_queue/self.queue_max_len,
                                       oldest_task_age/24,
                                       average_task_age/24,
                                       task_age_histogram])), dtype=np.float32)
        
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