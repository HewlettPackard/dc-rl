import gym
import numpy as np
from utils.utils_cf import SequentialLoadChangeShave, normalize, sc_obs
import pandas as pd

class NoLoadShiftEnv(gym.Env):
    def __init__(self,future=False,
                 future_power=True,
                 max_capacity=100.0, 
                 future_steps=4,
                 cpu_data=[],
                 flexible_workload_ratio=0.1,
                 carbon_data=None,
                 n_vars_energy=0,
                 n_vars_battery=1,
                 training_days=366, 
                 location='NYIS',
                 init_day=0):
        
        self.cpu_data = cpu_data
        self.carbon_data = carbon_data
        self.init_day = init_day
        self.time_step = 0
        self.action_space = gym.spaces.Discrete(2)

        if future:
            self.observation_space = gym.spaces.Box(low=-5e1, high=5e1, shape=(4+future_steps,),dtype=np.float32)
        elif future_power:
            self.observation_space = gym.spaces.Box(low=-5e1, high=5e1, shape=(8+(2*future_steps+n_vars_energy+n_vars_battery),),dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=-5e1, high=5e1, shape=(8,),dtype=np.float32)
        
        self.max_step = training_days*24*4 
        self.max_capacity = max_capacity
        self.min_capacity = min(cpu_data)
        self.day = 0
        self.n_vars_energy = n_vars_energy
        self.n_vars_battery = n_vars_battery
        self.temp_state = None

    def reset(self):
        self.data_index = 0 + self.init_day
        done = False
        self.init_load_left = 0
        self.current_hour = 0.0
        self.day = 0

        info = {'load': cpu_data,
                'carbon_data': carbon_data,
                'action': -1,
                'info_load_left': 0,}
        
        state = ''
        return state, info

    def update_state(self):
        self.temp_state[-(self.n_vars_energy + self.n_vars_battery):] = np.hstack((self.energy_vars, self.battery_vars))
        return self.temp_state
    
    def set_energy_vars(self, energy_vars):
        self.energy_vars = energy_vars
    
    def set_battery_vars(self, battery_vars):
        self.battery_vars = battery_vars
        
    def step(self, action):
        if self.current_hour >= 24:
            self.current_hour=0
            self.day += 1
        alarm = 0
        if self.current_hour >= 23:
            alarm = 1
        done = False
        self.data_index+=1
        if self.data_index >= self.max_step:
            done = True
        cpu_data = self.cpu_data[self.data_index]
        carbon_data = self.carbon_data[self.data_index]
        total_carbon_number = cpu_data*carbon_data

        info = {"total_carbon_number":total_carbon_number,
                'load': p_cpu_data,
                'carbon_data': carbon_data,
                'normalized_ci': normalize(carbon_data, self.ci_min, self.ci_max),
                'action': action,
                'info_load_left': 0,}

        stata = []
        self.current_hour+=0.25
        truncated = False
        return state, 0, done, truncated, info
