import numpy as np
import gymnasium as gym
import envs.battery_model as batt
from utils import reward_creator

class BatteryEnvFwd(gym.Env):
    def __init__(self, env_config) -> None:
        super(BatteryEnvFwd, self).__init__()

        n_fwd_steps = env_config['n_fwd_steps']
        max_bat_cap = env_config['max_bat_cap']
        charging_rate = env_config['charging_rate']
        self.episodes_24hr = env_config['24hr_episodes']
        self.starting_point = int(env_config['start_point'])
        self.reward_method = reward_creator.get_reward_method(env_config['reward_method'] if 'reward_method' in env_config.keys() else 'default_bat_reward')
        self.end_point = self.starting_point + 30 * 96
        self.observation_space = gym.spaces.Box(low=np.float32(-2 * np.ones(1 + 1 + 4 + n_fwd_steps)),
                                                high=np.float32(2 * np.ones(1 + 1 + 4 + n_fwd_steps)))
        self.max_dc_pw = 7.24
        self.action_space = gym.spaces.Discrete(3)
        self._action_to_direction = {0: 'charge', 1: 'discharge', 2: 'idle'}
        other_states_max = np.array([self.max_dc_pw, max_bat_cap])
        other_states_min = np.array([0.1, 0])
        self.observation_max = other_states_max
        self.observation_min = other_states_min
        self.delta = self.observation_max - self.observation_min
        self.battery = batt.Battery2(capacity=max_bat_cap, current_load=0 * max_bat_cap)
        self.n_fwd_steps = n_fwd_steps
        self.charging_rate = charging_rate
        self.spot_CI = None
        self.ma_CI = None
        self.eta = 0.7
        self.dataset_end = True
        self.dcload = 0
        self.temp_state = None
        self.var_to_dc = 0
        self.max_bat_cap = max_bat_cap
        self.total_energy_with_battery = 0
        self.ci = 0
        self.ci_n = []
        self.dcload_max = env_config['dcload_max']
        self.dcload_min = env_config['dcload_min']
        
    def reset(self, *, seed=None, options=None):
        self.current_step = self.starting_point
        self.raw_obs = self._hist_data_collector()
        #self.ep_len_intervals = 0
        self.dcload = 0
        self.temp_state = self._process_obs(self.raw_obs)
        return self.temp_state, {
            'action': -1,
            'avg_dc_power_mw': self.raw_obs[0],
            'Grid_CI': 0,
            'total_energy_with_battery': 0,
            'CO2_footprint': 0,
            'avg_CI': 0,
            'battery SOC': self.battery.current_load,
            'total_energy_with_battery': 0
        }

    def step(self, action_id):
        action_instantaneous = self._action_to_direction[action_id]
        self.discharge_energy = self._simulate_battery_operation(self.battery, action_instantaneous,
                                                                  charging_rate=self.charging_rate)
        
        self.CO2_total = self.CO2_footprint(self.dcload, self.ci, action_instantaneous, self.discharge_energy)
        self.reward = self.reward_method(params={'total_energy_with_battery':self.total_energy_with_battery,
                                                 'norm_CI':self.ci_n,
                                                 'a_t':action_instantaneous,
                                                 'dcload_min':self.dcload_min,
                                                 'dcload_max':self.dcload_max,
                                                 'battery_current_load':self.battery.current_load,
                                                 'max_bat_cap':self.max_bat_cap,
                                                 'battery_current_load':self.battery.current_load,
                                                 'max_bat_cap':self.max_bat_cap})
        # self.current_step += 1
        self.raw_obs = self._hist_data_collector()
        self.temp_state = self._process_obs(self.raw_obs)
        #self.ep_len_intervals += 1
        # self.dataset_end = self.current_step == self.end_point
        # if self.episodes_24hr:
        #     done = self.dataset_end or (self.ep_len_intervals >= 24 * 4 * 7)
        # else:
        #     done = self.dataset_end

        # if self.dataset_end:
        #     self.current_step = self.starting_point

        self.info = {
            'action': action_id,
            'battery SOC': self.battery.current_load,
            'CO2_footprint': self.CO2_total,
            'avg_CI': self.ci,
            'avg_dc_power_mw': self.raw_obs[0],
            'step_reward': self.reward,
            'total_energy_with_battery': self.total_energy_with_battery
        }
        #Done and truncated are managed by the main class, implement individual function if needed
        truncated = False
        done = False 
        return self.temp_state, self.reward, done, truncated, self.info

    def update_ci(self, ci, ci_n):
        self.ci = ci
        self.ci_n = ci_n

    def get_dcload(self):
        raise NotImplementedError

    def _process_obs(self, state):
        scaled_value = (state - self.observation_min) / self.delta
        return scaled_value

    def _process_action(self, action_id):
        return self._action_to_direction[action_id]

    def update_state(self):
        self.temp_state[0] = self.dcload
        return self.temp_state

    def set_dcload(self, dc_load):
        self.dcload = dc_load

    def _hist_data_collector(self):
        raw_obs = np.array([self.dcload, self.battery.current_load])
        return raw_obs

    def _simulate_battery_operation(self, battery, battery_action, charging_rate=None):
        discharge_energy = 0
        if battery_action == 'charge':
            self.var_to_dc = battery.charge(battery.capacity, self.charging_rate_modifier(battery) * 15 / 60)
        elif battery_action == 'discharge':
            discharge_energy = battery.discharge(battery.capacity, self.discharging_rate_modifier(battery) * 15 / 60,
                                                 self.dcload * 0.25)
            self.var_to_dc = -discharge_energy
        else:
            discharge_energy = 0
            self.var_to_dc = 0
        self.var_to_dc = self.var_to_dc / self.max_bat_cap
        return discharge_energy

    def CO2_footprint(self, dc_load, ci, a_t, discharge_energy):
        if a_t == 'charge':
            self.total_energy_with_battery = dc_load * 1e3 * 0.25 + self.battery.charging_load * 1e3
            self.battery.charging_load = 0  # *Added*
            CO2_footprint = (self.total_energy_with_battery) * ci
        elif a_t == 'discharge':
            assert dc_load * 1e3 * 0.25 >= discharge_energy * 1e3, "Battery discharge rate should not be higher than the datacenter energy consumption rate"
            self.total_energy_with_battery = dc_load * 1e3 * 0.25 - discharge_energy * 1e3
            CO2_footprint = max(self.total_energy_with_battery, 0) * ci  # (KWh) * gCO2/KWh
        else:
            self.total_energy_with_battery = dc_load * 1e3 * 0.25
            CO2_footprint = self.total_energy_with_battery * ci

        return CO2_footprint

    def charging_rate_modifier(self, battery):
        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        curr_load = battery.current_load
        bat_max, bat_min = battery.capacity, 0
        sigmoid_max, sigmoid_min = 4, -4
        scaled_curr_load = (curr_load - bat_min) * (sigmoid_max - sigmoid_min) / (bat_max - bat_min) + sigmoid_min
        charging_rate = 0.3

        return charging_rate

    def discharging_rate_modifier(self, battery):
        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        curr_load = battery.current_load
        bat_max, bat_min = battery.capacity, 0
        sigmoid_max, sigmoid_min = 4, -4
        scaled_curr_load = (curr_load - bat_min) * (sigmoid_max - sigmoid_min) / (bat_max - bat_min) + sigmoid_min
        discharging_rate = 0.3

        return discharging_rate

    def cal_maxmin(self):
        self.dcload_max, self.dcload_min = self.max_dc_pw/4, 0.2/4  # /4 because we have 15 minutes time interval and we are using this to normalize MWH

    
    def update_dcload_ranges(self, current_dc_load):
        current_dc_load = current_dc_load/4
        if current_dc_load > self.dcload_max:
            self.dcload_max = current_dc_load
            print('max dcload updated to: ', self.dcload_max)
        elif current_dc_load < self.dcload_min:
            self.dcload_min = current_dc_load
            print('min dcload updated to: ', self.dcload_min)
        
        
