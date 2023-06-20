import numpy as np
import gymnasium as gym
import envs.battery_model as batt
from utils import reward_creator

class BatteryEnvFwd(gym.Env):
    def __init__(self, env_config) -> None:
        """Creates battery envrionemnt

        Args:
            env_config (dict): Customizable environment confing.
                n_fwd_steps(int): Number of forward forecast steps available
                max_bat_cap(float): Maximun battery capacity in MW
                charging_rate(float): Rate of charge of the battery
                reward_metod(function): Method used to calculate the reward
        """
        super(BatteryEnvFwd, self).__init__()
        n_fwd_steps = env_config['n_fwd_steps']
        max_bat_cap = env_config['max_bat_cap']
        charging_rate = env_config['charging_rate']
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
        """
        Reset `BatteryEnvFwd` to initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Environment options.

        Returns:
            temp_state (List[float]): Current state of the environmment
            info (dict): A dictionary that containing additional information about the environment state
        """
        self.raw_obs = self._hist_data_collector()
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
        """ Step function
        Args:
            action_id (int): the action id
        Returns:
            obs (list): Current state of the environmment
            reward (float): reward value.
            done (bool): A boolean value signaling the if the episode has ended.
            info (dict): A dictionary that containing additional information about the environment state
        """
        action_instantaneous = self._action_to_direction[action_id]
        self.discharge_energy = self._simulate_battery_operation(self.battery, action_instantaneous,
                                                                  charging_rate=self.charging_rate)
        
        self.CO2_total = self.CO2_footprint(self.dcload, self.ci, action_instantaneous, self.discharge_energy)

        self.raw_obs = self._hist_data_collector()
        self.temp_state = self._process_obs(self.raw_obs)
        
        self.reward = 0

        self.info = {
            'bat_action': action_id,
            'bat_SOC': self.battery.current_load,
            'bat_CO2_footprint': self.CO2_total,
            'bat_avg_CI': self.ci,
            'bat_total_energy_without_battery_KWh': self.dcload * 1e3 * 0.25,
            'bat_total_energy_with_battery_KWh': self.total_energy_with_battery,
            'bat_max_bat_cap': self.max_bat_cap,
            'bat_a_t': action_instantaneous,
            'bat_dcload_min': self.dcload_min,
            'bat_dcload_max': self.dcload_max,
        }
        #Done and truncated are managed by the main class, implement individual function if needed
        truncated = False
        done = False 
        return self.temp_state, self.reward, done, truncated, self.info

    def update_ci(self, ci, ci_n):
        """Sets internal CIs values.
        """
        self.ci = ci
        self.ci_n = ci_n

    def _process_obs(self, state):
        """Normalizes observations

        Args:
            state (List[float]): Current environment state.

        Returns:
            normalized_observations (List[float])
        """
        scaled_value = (state - self.observation_min) / self.delta
        return scaled_value

    def _process_action(self, action_id):
        """Maps agent actions to actoniable action for the model

        Args:
            action_id (int): Action to take.

        Returns:
            normalized_observations (string)
        """
        return self._action_to_direction[action_id]

    def update_state(self):
        """Updates obsevation with current DC energy consumption

        Returns:
            normalized_observations (string)
        """
        self.temp_state[0] = self.dcload
        return self.temp_state

    def set_dcload(self, dc_load):
        """Set the current DC energy consumption

        Args:
            dc_load float: DC energy consumption.

        """
        self.dcload = dc_load

    def _hist_data_collector(self):
        """Generates the observation for the agent

        Returns:
            raw_obs (List[Float]): Current state observation
        """

        raw_obs = np.array([self.dcload, self.battery.current_load])
        return raw_obs

    def _simulate_battery_operation(self, battery, battery_action, charging_rate=None):
        """Simulates battery operation

        Args:
            battery (Class): Battery model.
            battery_action (string): Desired action.
            charging_rate (string): Battery charging rate.

        Returns:
            discharge_energy (float): Output energy.
        """
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
        """Calculates carbon footprint

        Args:
            dc_load (float): Total energy consumption of the DC.
            ci (float): Carbon intensity at current time step.
            a_t (string): Agent's action.
            discharge_energy (float): Amount of energy to be discharged

        Returns:
            CO2_footprint (float): Carbon footprint produced at the current time step
        """
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
        """Calculates the battery state depeding charging rate

        Args:
            battery (batt.Battery2): Battery model

        Returns:
            charging_rate (float): Battery charging rate
        """
        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        curr_load = battery.current_load
        bat_max, bat_min = battery.capacity, 0
        sigmoid_max, sigmoid_min = 4, -4
        scaled_curr_load = (curr_load - bat_min) * (sigmoid_max - sigmoid_min) / (bat_max - bat_min) + sigmoid_min
        charging_rate = 0.3

        return charging_rate

    def discharging_rate_modifier(self, battery):
        """Calculates the battery state depeding discharging rate

        Args:
            battery (batt.Battery2): Battery model

        Returns:
            discharging_rate (float): Battery discharging rate
        """
        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        curr_load = battery.current_load
        bat_max, bat_min = battery.capacity, 0
        sigmoid_max, sigmoid_min = 4, -4
        scaled_curr_load = (curr_load - bat_min) * (sigmoid_max - sigmoid_min) / (bat_max - bat_min) + sigmoid_min
        discharging_rate = 0.3

        return discharging_rate
        
        
