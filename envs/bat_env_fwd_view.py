import numpy as np
import gymnasium as gym
from gymnasium import spaces

import envs.battery_model as batt
from utils import reward_creator

class BatteryEnvFwd(gym.Env):
    def __init__(self, env_config) -> None:
        """Creates battery envrionemnt

        Args:
            env_config (dict): Customizable environment confing.
                n_fwd_steps(int): Number of forward forecast steps available
                max_bat_cap(float): Maximun battery capacity in MWh
                charging_rate(float): Rate of charge of the battery
                reward_metod(function): Method used to calculate the reward
        """
        super(BatteryEnvFwd, self).__init__()
        n_fwd_steps = env_config['n_fwd_steps']
        max_bat_cap = env_config['max_bat_cap']
        charging_rate = env_config['charging_rate']
        self.observation_space = spaces.Box(low=np.float32(-1.0 * np.ones(13)),
                                            high=np.float32(1.0 * np.ones(13)))
        
        self.max_dc_pw_MW = env_config['max_dc_pw_MW'] # 7.24  # in MW
        self.action_space = spaces.Discrete(3)
        self._action_to_direction = {0: 'charge', 1: 'discharge', 2: 'idle'}
        
        other_states_max = np.array([self.max_dc_pw_MW, max_bat_cap])
        other_states_min = np.array([0.0, 0])
        
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

        # On-site PV (kW AC) available in the current step; set by SustainDC with set_pv(). See CO2_footprint().
        self.pv_kw = 0.0
        # PV follows the standard self-consumption priority: DC load first, then the battery (automatically, within its
        # charge limit, whatever the action), then the grid. pv_export_credit True: that last part is exported and
        # credited at the current CI (net CO2 can go negative). False (default): it is curtailed.
        self.pv_export_credit = env_config.get('pv_export_credit', False)
        self.pv_flows = self._pv_flows(0.0, 0.0, 0.0, 0.0)
        # What the battery actually did in the last step: 'charge', 'discharge', 'idle' or 'pv_charge' (absorbing PV
        # surplus while the action was idle/discharge). Without PV this always equals the action.
        self.bat_mode = 'idle'
        
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
        self.battery.reset() # Reset the battery with a random current_load between 0 and 25% max capacity
        self.energy_added_removed = []

        self.dcload = self.dcload_min
        self.raw_obs = self._hist_data_collector()
        
        self.temp_state = self._process_obs(self.raw_obs)
        return self.temp_state, {
            'action': -1,
            'avg_dc_power_mw': self.raw_obs[0],
            'Grid_CI': 0,
            'total_energy_with_battery': 0,
            'CO2_footprint': 0,
            'avg_CI': 0,
            'bat_SOC': self.get_battery_soc(),
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
        
        # Change the action for a random action
        # action_id = np.random.randint(self.action_space.n)
        # print(f'Warning, using random action {action_id} in the bat environment')
        # print(f'Using action {action_id} in the bat environment')

        action_instantaneous = self._action_to_direction[action_id]
        self.discharge_energy = self._simulate_battery_operation(self.battery, action_instantaneous,
                                                                  charging_rate=self.charging_rate)
        
        self.CO2_total = self.CO2_footprint(self.dcload, self.ci, self.bat_mode, self.discharge_energy)

        self.raw_obs = self._hist_data_collector()
        self.temp_state = self._process_obs(self.raw_obs)
        
        self.reward = 0

        self.info = {
            'bat_action': action_id,
            'bat_SOC': self.get_battery_soc(),
            'bat_CO2_footprint': self.CO2_total,
            'bat_avg_CI': self.ci,
            'bat_total_energy_without_battery_KWh': self.dcload * 1e3 * 0.25,
            'bat_total_energy_with_battery_KWh': self.total_energy_with_battery,
            'bat_max_bat_cap': self.max_bat_cap,
            'bat_a_t': action_instantaneous,
            'bat_mode': self.bat_mode,
            'bat_dcload_min': self.dcload_min,
            'bat_dcload_max': self.dcload_max,
            **self.pv_flows,
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
        return np.float32(scaled_value)

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

    def set_pv(self, pv_kw):
        """Set the on-site PV AC power (kW) available in the current step."""
        self.pv_kw = float(pv_kw)

    def set_dcload(self, dc_load):
        """Set the current DC energy consumption

        Args:
            dc_load float: DC energy consumption.

        """
        self.dcload = dc_load
        
    def get_battery_soc(self,):
        """Get the current battery state of charge

        Returns:
            battery_soc float: Battery state of charge
        """
        return self.battery.get_battery_soc()
    

    def _hist_data_collector(self):
        """Generates the observation for the agent

        Returns:
            raw_obs (List[Float]): Current state observation
        """

        raw_obs = np.array([self.dcload, self.battery.current_load])
        return raw_obs

    def _simulate_battery_operation(self, battery, battery_action, charging_rate=None):
        """Simulates battery operation

        PV surplus (PV beyond the DC load) is stored automatically, as in standard self-consumption control:
            charge:    charge at the full per-step rate; PV surplus is used first, the rest comes from the grid.
            idle:      no grid charging, but PV surplus is absorbed (up to the same per-step charge limit).
            discharge: cover the net load (DC load - PV). With a PV surplus there is no net load, so the battery
                       absorbs the surplus instead, as with idle.
        Without PV this is the original charge / discharge / idle behaviour.

        Args:
            battery (Class): Battery model.
            battery_action (string): Desired action.
            charging_rate (string): Battery charging rate.

        Returns:
            discharge_energy (float): Output energy.
        """
        discharge_energy = 0
        surplus_MWh = max(self.pv_kw / 1e3 - self.dcload, 0) * 0.25
        if battery_action == 'charge':
            self.bat_mode = 'charge'
            self.var_to_dc = battery.charge(battery.capacity, self.charging_rate_modifier(battery) * 15 / 60)
        elif surplus_MWh > 0:
            self.bat_mode = 'pv_charge'
            self.var_to_dc = self._charge_from_pv(battery, surplus_MWh)
        elif battery_action == 'discharge':
            self.bat_mode = 'discharge'
            # Only the load not already covered by PV can be served by the battery (no discharging into curtailment).
            net_dcload = max(self.dcload - self.pv_kw / 1e3, 0)
            discharge_energy = battery.discharge(battery.capacity, self.discharging_rate_modifier(battery) * 15 / 60,
                                                 net_dcload / 4)
            self.var_to_dc = -discharge_energy
        else:
            self.bat_mode = 'idle'
            discharge_energy = 0
            self.var_to_dc = 0
        self.var_to_dc = self.var_to_dc / self.max_bat_cap
        return discharge_energy

    def _charge_from_pv(self, battery, energy_MWh):
        """Charge the battery with at most energy_MWh of PV surplus, within the same per-step limit as the charge action.

        Returns:
            float: Battery energy content after charging (MWh), like Battery2.charge.
        """
        T_u = self.charging_rate_modifier(battery) * 15 / 60
        if T_u <= 0:
            battery.charging_load = 0
            return battery.current_load
        # Battery2.charge stores min(max_charge, input_load) * eff_c * T_u, so this input stores at most energy_MWh.
        return battery.charge(min(battery.capacity, energy_MWh / (battery.eff_c * T_u)), T_u)

    def CO2_footprint(self, dc_load, ci, a_t, discharge_energy):
        """Calculates the grid energy and carbon footprint of the current 15-minute step.

        Energy balance (kWh): L = DC load, P = PV, C = battery charging, D = battery discharging.
            Self-consumption priority: PV first serves the DC load (pv_self = min(P, L)), then charges the battery
            (pv_to_battery = min(P - pv_self, C); the battery takes PV surplus whatever the action, see
            _simulate_battery_operation); what is left is curtailed (or exported, if pv_export_credit).
            Discharging is capped at the net load L - pv_self. Charged energy is counted 1:1 against the grid or PV
            (as in the original model, where charging efficiency only affects the stored energy).
            grid = L - pv_self - D + C - pv_to_battery >= 0, and CO2 = grid * ci.
        With no PV this reduces to the original accounting: L + C (charge), L - D (discharge), L (idle).

        Args:
            dc_load (float): DC power (MW) over the step.
            ci (float): Carbon intensity at current time step (gCO2/kWh).
            a_t (string): What the battery did: 'charge', 'discharge', 'idle' or 'pv_charge'.
            discharge_energy (float): Energy discharged by the battery (MWh).

        Returns:
            CO2_footprint (float): Carbon footprint produced at the current time step (gCO2e)
        """
        load = dc_load * 1e3 * 0.25
        pv = self.pv_kw * 0.25
        pv_self = min(pv, load)
        charge = discharge = 0.0
        if a_t in ('charge', 'pv_charge'):
            charge = self.battery.charging_load * 1e3
            self.energy_added_removed.append(charge)
            self.battery.charging_load = 0  # *Added*
        elif a_t == 'discharge':
            discharge = discharge_energy * 1e3
            assert load - pv_self >= discharge - 1e-9, "Battery discharge should not be higher than the net (after PV) DC energy consumption"
            self.energy_added_removed.append(-1.0 * discharge)
        if a_t == 'pv_charge':
            # PV-only charging by construction; drop float round-off from the MWh <-> rate conversions.
            charge = min(charge, pv - pv_self)
        pv_to_battery = min(pv - pv_self, charge)
        surplus = pv - pv_self - pv_to_battery

        # Grouped as (load not covered by PV or battery) + (charging not covered by PV), so full PV coverage gives an
        # exact 0, and without PV the float operations are the original ones (load + charge, load - discharge).
        self.total_energy_with_battery = max((load - pv_self - discharge) + (charge - pv_to_battery), 0)
        CO2_footprint = self.total_energy_with_battery * ci  # (KWh) * gCO2/KWh
        if self.pv_export_credit:
            CO2_footprint -= surplus * ci
        self.pv_flows = self._pv_flows(pv, pv_self, pv_to_battery, surplus)
        return CO2_footprint

    def _pv_flows(self, pv, pv_self, pv_to_battery, surplus):
        """Info entries describing where the PV energy of the step went (kWh), plus the grid import."""
        exported = surplus if self.pv_export_credit else 0.0
        return {
            'pv_power_kW': pv / 0.25,
            'pv_energy_KWh': pv,
            'pv_self_consumed_KWh': pv_self,
            'pv_to_battery_KWh': pv_to_battery,
            'pv_curtailed_KWh': surplus - exported,
            'pv_exported_KWh': exported,
            'grid_energy_KWh': self.total_energy_with_battery,
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def charging_rate_modifier(self, battery):
        """Calculates the battery state depending on the charging rate

        Args:
            battery (batt.Battery2): Battery model

        Returns:
            charging_rate (float): Battery charging rate
        """
        curr_load = battery.current_load
        bat_max, bat_min = battery.capacity, 0
        scaled_curr_load = (curr_load - bat_min) / (bat_max - bat_min)  # Scale current load to [0, 1]

        # Use a sigmoid function to model the charging rate
        charging_rate = 0.5*(1 - self.sigmoid(10* (scaled_curr_load - 0.5)))  # Shift and scale sigmoid to model charging rate

        return np.round(charging_rate, 4)

    def discharging_rate_modifier(self, battery):
        """Calculates the battery state depending on the discharging rate

        Args:
            battery (batt.Battery2): Battery model

        Returns:
            discharging_rate (float): Battery discharging rate
        """
        curr_load = battery.current_load
        bat_max, bat_min = battery.capacity, 0
        scaled_curr_load = (curr_load - bat_min) / (bat_max - bat_min)  # Scale current load to [0, 1]

        # Use a sigmoid function to model the discharging rate
        discharging_rate = 4*self.sigmoid(10 * (scaled_curr_load - 0.25))  # Shift and scale sigmoid to model discharging rate

        return max(0.5, discharging_rate)
        
