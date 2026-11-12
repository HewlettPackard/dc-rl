import numpy as np
import gymnasium as gym
from gymnasium import spaces

import envs.battery_model as batt
from utils import reward_creator

# PV actions when PV is dispatched by the agent (SustainDC include_pv=True). The battery agent's action is then
# Discrete(9): action = battery_action + 3 * pv_action, so actions 0-2 are the usual battery actions with PV 'use'.
PV_ACTIONS = ('use', 'battery', 'resell')
N_BAT_ACTIONS = 3


def combine_bat_pv_action(bat_action, pv_action):
    """Battery agent action (Discrete(9)) from a battery action (0 charge, 1 discharge, 2 idle) and a PV action
    (0 use, 1 battery, 2 resell)."""
    return int(bat_action) + N_BAT_ACTIONS * int(pv_action)


class BatteryEnvFwd(gym.Env):
    def __init__(self, env_config) -> None:
        """Creates battery envrionemnt

        Args:
            env_config (dict): Customizable environment confing.
                n_fwd_steps(int): Number of forward forecast steps available
                max_bat_cap(float): Battery energy capacity in MWh
                battery_config(dict, optional): Battery specs, see envs.battery_model.DEFAULT_BATTERY_CONFIG
                    (C-rate, efficiencies, SoC window, charge taper).
        """
        super(BatteryEnvFwd, self).__init__()
        n_fwd_steps = env_config['n_fwd_steps']
        max_bat_cap = env_config['max_bat_cap']
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
        
        self.battery = batt.Battery(capacity=max_bat_cap, battery_config=env_config.get('battery_config'))
        self.n_fwd_steps = n_fwd_steps
        self.step_hours = 0.25  # 15-minute steps
        self.spot_CI = None
        self.ma_CI = None
        self.eta = 0.7
        self.dataset_end = True
        self.dcload = 0
        self.temp_state = None
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
        # Agent-dispatched PV (set_pv_dispatch(True), i.e. SustainDC's include_pv=True): the self-consumption rule
        # above is replaced by a PV action chosen every step, and the action space becomes Discrete(9). See
        # _dispatch_step(). pv_resell_credit is the fraction of the current CI credited per kWh of resold PV.
        self.pv_dispatch = False
        self.pv_resell_credit = 1.0
        self.pv_action = None  # name of the last PV action ('use' | 'battery' | 'resell'), None without dispatch
        self.pv_flows = self._pv_flows(0.0, 0.0, 0.0, 0.0)
        # What the battery actually did in the last step: 'charge', 'discharge', 'idle' or 'pv_charge' (absorbing PV
        # surplus while the action was idle/discharge). Without PV this always equals the action.
        self.bat_mode = 'idle'
        self.charge_draw = 0.0   # energy drawn by the battery this step (MWh, from the grid and/or PV)
        
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
        self.battery.reset()  # Start at the bottom of the usable SoC window
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

        bat_id, pv_id = self.split_action(action_id)
        action_instantaneous = self._action_to_direction[bat_id]
        if self.pv_dispatch:
            self.pv_action = PV_ACTIONS[pv_id]
            self.CO2_total = self._dispatch_step(action_instantaneous, self.pv_action)
        else:
            self.discharge_energy = self._simulate_battery_operation(self.battery, action_instantaneous)
            self.CO2_total = self.CO2_footprint(self.dcload, self.ci, self.bat_mode, self.discharge_energy)

        self.raw_obs = self._hist_data_collector()
        self.temp_state = self._process_obs(self.raw_obs)
        
        self.reward = 0

        self.info = {
            'bat_action': bat_id,
            'pv_action': pv_id if self.pv_dispatch else -1,       # 0 use, 1 battery, 2 resell; -1 = not dispatched
            'pv_a_t': self.pv_action if self.pv_dispatch else 'auto',
            'bat_SOC': self.get_battery_soc(),
            'bat_CO2_footprint': self.CO2_total,
            'bat_avg_CI': self.ci,
            'bat_total_energy_without_battery_KWh': self.dcload * 1e3 * 0.25,
            'bat_total_energy_with_battery_KWh': self.total_energy_with_battery,
            'bat_max_bat_cap': self.max_bat_cap,
            'bat_a_t': action_instantaneous,
            'bat_mode': self.bat_mode,
            'bat_energy_in_KWh': self.charge_draw * 1e3,        # drawn to charge (grid and/or PV), incl. losses
            'bat_energy_out_KWh': self.discharge_energy * 1e3,  # delivered to the DC
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

    def set_pv_dispatch(self, enabled, resell_credit=1.0):
        """Turn agent-dispatched PV on/off. On: the action space becomes Discrete(9) (battery x PV action)."""
        self.pv_dispatch = bool(enabled)
        self.pv_resell_credit = float(resell_credit)
        self.action_space = spaces.Discrete(N_BAT_ACTIONS * (len(PV_ACTIONS) if self.pv_dispatch else 1))

    def split_action(self, action_id):
        """(battery action id, PV action id) of an agent action. Without PV dispatch the PV id is None."""
        action_id = int(action_id)
        if not self.pv_dispatch:
            return action_id, None
        return action_id % N_BAT_ACTIONS, action_id // N_BAT_ACTIONS

    def _dispatch_step(self, battery_action, pv_action):
        """One step with agent-dispatched PV. Sets the battery, energy and PV-flow state and returns the CO2 (g).

        PV actions (P = PV energy of the step, L = DC load energy):
            use:     PV supplies the DC load. Any surplus can feed a battery 'charge' (before the grid); the rest is
                     curtailed. The battery follows its own action (discharge covers L - PV used).
            battery: PV goes to the battery first, up to its charge limit (a 'charge' action tops it up from the
                     grid; 'discharge' is overridden, the battery cannot charge and discharge at once). PV the
                     battery cannot take supplies the DC load; anything left is curtailed.
            resell:  all of P is sold to the grid and credited at pv_resell_credit x the current CI (it is
                     renewable energy that displaces grid energy). The DC load and any battery action are then
                     served by the grid / battery as if there were no PV.
        Grid import = L - PV used - discharge + charge - PV to battery (>= 0); CO2 = grid x CI - resold x credit x CI.
        """
        dt = self.step_hours
        battery = self.battery
        load = self.dcload * dt                     # MWh
        pv = self.pv_kw / 1e3 * dt                  # MWh
        pv_self = pv_to_battery = resold = 0.0
        discharge = 0.0
        self.charge_draw = 0.0

        if pv_action == 'battery':
            if battery_action == 'charge':
                self.bat_mode = 'charge'
                self.charge_draw = battery.charge(dt)
            else:
                self.bat_mode = 'pv_charge'
                self.charge_draw = battery.charge(dt, max_draw=pv)
            pv_to_battery = min(pv, self.charge_draw)
            pv_self = min(pv - pv_to_battery, load)
        else:
            if pv_action == 'use':
                pv_self = min(pv, load)
            else:  # resell
                resold = pv
            surplus = pv - pv_self - resold
            if battery_action == 'charge':
                self.bat_mode = 'charge'
                self.charge_draw = battery.charge(dt)
                pv_to_battery = min(surplus, self.charge_draw)
            elif battery_action == 'discharge':
                self.bat_mode = 'discharge'
                discharge = battery.discharge(dt, max_out=load - pv_self)
            else:
                self.bat_mode = 'idle'
        self.discharge_energy = discharge

        to_kwh = 1e3
        load, pv, pv_self, pv_to_battery, resold = (x * to_kwh for x in (load, pv, pv_self, pv_to_battery, resold))
        charge, discharge = self.charge_draw * to_kwh, discharge * to_kwh
        if charge > 0:
            self.energy_added_removed.append(charge)
        elif discharge > 0:
            self.energy_added_removed.append(-discharge)
        curtailed = max(pv - pv_self - pv_to_battery - resold, 0.0)

        self.total_energy_with_battery = max((load - pv_self - discharge) + (charge - pv_to_battery), 0.0)
        credit = resold * self.pv_resell_credit * self.ci
        self.pv_flows = self._pv_flows(pv, pv_self, pv_to_battery, curtailed + resold, exported=resold,
                                       credit=credit)
        return self.total_energy_with_battery * self.ci - credit

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

    def _simulate_battery_operation(self, battery, battery_action):
        """Simulates battery operation over one step (see envs/battery_model.py for the battery physics).

        PV surplus (PV beyond the DC load) is stored automatically, as in standard self-consumption control:
            charge:    charge at the battery's full power; PV surplus is used first, the rest comes from the grid.
            idle:      no grid charging, but PV surplus is absorbed (within the same charge limit).
            discharge: cover the net load (DC load - PV). With a PV surplus there is no net load, so the battery
                       absorbs the surplus instead, as with idle.

        Args:
            battery (envs.battery_model.Battery): Battery model.
            battery_action (string): Desired action.

        Returns:
            discharge_energy (float): Energy delivered to the DC (MWh).
        """
        dt = self.step_hours
        discharge_energy = 0.0
        self.charge_draw = 0.0
        surplus_MWh = max(self.pv_kw / 1e3 - self.dcload, 0) * dt
        if battery_action == 'charge':
            self.bat_mode = 'charge'
            self.charge_draw = battery.charge(dt)
        elif surplus_MWh > 0:
            self.bat_mode = 'pv_charge'
            self.charge_draw = battery.charge(dt, max_draw=surplus_MWh)
        elif battery_action == 'discharge':
            self.bat_mode = 'discharge'
            # Only the load not already covered by PV can be served by the battery (no discharging into curtailment).
            net_dcload_MWh = max(self.dcload - self.pv_kw / 1e3, 0) * dt
            discharge_energy = battery.discharge(dt, max_out=net_dcload_MWh)
        else:
            self.bat_mode = 'idle'
        return discharge_energy

    def CO2_footprint(self, dc_load, ci, a_t, discharge_energy):
        """Calculates the grid energy and carbon footprint of the current 15-minute step.

        Energy balance (kWh): L = DC load, P = PV, C = battery charging, D = battery discharging.
            Self-consumption priority: PV first serves the DC load (pv_self = min(P, L)), then charges the battery
            (pv_to_battery = min(P - pv_self, C); the battery takes PV surplus whatever the action, see
            _simulate_battery_operation); what is left is curtailed (or exported, if pv_export_credit).
            Discharging is capped at the net load L - pv_self. C is the energy drawn to charge (the battery stores
            C x eff_charge), D the energy delivered (it removes D / eff_discharge from storage), so conversion losses
            are paid for by the grid / PV.
            grid = L - pv_self - D + C - pv_to_battery >= 0, and CO2 = grid * ci.

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
            charge = self.charge_draw * 1e3
            self.energy_added_removed.append(charge)
        elif a_t == 'discharge':
            discharge = discharge_energy * 1e3
            assert load - pv_self >= discharge - 1e-9, "Battery discharge should not be higher than the net (after PV) DC energy consumption"
            self.energy_added_removed.append(-1.0 * discharge)
        if a_t == 'pv_charge':
            # PV-only charging by construction; drop float round-off from the MWh <-> kWh conversions.
            charge = min(charge, pv - pv_self)
        pv_to_battery = min(pv - pv_self, charge)
        surplus = pv - pv_self - pv_to_battery

        # Grouped as (load not covered by PV or battery) + (charging not covered by PV), so full PV coverage gives an
        # exact 0.
        self.total_energy_with_battery = max((load - pv_self - discharge) + (charge - pv_to_battery), 0)
        CO2_footprint = self.total_energy_with_battery * ci  # (KWh) * gCO2/KWh
        credit = surplus * ci if self.pv_export_credit else 0.0
        CO2_footprint -= credit
        self.pv_flows = self._pv_flows(pv, pv_self, pv_to_battery, surplus, credit=credit)
        return CO2_footprint

    def _pv_flows(self, pv, pv_self, pv_to_battery, surplus, exported=None, credit=0.0):
        """Info entries describing where the PV energy of the step went (kWh), plus the grid import.
        surplus = PV not used by the DC or the battery; exported = the part of it sold to the grid (default: all of
        it if pv_export_credit, else none); credit = CO2 credited for the export (g)."""
        if exported is None:
            exported = surplus if self.pv_export_credit else 0.0
        return {
            'pv_export_credit_gCO2': credit,
            'pv_power_kW': pv / 0.25,
            'pv_energy_KWh': pv,
            'pv_self_consumed_KWh': pv_self,
            'pv_to_battery_KWh': pv_to_battery,
            'pv_curtailed_KWh': surplus - exported,
            'pv_exported_KWh': exported,
            'grid_energy_KWh': self.total_energy_with_battery,
        }
