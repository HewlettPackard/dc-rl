"""Stationary lithium-ion (LFP) battery energy storage system (BESS) model.

The battery is rated by duration: power = c_rate x capacity, the same for charging and discharging (the inverter
limits power in both directions). A 0.5C battery is a "2-hour" battery, the most common size for commercial
behind-the-meter storage; 0.25C (4-hour) is typical of grid-scale storage, 1C (1-hour) of power-oriented systems.

- Usable state of charge (SoC) window [soc_min, soc_max] to limit degradation.
- Full charging power up to taper_start, then a linear taper to 0 at soc_max (the constant-voltage phase).
- One-way efficiencies eff_charge / eff_discharge (0.95 each, ~90% round trip, AC-AC including the inverter).
  Energy drawn from the grid/PV to charge is `draw`, of which `draw * eff_charge` is stored; delivering `out`
  removes `out / eff_discharge` from storage.

The limit functions are vectorized (numpy arrays of stored energy) so planners (baseline_policies.BatOracle) use
exactly the same physics as the environment.
"""
import numpy as np

DEFAULT_BATTERY_CONFIG = {
    'c_rate': 0.5,           # charge/discharge power as a fraction of the energy capacity per hour (0.5 = 2-hour battery)
    'eff_charge': 0.95,      # one-way charging efficiency
    'eff_discharge': 0.95,   # one-way discharging efficiency
    'soc_min': 0.05,         # lowest usable SoC
    'soc_max': 0.95,         # highest usable SoC
    'taper_start': 0.90,     # SoC above which charging power tapers linearly to 0 at soc_max
}


def make_battery_config(battery_config=None):
    """Fill in defaults and validate a battery config dict."""
    battery_config = dict(battery_config or {})
    unknown = set(battery_config) - set(DEFAULT_BATTERY_CONFIG)
    if unknown:
        raise ValueError(f"Unknown battery_config keys {sorted(unknown)}; valid keys: {sorted(DEFAULT_BATTERY_CONFIG)}")
    cfg = {**DEFAULT_BATTERY_CONFIG, **battery_config}
    if not cfg['c_rate'] > 0:
        raise ValueError(f"battery_config['c_rate'] must be > 0, got {cfg['c_rate']!r}")
    if not (0 < cfg['eff_charge'] <= 1 and 0 < cfg['eff_discharge'] <= 1):
        raise ValueError("battery_config efficiencies must be in (0, 1]")
    if not 0 <= cfg['soc_min'] < cfg['taper_start'] <= cfg['soc_max'] <= 1:
        raise ValueError("battery_config needs 0 <= soc_min < taper_start <= soc_max <= 1")
    return cfg


class Battery:
    """Stationary Li-ion battery.

    Args:
        capacity (float): Nameplate energy capacity (MWh).
        battery_config (dict, optional): See DEFAULT_BATTERY_CONFIG.

    Attributes:
        current_load (float): Stored energy (MWh).
    """

    def __init__(self, capacity, battery_config=None):
        self.capacity = capacity
        cfg = make_battery_config(battery_config)
        self.config = cfg
        self.c_rate = cfg['c_rate']
        self.eff_charge, self.eff_discharge = cfg['eff_charge'], cfg['eff_discharge']
        self.soc_min, self.soc_max, self.taper_start = cfg['soc_min'], cfg['soc_max'], cfg['taper_start']
        self.power_mw = self.c_rate * capacity
        self.reset()

    def reset(self):
        """Start every episode at the bottom of the usable SoC window."""
        self.current_load = self.soc_min * self.capacity

    def get_battery_soc(self):
        """State of charge (0...1) of the battery."""
        return self.current_load / self.capacity

    def charge_limit(self, stored, dt):
        """Most energy (MWh) the battery can draw over dt hours from the given stored energy (vectorized)."""
        stored = np.asarray(stored, dtype=float)
        soc = stored / self.capacity
        taper = np.clip((self.soc_max - soc) / (self.soc_max - self.taper_start), 0.0, 1.0) if self.soc_max > self.taper_start \
            else (soc < self.soc_max).astype(float)
        headroom = np.maximum(self.soc_max * self.capacity - stored, 0.0) / self.eff_charge
        return np.minimum(self.power_mw * taper * dt, headroom)

    def discharge_limit(self, stored, dt):
        """Most energy (MWh) the battery can deliver over dt hours from the given stored energy (vectorized)."""
        stored = np.asarray(stored, dtype=float)
        available = np.maximum(stored - self.soc_min * self.capacity, 0.0) * self.eff_discharge
        return np.minimum(self.power_mw * dt, available)

    def charge(self, dt, max_draw=np.inf):
        """Charge for dt hours, drawing at most max_draw MWh. Returns the energy drawn (MWh)."""
        draw = float(min(self.charge_limit(self.current_load, dt), max_draw))
        self.current_load += draw * self.eff_charge
        return draw

    def discharge(self, dt, max_out=np.inf):
        """Discharge for dt hours, delivering at most max_out MWh. Returns the energy delivered (MWh)."""
        out = float(min(self.discharge_limit(self.current_load, dt), max_out))
        self.current_load = max(self.current_load - out / self.eff_discharge, self.soc_min * self.capacity)
        return out
