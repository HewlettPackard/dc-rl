"""On-site solar PV generation computed from the location's EPW weather file with pvlib.

The EPW files in data/Weather/ already contain hourly global/direct/diffuse irradiance (GHI/DNI/DHI), air temperature
and wind speed, and their LOCATION header gives the site's latitude, longitude, UTC offset and elevation. PV output is
therefore physically consistent with the temperatures the cooling model sees (cloudy hours are also cool hours).

Model chain (pvlib): solar position -> Perez transposition to the plane of array -> physical (Fresnel) incidence-angle
losses -> SAPM cell temperature -> PVWatts DC power (temperature coefficient gamma_pdc) -> flat system losses ->
PVWatts inverter (clipping at the AC rating = DC rating / dc_ac_ratio).
"""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

# PV system specification. capacity_kWp = 0 disables PV (no generation; the simulation is unchanged).
DEFAULT_PV_CONFIG = {
    'capacity_kWp': 0.0,         # DC nameplate rating of the array (kW peak, at STC)
    'surface_tilt': None,        # panel tilt from horizontal (deg); None = site latitude. Fixed mount only.
    'surface_azimuth': 180.0,    # panel azimuth (deg, 180 = facing south); for single_axis, the tracker axis azimuth
    'tracking': 'fixed',         # 'fixed' or 'single_axis' (horizontal N-S axis, backtracking)
    'max_angle': 60.0,           # single_axis: maximum rotation (deg)
    'gcr': 0.35,                 # single_axis: ground coverage ratio (for backtracking)
    'gamma_pdc': -0.0035,        # power temperature coefficient (1/degC); -0.0035 is typical for crystalline Si
    'temp_model': 'open_rack_glass_polymer',  # SAPM mounting: open_rack_glass_glass, open_rack_glass_polymer,
                                              # close_mount_glass_glass, insulated_back_glass_polymer
    'system_losses_pct': 14.0,   # soiling, wiring, mismatch, availability, ... (PVWatts default is ~14%)
    'dc_ac_ratio': 1.2,          # DC rating / inverter AC rating
    'inverter_eff_nom': 0.96,    # nominal inverter efficiency
    'albedo': None,              # ground reflectance; None = median valid EPW albedo, else 0.2
}

TRACKING_MODES = ('fixed', 'single_axis')


def make_pv_config(pv_config=None):
    """Fill in defaults and validate a PV config dict."""
    pv_config = dict(pv_config or {})
    unknown = set(pv_config) - set(DEFAULT_PV_CONFIG)
    if unknown:
        raise ValueError(f"Unknown pv_config keys {sorted(unknown)}; valid keys: {sorted(DEFAULT_PV_CONFIG)}")
    cfg = {**DEFAULT_PV_CONFIG, **pv_config}
    if not cfg['capacity_kWp'] >= 0:
        raise ValueError(f"pv_config['capacity_kWp'] must be >= 0, got {cfg['capacity_kWp']!r}")
    if cfg['tracking'] not in TRACKING_MODES:
        raise ValueError(f"pv_config['tracking'] must be one of {TRACKING_MODES}, got {cfg['tracking']!r}")
    if not cfg['dc_ac_ratio'] > 0 or not 0 < cfg['inverter_eff_nom'] <= 1:
        raise ValueError("pv_config needs dc_ac_ratio > 0 and 0 < inverter_eff_nom <= 1")
    if not 0 <= cfg['system_losses_pct'] < 100:
        raise ValueError(f"pv_config['system_losses_pct'] must be in [0, 100), got {cfg['system_losses_pct']!r}")
    return cfg


def ac_rating_kw(pv_config):
    """AC (inverter) rating in kW: the maximum AC power the system can deliver."""
    cfg = make_pv_config(pv_config)
    return cfg['capacity_kWp'] / cfg['dc_ac_ratio']


def _epw_path(weather_file):
    return weather_file if os.path.isfile(weather_file) else os.path.join(PATH, 'data', 'Weather', weather_file)


def compute_pv_ac_kw(weather_file, pv_config=None):
    """Hourly AC output (kW) of the PV system over the EPW year.

    Args:
        weather_file (str): EPW filename in data/Weather/ (as returned by utils.utils_cf.obtain_paths) or a path.
        pv_config (dict, optional): PV specs, see DEFAULT_PV_CONFIG.

    Returns:
        np.ndarray: 8760 values, row i = the EPW's row i (the hour ending at hour i + 1, local standard time), i.e.
            the same indexing as the temperature data read by utils.managers.Weather_Manager.
    """
    cfg = make_pv_config(pv_config)
    return _compute_cached(_epw_path(weather_file), json.dumps(cfg, sort_keys=True)).copy()


@lru_cache(maxsize=32)
def _compute_cached(epw_path, cfg_json):
    cfg = json.loads(cfg_json)
    weather, meta = _read_epw(epw_path)
    n = len(weather)
    if cfg['capacity_kWp'] == 0:
        return np.zeros(n)

    import pvlib
    from pvlib.location import Location
    from pvlib.modelchain import ModelChain
    from pvlib.pvsystem import Array, FixedMount, PVSystem, SingleAxisTrackerMount
    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

    location = Location(meta['latitude'], meta['longitude'], tz=weather.index.tz, altitude=meta['altitude'])

    if cfg['tracking'] == 'fixed':
        tilt = abs(meta['latitude']) if cfg['surface_tilt'] is None else cfg['surface_tilt']
        mount = FixedMount(surface_tilt=tilt, surface_azimuth=cfg['surface_azimuth'])
    else:
        mount = SingleAxisTrackerMount(axis_tilt=0, axis_azimuth=cfg['surface_azimuth'], max_angle=cfg['max_angle'],
                                       backtrack=True, gcr=cfg['gcr'])

    if cfg['albedo'] is not None:
        albedo = cfg['albedo']
    else:
        valid = weather['albedo'][(weather['albedo'] > 0) & (weather['albedo'] < 1)] if 'albedo' in weather else []
        albedo = float(np.median(valid)) if len(valid) else 0.2

    pdc0_w = cfg['capacity_kWp'] * 1e3
    pac0_w = pdc0_w / cfg['dc_ac_ratio']
    array = Array(mount=mount, albedo=albedo,
                  module_parameters={'pdc0': pdc0_w, 'gamma_pdc': cfg['gamma_pdc']},
                  temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm'][cfg['temp_model']])
    system = PVSystem(arrays=[array],
                      inverter_parameters={'pdc0': pac0_w / cfg['inverter_eff_nom'], 'eta_inv_nom': cfg['inverter_eff_nom']})
    mc = ModelChain(system, location, transposition_model='perez', aoi_model='physical', spectral_model='no_loss',
                    dc_model='pvwatts', ac_model='pvwatts', losses_model='no_loss', temperature_model='sapm')

    # EPW values are integrals over the hour ending at the timestamp: evaluate the sun at the middle of that hour.
    mid_hour = weather[['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']].copy()
    mid_hour.index = weather.index - pd.Timedelta(minutes=30)
    mc.run_model(mid_hour)

    # Flat system losses are applied to DC power before the inverter (as in PVWatts), so they also reduce clipping.
    dc_w = np.asarray(mc.results.dc, dtype=float) * (1 - cfg['system_losses_pct'] / 100)
    ac_w = pvlib.inverter.pvwatts(dc_w, pac0_w / cfg['inverter_eff_nom'], eta_inv_nom=cfg['inverter_eff_nom'])
    ac_kw = np.clip(np.nan_to_num(np.asarray(ac_w, dtype=float)), 0.0, pac0_w) / 1e3
    return ac_kw


def _read_epw(epw_path):
    """EPW data (tz-aware hourly index in local standard time) and metadata via pvlib, coerced to a non-leap year."""
    from pvlib.iotools import read_epw
    weather, meta = read_epw(epw_path, coerce_year=2023)
    if len(weather) != 8760:
        raise ValueError(f"{epw_path}: expected 8760 hourly rows, got {len(weather)}")
    return weather, meta
