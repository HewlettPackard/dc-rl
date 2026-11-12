"""Per-step decisions taken during evaluation, with the context needed to read them.

One row per step and episode: the three agents' actions plus what they did (the battery's actual energy in/out and
state of charge, the work run vs submitted, the setpoint) and the conditions they reacted to (carbon intensity,
weather, queue, PV). Written by the in-training evaluator (<run>/eval/decisions/) and eval_compare.py
(<out_dir>/decisions.csv.gz).

Action codes:  ls: 0 = defer flexible work, 1 = do nothing, 2 = run deferred work
               dc: 0 = lower setpoint, 1 = hold, 2 = raise setpoint
               bat: 0 = charge, 1 = discharge, 2 = idle
"""
import pandas as pd

# column -> info key (merged per-step info dict of SustainDC)
INFO_COLS = {
    'day': 'day', 'hour': 'hour',
    'ci': 'bat_avg_CI',                                   # grid carbon intensity (gCO2/kWh)
    'outside_temp': 'outside_temp',
    'work_submitted': 'ls_original_workload',             # utilization that arrived this step
    'work_executed': 'ls_shifted_workload',               # utilization actually run this step
    'queue': 'ls_tasks_in_queue', 'dropped': 'ls_tasks_dropped',
    'setpoint': 'dc_crac_setpoint',
    'it_kw': 'dc_ITE_total_power_kW', 'hvac_kw': 'dc_HVAC_total_power_kW',
    'dc_kwh': 'bat_total_energy_without_battery_KWh',     # DC consumption, before PV and battery
    'grid_kwh': 'bat_total_energy_with_battery_KWh',      # drawn from the grid, after PV and battery
    'co2_g': 'bat_CO2_footprint',
    'soc': 'bat_SOC',
    'bat_in_kwh': 'bat_energy_in_KWh',                    # drawn to charge (grid and/or PV), incl. losses
    'bat_out_kwh': 'bat_energy_out_KWh',                  # delivered to the DC
    'pv_kwh': 'pv_energy_KWh', 'pv_self_kwh': 'pv_self_consumed_KWh', 'pv_to_bat_kwh': 'pv_to_battery_KWh',
    'pv_curtailed_kwh': 'pv_curtailed_KWh', 'pv_exported_kwh': 'pv_exported_KWh',
    'pv_action': 'pv_action',                             # 0 use, 1 battery, 2 resell; -1 = include_pv off
}
ACTION_COLS = {'act_ls': 'agent_ls', 'act_dc': 'agent_dc', 'act_bat': 'agent_bat'}


class DecisionRecorder:
    """Pass as run_episode(..., trace=recorder); one recorder per episode."""

    def __init__(self, sdc=None):
        self.sdc = sdc
        self.rows = []

    def __call__(self, k, info, actions):
        row = {'t': k}
        row.update({c: int(actions[a]) for c, a in ACTION_COLS.items()})
        row.update({c: info.get(key) for c, key in INFO_COLS.items()})
        if self.sdc is not None:
            row['reward_team'] = (self.sdc.ls_reward + self.sdc.dc_reward + self.sdc.bat_reward) / 3
        self.rows.append(row)


def frame(rows_by_episode, **labels):
    """[(seed, rows)] -> one DataFrame with seed (and any label columns, e.g. policy=...) first."""
    parts = []
    for seed, rows in rows_by_episode:
        df = pd.DataFrame(rows)
        df.insert(0, 'seed', seed)
        for i, (k, v) in enumerate(labels.items()):
            df.insert(i, k, v)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def battery_summary(df):
    """How a policy uses the battery, from its decisions: action shares, the share of charge/discharge decisions
    that moved energy (charging at full or discharging at empty does nothing), energy moved, equivalent cycles and
    the carbon intensity when charging vs discharging (arbitrage needs charge CI < discharge CI)."""
    if df.empty:
        return {}
    bat = df['act_bat'] % 3  # with include_pv the battery agent's action is battery + 3 x PV
    ch, dis = bat == 0, bat == 1
    moved_in, moved_out = df['bat_in_kwh'] > 1e-6, df['bat_out_kwh'] > 1e-6
    n_ep = df['seed'].nunique()
    out = {
        'share_charge': ch.mean(), 'share_discharge': dis.mean(), 'share_idle': (bat == 2).mean(),
        'charge_effective': (ch & moved_in).sum() / max(ch.sum(), 1),
        'discharge_effective': (dis & moved_out).sum() / max(dis.sum(), 1),
        'charged_mwh_per_ep': df['bat_in_kwh'].sum() / 1e3 / n_ep,
        'discharged_mwh_per_ep': df['bat_out_kwh'].sum() / 1e3 / n_ep,
        'soc_min': df['soc'].min(), 'soc_mean': df['soc'].mean(), 'soc_max': df['soc'].max(),
        'ci_when_charging': (df.loc[moved_in, 'ci'] * df.loc[moved_in, 'bat_in_kwh']).sum()
        / max(df.loc[moved_in, 'bat_in_kwh'].sum(), 1e-9) if moved_in.any() else float('nan'),
        'ci_when_discharging': (df.loc[moved_out, 'ci'] * df.loc[moved_out, 'bat_out_kwh']).sum()
        / max(df.loc[moved_out, 'bat_out_kwh'].sum(), 1e-9) if moved_out.any() else float('nan'),
        # grid CO2 avoided by discharging minus grid CO2 caused by charging (PV-charged energy is free)
        'net_co2_saved_kg_per_ep': ((df['bat_out_kwh'] * df['ci']).sum()
                                    - ((df['bat_in_kwh'] - df['pv_to_bat_kwh']).clip(lower=0) * df['ci']).sum())
        / 1e3 / n_ep,
    }
    if 'pv_action' in df and (df['pv_action'] >= 0).any():
        for k, name in enumerate(('use', 'battery', 'resell')):
            out[f'share_pv_{name}'] = (df['pv_action'] == k).mean()
    return out
