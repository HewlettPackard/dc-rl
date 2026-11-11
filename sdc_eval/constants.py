"""Constants shared by training, evaluation and the baselines."""
ENV_CFG = 'harl/configs/envs_cfgs/sustaindc.yaml'
AGENTS = ('agent_ls', 'agent_dc', 'agent_bat')
NOOP = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}  # ls: no shifting, dc: hold setpoint, bat: idle
N_ACTIONS = 3
STEP_HOURS = 0.25  # 15-minute steps

# Config keys that define the simulated scenario. Policies may differ in observation settings, not in these.
SCENARIO_KEYS = ['location', 'month', 'days_per_episode', 'workload_file', 'cintensity_file', 'weather_file',
                 'flexible_load', 'max_bat_cap_Mw', 'dc_config_file', 'datacenter_capacity_mw', 'timezone_shift',
                 'pv_config', 'pv_export_credit', 'battery_config']

# Disjoint episode seeds. Episode i of a set uses base + i.
TRAIN_EVAL_SEED = 10000  # in-training evaluation and best-model selection
FINAL_EVAL_SEED = 1000   # final reported comparison (eval_compare.py); never used for model selection
TUNE_SEED = 2000         # baseline parameter tuning (tune_baselines.py)
MPC_TRAIN_SEED = 3000    # MPC value-iteration data collection (train_mpc_value.py)


def seeds(base, n):
    return [base + i for i in range(n)]
