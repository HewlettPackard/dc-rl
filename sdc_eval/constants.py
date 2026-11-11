"""Constants shared by training, evaluation and the baselines."""
ENV_CFG = 'harl/configs/envs_cfgs/sustaindc.yaml'
AGENTS = ('agent_ls', 'agent_dc', 'agent_bat')
NOOP = {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2}  # ls: no shifting, dc: hold setpoint, bat: idle
N_ACTIONS = 3
STEP_HOURS = 0.25  # 15-minute steps
TASK_UTILIZATION = 0.01  # one task = 1 % of the data center's CPU for one step
# Dropped work is charged at this multiple of its marginal energy x current CI in the training objectives
# (co2_team_reward, the MPC's stage cost), so that dropping never pays. Evaluation (co2_work_adjusted) charges
# unexecuted work at 1x: it is accounting, not an incentive.
DROPPED_TASK_CO2_FACTOR = 2.0

# Config keys that define the simulated scenario. Policies may differ in observation settings, not in these.
SCENARIO_KEYS = ['location', 'month', 'days_per_episode', 'workload_file', 'cintensity_file', 'weather_file',
                 'flexible_load', 'max_bat_cap_Mw', 'dc_config_file', 'datacenter_capacity_mw', 'timezone_shift',
                 'pv_config', 'pv_export_credit', 'battery_config']

# Disjoint episode seeds. Episode i of a set uses base + i.
TRAIN_EVAL_SEED = 10000  # in-training evaluation and best-model selection
FINAL_EVAL_SEED = 1000   # final reported comparison (eval_compare.py); never used for model selection
TUNE_SEED = 2000         # baseline parameter tuning (tune_baselines.py)
MPC_TRAIN_SEED = 3000    # MPC value-iteration data collection (train_mpc_value.py)

# Default in-training evaluation interval (env steps summed over parallel envs), the same for every trainer.
EVAL_EVERY_STEPS = 100_000


def seeds(base, n):
    return [base + i for i in range(n)]
