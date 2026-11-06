"""Joint (de-duplicated) observation over the three SustainDC agents.

Per-agent observation layouts (built in sustaindc_env.SustainDC._create_*_state):
    ls  (26): hour sin/cos[0:2], ci[2], ci feats[3:10], oldest task age[10], avg task age[11],
              queue[12], workload[13], temp[14], temp feats[15:21], task-age histogram[21:26]
    dc  (14): hour[0:2], ci[2], ci feats[3:10], workload[10], next workload[11], temp[12], next temp[13]
              (+ current CRAC setpoint[14] when include_dc_setpoint is on)
    bat (13): hour[0:2], ci[2], ci feats[3:10], workload[10], temp[11], battery SoC[12]

Everything in dc/bat duplicates ls except dc's next workload / next temperature (and setpoint) and bat's SoC,
so the union is ls + those values: 29 dims, or 30 with the setpoint (appended last, so SoC stays at index 28).

The indices are absolute (from the start of each array), so this also works on observations that
supersuit has zero-padded at the end (as in the HARL wrapper).
"""
import numpy as np

LS_OBS_DIM = 26
DC_NEXT_WORKLOAD_IDX = 11
DC_NEXT_TEMP_IDX = 13
DC_SETPOINT_IDX = 14
BAT_SOC_IDX = 12

JOINT_OBS_DIM = LS_OBS_DIM + 3


def joint_obs_dim(include_dc_setpoint=False):
    return JOINT_OBS_DIM + int(include_dc_setpoint)


def build_joint_obs(ls_state, dc_state, bat_state, include_dc_setpoint=False):
    """Return the joint observation [ls(26), dc next workload, dc next temp, bat SoC(, dc setpoint)]."""
    dc_idx = [DC_NEXT_WORKLOAD_IDX, DC_NEXT_TEMP_IDX]
    parts = [np.asarray(ls_state)[:LS_OBS_DIM], np.asarray(dc_state)[dc_idx], np.asarray(bat_state)[[BAT_SOC_IDX]]]
    if include_dc_setpoint:
        parts.append(np.asarray(dc_state)[[DC_SETPOINT_IDX]])
    return np.hstack(parts).astype(np.float32)
