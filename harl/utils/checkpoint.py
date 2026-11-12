"""Periodic, resumable training checkpoints for the HARL runners (protection against preemption).

A checkpoint holds everything needed to continue training where it stopped: the networks *and* their optimizers
(including target networks and counters of the actor/critic objects), the value normalizer, the loop position, the
best evaluation reward so far, the random-number-generator states and, for off-policy algorithms, the replay buffer.
The environments are not saved: on resume they restart with new seeds (the partial episodes in progress are lost).

Layout, in <run_dir>/checkpoints/:
    ckpt_<steps>.pt            networks, optimizers, loop position, RNG states
    buffer_<steps>.npz         replay buffer (off-policy only)
Files are written under a temporary name and renamed, and the .pt file is written last, so a job killed in the middle
of a save never leaves a checkpoint that looks complete. Only the newest `keep` checkpoints are kept.

Resuming: run again with the same run directory (--run_id, e.g. the Slurm job id, which a requeued job keeps). The
runner then loads the newest checkpoint; it refuses to resume if the saved config.json differs from the new one.
"""
import glob
import json
import numbers
import os
import random
import re

import numpy as np
import torch

CKPT_DIR = 'checkpoints'


def checkpoint_dir(run_dir):
    return os.path.join(run_dir, CKPT_DIR)


def list_checkpoints(run_dir):
    """[(steps, path)] of the complete checkpoints of a run, oldest first."""
    out = []
    for path in glob.glob(os.path.join(checkpoint_dir(run_dir), 'ckpt_*.pt')):
        m = re.fullmatch(r'ckpt_(\d+)\.pt', os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def latest_checkpoint(run_dir):
    ckpts = list_checkpoints(run_dir)
    return ckpts[-1] if ckpts else (0, None)


def check_config(run_dir, args, algo_args, env_args):
    """Refuse to resume a run whose saved configuration differs from the new one."""
    from harl.utils.configs_tools import convert_json
    path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        saved = json.load(f)
    new = json.loads(json.dumps(convert_json({'main_args': args, 'algo_args': algo_args, 'env_args': env_args})))
    diff = sorted(set(_config_diff(saved, new)))
    if diff:
        raise SystemExit(f'Refusing to resume {run_dir}: the configuration differs from its config.json '
                         f'(keys: {diff}). Use another --run_id to start a new run.')


# Options that don't change what is trained: where outputs go and how hyperparameters were passed in.
_RUN_CONTROL_KEYS = {'main_args.out_root', 'main_args.config', 'main_args.overrides', 'main_args.load_config',
                     'main_args.checkpoint_interval', 'algo_args.logger.log_dir'}


def _config_diff(saved, new, path=''):
    """Dotted keys whose values differ. Keys absent from the saved config are options added to the code after the run
    started (e.g. the common evaluation settings): they are ignored, so older runs can still be resumed."""
    for k, v in saved.items():
        key = f'{path}.{k}' if path else k
        if key in _RUN_CONTROL_KEYS:
            continue
        if k not in new:
            yield key
        elif isinstance(v, dict) and isinstance(new[k], dict):
            yield from _config_diff(v, new[k], key)
        elif v != new[k]:
            yield key


# ---------------------------------------------------------------------------------------------------------------------
# State of the actor / critic objects
# ---------------------------------------------------------------------------------------------------------------------
def object_state(obj):
    """State of an actor/critic object: state_dicts of its modules and optimizers, plus its plain scalar attributes
    (e.g. update counters). Covers HAPPO/MAPPO (network + optimizer) and HAD3QN (networks, target networks, optimizers)."""
    state = {}
    for name, val in vars(obj).items():
        if isinstance(val, (torch.nn.Module, torch.optim.Optimizer)):
            state[name] = ('state_dict', val.state_dict())
        elif isinstance(val, torch.Tensor):
            state[name] = ('tensor', val.detach().cpu().clone())
        elif isinstance(val, (bool, numbers.Number, np.number)) and not isinstance(val, type):
            state[name] = ('value', val)
    return state


def load_object_state(obj, state):
    for name, (kind, val) in state.items():
        if not hasattr(obj, name):
            raise KeyError(f'{type(obj).__name__} has no attribute {name!r} saved in the checkpoint')
        if kind == 'state_dict':
            getattr(obj, name).load_state_dict(val)
        elif kind == 'tensor':
            cur = getattr(obj, name)
            if isinstance(cur, torch.Tensor):
                with torch.no_grad():
                    cur.copy_(val.to(cur.device))
            else:
                setattr(obj, name, val)
        else:
            setattr(obj, name, val)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


# ---------------------------------------------------------------------------------------------------------------------
# Replay buffer (off-policy)
# ---------------------------------------------------------------------------------------------------------------------
# Buffer arrays recomputed from the stored data before each sample (not state): end_flag comes from dones.
DERIVED_BUFFER_ARRAYS = {'end_flag'}


def buffer_arrays(buffer):
    """Flat {name: array} of a replay buffer's storage (only the filled part) plus its cursor."""
    n = int(buffer.cur_size)
    arrays = {'__idx': np.array(buffer.idx), '__cur_size': np.array(buffer.cur_size)}
    for name, val in vars(buffer).items():
        if name in DERIVED_BUFFER_ARRAYS:
            continue
        if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == buffer.buffer_size:
            arrays[name] = val[:n]
        elif isinstance(val, list) and val and all(isinstance(v, np.ndarray) and v.ndim >= 1
                                                   and len(v) == buffer.buffer_size for v in val):
            for i, v in enumerate(val):
                arrays[f'{name}__{i}'] = v[:n]
    return arrays


def load_buffer_arrays(buffer, arrays):
    n = int(arrays['__cur_size'])
    for key in arrays.files if hasattr(arrays, 'files') else arrays:
        if key.startswith('__'):
            continue
        if '__' in key:
            name, i = key.rsplit('__', 1)
            getattr(buffer, name)[int(i)][:n] = arrays[key]
        else:
            getattr(buffer, key)[:n] = arrays[key]
    buffer.idx = int(arrays['__idx'])
    buffer.cur_size = n


# ---------------------------------------------------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------------------------------------------------
def save_checkpoint(run_dir, steps, state, buffer=None, keep=2):
    """Write ckpt_<steps>.pt (and buffer_<steps>.npz) atomically; keep the newest `keep` checkpoints."""
    d = checkpoint_dir(run_dir)
    os.makedirs(d, exist_ok=True)
    if buffer is not None:
        tmp = os.path.join(d, f'buffer_{steps}.tmp.npz')
        np.savez(tmp, **buffer_arrays(buffer))
        os.replace(tmp, os.path.join(d, f'buffer_{steps}.npz'))
    tmp = os.path.join(d, f'ckpt_{steps}.pt.tmp')
    torch.save({'steps': steps, **state}, tmp)
    os.replace(tmp, os.path.join(d, f'ckpt_{steps}.pt'))  # written last: marks the checkpoint complete
    for old_steps, old in list_checkpoints(run_dir)[:-keep]:
        os.remove(old)
        old_buf = os.path.join(d, f'buffer_{old_steps}.npz')
        if os.path.exists(old_buf):
            os.remove(old_buf)
    print(f'Checkpoint saved at {steps} env steps: {d}', flush=True)


def load_checkpoint(path, buffer=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if buffer is not None:
        buf_path = os.path.join(os.path.dirname(path), f"buffer_{state['steps']}.npz")
        with np.load(buf_path) as arrays:
            load_buffer_arrays(buffer, arrays)
    return state


def trim_progress(run_dir, resumed_steps):
    """Drop progress.txt lines logged after the checkpoint we resume from (they will be logged again). Lines that do
    not start with a step count belong to the last step-count line before them."""
    path = os.path.join(run_dir, 'progress.txt')
    if not os.path.exists(path):
        return
    keep, cur = [], 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            head = line.split(',', 1)[0].strip()
            try:
                cur = float(head)
            except ValueError:
                pass
            if cur <= resumed_steps:
                keep.append(line)
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(keep)


def best_eval_path(save_dir):
    return os.path.join(save_dir, 'best_eval.json')


def record_best_eval(save_dir, reward, steps):
    """Remember the evaluation reward of the model saved in models/ (so a resumed run never overwrites a better one)."""
    with open(best_eval_path(save_dir), 'w', encoding='utf-8') as f:
        json.dump({'avg_eval_episode_reward': float(reward), 'steps': int(steps)}, f)


def saved_best_eval(save_dir):
    try:
        with open(best_eval_path(save_dir), encoding='utf-8') as f:
            return float(json.load(f)['avg_eval_episode_reward'])
    except (OSError, ValueError, KeyError):
        return -np.inf
