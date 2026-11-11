"""Policies for evaluation. Every policy reads the SustainDC state after reset/step and returns {agent: int action}.

The interface (duck-typed; see Policy):
    include_dc_setpoint, include_pv_obs   observation flags the policy was trained with (the eval env follows them)
    train_config                          the env config it was trained on ({} for baselines); checked against the
                                          eval scenario's SCENARIO_KEYS
    reset(seed)                           called right after SustainDC.reset() at the start of every episode
    act(sdc) -> {agent: int}              reads the raw SustainDC object, builds its own observation

Learned policies (SB3Policy, HARLPolicy) can be built from a run directory on disk or, during training, from the
live model (from_live), so the in-training evaluation uses exactly the same observation code as the final one.
"""
import glob
import json
import os
from typing import Protocol

import numpy as np
import torch

from sdc_eval.constants import NOOP
from sustaindc_single_env import AGENTS, decode_joint_action
from utils.joint_obs import PV_OBS_DIM, build_joint_obs


class Policy(Protocol):
    include_dc_setpoint: bool
    include_pv_obs: bool
    train_config: dict

    def reset(self, seed): ...

    def act(self, sdc) -> dict: ...


class NoOpPolicy:
    include_dc_setpoint = False
    include_pv_obs = False
    train_config = {}

    def reset(self, seed):
        pass

    def act(self, sdc):
        return dict(NOOP)


class RandomPolicy(NoOpPolicy):
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, sdc):
        return {agent: int(self.rng.integers(3)) for agent in AGENTS}


class SB3Policy:
    def __init__(self, run_dir, model_name='best_model'):
        from train_single_agent import ALGOS, load_obs_normalizer
        with open(os.path.join(run_dir, 'config.json')) as f:
            cfg = json.load(f)
        self.train_config = cfg['env_args']
        self.include_dc_setpoint = bool(self.train_config.get('include_dc_setpoint', False))
        self.include_pv_obs = bool(self.train_config.get('include_pv_obs', False))
        self.action_type = self.train_config.get('action_type', 'multidiscrete')
        model_path = os.path.join(run_dir, 'models', model_name)
        if model_name == 'best_model' and not os.path.exists(model_path + '.zip'):
            # No evaluation passed the best-model guard (e.g. every one dropped more tasks than noop).
            print(f'Note: {run_dir} has no best_model (see eval/evals.csv); using final_model.')
            model_path = os.path.join(run_dir, 'models', 'final_model')
        self.model = ALGOS[cfg['main_args']['algo']].load(model_path, device='cpu')
        self.normalize = load_obs_normalizer(model_path)  # VecNormalize stats saved with the model, if any

    @classmethod
    def from_live(cls, model, env_config, normalize=None):
        """Wrap a model being trained (in-training evaluation). normalize: obs -> normalized obs, e.g. the training
        VecNormalize's normalize_obs (reads its current statistics without updating them)."""
        self = cls.__new__(cls)
        self.train_config = env_config
        self.include_dc_setpoint = bool(env_config.get('include_dc_setpoint', False))
        self.include_pv_obs = bool(env_config.get('include_pv_obs', False))
        self.action_type = env_config.get('action_type', 'multidiscrete')
        self.model = model
        self.normalize = normalize or (lambda obs: obs)
        return self

    def reset(self, seed):
        pass

    def act(self, sdc):
        obs = build_joint_obs(sdc.ls_state, sdc.dc_state, sdc.bat_state, self.include_dc_setpoint, self.include_pv_obs)
        action, _ = self.model.predict(self.normalize(obs), deterministic=True)
        return decode_joint_action(action, self.action_type)


class HARLPolicy:
    """Deterministic HARL actors loaded from a run directory, fed the same padded per-agent observations as in training."""
    ON_POLICY = ('happo', 'mappo', 'haa2c', 'hatrpo')
    OFF_POLICY = ('hasac', 'had3qn')

    def __init__(self, run_dir):
        from gymnasium import spaces
        from harl.algorithms.actors import ALGO_REGISTRY
        with open(os.path.join(run_dir, 'config.json')) as f:
            cfg = json.load(f)
        self.algo = cfg['main_args']['algo']
        if self.algo not in self.ON_POLICY + self.OFF_POLICY:
            raise ValueError(f"HARL algo {self.algo!r} not supported here (discrete-action algos: {self.ON_POLICY + self.OFF_POLICY})")
        self.train_config = cfg['env_args']
        self.include_dc_setpoint = bool(self.train_config.get('include_dc_setpoint', False))
        self.include_pv_obs = bool(self.train_config.get('include_pv_obs', False))
        self.agents = list(self.train_config['agents'])
        model_args, algo_args = cfg['algo_args']['model'], cfg['algo_args']['algo']
        self.recurrent = model_args.get('use_recurrent_policy', False) or model_args.get('use_naive_recurrent_policy', False)
        # RNN state shape only matters for recurrent on-policy actors; off-policy configs (e.g. HAD3QN) name their
        # layers base_hidden_sizes instead of hidden_sizes.
        hidden = model_args.get('hidden_sizes') or model_args.get('base_hidden_sizes') or [1]
        self.rnn_shape = (1, model_args.get('recurrent_n', 1), hidden[-1])

        # Supersuit pads every active agent's observation with zeros up to the largest one.
        n_pv = PV_OBS_DIM * int(self.include_pv_obs)
        obs_dims = {'agent_ls': 26 + n_pv, 'agent_dc': 14 + int(self.include_dc_setpoint) + n_pv, 'agent_bat': 13 + n_pv}
        self.pad_dim = max(obs_dims[a] for a in self.agents)
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.pad_dim,), dtype=np.float32)
        act_space = spaces.Discrete(3)

        models_dir = 'models'
        if not os.path.exists(os.path.join(run_dir, 'models', 'actor_agent0.pt')) and \
                os.path.exists(os.path.join(run_dir, 'models_final', 'actor_agent0.pt')):
            # No evaluation passed the best-model guard (see eval/evals.csv): use the final model.
            print(f'Note: {run_dir} has no best model in models/; using models_final/.')
            models_dir = 'models_final'
        self.actors = []
        for i, _ in enumerate(self.agents):
            actor = ALGO_REGISTRY[self.algo]({**model_args, **algo_args}, obs_space, act_space, device=torch.device('cpu'))
            state = torch.load(os.path.join(run_dir, models_dir, f'actor_agent{i}.pt'), map_location='cpu')
            in_dim = next(v.shape[1] for k, v in state.items() if k.endswith('weight') and v.dim() == 2)
            if in_dim != self.pad_dim:
                raise ValueError(
                    f"{run_dir}: actor {i} expects {in_dim}-dim observations, but the current env gives "
                    f"{self.pad_dim} for this run's config. It was trained on an older observation layout and can't "
                    f"be evaluated on the current simulator (e.g. the 2024 trained_models/ use 14-dim observations).")
            actor.actor.load_state_dict(state)
            actor.actor.eval()
            self.actors.append(actor)

    @classmethod
    def from_live(cls, actors, algo, env_args, model_args, pad_dim):
        """Wrap the actors of a HARL run being trained (in-training evaluation). pad_dim: the padded observation
        size of the training envs."""
        self = cls.__new__(cls)
        self.algo = algo
        if algo not in cls.ON_POLICY + cls.OFF_POLICY:
            raise ValueError(f"HARL algo {algo!r} not supported (discrete-action algos: {cls.ON_POLICY + cls.OFF_POLICY})")
        self.train_config = env_args
        self.include_dc_setpoint = bool(env_args.get('include_dc_setpoint', False))
        self.include_pv_obs = bool(env_args.get('include_pv_obs', False))
        self.agents = list(env_args['agents'])
        self.recurrent = model_args.get('use_recurrent_policy', False) or model_args.get('use_naive_recurrent_policy', False)
        hidden = model_args.get('hidden_sizes') or model_args.get('base_hidden_sizes') or [1]
        self.rnn_shape = (1, model_args.get('recurrent_n', 1), hidden[-1])
        self.pad_dim = pad_dim
        self.actors = list(actors)
        return self

    def reset(self, seed):
        self.rnn_states = [np.zeros(self.rnn_shape, dtype=np.float32) for _ in self.agents]

    def _obs(self, sdc, agent):
        state = {'agent_ls': sdc.ls_state, 'agent_dc': sdc.dc_state, 'agent_bat': sdc.bat_state}[agent]
        if agent == 'agent_dc' and not self.include_dc_setpoint and sdc.include_dc_setpoint:
            state = np.delete(state, 14)  # the eval env appends the setpoint; this policy was trained without it
        padded = np.zeros((1, self.pad_dim), dtype=np.float32)
        padded[0, :len(state)] = state
        return padded

    @torch.no_grad()
    def act(self, sdc):
        actions = dict(NOOP)  # agents this run did not train keep the do-nothing action (as SustainDC does)
        masks = np.ones((1, 1), dtype=np.float32)
        for i, (agent, actor) in enumerate(zip(self.agents, self.actors)):
            obs = self._obs(sdc, agent)
            if self.algo in self.ON_POLICY:
                a, rnn = actor.act(obs, self.rnn_states[i], masks, None, deterministic=True)
                self.rnn_states[i] = rnn.cpu().numpy() if torch.is_tensor(rnn) else rnn
            elif self.algo == 'hasac':
                a = actor.get_actions(obs, None, stochastic=False)
            else:  # had3qn
                a = actor.get_actions(obs, epsilon_greedy=False)
            actions[agent] = int(np.asarray(a.cpu() if torch.is_tensor(a) else a).reshape(-1)[0])
        return actions


def split_label(spec):
    """'<spec>=<label>' -> (spec, label). Controller parameters also use '=' (rule:ls:ls_hi=0.8), so the label is the
    text after the last '=' unless that text is a parameter value: a number, an MPC keyword or a .npz path."""
    from mpc_value import PARAM_WORDS
    head, sep, tail = spec.rpartition('=')
    if not sep or tail in PARAM_WORDS or tail.endswith('.npz'):
        return spec, ''
    try:
        float(tail)
        return spec, ''
    except ValueError:
        return head, tail


def make_policy(spec):
    spec, label = split_label(spec)
    kind, _, rest = spec.partition(':')
    if kind == 'noop':
        policy = NoOpPolicy()
    elif kind == 'random':
        policy = RandomPolicy()
    elif kind == 'sb3':
        run_dir, _, model = rest.partition(':')
        policy = SB3Policy(run_dir, model or 'best_model')
    elif kind == 'harl':
        policy = HARLPolicy(rest)
    elif kind in ('rule', 'oracle', 'setpoint'):
        from baseline_policies import make_baseline_policy
        policy = make_baseline_policy(kind, rest)
    elif kind == 'mpc':
        from mpc_value import make_mpc_policy
        policy = make_mpc_policy(rest)
    else:
        raise ValueError(f'Unknown policy spec {spec!r}')
    return label or spec, policy


def expand_specs(specs):
    """Expand harl:<glob> specs into one spec per run directory, labelled <label>_s<seed>."""
    out = []
    for spec in specs:
        body, label = split_label(spec)
        kind, _, rest = body.partition(':')
        if kind == 'harl' and any(ch in rest for ch in '*?['):
            dirs = sorted(d for d in glob.glob(rest) if os.path.isfile(os.path.join(d, 'config.json')))
            if not dirs:
                raise ValueError(f'No HARL run directories match {rest!r}')
            base = label or os.path.basename(os.path.dirname(dirs[0]))
            for d in dirs:
                with open(os.path.join(d, 'config.json')) as f:
                    seed = json.load(f)['algo_args']['seed']['seed']
                out.append(f'harl:{d}={base}_s{seed}')
        else:
            out.append(spec)
    return out
