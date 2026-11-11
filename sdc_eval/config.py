"""Hyperparameters from a config file and/or the command line, the same way for every trainer.

    --config my_run.yaml           a YAML file with sections, e.g.
                                       env:  {location: ny, month: 6, pv_config: {capacity_kWp: 500}}
                                       algo: {learning_rate: 1.0e-4, ent_coef: 0.01, net_arch: [64, 64]}
    --set algo.learning_rate=1e-4  one value (repeatable); dotted path into the sections, value parsed as YAML

Precedence: trainer defaults < --config < the trainer's own flags (--learning_rate, --location, ...) < --set.
Keys must already exist in the defaults (typos fail instead of being silently ignored), except in sections the
trainer declares open (e.g. SB3 'algo', checked against the algorithm's constructor instead).

The resolved values are saved with the run (config.json, and flattened in run_meta.json 'hparams' and TensorBoard's
HParams tab) by sdc_eval.logger.RunLogger.
"""
import copy

import yaml


def add_config_args(parser):
    parser.add_argument('--config', default=None,
                        help='YAML file of hyperparameters with sections env:, algo:, ... (see sdc_eval/config.py).')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='SECTION.KEY=VALUE',
                        help='Override one hyperparameter, e.g. --set algo.learning_rate=1e-4 (repeatable).')


def _numbers(v):
    """PyYAML (YAML 1.1) reads 1e-4 (no decimal point) as a string: turn such strings into floats."""
    if isinstance(v, dict):
        return {k: _numbers(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_numbers(x) for x in v]
    if isinstance(v, str):
        try:
            return float(v) if any(c in v for c in '.eE') else v
        except ValueError:
            return v
    return v


def parse_overrides(items):
    """['algo.lr=1e-4', 'env.pv_config.capacity_kWp=500'] -> {'algo': {'lr': 1e-4}, 'env': {'pv_config': {...}}}"""
    out = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f'--set expects SECTION.KEY=VALUE, got {item!r}')
        path, value = item.split('=', 1)
        keys = path.strip().split('.')
        if len(keys) < 2:
            raise ValueError(f'--set {item!r}: give the section too, e.g. algo.{path}')
        d = out
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = _numbers(yaml.safe_load(value))
    return out


def load_config(path):
    if not path:
        return {}
    with open(path) as f:
        cfg = _numbers(yaml.safe_load(f) or {})
    if not isinstance(cfg, dict):
        raise ValueError(f'{path}: expected a mapping of sections (env:, algo:, ...)')
    return cfg


def merge(base, override, open_sections=(), path=''):
    """Recursively merge `override` into a copy of `base`. Unknown keys raise, except under open sections."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        where = f'{path}.{k}' if path else k
        if k not in out and path.split('.')[0] not in open_sections and where not in open_sections:
            known = ', '.join(sorted(map(str, out))) if isinstance(out, dict) else ''
            raise KeyError(f'Unknown hyperparameter {where!r}. Known keys here: {known}')
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v, open_sections, where)
        else:
            out[k] = v
    return out


def flatten(d, prefix=''):
    """{'algo': {'lr': 1e-4, 'net': [64, 64]}} -> {'algo.lr': 1e-4, 'algo.net': '[64, 64]'} (scalars for HParams)."""
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[key] = v
        else:
            out[key] = str(v)
    return out


def explicit_dests(parser, argv):
    """argparse dests whose flags were given on the command line (as opposed to left at their defaults)."""
    given = {a.split('=', 1)[0] for a in argv if a.startswith('-')}
    return {a.dest for a in parser._actions if any(o in given for o in a.option_strings)}


def resolve_hparams(defaults, config_path, overrides, explicit=(), open_sections=()):
    """defaults < --config file < explicitly given flags < --set.

    defaults: {section: {key: value}} already holding the command-line values; explicit: (section, key) pairs that
    were given as flags, so a config file does not override them.
    """
    cfg = merge(defaults, load_config(config_path), open_sections)
    for section, key in explicit:
        cfg[section][key] = copy.deepcopy(defaults[section][key])
    return merge(cfg, parse_overrides(overrides), open_sections)
