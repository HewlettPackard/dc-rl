"""Where runs and reports are written.

The output root is, first match wins:
    1. --out_root <dir> on the command line
    2. the SDC_OUT_ROOT environment variable (e.g. export SDC_OUT_ROOT=$SCRATCH/sustaindc-rl)
    3. ./results

Layout under the root, the same for every algorithm:
    <root>/sustaindc/<location>/<algo>/<exp_name>/seed-XXXXX-<run_id>/  one training run (algo: happo, single_ppo, mpc, ...)
    <root>/eval_compare/<timestamp>/                                eval_compare.py reports
    <root>/compare/<timestamp>/                                     compare_runs.py reports
"""
import datetime
import os

ENV_VAR = 'SDC_OUT_ROOT'
DEFAULT_ROOT = 'results'
RUNS_SUBDIR = 'sustaindc'  # same as the existing HARL and SB3 layouts, so older runs and resumes keep working


def resolve_root(cli_value=None):
    root = cli_value or os.environ.get(ENV_VAR) or DEFAULT_ROOT
    return os.path.abspath(os.path.expanduser(os.path.expandvars(root)))


def add_out_root_arg(parser, *aliases):
    """Add --out_root (plus optional legacy aliases such as --log_dir) to an argparse parser."""
    parser.add_argument('--out_root', *aliases, dest='out_root', default=None,
                        help=f'Output root (default: ${ENV_VAR} if set, else ./{DEFAULT_ROOT}).')


def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def run_dir(root, location, algo, exp_name, seed, run_id=None):
    return os.path.join(root, RUNS_SUBDIR, location, algo, exp_name, f'seed-{seed:0>5}-{run_id or timestamp()}')


def report_dir(root, kind, name=None):
    """kind: 'eval_compare' or 'compare'."""
    return os.path.join(root, kind, name or timestamp())
