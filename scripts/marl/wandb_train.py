"""Run HARL's train_sustaindc.py with live wandb logging (no changes to the training code).

wandb.init(sync_tensorboard=True) mirrors everything HARL writes with its TensorBoard writer into wandb.
Used by scripts/marl/launch.sh (WANDB=1). Usage, from the root of a SustainDC checkout:
    python scripts/marl/wandb_train.py --algo happo --exp_name baseline_v1 --location az --seed 1 [other train_sustaindc.py args]
Environment variables: WANDB_PROJECT (default sustaindc-rl), WANDB_ENTITY (default: your wandb default entity).
If wandb cannot be reached, the run logs offline; sync later with `wandb sync <dir>`.
"""
import glob
import json
import os
import runpy
import subprocess
import sys

import wandb

argv = sys.argv[1:]
args = dict(zip([a[2:] for a in argv[0::2]], argv[1::2]))
algo, exp, loc, seed = args.get('algo', 'happo'), args.get('exp_name', 'installtest'), args.get('location', 'ca'), args.get('seed', '1')
code_dir = os.getcwd()
commit = subprocess.run(['git', '-C', code_dir, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()

# A job requeued after preemption keeps its Slurm job id: reuse it as the wandb run id so the resumed training continues
# the same wandb run (steps already logged before the checkpoint are skipped by wandb).
job_id = os.environ.get('SLURM_JOB_ID')
init_kwargs = dict(
    id=f"{exp}_{algo}_{loc}_s{seed}_{job_id}" if job_id else None,
    resume='allow' if job_id else None,
    project=os.environ.get('WANDB_PROJECT', 'sustaindc-rl'),
    entity=os.environ.get('WANDB_ENTITY') or None,
    name=f"{exp}_{algo}_{loc}_s{seed}",
    group=f"{exp}/{algo}/{loc}",
    job_type=exp,
    tags=[exp, algo, loc, 'marl'],
    config={'cli_args': args, 'algo': algo, 'location': loc, 'seed': int(seed), 'exp_name': exp,
            'code_commit': commit, 'code_dir': code_dir,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': os.uname().nodename},
    sync_tensorboard=True,
    dir=args.get('log_dir', '.'),
)
try:
    run = wandb.init(mode='online', settings=wandb.Settings(init_timeout=180), **init_kwargs)
except Exception as ex:  # no network on the node: keep training, log offline
    print(f"wandb online init failed ({ex!r}); logging offline", flush=True)
    run = wandb.init(mode='offline', **init_kwargs)
print(f"wandb run: {run.url or run.dir}", flush=True)

exit_code = 0
try:
    sys.path.insert(0, code_dir)
    sys.argv = [os.path.join(code_dir, 'train_sustaindc.py')] + argv
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as ex:
    exit_code = ex.code or 0
except BaseException:
    exit_code = 1
    raise
finally:
    # Attach HARL's full saved config (algo/env/main args) to the wandb run.
    pattern = os.path.join(args.get('log_dir', 'results'), 'sustaindc', loc, algo, exp, f"seed-{int(seed):05d}-*", 'config.json')
    configs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if configs:
        with open(configs[-1]) as f:
            run.config.update({'harl': json.load(f), 'run_dir': os.path.dirname(configs[-1])}, allow_val_change=True)
    run.finish(exit_code=exit_code)
