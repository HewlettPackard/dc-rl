"""Checks the resumable HARL checkpoints (harl/utils/checkpoint.py) for HAPPO, MAPPO and HAD3QN.
Run from the repo root: python tests/verify_checkpoint_resume.py"""
import glob, os, shutil, signal, subprocess, sys, tempfile, time
sys.path.insert(0, os.getcwd())
import numpy as np, torch

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.utils import checkpoint as ckpt
from harl.runners import RUNNER_REGISTRY

TMP = tempfile.mkdtemp(prefix='ckpt_test_', dir=os.environ.get('SLURM_TMPDIR') or None)
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


ON = dict(num_env_steps=96 * 6, episode_length=96, eval_interval=2, hidden_sizes=[16, 16])
OFF = dict(num_env_steps=1200, warmup_steps=200, train_interval=100, eval_interval=500, buffer_size=5000,
           batch_size=32, base_hidden_sizes=[16, 16])
COMMON = dict(n_rollout_threads=1, n_eval_rollout_threads=1, eval_episodes=1, days_per_episode=1, location='az',
              log_dir=TMP, seed=1)


def make_runner(algo, run_id, interval, **extra):
    algo_args, env_args = get_defaults_yaml_args(algo, 'sustaindc')
    update_args({**COMMON, **(OFF if algo == 'had3qn' else ON), **extra}, algo_args, env_args)
    args = {'algo': algo, 'env': 'sustaindc', 'exp_name': 'ckpt_test', 'load_config': '', 'run_id': run_id,
            'checkpoint_interval': interval}
    return RUNNER_REGISTRY[algo](args, algo_args, env_args)


def same_state(a, b):
    """Recursively compare state_dicts / tensors / scalars."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    return a == b


def steps_in_progress(run_dir):
    out = []
    for line in open(os.path.join(run_dir, 'progress.txt')):
        try:
            out.append(float(line.split(',', 1)[0]))
        except ValueError:
            pass
    return out


def progress_ok(steps, off):
    """Steps never go back, and no step is logged by two training segments. (Off-policy runs write each evaluation
    twice, from the runner and from the logger, so a step may appear twice there.)"""
    from collections import Counter
    return steps == sorted(steps) and max(Counter(steps).values(), default=0) <= (2 if off else 1)


for algo in ('happo', 'mappo', 'had3qn'):
    off = algo == 'had3qn'
    interval = 400 if off else 192
    print(f"{algo}: train with periodic checkpoints, then resume")
    r = make_runner(algo, 'r1', interval)
    run_dir = r.run_dir
    r.run()
    r.close()
    ckpts = ckpt.list_checkpoints(run_dir)
    expect_last = 1200 if off else 576  # newest multiple of the interval reached (warmup 200 + 1200 steps)
    check("checkpoints saved, only the newest 2 kept", len(ckpts) == 2 and ckpts[-1][0] == expect_last,
          f"{[s for s, _ in ckpts]}")
    if off:
        check("replay buffer saved with each checkpoint",
              all(os.path.exists(os.path.join(run_dir, 'checkpoints', f'buffer_{s}.npz')) for s, _ in ckpts))
    check("no temporary files left", not glob.glob(os.path.join(run_dir, 'checkpoints', '*.tmp*')))

    # resume from the older checkpoint (as if the job had been killed before the last one): drop the newest
    steps_old, path_old = ckpts[0]
    os.remove(ckpts[-1][1])
    saved = torch.load(path_old, map_location='cpu', weights_only=False)
    r2 = make_runner(algo, 'r1', interval)
    check("resumes from the newest remaining checkpoint", r2.resume_path == path_old, f"{steps_old} steps")
    ok = all(same_state(saved['actors'][i][name][1], getattr(a, name).state_dict())
             for i, a in enumerate(r2.actor) for name in saved['actors'][i] if saved['actors'][i][name][0] == 'state_dict')
    check("actor networks and optimizers (and target nets) restored exactly", ok)
    ok = all(same_state(val, getattr(r2.critic, name).state_dict())
             for name, (kind, val) in saved['critic'].items() if kind == 'state_dict')
    check("critic network and optimizer restored exactly", ok)
    if off:
        with np.load(os.path.join(run_dir, 'checkpoints', f'buffer_{steps_old}.npz')) as z:
            n = int(z['__cur_size'])
            ok = n == r2.buffer.cur_size and int(z['__idx']) == r2.buffer.idx and \
                np.array_equal(z['rewards'], r2.buffer.rewards[:n]) and \
                np.array_equal(z['obs__0'], r2.buffer.obs[0][:n])
        check("replay buffer restored (contents, size, cursor)", ok, f"{n} transitions")
        check("update counter restored", r2.total_it == saved['total_it'], f"{r2.total_it}")
        check("loop resumes after the checkpointed step", r2.resume_step == saved['step'])
    else:
        check("loop resumes after the checkpointed iteration", r2.resume_episode == saved['episode'])
    check("progress.txt trimmed to the checkpoint", max(steps_in_progress(run_dir) or [0]) <= steps_old)
    r2.run()
    r2.close()
    steps = steps_in_progress(run_dir)
    check("resumed run finishes; progress.txt stays monotonic without duplicates",
          progress_ok(steps, off) and ckpt.latest_checkpoint(run_dir)[0] == expect_last,
          f"{len(steps)} rows, last checkpoint {ckpt.latest_checkpoint(run_dir)[0]}")

    try:
        make_runner(algo, 'r1', interval, lr=0.123)
        refused = False
    except SystemExit as ex:
        refused = 'Refusing to resume' in str(ex)
    check("refuses to resume with a different configuration", refused)

print("Real preemption: SIGKILL a training process, rerun the same command")
cmd = [sys.executable, 'train_sustaindc.py', '--algo', 'had3qn', '--exp_name', 'ckpt_kill', '--run_id', 'k1',
       '--checkpoint_interval', '400', '--log_dir', TMP] + \
      sum([[f'--{k}', str(v)] for k, v in {**COMMON, **OFF, 'num_env_steps': 2400}.items() if k != 'log_dir'], [])
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
pattern = os.path.join(TMP, 'sustaindc', '*', 'had3qn', 'ckpt_kill', 'seed-00001-k1', 'checkpoints', 'ckpt_*.pt')
t0 = time.time()
while not glob.glob(pattern) and proc.poll() is None and time.time() - t0 < 900:
    time.sleep(1)
time.sleep(2)
killed = proc.poll() is None
if killed:
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
check("process killed after its first checkpoint", killed and glob.glob(pattern))
out = subprocess.run(cmd, capture_output=True, text=True)
check("rerun resumes from the checkpoint and completes", out.returncode == 0 and 'Resuming from' in out.stdout,
      f"exit {out.returncode}")
kd = os.path.dirname(os.path.dirname(glob.glob(pattern)[0]))
steps = steps_in_progress(kd)
check("progress.txt monotonic across the kill", progress_ok(steps, True), f"{len(steps)} rows")

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
