"""Learn the terminal value of mpc_value.MPCPolicy by approximate policy iteration.

    iteration 0: run the MPC with the naive terminal value on training episodes, recording states and stage costs
    fit V_1 (load-shifting and battery heads) to the n-step average-cost returns of that data
    iteration 1: run the MPC with V_1, fit V_2 on the new data, ...

Policy evaluation (per head i in {ls, bat}), with the stage costs d_i defined in mpc_value.py:
    target(s_t) = sum_{k<n} (d_i[t+k] - rho_i) + V(s_{t+n}),   rho_i = mean d_i over the data
V is refit `--fits` times on the same data, each time bootstrapping from the previous fit (the first fit bootstraps from
the value the data was collected with). States in the last n steps of an episode are not used as training samples.

Training episodes use seeds --seed, --seed+1, ... in every round (keep them disjoint from the evaluation seeds,
1000..1009 by default).
Each iteration also runs `noop` on the same seeds and reports the paired change in work-adjusted CO2, so the log shows
whether the policy improves from one iteration to the next. The last round only runs the last value (no fit). The value
of the best round on these training episodes is copied to value.npz.

Logging and model selection use the common suite (sdc_eval/), like every other trainer: rollout/* (each training
episode's physical metrics), health/value_fit_<head>/* (rho, target std, R2 train/val), perf/*, and eval/*: each
iteration's MPC (without exploration) is evaluated on the fixed TRAIN_EVAL_SEED episodes (disjoint from the training
and the final eval_compare seeds); value.npz = the value of the best evaluation (lowest work-adjusted CO2 among those
that don't drop more tasks than noop). env_steps = environment steps of MPC data collected so far.

Hyperparameters: flags, or --config <yaml> (sections env: and algo:, algo keys = the flag names) and
--set algo.H=24 (sdc_eval/config.py). Runs go to <out_root>/sustaindc/<location>/mpc/<exp_name>/seed-XXXXX-<run_id>
(out_root: --out_root, else $SDC_OUT_ROOT, else ./results), or to --out_dir if given.

Example (16 CPUs):
    python train_mpc_value.py --location az --iters 4 --episodes 32 --workers 16 --exp_name per_unit
Then evaluate:
    python eval_compare.py --location az --n_episodes 10 --policy noop \\
        --policy mpc:value=<run_dir>/value.npz=mpc_learned
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import yaml

from sdc_eval.constants import ENV_CFG, MPC_TRAIN_SEED, STEP_HOURS, TRAIN_EVAL_SEED
from sdc_eval.metrics import EpisodeAccumulator
from sdc_eval.config import add_config_args, explicit_dests, resolve_hparams
from sdc_eval.evaluator import FixedSeedEvaluator
from sdc_eval.logger import RunLogger
from sdc_eval.paths import resolve_root, run_dir as run_dir_for
from sdc_eval.rollout import reset_reward_history

# Flags that are hyperparameters of the method (config section 'algo').
ALGO_KEYS = ('iters', 'episodes', 'seed', 'H', 'forecast', 'dc_setpoint', 'explore', 'nstep', 'fits', 'val_episodes',
             'init_value', 'form')


# ---------------------------------------------------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------------------------------------------------
def make_env(scenario):
    from sustaindc_env import SustainDC
    from sustaindc_single_env import AGENTS
    return SustainDC({**scenario, 'agents': list(AGENTS), 'include_dc_setpoint': False})


def collect_episode(job):
    """Run one MPC episode and the noop episode with the same seed. Returns recorded arrays and metrics."""
    import mpc_value as M
    from sdc_eval.policies import NoOpPolicy
    from sdc_eval.rollout import run_episode
    torch.set_num_threads(1)
    scenario, seed, mpc_kwargs = job['scenario'], job['seed'], job['mpc']
    sdc = make_env(scenario)
    noop, _ = run_episode(sdc, NoOpPolicy(), seed)

    policy = M.MPCPolicy(record=True, **mpc_kwargs)
    random.seed(seed)
    np.random.seed(seed)
    reset_reward_history()
    sdc.reset()
    policy.reset(seed)
    rec = {k: [] for k in ('queue', 'q', 'soc', 'ctx', 'd_ls', 'd_bat', 'task_cost', 'ci')}
    acc = EpisodeAccumulator(record_inputs=False)  # the common per-episode metrics (sdc_eval.metrics)
    submitted = ci_sum = 0.0
    steps = 0
    plan_time = 0.0
    t0 = time.time()
    while True:
        t_plan = time.time()
        actions = policy.act(sdc)
        plan_time += time.time() - t_plan
        last = policy.last
        _, _, _, truncs, info = sdc.step(actions)
        i = info['agent_ls']
        acc.step(i, actions, (sdc.ls_reward, sdc.dc_reward, sdc.bat_reward))
        ci, coef = i['bat_avg_CI'], last['coef']
        u, w = i['ls_shifted_workload'], i['ls_original_workload']
        d_ls = ci * (M.energy_kwh(coef, u) - M.energy_kwh(coef, w)) / 1e3
        d_ls += M.DROP_PENALTY * i['ls_tasks_dropped'] * M.task_kwh(coef, u) * ci / 1e3
        d_bat = ci * (i['bat_total_energy_with_battery_KWh'] - i['bat_total_energy_without_battery_KWh']) / 1e3
        task_cost = M.task_kwh(coef, u) * ci / 1e3  # kg CO2 of one task run at this step
        for k, v in (('queue', last['queue']), ('q', last['q']), ('soc', last['soc']), ('ctx', last['ctx']),
                     ('d_ls', d_ls), ('d_bat', d_bat), ('task_cost', task_cost), ('ci', ci)):
            rec[k].append(v)
        submitted += w
        ci_sum += ci
        steps += 1
        if truncs['__all__']:
            break
    episode_metrics = acc.finalize(sdc)
    co2_adj = episode_metrics['co2_work_adjusted']
    data = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    # end of episode: queued work is charged like eval_compare.py does (marginal energy x mean CI); stored energy is lost
    data['q_final'] = M.queue_counts(sdc.ls_env)
    battery = sdc.bat_env.battery  # 'soc' entries are usable stored energy (MWh above the SoC floor), as MPCPolicy records
    data['soc_final'] = float(battery.current_load - battery.soc_min * battery.capacity)
    data['eff_discharge'] = float(battery.eff_discharge)
    data['end_task_cost'] = float(M.task_kwh(policy.last['coef'], submitted / steps) * ci_sum / steps / 1e3)
    data['cap'] = float(sdc.bat_env.max_bat_cap)
    metrics = {'seed': seed, 'co2_work_adjusted': co2_adj, 'noop_co2_work_adjusted': noop['co2_work_adjusted'],
               'change_pct': 100 * (co2_adj / noop['co2_work_adjusted'] - 1),
               'dropped': episode_metrics['tasks_dropped'], 'queued_end': episode_metrics['tasks_queued_end'],
               'seconds': time.time() - t0, 'steps': steps, 'plan_ms_per_step': 1e3 * plan_time / steps,
               'episode_metrics': episode_metrics}
    return data, metrics


# ---------------------------------------------------------------------------------------------------------------------
# Value fitting
# ---------------------------------------------------------------------------------------------------------------------
def features(data):
    x_ls = np.hstack([data['queue'], data['ctx']])
    x_bat = np.hstack([data['soc'][:, None], data['ctx']])
    return x_ls, x_bat


def naive_values(data, ci_scale):
    """NaiveValue at recorded states: queued tasks at the past day's mean CI; battery energy at that CI (kg).
    The marginal task energy is not recorded per state, so a constant is used (only the first fit uses this)."""
    ci_mean = data['ctx'][:, 2] * ci_scale
    tasks = data['queue'][:, 7] * 1000
    return tasks * data['task_kwh'] * ci_mean / 1e3, -data['soc'] * data['eff_discharge'] * ci_mean


def nstep_targets(d, v, n, ends):
    """target[t] = sum_{k<n} (d[t+k] - rho) + v[t+n] for t with t + n inside the same episode (ends: episode end
    indices, exclusive). Returns (indices, targets, rho)."""
    rho = d.mean()
    idx, tgt = [], []
    start = 0
    for end in ends:
        dd = d[start:end] - rho
        c = np.concatenate([[0.0], np.cumsum(dd)])
        t = np.arange(start, end - n)
        idx.append(t)
        tgt.append(c[t - start + n] - c[t - start] + v[t + n])
        start = end
    return np.concatenate(idx), np.concatenate(tgt), rho


def per_unit_targets(ep):
    """Monte Carlo targets of the per-unit values for one recorded episode.

    Queue: tasks are tracked first-in first-out (the env pops the oldest first and overdue tasks are the oldest). Each
    task gets the CO2 of the step that ran it (task_cost) or, if still queued at the end, the episode-end cost. For each
    state and age bucket the target is the mean cost of the tasks in that bucket, weighted by their number.
    Battery: stored energy is discharged first-in first-out as well, so the usable energy stored at t (soc_t, MWh above
    the SoC floor) is the next soc_t MWh to be discharged; its target is the mean CI of those discharges times the
    discharge efficiency, i.e. grid CO2 displaced per stored kWh (energy never discharged counts as 0). With an empty
    battery it is that of the next discharge (the marginal value of a small amount of stored energy).
    Returns (ls inputs, ls targets kg/task, ls weights) and (bat inputs, bat targets g per stored kWh).
    """
    import mpc_value as M
    q, cost, ctx, soc, ci = ep['q'], ep['task_cost'], ep['ctx'], ep['soc'], ep['ci']
    T = len(q)
    q_next = np.vstack([q[1:], ep['q_final'][None]])
    length = q.sum(axis=1)
    added = q_next[:, 1]                      # tasks deferred at step t are 1 step old at t+1
    removed = length + added - q_next.sum(axis=1)
    assert np.all(removed >= -1e-9), 'queue bookkeeping out of sync'
    n_ids = int(added.sum() + length[0])
    enq = np.empty(n_ids)
    task_cost = np.full(n_ids, ep['end_task_cost'])
    # tasks already queued at t=0 (empty in practice): enqueue steps from their ages
    ages0 = np.repeat(np.arange(M.AGE_BINS)[::-1], q[0][::-1].astype(int))
    enq[:len(ages0)] = -ages0
    head, tail = 0, len(ages0)
    heads, tails = np.empty(T, int), np.empty(T, int)
    for t in range(T):
        heads[t], tails[t] = head, tail
        r, a = int(round(removed[t])), int(round(added[t]))
        task_cost[head:head + r] = cost[t]
        head += r
        enq[tail:tail + a] = t
        tail += a
    csum = np.concatenate([[0.0], np.cumsum(task_cost)])
    edges = np.array([0, 16, 32, 48, 64, 80, 97, np.inf])
    xs, ys, ws = [], [], []
    for t in range(T):
        h, tl = heads[t], tails[t]
        if tl == h:
            continue
        e = enq[h:tl]                          # ascending: oldest first
        for b in range(M.N_BUCKETS):           # ages in [edges[b], edges[b+1]) <=> enq in (t - edges[b+1], t - edges[b]]
            lo = h + np.searchsorted(e, t - edges[b + 1], side='right')
            hi = h + np.searchsorted(e, t - edges[b], side='right')
            if hi > lo:
                xs.append((b, length[t], t))
                ys.append((csum[hi] - csum[lo]) / (hi - lo))
                ws.append(hi - lo)
    xs = np.array(xs, dtype=float).reshape(-1, 3)
    x_ls = M.per_task_inputs(xs[:, 1], ctx[xs[:, 2].astype(int)]) if len(xs) else np.zeros((0, M.N_BUCKETS + 13))
    # per_task_inputs returns all buckets per row; keep the one of each sample
    if len(xs):
        x_ls = x_ls.reshape(len(xs), M.N_BUCKETS, -1)[np.arange(len(xs)), xs[:, 0].astype(int)]
    # battery
    soc_next = np.append(soc[1:], ep['soc_final'])
    dis = np.clip(soc - soc_next, 0, None)
    D = np.concatenate([[0.0], np.cumsum(dis)])           # cumulative discharge before step t
    P = np.concatenate([[0.0], np.cumsum(dis * ci)])

    def G(x):  # integral of CI over the discharge coordinate (0 beyond the last discharge)
        k = np.clip(np.searchsorted(D, x, side='right') - 1, 0, T - 1)
        return np.where(x >= D[-1], P[-1], P[k] + (x - D[k]) * ci[k])

    s = np.maximum(soc, 0.01 * ep['cap'])
    y_bat = (G(D[:-1] + s) - G(D[:-1])) / s * ep.get('eff_discharge', 1.0)
    x_bat = np.hstack([soc[:, None], ctx])
    return (x_ls, np.array(ys), np.array(ws, dtype=float)), (x_bat, y_bat)


def fit_mlp(x, y, val_mask, hidden=(128, 128), epochs=40, lr=1e-3, batch=None, seed=0, weights=None):
    """Fit a SiLU MLP (standardized inputs/targets, optional sample weights). Returns (numpy params dict, train R^2,
    validation R^2), R^2 weighted like the loss."""
    torch.manual_seed(seed)
    x_mean, x_std = x.mean(0), x.std(0) + 1e-6
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    wt = torch.tensor(w, dtype=torch.float32)[:, None]
    y_mean = float(np.average(y, weights=w))
    y_std = float(np.sqrt(np.average((y - y_mean) ** 2, weights=w))) + 1e-9
    xt = torch.tensor((x - x_mean) / x_std, dtype=torch.float32)
    yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)[:, None]
    dims = [x.shape[1], *hidden, 1]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.SiLU()]
    net = torch.nn.Sequential(*layers[:-1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = np.where(~val_mask)[0]
    batch = batch or int(np.clip(len(tr) // 50, 64, 1024))  # at least ~50 updates per epoch
    for _ in range(epochs):
        perm = np.random.default_rng(seed + _).permutation(tr)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            loss = (wt[b] * (net(xt[b]) - yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        pred = net(xt).numpy()[:, 0]
    yn = yt.numpy()[:, 0]

    def r2(m):
        mu = np.average(yn[m], weights=w[m])
        return float(1 - np.average((pred[m] - yn[m]) ** 2, weights=w[m]) / np.average((yn[m] - mu) ** 2, weights=w[m]))
    params = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': np.array(y_mean), 'y_std': np.array(y_std)}
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    for i, lin in enumerate(linears):
        params[f'W{i}'] = lin.weight.detach().numpy().T.astype(np.float64)
        params[f'b{i}'] = lin.bias.detach().numpy().astype(np.float64)
    return params, r2(~val_mask), (r2(val_mask) if val_mask.any() else float('nan'))


def fit_values(data, ends, prev_value, n, fits, ci_scale, val_episodes, log):
    """Fitted policy evaluation for both heads. prev_value: path of the value used to collect the data, or 'naive'."""
    import mpc_value as M
    x_ls, x_bat = features(data)
    episode_of = np.searchsorted(ends, np.arange(len(x_ls)), side='right')
    val_mask_all = episode_of >= len(ends) - val_episodes
    if prev_value == 'naive':
        v_ls, v_bat = naive_values(data, ci_scale)
    else:
        prev = M.LearnedValue(prev_value)
        v_ls, v_bat = prev.v_ls(x_ls), prev.v_bat(x_bat)
    out = {'ci_scale': np.array(ci_scale)}
    for f in range(fits):
        new = {}
        for head, x, d, v in (('ls', x_ls, data['d_ls'], v_ls), ('bat', x_bat, data['d_bat'], v_bat)):
            idx, y, rho = nstep_targets(d, v, n, ends)
            params, r2_tr, r2_val = fit_mlp(x[idx], y, val_mask_all[idx], seed=f)
            new[head] = M.MLP({f'{head}_{k}': p for k, p in params.items()}, f'{head}_')
            out.update({f'{head}_{k}': p for k, p in params.items()})
            log.append({'fit': f, 'head': head, 'rho': float(rho), 'target_std': float(y.std()),
                        'r2_train': r2_tr, 'r2_val': r2_val, 'samples': int(len(idx))})
            print(f'    fit {f} {head}: rho {rho:+.4f} kg/step, target std {y.std():.2f} kg, '
                  f'R2 train {r2_tr:.3f} val {r2_val:.3f} ({len(idx)} samples)', flush=True)
        v_ls, v_bat = new['ls'](x_ls), new['bat'](x_bat)
    return out


def fit_per_unit(episodes, ci_scale, val_episodes, log):
    """Fit the per-unit values c_ls (kg per queued task) and c_bat (g/kWh of stored energy) to Monte Carlo targets."""
    parts = [per_unit_targets(ep) for ep in episodes]
    out = {'ci_scale': np.array(ci_scale), 'form': np.array('per_unit')}
    n_ep = len(parts)
    for head, k in (('ls', 0), ('bat', 1)):
        xs = [p[k][0] for p in parts]
        x = np.vstack(xs)
        y = np.concatenate([p[k][1] for p in parts])
        w = np.concatenate([p[k][2] for p in parts]) if head == 'ls' else None
        val = np.concatenate([np.full(len(xi), i >= n_ep - val_episodes) for i, xi in enumerate(xs)])
        params, r2_tr, r2_val = fit_mlp(x, y, val, weights=w)
        out.update({f'{head}_{key}': p for key, p in params.items()})
        unit = 'kg/task' if head == 'ls' else 'g/kWh'
        mean = float(np.average(y, weights=w))
        log.append({'head': head, 'target_mean': mean, 'target_std': float(y.std()), 'r2_train': r2_tr,
                    'r2_val': r2_val, 'samples': int(len(y))})
        print(f'    {head}: target mean {mean:.3f} {unit}, std {y.std():.3f}, R2 train {r2_tr:.3f} val {r2_val:.3f} '
              f'({len(y)} samples)', flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument('--location', required=True)
    p.add_argument('--month', type=int, default=None, help='Default: the env config month (6, July).')
    p.add_argument('--days_per_episode', type=int, default=30)
    p.add_argument('--exp_name', default='mpc_value')
    p.add_argument('--run_id', default=None, help='Run directory suffix (default: a timestamp).')
    p.add_argument('--out_root', default=None, help='Output root (default: $SDC_OUT_ROOT if set, else ./results).')
    p.add_argument('--out_dir', default=None, help='Write the run here instead of the standard run directory.')
    add_config_args(p)
    p.add_argument('--iters', type=int, default=4, help='Policy-iteration rounds (data collection + fit).')
    p.add_argument('--episodes', type=int, default=32, help='Training episodes per iteration.')
    p.add_argument('--seed', type=int, default=MPC_TRAIN_SEED, help='Training episodes use seeds seed, seed+1, ...')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--H', type=int, default=16)
    p.add_argument('--forecast', default='realistic')
    p.add_argument('--dc_setpoint', type=float, default=15.0)
    p.add_argument('--explore', type=float, default=0.1, help='Random-action probability while collecting data.')
    p.add_argument('--nstep', type=int, default=96)
    p.add_argument('--fits', type=int, default=3, help='Fitted policy-evaluation rounds per iteration.')
    p.add_argument('--val_episodes', type=int, default=4)
    p.add_argument('--init_value', default='naive', help='Terminal value of iteration 0 (naive or a .npz).')
    p.add_argument('--form', default='per_unit', choices=['per_unit', 'mlp'],
                   help='per_unit: learned per-task / per-MWh values (Monte Carlo); mlp: unstructured n-step values.')
    p.add_argument('--n_eval_episodes', type=int, default=5,
                   help='Fixed-seed evaluation episodes per iteration (seeds TRAIN_EVAL_SEED + i); 0 = no evaluation '
                        '(value.npz then falls back to the best iteration on its own training episodes).')
    p.add_argument('--eval_days', type=int, default=None, help='Days per evaluation episode (default: days_per_episode).')
    p.add_argument('--drop_tolerance', type=float, default=1.0,
                   help='Best-model guard: max mean tasks dropped per eval episode above noop.')
    args = p.parse_args()

    # Hyperparameters: defaults < --config < explicit flags < --set (sdc_eval/config.py)
    with open(ENV_CFG) as f:
        scenario = yaml.safe_load(f)
    scenario['location'] = args.location
    scenario['days_per_episode'] = args.days_per_episode
    if args.month is not None:
        scenario['month'] = args.month
    explicit = explicit_dests(p, sys.argv[1:])
    hp = resolve_hparams(
        {'env': scenario, 'algo': {k: getattr(args, k) for k in ALGO_KEYS}}, args.config, args.overrides,
        explicit=[('algo', k) for k in ALGO_KEYS if k in explicit] +
                 [('env', k) for k in ('location', 'days_per_episode', 'month') if k in explicit])
    scenario = hp['env']
    for k, v in hp['algo'].items():
        setattr(args, k, v)

    out_dir = args.out_dir or run_dir_for(resolve_root(args.out_root), scenario['location'], 'mpc', args.exp_name,
                                          args.seed, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Run dir: {out_dir}', flush=True)
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump({**vars(args), 'scenario': scenario}, f, indent=2)
    rl = RunLogger(out_dir, 'mpc', 'mpc', env_cfg=scenario, seed=args.seed,
                   hparams={'algo': hp['algo'], 'env': scenario,
                            'eval': {'n_eval_episodes': args.n_eval_episodes, 'eval_days': args.eval_days}})
    evaluator = (FixedSeedEvaluator(scenario, out_dir, rl, n_episodes=args.n_eval_episodes, seed=TRAIN_EVAL_SEED,
                                    drop_tolerance=args.drop_tolerance, days_per_episode=args.eval_days)
                 if args.n_eval_episodes > 0 else None)

    from mpc_value import DAY
    sdc = make_env(scenario)
    ci_scale = float(np.mean(sdc.ci_m.original_data))
    del sdc

    value = args.init_value
    history = []
    env_steps = 0
    t_start = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(args.workers) as pool:
        for it in range(args.iters + 1):
            mpc_kwargs = {'H': args.H, 'value': value, 'forecast': args.forecast, 'dc_setpoint': args.dc_setpoint,
                          'explore': args.explore}
            # the same training episodes every round, so rounds are compared episode by episode
            seeds = [args.seed + e for e in range(args.episodes)]
            print(f'Iteration {it}: running the MPC with value={value} on {len(seeds)} episodes', flush=True)
            t0 = time.time()
            results = pool.map(collect_episode, [{'scenario': scenario, 'seed': s, 'mpc': mpc_kwargs} for s in seeds])
            metrics = [m for _, m in results]
            for m in metrics:  # rollout/*: each training episode, in seed order
                env_steps += m['steps']
                rl.episode(m['episode_metrics'], env_steps)
            ch = np.array([m['change_pct'] for m in metrics])
            half = 1.96 * ch.std(ddof=1) / np.sqrt(len(ch)) if len(ch) > 1 else float('nan')
            entry = {'iteration': it, 'value': value, 'explore': mpc_kwargs['explore'], 'env_steps': env_steps,
                     'change_pct_mean': float(ch.mean()), 'change_pct_ci95': float(half),
                     'dropped_mean': float(np.mean([m['dropped'] for m in metrics])),
                     'queued_end_mean': float(np.mean([m['queued_end'] for m in metrics])),
                     'minutes': (time.time() - t0) / 60,
                     'episodes': [{k: v for k, v in m.items() if k != 'episode_metrics'} for m in metrics]}
            print(f'  work-adjusted CO2 vs noop on these episodes: {ch.mean():+.2f}% ± {half:.2f} '
                  f'(dropped {entry["dropped_mean"]:.0f}, queued at end {entry["queued_end_mean"]:.0f}; '
                  f'{entry["minutes"]:.1f} min)', flush=True)
            rl.scalars('rollout', {'co2_work_adjusted_pct_vs_noop': entry['change_pct_mean']}, env_steps)
            rl.scalars('health', {'iteration': it}, env_steps)
            rl.scalars('perf', {'collect_minutes': entry['minutes'], 'wall_time': time.time() - t_start,
                                'plan_ms_per_step': float(np.mean([m['plan_ms_per_step'] for m in metrics]))},
                       env_steps)
            if evaluator is not None:
                # this iteration's value, without exploration, on the fixed evaluation episodes
                spec = (f"mpc:H={args.H},value={value},forecast={args.forecast},dc_setpoint={args.dc_setpoint},"
                        f"explore=0")
                out = evaluator.evaluate(None, env_steps, save_fn=lambda tag, v=value: save_value(out_dir, v),
                                         spec=spec, pool=pool)
                entry['eval'] = {k: out[k] for k in ('co2_work_adjusted', 'co2_work_adjusted_pct_vs_noop',
                                                     'tasks_dropped', 'eligible')}
                print(f"  fixed-seed eval: {out['co2_work_adjusted_pct_vs_noop']:+.2f}% vs noop, dropped "
                      f"{out['tasks_dropped']:.0f}" + (' -> new best' if evaluator.best is not None
                                                         and evaluator.best['env_steps'] == env_steps else ''), flush=True)
            if it == args.iters:
                history.append(entry)
                break
            fit_log = []
            if args.form == 'per_unit':
                params = fit_per_unit([d for d, _ in results], ci_scale, args.val_episodes, fit_log)
                entry['fits'] = fit_log
                log_fits(rl, fit_log, env_steps)
                history.append(entry)
                path = os.path.join(out_dir, f'value_iter{it + 1}.npz')
                np.savez(path, **params)
                value = path
                with open(os.path.join(out_dir, 'history.json'), 'w') as f:
                    json.dump(history, f, indent=2)
                rl.flush()
                continue
            keys = ('queue', 'soc', 'ctx', 'd_ls', 'd_bat')
            data = {k: np.concatenate([d[k] for d, _ in results]) for k in keys}
            ends = np.cumsum([len(d['d_ls']) for d, _ in results])
            # marginal task energy for the naive bootstrap: from the energy model at the mean utilization
            data['task_kwh'] = np.full(len(data['d_ls']), np.nan)
            data['eff_discharge'] = results[0][0]['eff_discharge']
            if value == 'naive':
                import mpc_value as M
                sdc = make_env(scenario)
                random.seed(args.seed)
                np.random.seed(args.seed)
                sdc.reset()
                coef = M.energy_coefficients(sdc, args.dc_setpoint)
                data['task_kwh'][:] = M.task_kwh(coef, 0.5)
                del sdc
            params = fit_values(data, ends, value, args.nstep, args.fits, ci_scale, args.val_episodes, fit_log)
            entry['fits'] = fit_log
            log_fits(rl, [f for f in fit_log if f['fit'] == args.fits - 1], env_steps)  # the last fit of each head
            history.append(entry)
            path = os.path.join(out_dir, f'value_iter{it + 1}.npz')
            np.savez(path, **params)
            value = path
            with open(os.path.join(out_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            rl.flush()

    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print('\nIteration summary (work-adjusted CO2 vs noop; training episodes | fixed-seed evaluation):')
    for e in history:
        ev = e.get('eval')
        print(f"  iter {e['iteration']} value={os.path.basename(str(e['value']))} explore={e['explore']}: "
              f"{e['change_pct_mean']:+.2f}% ± {e['change_pct_ci95']:.2f}"
              + (f" | {ev['co2_work_adjusted_pct_vs_noop']:+.2f}%" if ev else ''))
    if evaluator is None:
        # No evaluation episodes: the best iteration on its own training episodes (all collected with exploration).
        best = min(history, key=lambda e: e['change_pct_mean'])
        save_value(out_dir, best['value'])
    rl.close(evaluator.best if evaluator is not None else None)
    if os.path.exists(os.path.join(out_dir, 'value.npz')):
        print(f'Saved {out_dir}/value.npz')
    else:
        print('The naive value was best (or no evaluation passed the drop guard); no value.npz saved.')


def save_value(out_dir, value):
    """value.npz = the given iteration's value; the naive value has no file, so value.npz is removed."""
    import shutil
    target = os.path.join(out_dir, 'value.npz')
    if value == 'naive':
        if os.path.exists(target):
            os.remove(target)
    else:
        shutil.copy(value, target)


def log_fits(rl, fit_log, env_steps):
    """health/value_fit_<head>/*: diagnostics of the value fit made from this iteration's data."""
    for f in fit_log:
        rl.scalars(f"health/value_fit_{f['head']}",
                   {k: v for k, v in f.items() if k not in ('head', 'fit')}, env_steps)


if __name__ == '__main__':
    main()
