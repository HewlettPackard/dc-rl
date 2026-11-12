"""Compare training runs of any algorithms (HARL, SB3, MPC) from their common logs (sdc_eval: metrics.csv,
run_meta.json, eval/evals.csv).

    python compare_runs.py '<root>/sustaindc/ca/happo/v1/seed-*' '<root>/sustaindc/ca/single_ppo/v1/seed-*=ppo' \\
        '<root>/sustaindc/ca/mpc/per_unit/seed-*'

Each argument is a run directory or a quoted glob, optionally followed by =<label>. Runs are grouped (default: by
label, else <algo>/<exp_name>) and seeds of a group are averaged: curves show the mean and a 95% t-interval over runs.
--group_by can group by any recorded hyperparameter too, e.g. --group_by algo,algo.learning_rate.

Writes to <out_root>/compare/<timestamp>/ (or --out_dir):
    eval.png        fixed-seed evaluation over env steps: work-adjusted CO2 vs noop (%), tasks dropped per episode
    rollout.png     the training episodes' work-adjusted CO2 and tasks dropped (smoothed)
    health.png      training-health diagnostics present in the runs (entropy, KL, clip fraction, explained
                    variance, gradient norms, value/Q losses, Q values, TD error, value-fit R2, ...)
    summary.csv/md  per group: runs, env steps, final and best eval, mean over training (area under the curve),
                    env steps to reach --threshold % vs noop, tasks dropped, wall time, throughput
    hparams.csv     the hyperparameters that differ between runs
    curves.csv      the plotted group curves (tidy), to re-plot elsewhere
The final comparison of trained models stays eval_compare.py (FINAL_EVAL_SEED episodes, never used for selection).
"""
import argparse
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

from sdc_eval.paths import add_out_root_arg, report_dir, resolve_root

# Reference categorical palette (light mode), fixed order; a group keeps its slot in every panel.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#1f1f1e', '#5f5e5a', '#e6e5e0'

EVAL_PANELS = [('eval/co2_work_adjusted_pct_vs_noop', 'Work-adjusted CO2 vs noop (%)', 'lower is better'),
               ('eval/tasks_dropped', 'Tasks dropped per eval episode', 'lower is better')]
ROLLOUT_PANELS = [('rollout/co2_work_adjusted', 'Training episodes: work-adjusted CO2 (kg)', 'smoothed'),
                  ('rollout/tasks_dropped', 'Training episodes: tasks dropped', 'smoothed')]
HEALTH_PANELS = [  # (key, title, log scale)
    ('health/entropy_norm', 'Policy entropy (1 = uniform)', False),
    ('health/approx_kl', 'Policy change per update (approx. KL)', True),
    ('health/clip_fraction', 'Clip fraction', False),
    ('health/explained_variance', 'Explained variance of the critic', False),
    ('health/grad_norm', 'Actor gradient norm (pre-clip)', True),
    ('health/critic_grad_norm', 'Critic gradient norm (pre-clip)', True),
    ('train/value_loss', 'Value loss', True),
    ('train/q_loss', 'Q loss', True),
    ('health/q_mean', 'Mean Q value', False),
    ('health/td_error_abs', 'Mean |TD error|', True),
    ('health/epsilon', 'Exploration epsilon', False),
    ('health/alpha', 'Entropy temperature alpha', True),
    ('health/value_fit_ls/r2_val', 'MPC value fit R2 (ls, validation)', False),
    ('health/value_fit_bat/r2_val', 'MPC value fit R2 (battery, validation)', False),
    ('perf/sps', 'Env steps per second', True),
]


# ---------------------------------------------------------------------------------------------------------------------
def find_runs(specs):
    runs = []
    for spec in specs:
        pattern, label = spec, None
        if '=' in spec and not os.path.exists(spec):
            pattern, label = spec.rsplit('=', 1)
        dirs = sorted(d for d in glob.glob(os.path.expanduser(pattern))
                      if os.path.isfile(os.path.join(d, 'metrics.csv')))
        if not dirs:
            print(f'Warning: no run directories with metrics.csv match {pattern!r}')
        for d in dirs:
            meta = {}
            if os.path.exists(os.path.join(d, 'run_meta.json')):
                with open(os.path.join(d, 'run_meta.json')) as f:
                    meta = json.load(f)
            exp = os.path.basename(os.path.dirname(os.path.abspath(d)))
            runs.append({'dir': d, 'label': label or f"{meta.get('algo', '?')}/{exp}", 'algo': meta.get('algo'),
                         'exp': exp, 'seed': meta.get('seed'), 'hparams': meta.get('hparams', {}), 'meta': meta,
                         'df': pd.read_csv(os.path.join(d, 'metrics.csv'))})
    return runs


def group_key(run, fields):
    parts = []
    for f in fields:
        if f in ('label', 'algo', 'exp', 'seed'):
            parts.append(str(run[f]))
        else:
            parts.append(f"{f.split('.')[-1]}={run['hparams'].get(f)}")
    return ' '.join(parts)


def series(run, key):
    s = run['df'][run['df']['key'] == key]
    if s.empty:
        return None
    s = s.groupby('env_steps', as_index=False)['value'].mean().sort_values('env_steps')
    return s['env_steps'].to_numpy(float), s['value'].to_numpy(float)


def smooth(y, frac=0.05):
    w = max(int(len(y) * frac), 1)
    return pd.Series(y).rolling(w, min_periods=1, center=True).mean().to_numpy() if w > 1 else y


def t_half(values):
    from scipy import stats
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float('nan')
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def group_curve(runs, key, n_grid=200, smooth_frac=0.0):
    """Mean and 95% CI over runs on a common env-step grid (the steps all runs reached)."""
    per_run = [s for s in (series(r, key) for r in runs) if s is not None and len(s[0])]
    if not per_run:
        return None
    lo = max(x[0] for x, _ in per_run)
    hi = min(x[-1] for x, _ in per_run)
    if hi < lo:
        return None
    if len(per_run) == 1:  # keep the run's own points
        x, y = per_run[0]
        y = smooth(y, smooth_frac) if smooth_frac else y
        return x, y, np.full_like(y, np.nan), 1
    pts = sorted(set(np.concatenate([x[(x >= lo) & (x <= hi)] for x, _ in per_run])))
    grid = np.array(pts) if len(pts) <= n_grid else np.linspace(lo, hi, n_grid)
    ys = np.array([np.interp(grid, x, smooth(y, smooth_frac) if smooth_frac else y) for x, y in per_run])
    half = np.array([t_half(ys[:, j]) for j in range(len(grid))])
    return grid, ys.mean(0), half, len(per_run)


# ---------------------------------------------------------------------------------------------------------------------
def style_axes(ax, title, subtitle=None, log=False):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=10, color=INK, pad=14 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, color=INK_2, va='bottom')
    ax.set_xlabel('env steps', fontsize=8, color=INK_2)
    if log:
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(_step_fmt())


def _step_fmt():
    from matplotlib.ticker import FuncFormatter

    def f(v, _):
        for div, suf in ((1e9, 'B'), (1e6, 'M'), (1e3, 'k')):
            if abs(v) >= div:
                return f'{v / div:g}{suf}'
        return f'{v:g}'
    return FuncFormatter(f)


def plot_panels(groups, colors, panels, path, smooth_frac=0.0, curves_out=None, ref_zero=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    avail = []
    for p in panels:
        curves = {g: group_curve(rs, p[0], smooth_frac=smooth_frac) for g, rs in groups.items()}
        curves = {g: c for g, c in curves.items() if c is not None}
        if curves:
            avail.append((p, curves))
    if not avail:
        return []
    ncol = min(3, len(avail))
    nrow = int(np.ceil(len(avail) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow + 0.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    direct = len(groups) <= 4
    from matplotlib.lines import Line2D
    for ax, (p, curves) in zip(axes.flat, avail):
        ends = []  # (group, x, y) of each line's last point, for the direct labels
        key, title = p[0], p[1]
        log = p[2] is True if len(p) > 2 and isinstance(p[2], bool) else False
        subtitle = p[2] if len(p) > 2 and isinstance(p[2], str) else None
        style_axes(ax, title, subtitle, log=log)
        if key in ref_zero:
            ax.axhline(0, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        for g, (x, y, half, n) in curves.items():
            c = colors[g]
            if log:
                y = np.where(y > 0, y, np.nan)
            ax.plot(x, y, color=c, linewidth=2, solid_capstyle='round', label=f'{g} (n={n})')
            if n > 1:
                lo_b, hi_b = y - half, y + half
                if log:
                    lo_b = np.where(lo_b > 0, lo_b, np.nan)
                ax.fill_between(x, lo_b, hi_b, color=c, alpha=0.15, linewidth=0)
            if len(x) == 1:
                ax.plot(x, y, 'o', color=c, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
            if direct and np.isfinite(y).any():
                j = np.where(np.isfinite(y))[0][-1]
                ends.append((g, x[j], y[j]))
            if curves_out is not None:
                curves_out.extend({'panel': key, 'group': g, 'env_steps': xi, 'mean': yi, 'ci95': hi, 'runs': n}
                                  for xi, yi, hi in zip(x, y, half))
        place_end_labels(fig, ax, ends)
    for ax in list(axes.flat)[len(avail):]:
        ax.set_visible(False)
    legend_in = 0.0
    if len(groups) >= 2:  # one legend for the figure, with every group (whichever panels it appears in)
        n_runs = {g: len(rs) for g, rs in groups.items()}
        handles = [Line2D([], [], color=colors[g], linewidth=2) for g in groups]
        labels = [f'{g} (n={n_runs[g]})' for g in groups]
        ncol = min(len(labels), 4)
        legend_in = 0.25 * int(np.ceil(len(labels) / ncol)) + 0.15
        fig.legend(handles, labels, loc='upper left', ncol=ncol, frameon=False, fontsize=8, labelcolor=INK,
                   bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - legend_in / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return [p[0] for p, _ in avail]


# ---------------------------------------------------------------------------------------------------------------------
def place_end_labels(fig, ax, ends, min_gap_px=11):
    """Direct labels at the end of each line, pushed apart vertically so they don't overlap."""
    if not ends:
        return
    fig.canvas.draw()
    pts = sorted(((ax.transData.transform((x, y))[1], g, x, y) for g, x, y in ends), key=lambda t: t[0])
    placed = []
    for ypix, g, x, y in pts:
        if placed and ypix - placed[-1][0] < min_gap_px:
            ypix = placed[-1][0] + min_gap_px
        placed.append((ypix, g, x, y))
    for ypix, g, x, y in placed:
        y0 = ax.transData.transform((x, y))[1]
        ax.annotate(g, (x, y), xytext=(4, ypix - y0), textcoords='offset pixels', fontsize=7.5, color=INK_2,
                    va='center', annotation_clip=False)


def summarize(groups, threshold):
    rows = []
    for g, runs in groups.items():
        final, best, auc, reach, dropped, wall, sps, steps = [], [], [], [], [], [], [], []
        for r in runs:
            s = series(r, 'eval/co2_work_adjusted_pct_vs_noop')
            el = series(r, 'eval/eligible')
            dr = series(r, 'eval/tasks_dropped')
            if s is not None:
                x, y = s
                final.append(y[-1])
                ok = el[1] > 0.5 if el is not None and len(el[1]) == len(y) else np.ones_like(y, bool)
                best.append(y[ok].min() if ok.any() else np.nan)
                auc.append(np.trapz(y, x) / (x[-1] - x[0]) if len(x) > 1 else y[0])
                hit = np.where(y <= threshold)[0]
                reach.append(x[hit[0]] if len(hit) else np.nan)
            if dr is not None:
                dropped.append(dr[1][-1])
            wt = series(r, 'perf/wall_time')
            if wt is not None:
                wall.append(wt[1].max() / 3600)
            sp = series(r, 'perf/sps')
            if sp is not None:
                sps.append(np.median(sp[1]))
            steps.append(r['df']['env_steps'].max())
        n_reach = int(np.sum(~np.isnan(reach))) if reach else 0
        rows.append({
            'group': g, 'runs': len(runs), 'env_steps': int(np.min(steps)),
            'final_pct_vs_noop': np.nanmean(final) if final else np.nan, 'final_ci95': t_half(final),
            'best_pct_vs_noop': np.nanmean(best) if best else np.nan, 'best_ci95': t_half(best),
            'mean_over_training_pct': np.nanmean(auc) if auc else np.nan,
            f'steps_to_{threshold:g}pct': np.nanmedian(reach) if n_reach else np.nan,
            'runs_reaching': f'{n_reach}/{len(runs)}',
            'final_tasks_dropped': np.nanmean(dropped) if dropped else np.nan,
            'wall_hours': np.nanmean(wall) if wall else np.nan,
            'median_sps': np.nanmean(sps) if sps else np.nan,
        })
    return pd.DataFrame(rows)


def differing_hparams(runs):
    keys = sorted(set().union(*(r['hparams'].keys() for r in runs)))
    table = pd.DataFrame([{'run': r['dir'], 'group': r['group'], **{k: r['hparams'].get(k) for k in keys}}
                          for r in runs])
    diff = [k for k in keys if table[k].astype(str).nunique() > 1]
    return table[['group', 'run'] + diff]


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument('runs', nargs='+', help="Run directories or quoted globs, each optionally '=<label>'.")
    p.add_argument('--group_by', default='label',
                   help='Comma-separated: label, algo, exp, seed or hyperparameter keys from run_meta.json '
                        '(e.g. algo.learning_rate).')
    p.add_argument('--threshold', type=float, default=-5.0,
                   help='Sample efficiency: env steps until the eval first reaches this %% change vs noop.')
    p.add_argument('--smooth', type=float, default=0.05, help='Rolling-mean window for rollout/health curves, '
                                                               'as a fraction of the points.')
    p.add_argument('--out_dir', default=None, help='Default: <out_root>/compare/<timestamp>')
    add_out_root_arg(p)
    args = p.parse_args()

    runs = find_runs(args.runs)
    if not runs:
        p.error('no runs found')
    fields = [f.strip() for f in args.group_by.split(',') if f.strip()]
    groups = {}
    for r in runs:
        r['group'] = group_key(r, fields)
        groups.setdefault(r['group'], []).append(r)
    if len(groups) > len(PALETTE):
        print(f'Note: {len(groups)} groups but {len(PALETTE)} colors; plotting the first {len(PALETTE)} '
              f'(all are in the tables). Narrow the selection or use --group_by.')
    colors = {g: PALETTE[i] for i, g in enumerate(list(groups)[:len(PALETTE)])}
    plot_groups = {g: groups[g] for g in colors}
    hashes = {r['meta'].get('scenario_hash') for r in runs} - {None}
    if len(hashes) > 1:
        print(f'Warning: the runs were trained on {len(hashes)} different scenarios (scenario_hash); '
              f'curves are not directly comparable.')

    out = args.out_dir or report_dir(resolve_root(args.out_root), 'compare')
    os.makedirs(out, exist_ok=True)
    curves = []
    drawn = plot_panels(plot_groups, colors, EVAL_PANELS, os.path.join(out, 'eval.png'), curves_out=curves,
                        ref_zero=('eval/co2_work_adjusted_pct_vs_noop',))
    drawn += plot_panels(plot_groups, colors, ROLLOUT_PANELS, os.path.join(out, 'rollout.png'),
                         smooth_frac=args.smooth, curves_out=curves)
    drawn += plot_panels(plot_groups, colors, HEALTH_PANELS, os.path.join(out, 'health.png'),
                         smooth_frac=args.smooth, curves_out=curves)
    pd.DataFrame(curves).to_csv(os.path.join(out, 'curves.csv'), index=False)

    summary = summarize(groups, args.threshold)
    summary.to_csv(os.path.join(out, 'summary.csv'), index=False)
    hp = differing_hparams(runs)
    hp.to_csv(os.path.join(out, 'hparams.csv'), index=False)
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write(f'# Run comparison\n\n{len(runs)} runs in {len(groups)} groups. Percentages: fixed-seed '
                f'evaluation, work-adjusted CO2 vs noop (negative = better); ± is a 95% t-interval over runs.\n\n')
        f.write(summary.to_markdown(index=False, floatfmt='.3g') + '\n')
    pd.set_option('display.width', 200)
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print(f'\nPanels drawn: {len(drawn)}. Saved eval.png, rollout.png, health.png, summary.csv/md, hparams.csv, '
          f'curves.csv in {out}')


if __name__ == '__main__':
    main()
