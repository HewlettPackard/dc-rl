"""Compare policies (MARL, single-agent, baselines) on the same fixed set of SustainDC evaluation episodes.

Rewards are not comparable across setups (reward sharing, the global energy-normalization history), so every
policy is scored on physical metrics instead: CO2, IT / HVAC / total energy, water, dropped tasks and tasks
still queued at the end of the episode.

Each episode is fixed by its seed: `random` and `np.random` are seeded right before SustainDC.reset(), which is
where all of the simulator's randomness is drawn (start day/hour, workload/weather noise, initial task queue).
The harness checks that the exogenous inputs (workload, weather, carbon intensity) match across policies.

Run from the repo root, e.g.:
    python eval_compare.py --policy noop \\
        --policy sb3:results/sustaindc/ca/single_ppo/ppo_v1/seed-00001-job123 \\
        --policy harl:trained_models/sustaindc/ca/happo/happo3agents/seed-00001-2024-06-03-17-55-39 \\
        --n_episodes 5 --location ca --month 6

Policy specs:
    noop                          do nothing (ls: no shifting, dc: hold setpoint, bat: idle)
    random                        uniformly random actions (seeded per episode)
    sb3:<run_dir>[:<model>]       a train_single_agent.py run; <model> defaults to best_model
    harl:<run_dir>                a HARL run directory containing config.json and models/actor_agent*.pt;
                                  a glob (quote it) expands to one policy per run, labelled <label>_s<seed>
    rule:<subsystems>[:<params>]  rule-based controllers (past CI only), e.g. rule:ls,dc,bat or rule:bat:bat_lo=0.2
    oracle:<subsystems>[:<params>] perfect-forecast controllers (true future CI/workload/weather), e.g. oracle:ls,dc,bat
    mpc[:<params>]                MPC with a terminal value, e.g. mpc:H=16,value=<value.npz>,forecast=realistic
                                  (see mpc_value.py)
    setpoint:<C>                  hold the CRAC setpoint at <C>, everything else does nothing
Add =<label> to any spec to name it in the table, e.g. harl:<dir>=happo. See baseline_policies.py for the controllers.

Each policy is compared with `noop` episode by episode (paired), which removes the episode-to-episode variation:
the table reports the mean change and its 95% confidence interval over episodes. Runs labelled <label>_s<seed> are
also pooled per <label> (episode-wise mean over seeds).

Several result folders (e.g. from parallel jobs on the same seeds) can be merged afterwards:
    python eval_compare.py --combine results/eval_compare/a results/eval_compare/b --out_dir results/eval_compare/all
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch
import yaml

from sdc_eval.constants import ENV_CFG, FINAL_EVAL_SEED, NOOP, SCENARIO_KEYS, STEP_HOURS
from sdc_eval.metrics import METRICS, PAIRED_METRICS, aggregate, marginal_kwh_per_util_step
from sdc_eval.paths import add_out_root_arg, report_dir, resolve_root
from sdc_eval.policies import (HARLPolicy, NoOpPolicy, RandomPolicy, SB3Policy, expand_specs, make_policy,
                               split_label)
from sdc_eval.decisions import DecisionRecorder, battery_summary, frame as decision_frame
from sdc_eval.rollout import run_episode
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

# Columns printed in the terminal table (all METRICS are saved to episodes.csv and summary.json).
TABLE_METRICS = [(n, u) for n, u in METRICS[:13]] + [('pue', '-'), ('bat_cycles', 'cycles')]


def paired_report(results, seeds, base='noop'):
    """Episode-by-episode comparison with the baseline: mean % change in CO2 / energy / water with a 95% CI (t-dist),
    plus pooled rows for runs labelled <label>_s<seed>. Returns a dict for summary.json."""
    from scipy import stats
    if base not in results:
        return {}
    b = results[base]

    def pct(eps, key):
        return np.array([100 * (e[key] / be[key] - 1) for e, be in zip(eps, b)])

    groups = {}
    for label in results:
        if label != base and '_s' in label and label.rsplit('_s', 1)[1].isdigit():
            groups.setdefault(label.rsplit('_s', 1)[0], []).append(label)
    rows = {label: results[label] for label in results if label != base}
    for g, labels in groups.items():
        if len(labels) > 1:  # pooled: per episode, average the metric over the runs (seeds) of the group
            rows[f'{g} (mean of {len(labels)} runs)'] = [
                {k: float(np.mean([results[l][i][k] for l in labels])) for k in results[labels[0]][i]}
                for i in range(len(seeds))]

    report = {}
    print(f'\nPaired comparison with {base} over {len(seeds)} episodes (mean change, 95% CI; negative = less):')
    print(f"  {'policy':44s}{'CO2 (work-adjusted)':>24s}{'CO2':>24s}{'total energy':>24s}{'water':>24s}{'dropped':>10s}{'queued end':>12s}")
    for label, eps in rows.items():
        cells, rep = [], {}
        for key in PAIRED_METRICS:
            d = pct(eps, key)
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
            rep[key] = {'mean_pct': float(d.mean()), 'ci95_pct': float(half), 'per_episode_pct': d.tolist()}
            cells.append(f'{d.mean():+7.2f}% ± {half:5.2f}')
        rep['tasks_dropped'] = float(np.mean([e['tasks_dropped'] for e in eps]))
        rep['tasks_queued_end'] = float(np.mean([e['tasks_queued_end'] for e in eps]))
        report[label] = rep
        print(f"  {label[:44]:44s}" + ''.join(f'{c:>24s}' for c in cells)
              + f"{rep['tasks_dropped']:10.0f}{rep['tasks_queued_end']:12.0f}")
    return report


def combine(dirs, out_dir):
    """Merge episodes.csv files from several eval_compare runs on the same seeds and report them together."""
    import csv
    results, fps, seeds = {}, {}, None
    for d in dirs:
        with open(os.path.join(d, 'episodes.csv')) as f:
            for row in csv.DictReader(f):
                label = row['policy']
                # Metrics added later are missing from older CSVs: read them as NaN.
                results.setdefault(label, {})[int(row['seed'])] = {n: float(row.get(n) or 'nan') for n, _ in METRICS}
                fps.setdefault(int(row['seed']), set()).add(row['inputs_fingerprint'])
    seeds = sorted(next(iter(results.values())))
    if any(sorted(r) != seeds for r in results.values()):
        raise ValueError('Policies were evaluated on different seeds; cannot combine.')
    same_inputs = all(len(v) == 1 for v in fps.values())
    print(f"Combined {len(results)} policies from {len(dirs)} folders, {len(seeds)} episodes (seeds {seeds[0]}..{seeds[-1]}). "
          f"Exogenous inputs identical across policies: {'yes' if same_inputs else 'NO, comparison is not fair'}")
    results = {label: [r[s] for s in seeds] for label, r in results.items()}
    summary = {label: aggregate(eps) for label, eps in results.items()}
    print_table(summary)
    paired = paired_report(results, seeds)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'sources': dirs, 'seeds': seeds, 'inputs_identical': same_inputs, 'summary': summary,
                   'paired_vs_noop': paired}, f, indent=2)
    print(f'\nSaved {out_dir}/summary.json')


def print_battery(battery):
    """How each policy uses the battery (from its decisions; sdc_eval.decisions.battery_summary)."""
    print('\nBattery use (per episode; "effective" = share of charge/discharge decisions that moved energy):')
    print(f"  {'policy':34s}{'charge':>8s}{'disch.':>8s}{'idle':>7s}{'eff. ch':>9s}{'eff. dis':>9s}"
          f"{'in MWh':>8s}{'out MWh':>8s}{'SoC min-max':>13s}{'CI ch/dis g/kWh':>18s}{'net CO2 kg':>12s}")
    for label, b in battery.items():
        if not b:
            continue
        soc = f"{b['soc_min']:.2f}-{b['soc_max']:.2f}"
        ci = f"{b['ci_when_charging']:.0f}/{b['ci_when_discharging']:.0f}"
        print(f"  {label[:34]:34s}{b['share_charge']:8.1%}{b['share_discharge']:8.1%}{b['share_idle']:7.1%}"
              f"{b['charge_effective']:9.1%}{b['discharge_effective']:9.1%}{b['charged_mwh_per_ep']:8.2f}"
              f"{b['discharged_mwh_per_ep']:8.2f}{soc:>13s}{ci:>18s}{b['net_co2_saved_kg_per_ep']:12.1f}")


def print_table(summary):
    width = max(len(l) for l in summary) + 2
    print('\n' + 'policy'.ljust(width) + ''.join(f'{f"{n} ({u})":>26}' for n, u in TABLE_METRICS))
    for label, s in summary.items():
        print(label.ljust(width) + ''.join(f'{f"{s[n][0]:.1f} ± {s[n][1]:.1f}":>26}' for n, _ in TABLE_METRICS))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument('--policy', action='append', help='Policy spec (repeatable), see above.')
    p.add_argument('--combine', nargs='+', default=None, help='Merge the episodes.csv of these eval_compare folders.')
    p.add_argument('--n_episodes', type=int, default=5)
    p.add_argument('--seed', type=int, default=FINAL_EVAL_SEED, help='Episode i uses seed + i.')
    p.add_argument('--location', default=None)
    p.add_argument('--month', type=int, default=None)
    p.add_argument('--days_per_episode', type=int, default=None)
    p.add_argument('--pv_capacity_kWp', type=float, default=None,
                   help="Override the env config's on-site PV capacity (pv_config.capacity_kWp; 0 = no PV).")
    p.add_argument('--out_dir', default=None, help='Default: <out_root>/eval_compare/<timestamp>')
    p.add_argument('--save_decisions', action=argparse.BooleanOptionalAction, default=True,
                   help='Save every step\'s decisions (actions + context) to <out_dir>/decisions.csv.gz '
                        '(sdc_eval/decisions.py) and print how each policy uses the battery.')
    add_out_root_arg(p)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.combine:
        combine(args.combine, args.out_dir or report_dir(resolve_root(args.out_root), 'eval_compare', 'combined'))
        return
    if not args.policy:
        p.error('--policy is required (or use --combine)')

    with open(ENV_CFG) as f:
        scenario = yaml.safe_load(f)
    for key in ['location', 'month', 'days_per_episode']:
        if getattr(args, key) is not None:
            scenario[key] = getattr(args, key)
    if args.pv_capacity_kWp is not None:
        scenario['pv_config'] = {**scenario.get('pv_config', {}), 'capacity_kWp': args.pv_capacity_kWp}
    seeds = [args.seed + i for i in range(args.n_episodes)]

    policies = [make_policy(s) for s in expand_specs(args.policy)]
    for label, pol in policies:
        diff = {k: (pol.train_config[k], scenario.get(k)) for k in SCENARIO_KEYS
                if k in pol.train_config and pol.train_config[k] != scenario.get(k)}
        if diff:
            print(f"Note: '{label}' was trained on a different scenario (trained vs eval): {diff}")

    print(f"Scenario: location={scenario['location']} month={scenario['month']} days={scenario['days_per_episode']} | "
          f"{len(seeds)} episodes, seeds {seeds[0]}..{seeds[-1]}")
    results, fingerprints, decision_frames = {}, {}, []
    for label, pol in policies:
        # Observation flags follow the policy; everything that shapes the episode comes from the scenario.
        sdc = SustainDC({**scenario, 'agents': list(AGENTS), 'include_dc_setpoint': pol.include_dc_setpoint,
                         'include_pv_obs': pol.include_pv_obs})
        results[label], fingerprints[label] = [], []
        episode_rows = []
        for seed in seeds:
            rec = DecisionRecorder(sdc) if args.save_decisions else None
            m, fp = run_episode(sdc, pol, seed, trace=rec)
            results[label].append(m); fingerprints[label].append(fp)
            if rec is not None:
                episode_rows.append((seed, rec.rows))
            print(f"  {label} seed {seed}: CO2 {m['co2']:.1f} kg, dropped {m['tasks_dropped']:.0f}, "
                  f"queued at end {m['tasks_queued_end']:.0f}", flush=True)
        if args.save_decisions:
            decision_frames.append(decision_frame(episode_rows, policy=label))

    ref = next(iter(fingerprints.values()))
    same_inputs = all(fp == ref for fp in fingerprints.values())
    print('\nExogenous inputs identical across policies: ' + ('yes' if same_inputs else 'NO, comparison is not fair'))

    summary = {label: aggregate(eps) for label, eps in results.items()}
    print_table(summary)
    base = next((l for l, pol in policies if isinstance(pol, NoOpPolicy) and type(pol) is NoOpPolicy), None)
    if base:
        print(f'\nChange vs {base} (mean over episodes; negative = less):')
        for label, s in summary.items():
            if label == base:
                continue
            ch = [f"{n} {100 * (s[n][0] / summary[base][n][0] - 1):+.1f}%" for n in ('co2', 'total_energy', 'water')
                  if summary[base][n][0]]
            print(f"  {label}: {', '.join(ch)} | dropped {s['tasks_dropped'][0]:.0f} vs {summary[base]['tasks_dropped'][0]:.0f}, "
                  f"queued at end {s['tasks_queued_end'][0]:.0f} vs {summary[base]['tasks_queued_end'][0]:.0f}")


    paired = paired_report(results, seeds) if any(l == 'noop' for l, _ in policies) else {}
    battery = {}
    if decision_frames:
        battery = {label: battery_summary(df) for (label, _), df in zip(policies, decision_frames)}
        print_battery(battery)

    out_dir = args.out_dir or report_dir(resolve_root(args.out_root), 'eval_compare')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'episodes.csv'), 'w') as f:
        f.write('policy,seed,' + ','.join(n for n, _ in METRICS) + ',inputs_fingerprint\n')
        for label, eps in results.items():
            for seed, e, fp in zip(seeds, eps, fingerprints[label]):
                f.write(f'{label},{seed},' + ','.join(f'{e[n]:.6g}' for n, _ in METRICS) + f',{fp}\n')
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'scenario': scenario, 'seeds': seeds, 'policies': args.policy,
                   'inputs_identical': same_inputs, 'summary': summary, 'paired_vs_noop': paired,
                   'battery_usage': battery}, f, indent=2, default=float)
    if decision_frames:
        pd.concat(decision_frames, ignore_index=True).to_csv(os.path.join(out_dir, 'decisions.csv.gz'), index=False)
    print(f'\nSaved {out_dir}/episodes.csv and summary.json' + (' and decisions.csv.gz' if decision_frames else ''))


if __name__ == '__main__':
    main()
