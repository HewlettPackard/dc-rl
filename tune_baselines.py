"""Tune the rule-based / oracle controller parameters (baseline_policies.py) on tuning episodes.

The tuning seeds are disjoint from eval_compare's default evaluation seeds (1000+), so the evaluation stays unbiased.
Configurations are ranked by work-adjusted CO2 (unexecuted work charged at its marginal energy x mean CI), and
configurations that drop tasks are excluded: a controller must not reduce CO2 by discarding work.

Run from the repo root, e.g.:
    python tune_baselines.py --location az --out results/tune_baselines/az.json
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import yaml

from baseline_policies import make_baseline_policy
from eval_compare import ENV_CFG, NoOpPolicy, run_episode
from sustaindc_env import SustainDC
from sustaindc_single_env import AGENTS

GRIDS = {
    'setpoint': [f'setpoint:{s}' for s in (15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 21.6)],
    'rule_ls': [f'rule:ls:ls_hi={hi},ls_lo={lo}' for hi in (0.5, 0.6, 0.7, 0.8, 0.9) for lo in (0.2, 0.3, 0.4, 0.5, 0.6) if lo < hi],
    'oracle_ls': [f'oracle:ls:ls_hi={hi},ls_lo={lo}' for hi in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for lo in (0.1, 0.2, 0.3, 0.4, 0.5) if lo < hi],
    'rule_bat': [f'rule:bat:bat_lo={lo},bat_hi={hi}' for lo in (0.2, 0.3, 0.4) for hi in (0.6, 0.7, 0.8)],
}


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--location', required=True)
    p.add_argument('--month', type=int, default=None)
    p.add_argument('--days_per_episode', type=int, default=7)
    p.add_argument('--seeds', type=int, nargs='+', default=[2000, 2001, 2002, 2003, 2004])
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)

    with open(ENV_CFG) as f:
        scenario = yaml.safe_load(f)
    scenario.update(location=args.location, days_per_episode=args.days_per_episode, agents=list(AGENTS),
                    include_dc_setpoint=False)
    if args.month is not None:
        scenario['month'] = args.month
    sdc = SustainDC(dict(scenario))

    base = [run_episode(sdc, NoOpPolicy(), s)[0] for s in args.seeds]
    results = {}
    for family, specs in GRIDS.items():
        for spec in specs:
            pol = make_baseline_policy(*spec.split(':', 1))
            eps = [run_episode(sdc, pol, s)[0] for s in args.seeds]
            co2 = float(np.mean([100 * (e['co2_work_adjusted'] / b['co2_work_adjusted'] - 1) for e, b in zip(eps, base)]))
            dropped = float(np.sum([e['tasks_dropped'] for e in eps]))
            queued = float(np.mean([e['tasks_queued_end'] for e in eps]))
            results[spec] = {'family': family, 'co2_pct': co2, 'tasks_dropped': dropped, 'tasks_queued_end': queued}
            print(f"{spec:42s} work-adj. CO2 {co2:+6.2f}%  dropped {dropped:5.0f}  queued at end {queued:6.1f}", flush=True)

    best = {}
    for family in GRIDS:
        ok = {s: r for s, r in results.items() if r['family'] == family and r['tasks_dropped'] == 0}
        if ok:
            spec = min(ok, key=lambda s: ok[s]['co2_pct'])
            best[family] = {'spec': spec, **ok[spec]}
            print(f"best {family}: {spec} ({ok[spec]['co2_pct']:+.2f}% CO2)")
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'scenario': scenario, 'seeds': args.seeds, 'results': results, 'best': best}, f, indent=2)


if __name__ == '__main__':
    main()
