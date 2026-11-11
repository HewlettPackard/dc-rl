"""Common evaluation and training-metrics suite shared by every algorithm (HARL, SB3, MPC, baselines).

    constants  scenario file, do-nothing action, step length, seed sets
    paths      where runs and reports are written (--out_root > $SDC_OUT_ROOT > ./results)
    metrics    the metric registry (names, units) and the per-episode accumulator
    rollout    run_episode(): one seeded episode of a policy on SustainDC
    policies   the Policy interface and the policy specs used by eval_compare.py
    logger     RunLogger: TensorBoard + metrics.csv + run_meta.json, one schema for every algorithm
    evaluator  FixedSeedEvaluator: in-training evaluation on fixed seeds and best/final model selection
"""
