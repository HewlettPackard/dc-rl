#!/bin/bash
# Score policies on the shared IFT 6162 evaluation protocol, so results from different methods (MPC, MARL,
# single-agent, rules) go into one paired table.
#
# Protocol (fixed here, do not change it per method):
#   10 episodes, seeds 1000..1009, 30 days each, July (month index 6), PV off, one location (az or ny).
#   Every run also scores noop on the same episodes; results are paired % changes vs noop, metric = work-adjusted CO2.
#
# Usage (from anywhere; runs in the repo root):
#   scripts/eval_protocol.sh LOC OUT_DIR SPEC [SPEC ...]
# e.g. a single-agent run (SB3, best_model by default):
#   scripts/eval_protocol.sh az results/eval_protocol/az/single_ppo "sb3:results/sustaindc/az/single_ppo/<exp>/seed-00001-<id>=single_ppo_s1"
# On Slurm:
#   sbatch -c 4 --mem 16G -t 12:00:00 scripts/eval_protocol.sh az <out_dir> "<spec>"
#
# SPEC is any eval_compare.py policy spec (sb3:, harl:, mpc:, rule:, ...); add =<label> to name it in the table.
# Name seeds <label>_s<seed> so they are also pooled per method.
# To merge folders from several runs (same LOC) into one table:
#   python eval_compare.py --combine <out_dir_1> <out_dir_2> ... --out_dir <combined_dir>
set -euo pipefail
if [ $# -lt 3 ]; then
    sed -n '2,21p' "$0"
    exit 1
fi
LOC=$1; OUT=$2; shift 2
case $LOC in az|ny) ;; *) echo "LOC must be az or ny (got $LOC)"; exit 1 ;; esac
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
cd "$(dirname "$0")/.."
POLICIES=(--policy noop)
for spec in "$@"; do POLICIES+=(--policy "$spec"); done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
echo "eval_protocol: $LOC | code $(git rev-parse --short HEAD 2>/dev/null || echo '?') | host $(hostname) | $(date)"
python -u eval_compare.py "${POLICIES[@]}" \
    --location "$LOC" --month 6 --days_per_episode 30 --pv_capacity_kWp 0 \
    --n_episodes 10 --seed 1000 --out_dir "$OUT"
