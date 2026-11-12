#!/bin/bash
# Evaluate the MARL runs of one experiment / algorithm / location on the shared protocol (scripts/eval_protocol.sh):
# noop + every seed of EXP/ALGO (their best model), 10 paired 30-day episodes. Seeds are labelled <ALGO>_<EXP>_s<seed>
# and also pooled per method.
#
# Usage:  scripts/marl/eval.sh EXP_NAME LOCATION ALGO
#   Slurm: sbatch -c 8 --mem 16G -t 12:00:00 scripts/marl/eval.sh marl_co2 az happo
# SDC_RUNS: as for scripts/marl/launch.sh (default: <repo>/results/marl_runs). Results go to
# $SDC_RUNS/EXP_NAME/eval/LOCATION/ALGO; merge several with: python eval_compare.py --combine <dirs> --out_dir <dir>
set -euo pipefail
if [ $# -ne 3 ]; then sed -n '2,10p' "$0"; exit 1; fi
EXP=$1; LOC=$2; ALGO=$3
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=$(cd "${SDC_RUNS:-$REPO/results/marl_runs}" && pwd)
DIRS=("$RUNS/$EXP"/runs/sustaindc/"$LOC"/"$ALGO"/"$EXP"/seed-*)
[ -d "${DIRS[0]}" ] || { echo "No runs found under $RUNS/$EXP/runs/sustaindc/$LOC/$ALGO/$EXP/"; exit 1; }
echo "${#DIRS[@]} run(s): ${DIRS[*]##*/}"
exec "$REPO/scripts/eval_protocol.sh" "$LOC" "$RUNS/$EXP/eval/$LOC/$ALGO" \
    "harl:$RUNS/$EXP/runs/sustaindc/$LOC/$ALGO/$EXP/seed-*=${ALGO}_${EXP}"
