#!/bin/bash
# One single-agent training run on main-cpu (8 CPUs is the per-user cap on this partition).
#
# Submit from the repo root:
#   mkdir -p results/slurm && sbatch scripts/slurm/train_single_main_cpu.sbatch
# Extra arguments are passed to train_single_agent.py and override the defaults below, e.g.:
#   sbatch scripts/slurm/train_single_main_cpu.sbatch --seed 2 --total_timesteps 10000000
#
# Checkpointing: the run directory is keyed on the Slurm job id, so if the job is preempted and requeued
# (or requeues itself near the time limit, see the trap below) it resumes from its latest checkpoint.
#SBATCH --job-name=sdc-single
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f sustaindc_env.py ] || { echo "Submit from the sustaindc-rl repo root"; exit 1; }

PYTHON=${PYTHON:-$HOME/.conda/envs/sustaindc/bin/python}
# One thread per process: parallelism comes from the env subprocesses, not from numpy/torch threads.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# 5 minutes before the time limit, requeue ourselves; the new job resumes from the latest checkpoint.
trap 'echo "[$(date)] Time limit near, requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1

echo "[$(date)] Job $SLURM_JOB_ID (restart count ${SLURM_RESTART_COUNT:-0}) on $(hostname)"
"$PYTHON" train_single_agent.py \
    --algo ppo \
    --n_envs $((SLURM_CPUS_PER_TASK - 1)) \
    --total_timesteps 5000000 \
    --eval_freq 250000 \
    --checkpoint_freq 250000 \
    --exp_name ppo_v1 \
    --seed 1 \
    --run_id "job${SLURM_JOB_ID}" \
    "$@" &
wait $!
