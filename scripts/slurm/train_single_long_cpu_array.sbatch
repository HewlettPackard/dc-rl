#!/bin/bash
# Seed sweep of single-agent training on long-cpu (no per-user CPU cap, lower priority, preemptible).
# Each array task is one seed: seed = SLURM_ARRAY_TASK_ID.
#
# Submit from the repo root:
#   mkdir -p results/slurm && sbatch scripts/slurm/train_single_long_cpu_array.sbatch
# Change the seeds with --array, and pass extra train_single_agent.py arguments after the script, e.g.:
#   sbatch --array=1-5 scripts/slurm/train_single_long_cpu_array.sbatch --total_timesteps 10000000
#
# Checkpointing: the run directory is keyed on the array job id + seed, so a preempted/requeued task resumes
# from its latest checkpoint instead of starting over.
#SBATCH --job-name=sdc-single-sweep
#SBATCH --partition=long-cpu
#SBATCH --array=1-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x-%A_%a.out
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

echo "[$(date)] Task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} (restart count ${SLURM_RESTART_COUNT:-0}) on $(hostname)"
"$PYTHON" train_single_agent.py \
    --algo ppo \
    --n_envs $((SLURM_CPUS_PER_TASK - 1)) \
    --total_timesteps 5000000 \
    --eval_freq 250000 \
    --checkpoint_freq 250000 \
    --exp_name ppo_sweep_v1 \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --run_id "job${SLURM_ARRAY_JOB_ID}" \
    "$@" &
wait $!
