#!/bin/bash
# Submit SustainDC MARL training runs to Slurm: one job per algorithm x location x seed.
#
# Usage (from anywhere):
#   scripts/marl/launch.sh EXP_NAME STEPS TIME "ALGOS" "LOCATIONS" "SEEDS"
# e.g.
#   REWARD=co2 scripts/marl/launch.sh marl_co2 20000000 1-00:00:00 "happo mappo" "az ny" "1 2 3"
#
# Settings (environment variables):
#   REWARD       co2 (default): co2_team_reward for all three agents | default: the original SustainDC rewards.
#                Always passed explicitly, so runs do not depend on the env config's default reward.
#   CODE         code directory the jobs run from (default: this checkout). Jobs read the code when they START, so
#                pin it if you keep editing: git worktree add ../code-$(git rev-parse --short HEAD) HEAD; CODE=...
#   PYTHON       python of your environment (default: `python` on PATH when you submit; activate the env first)
#   SDC_RUNS     where runs go (default: $CODE/results/marl_runs); runs of EXP_NAME go to $SDC_RUNS/EXP_NAME
#   WANDB        1 (default): live wandb logging via scripts/marl/wandb_train.py (needs `pip install wandb`) | 0: off
#   WANDB_PROJECT (default sustaindc-rl), WANDB_ENTITY (default: your wandb default entity)
#   SBATCH_ARGS  extra sbatch options, e.g. "-p long-cpu" or "--account=..."
#   EXTRA        extra train_sustaindc.py arguments for every run
#   DRY_RUN=1    write the job scripts but do not submit
#
# Each job: 16 CPUs, 32 GB, TIME for on-policy algorithms (HAPPO/MAPPO), 2 days for off-policy (HAD3QN).
# Preemption: jobs are submitted with --requeue and --run_id $SLURM_JOB_ID, so a requeued job resumes from its last
# checkpoint (every 500k env steps; --checkpoint_interval in EXTRA to change it).
# Outputs: $SDC_RUNS/EXP_NAME/{jobs,slurm,runs}, plus manifest.tsv (job id, algo, location, seed, steps, time).
# Evaluate finished runs with scripts/marl/eval.sh (the shared protocol of scripts/eval_protocol.sh).
set -euo pipefail
if [ $# -ne 6 ]; then sed -n '2,29p' "$0"; exit 1; fi
EXP=$1; STEPS=$2; TIME_ONPOL=$3; ALGOS=$4; LOCS=$5; SEEDS=$6

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE=$(cd "${CODE:-$SCRIPT_DIR/../..}" && pwd)
PY=${PYTHON:-$(command -v python || true)}
[ -n "$PY" ] || { echo "No python on PATH: activate your environment or set PYTHON=..."; exit 1; }
PY=$(command -v "$PY")
OUT=${SDC_RUNS:-$CODE/results/marl_runs}/$EXP
REWARD=${REWARD:-co2}
case $REWARD in
  co2)     REWARD_ARGS='--ls_reward co2_team_reward --dc_reward co2_team_reward --bat_reward co2_team_reward' ;;
  default) REWARD_ARGS='--ls_reward default_ls_reward --dc_reward default_dc_reward --bat_reward default_bat_reward' ;;
  *) echo "REWARD must be co2 or default (got $REWARD)"; exit 1 ;;
esac
WANDB=${WANDB:-1}
if [ "$WANDB" = 1 ]; then
  "$PY" -c 'import wandb' 2>/dev/null || { echo "wandb is not installed in $PY (pip install wandb, or WANDB=0)"; exit 1; }
  ENTRY="$CODE/scripts/marl/wandb_train.py"
else
  ENTRY="$CODE/train_sustaindc.py"
fi

# Shared PPO settings (HAPPO and MAPPO). lr 3e-4 and entropy 0.01: with 3e-5 / 0.1 the policies stayed near-uniform.
PPO_ARGS='--hidden_sizes [64,64] --initialization_method xavier_uniform_ --lr 0.0003 --critic_lr 0.0003 --clip_param 0.5 --actor_num_mini_batch 8 --critic_num_mini_batch 8 --entropy_coef 0.01 --eval_interval 10'
# Shared run settings (all algorithms). HAD3QN keeps its own settings from harl/configs/algos_cfgs/had3qn.yaml.
RUN_ARGS='--n_rollout_threads 12 --episode_length 1024 --n_eval_rollout_threads 5 --eval_episodes 5'

mkdir -p "$OUT/jobs" "$OUT/slurm"
echo "EXP $EXP | reward $REWARD | code $CODE @ $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '?') | python $PY | out $OUT"
for algo in $ALGOS; do for loc in $LOCS; do for seed in $SEEDS; do
  name=${algo}_${loc}_s${seed}
  extra="$RUN_ARGS $REWARD_ARGS ${EXTRA:-}"; time=$TIME_ONPOL
  if [[ $algo == happo || $algo == mappo ]]; then extra="$extra $PPO_ARGS"; fi
  if [[ $algo == had3qn ]]; then time=2-00:00:00; fi   # off-policy is ~1.5-2x slower per env step
  cat > "$OUT/jobs/$name.sh" <<JOB
#!/bin/bash
cd "$CODE"
echo "host \$(hostname)  start \$(date)  code \$(git rev-parse --short HEAD 2>/dev/null)"
export WANDB_PROJECT=${WANDB_PROJECT:-sustaindc-rl}${WANDB_ENTITY:+ WANDB_ENTITY=$WANDB_ENTITY}
"$PY" -u "$ENTRY" --algo $algo --exp_name $EXP --log_dir "$OUT/runs" \\
    --location $loc --seed $seed --num_env_steps $STEPS --run_id \$SLURM_JOB_ID $extra
echo "exit \$?  end \$(date)"
JOB
  chmod +x "$OUT/jobs/$name.sh"
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "(dry run) $OUT/jobs/$name.sh"; continue; fi
  # shellcheck disable=SC2086
  jid=$(sbatch --parsable -J "sdc-$EXP-$name" --cpus-per-task=16 --mem=32G --time=$time --requeue --open-mode=append \
        ${SBATCH_ARGS:-} -o "$OUT/slurm/$name.log" "$OUT/jobs/$name.sh")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$jid" "$algo" "$loc" "$seed" "$STEPS" "$(date -Is)" >> "$OUT/manifest.tsv"
  echo "$jid $name"
done; done; done
