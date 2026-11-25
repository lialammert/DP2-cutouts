#!/bin/bash
# process_cpu.sh
# Anna Preto
# AstroDeep Project
# Phases process (one task per patch) and assemble on cpu_p1, which needs no network: for when plan and fetch are
# already done, e.g. fetch run on the login node (`dp2-cutouts $DP2_CONFIG fetch`) to avoid the prepost queue.
# Usage: bash scripts/jeanzay/process_cpu.sh [--force]

set -euo pipefail
: "${DP2_VENV:?set DP2_VENV}" "${DP2_CONFIG:?set DP2_CONFIG}" "${DP2_CPU_ACCOUNT:?set DP2_CPU_ACCOUNT}"
force="${1:-}"
mkdir -p logs
source "$DP2_VENV/bin/activate"
N=$(python -c "
from dp2_cutouts.config import Config
from dp2_cutouts.pipeline.patches import load_patches
print(len(load_patches(Config.load('$DP2_CONFIG'))))")
# GPU settings of the shell (SBATCH_QOS, SBATCH_TIMELIMIT, SBATCH_ACCOUNT) would override these CPU jobs: dropped.
SB=(env -u SBATCH_QOS -u SBATCH_TIMELIMIT -u SBATCH_ACCOUNT sbatch)
common=(--parsable --account="$DP2_CPU_ACCOUNT" --partition=cpu_p1 --cpus-per-task=4 --hint=nomultithread)
P=$("${SB[@]}" "${common[@]}" --array=0-$((N - 1)) --time=04:00:00 --job-name=dp2-process \
    --output=logs/process_%A_%a.out \
    --wrap="source $DP2_VENV/bin/activate && dp2-cutouts $DP2_CONFIG process --index \$SLURM_ARRAY_TASK_ID $force")
A=$("${SB[@]}" "${common[@]}" --time=02:00:00 --job-name=dp2-assemble --output=logs/assemble_%j.out \
    --dependency=afterok:"$P" --wrap="source $DP2_VENV/bin/activate && dp2-cutouts $DP2_CONFIG assemble")
echo "process $P, assemble $A"
