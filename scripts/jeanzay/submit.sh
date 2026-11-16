#!/bin/bash
# submit.sh
# Anna Preto
# AstroDeep Project
# Submits the whole patch-by-patch pipeline on Jean Zay: list of patches (on the login node, needs internet), then
# plan, fetch and process as job arrays chained by dependencies, then the assembly of the final dataset.
# Usage: DP2_VENV=... DP2_CONFIG=... scripts/jeanzay/submit.sh [max simultaneous fetch tasks, default 4]
# A rerun skips every patch and phase already done with the same configuration.

set -euo pipefail
: "${DP2_VENV:?set DP2_VENV}" "${DP2_CONFIG:?set DP2_CONFIG}"
export DP2_VENV DP2_CONFIG
FETCH_MAX="${1:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p logs

source "$DP2_VENV/bin/activate"
dp2-cutouts "$DP2_CONFIG" patches
N=$(python -c "
from dp2_cutouts.config import Config
from dp2_cutouts.pipeline.patches import load_patches
print(len(load_patches(Config.load('$DP2_CONFIG'))))")
LAST=$((N - 1))
echo "$N patches"

PLAN=$(sbatch --parsable --array=0-"$LAST" "$HERE/plan.slurm")
FETCH=$(sbatch --parsable --array=0-"$LAST"%"$FETCH_MAX" --dependency=afterany:"$PLAN" "$HERE/fetch.slurm")
PROCESS=$(sbatch --parsable --array=0-"$LAST" --dependency=afterany:"$FETCH" "$HERE/process.slurm")
ASSEMBLE=$(sbatch --parsable --partition=prepost --account="$(sed -n 's/^#SBATCH --account=//p' "$HERE/plan.slurm")" \
  --cpus-per-task=4 --hint=nomultithread --time=10:00:00 --output=logs/assemble_%j.out \
  --dependency=afterany:"$PROCESS" \
  --wrap "source $DP2_VENV/bin/activate && dp2-cutouts $DP2_CONFIG assemble")
echo "plan $PLAN, fetch $FETCH, process $PROCESS, assemble $ASSEMBLE"
echo "status of each patch: \$(data_root)/manifests/<tract>_<patch>.json"
