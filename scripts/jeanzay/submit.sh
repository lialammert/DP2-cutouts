#!/bin/bash
# submit.sh
# Anna Preto
# AstroDeep Project
# Submits the whole patch-by-patch pipeline on Jean Zay: list of patches (on the login node, needs internet), then
# plan, fetch and process as job arrays chained by dependencies, then the assembly of the final dataset.
# Usage: DP2_VENV=... DP2_CONFIG=... DP2_CPU_ACCOUNT=<projet>@cpu scripts/jeanzay/submit.sh [max simultaneous fetch
# tasks, default 4]
# A rerun skips every patch and phase already done with the same configuration (a new wave of patches: change the
# patch list of patches.names_file and rerun).

set -euo pipefail
: "${DP2_VENV:?set DP2_VENV}" "${DP2_CONFIG:?set DP2_CONFIG}" "${DP2_CPU_ACCOUNT:?set DP2_CPU_ACCOUNT=<projet>@cpu}"
export DP2_VENV DP2_CONFIG
FETCH_MAX="${1:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p logs

source "$DP2_VENV/bin/activate"
# fast3tree (copaings) compiles its C library into site-packages on first use; array tasks starting together race on
# the same .c/.so files (FileNotFoundError on the .c, AttributeError on a half-written .so). Build it once here.
python -c "from fast3tree.make import make_lib; print('fast3tree:', make_lib(3, True))"
dp2-cutouts "$DP2_CONFIG" patches
N=$(python -c "
from dp2_cutouts.config import Config
from dp2_cutouts.pipeline.patches import load_patches
print(len(load_patches(Config.load('$DP2_CONFIG'))))")
LAST=$((N - 1))
echo "$N patches"

# The account is given here (the headers say CHANGE_ME), and the GPU settings a shell may export (SBATCH_QOS,
# SBATCH_TIMELIMIT, SBATCH_ACCOUNT) are dropped: they would override the headers of these CPU jobs.
SB=(env -u SBATCH_QOS -u SBATCH_TIMELIMIT -u SBATCH_ACCOUNT sbatch --parsable --account="$DP2_CPU_ACCOUNT")
PLAN=$("${SB[@]}" --array=0-"$LAST" "$HERE/plan.slurm")
FETCH=$("${SB[@]}" --array=0-"$LAST"%"$FETCH_MAX" --dependency=afterany:"$PLAN" "$HERE/fetch.slurm")
PROCESS=$("${SB[@]}" --array=0-"$LAST" --dependency=afterany:"$FETCH" "$HERE/process.slurm")
ASSEMBLE=$("${SB[@]}" --partition=prepost \
  --cpus-per-task=4 --hint=nomultithread --time=10:00:00 --output=logs/assemble_%j.out \
  --dependency=afterany:"$PROCESS" \
  --wrap "source $DP2_VENV/bin/activate && dp2-cutouts $DP2_CONFIG assemble")
echo "plan $PLAN, fetch $FETCH, process $PROCESS, assemble $ASSEMBLE"
echo "status of each patch: \$(data_root)/manifests/<tract>_<patch>.json"
