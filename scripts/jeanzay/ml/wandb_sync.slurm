#!/bin/bash
# wandb_sync.slurm
# Anna Preto
# AstroDeep Project
# Uploads the offline W&B runs of the deblending campaign (deblend/tracking.py) every few minutes, from the prepost
# partition, which has internet access (the GPU nodes do not). Each pass syncs the run folders written to since the
# previous pass (wandb sync appends to the run of the same id). Stops at its walltime; submit it again for longer.
# Usage (login node, venv and wandb lib as for the trainings):
#   export DP2_WANDB_LIB=$SCRATCH/pylibs/wandb WANDB_DIR=$SCRATCH/wandb
#   sbatch --account=fqn@cpu scripts/jeanzay/ml/wandb_sync.slurm
#SBATCH --job-name=dp2-wandb-sync
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/wandb_sync_%j.out

set -uo pipefail
: "${DP2_ML_VENV:?set DP2_ML_VENV}" "${DP2_WANDB_LIB:?set DP2_WANDB_LIB}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
export WANDB_CACHE_DIR="$WANDB_DIR/.cache" WANDB_CONFIG_DIR="$WANDB_DIR/.config" WANDB_SILENT=true
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$DP2_WANDB_LIB"
source "$DP2_ML_VENV/bin/activate"
WB="python -m wandb"
EVERY="${DP2_WANDB_SYNC_EVERY:-300}"
echo "[wandb-sync] $(date) every ${EVERY} s from $WANDB_DIR; internet: $(curl -s -o /dev/null -w '%{http_code}' https://api.wandb.ai || echo none)"
stamp="$WANDB_DIR/.last_sync"
while true; do
    new=$(mktemp)
    touch "$new"
    for d in "$WANDB_DIR"/wandb/offline-run-*; do
        [[ -d "$d" ]] || continue
        if [[ ! -e "$stamp" ]] || [[ -n "$(find "$d" -newer "$stamp" -print -quit)" ]]; then
            $WB sync "$d" > /dev/null 2>&1 && echo "[wandb-sync] $(date +%H:%M) $(basename "$d")" \
                || echo "[wandb-sync] $(date +%H:%M) FAILED $(basename "$d")"
        fi
    done
    mv "$new" "$stamp"
    sleep "$EVERY"
done
