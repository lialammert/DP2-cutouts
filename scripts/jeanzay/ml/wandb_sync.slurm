#!/bin/bash
# wandb_sync.slurm
# Anna Preto
# AstroDeep Project
# Uploads the offline W&B runs of the deblending campaign (deblend/tracking.py) every few minutes, from the prepost
# partition, which has internet access (the GPU nodes do not). Each pass syncs the run folders written to since the
# previous pass (wandb sync appends to the run of the same id). Stops at its walltime; submit it again for longer.
# Usage, either in tmux on a login node (no hours billed; restart it after a reboot of the node) or as a job:
#   export DP2_WANDB_LIB=$SCRATCH/pylibs/wandb WANDB_DIR=$SCRATCH/wandb
#   bash scripts/jeanzay/ml/wandb_sync.slurm                          (in tmux), or
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
# WANDB_CONFIG_DIR stays the default (~/.config/wandb, a few bytes): it holds the server of the account (base_url,
# e.g. a team instance), with which ~/.netrc holds the key.
export WANDB_CACHE_DIR="$WANDB_DIR/.cache" WANDB_SILENT=true
mkdir -p "$WANDB_DIR"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$DP2_WANDB_LIB"
source "$DP2_ML_VENV/bin/activate"
WB="python -m wandb"
EVERY="${DP2_WANDB_SYNC_EVERY:-300}"
echo "[wandb-sync] $(date) every ${EVERY} s from $WANDB_DIR"
# Which account and server the uploads go to: a query that never prompts (stdin closed); fails when not logged in.
who=$(timeout 60 python -c 'import wandb; a = wandb.Api(); print(a.default_entity, "@", a.settings["base_url"])'       </dev/null 2>/dev/null) || who="NOT LOGGED IN: run python -m wandb login (with DP2_WANDB_LIB on PYTHONPATH)"
echo "[wandb-sync] account: $who"
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
