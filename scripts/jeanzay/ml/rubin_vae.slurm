#!/bin/bash
# rubin_vae.slurm
# Anna Preto
# AstroDeep Project
# Reference Rubin VAE: DebVAder stage 1 trained from scratch on the Rubin stamps of D1, with debvader-keras' own
# trainer (scripts/train.py --stage 1), then phase 0 on the new run. The training set and its config are written
# beforehand by `dp2-ml <cfg> export-rubin-vae` (submit_ml.sh rubin-vae chains both).
#   DEBVADER_REPO     debvader-keras checkout holding scripts/train.py (default $WORK/debvader-keras)
#   EXPERIMENTS_DIR   where debvader-keras writes its runs (default $WORK/debvader-results)
#   RUBIN_VAE_RUN     run name (default: rubin_vae.run_name of the ML config)
# The run lands in $EXPERIMENTS_DIR/<run name>: that folder is the DEBVADER_RUN of everything after this job.
# A job that hits the 20 h limit is resubmitted as is: train.py --resume continues from the last checkpoint.
#SBATCH --job-name=dp2-rubin-vae
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=20:00:00
#SBATCH --output=logs/ml_gpu_%x_%j.out

source "${DP2_ML_SCRIPTS:-${SLURM_SUBMIT_DIR:-.}/scripts/jeanzay/ml}/env.sh"
DEBVADER_REPO="${DEBVADER_REPO:-$WORK/debvader-keras}"
if [[ ! -f "$DEBVADER_REPO/scripts/train.py" ]]; then
    echo "ERROR: no scripts/train.py in DEBVADER_REPO=$DEBVADER_REPO (set it to the debvader-keras checkout)" >&2
    exit 2
fi
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$WORK/debvader-results}"
RUN="${RUBIN_VAE_RUN:-$(python -c "
from dp2_cutouts.mlconfig import MLConfig
print(MLConfig.load('$DP2_ML_CONFIG')['rubin_vae']['run_name'])")}"
DATA="$DP2_ML_ROOT/datasets/rubin_vae"
if [[ ! -f "$DATA/debvader_config.json" ]]; then
    echo "ERROR: $DATA/debvader_config.json missing: run dp2-ml export-rubin-vae first" >&2
    exit 2
fi
echo "[rubin-vae] repo $DEBVADER_REPO | data $DATA | run $EXPERIMENTS_DIR/$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd "$DEBVADER_REPO"
srun python scripts/train.py --config "$DATA/debvader_config.json" --stage 1 --run-name "$RUN" --resume --single-gpu
cd - >/dev/null

# Phase 0 on the new run: the same checks as for paper_baseline, written to evaluations/phase0_<run>.
srun dp2-ml "$DP2_ML_CONFIG" phase0 --debvader-run "$EXPERIMENTS_DIR/$RUN" --run-name "phase0_$RUN"
echo "[rubin-vae] done: export DEBVADER_RUN=$EXPERIMENTS_DIR/$RUN"
