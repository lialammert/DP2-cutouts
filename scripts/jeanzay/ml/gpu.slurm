#!/bin/bash
# gpu.slurm
# Anna Preto
# AstroDeep Project
# dp2-ml commands that run the networks, on one A100 (one GPU: debvader-keras measured no speed-up from four on a
# model of this size). The VAE arguments (--debvader-run, --debvader-stats) are added from the environment.
#   sbatch --account=<projet>@a100 scripts/jeanzay/ml/gpu.slurm train-stage1 --ablation s1_latent_cross
# A training job that hits the 20 h limit is resubmitted as is: it resumes from <run>/backup.
# AFTER_TRAIN=eval-stage1 (or eval-stage2) runs the evaluation in the same job once training has finished.
#SBATCH --job-name=dp2-ml-gpu
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=20:00:00
#SBATCH --output=logs/ml_gpu_%x_%j.out

# Slurm runs a copy of this file from its spool folder: env.sh is found from DP2_ML_SCRIPTS or the submit folder.
source "${DP2_ML_SCRIPTS:-${SLURM_SUBMIT_DIR:-.}/scripts/jeanzay/ml}/env.sh"
: "${DEBVADER_RUN:?set DEBVADER_RUN to the paper_baseline run folder}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
srun dp2-ml "$DP2_ML_CONFIG" "$@" "${DEBVADER_ARGS[@]}"
if [[ -n "${AFTER_TRAIN:-}" ]]; then
    # The evaluation accepts the same options and ignores those it does not use (--epochs, --no-resume).
    srun dp2-ml "$DP2_ML_CONFIG" "$AFTER_TRAIN" "${@:2}" "${DEBVADER_ARGS[@]}"
fi
