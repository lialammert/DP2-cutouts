#!/bin/bash
# deblend.slurm
# Anna Preto
# AstroDeep Project
# One step of the deblending campaign on one A100: dp2-ml <ml config> <command> [args], e.g.
#   sbatch --account=<projet>@a100 scripts/jeanzay/ml/deblend.slurm deblend-train k1
#   sbatch --account=<projet>@a100 scripts/jeanzay/ml/deblend.slurm deblend-eval d2 --split test
# Defaults to the 2 h dev QoS: a training cut by the walltime is resubmitted as is and resumes from <run>/backup
# (submit_deblend.sh chains several jobs with afterany); a finished run (DONE) returns at once.
# AFTER_CMD (optional) runs a second dp2-ml command in the same job once the first one succeeded (an evaluation).
#SBATCH --job-name=dp2-deblend
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=02:00:00
#SBATCH --output=logs/deblend_%x_%j.out

source "${DP2_ML_SCRIPTS:-${SLURM_SUBMIT_DIR:-.}/scripts/jeanzay/ml}/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
srun dp2-ml "$DP2_ML_CONFIG" "$@" ${DEBLEND_SET:+$DEBLEND_SET}
if [[ -n "${AFTER_CMD:-}" ]]; then
    # shellcheck disable=SC2086
    srun dp2-ml "$DP2_ML_CONFIG" $AFTER_CMD ${DEBLEND_SET:+$DEBLEND_SET}
fi
