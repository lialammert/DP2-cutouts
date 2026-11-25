#!/bin/bash
# cpu.slurm
# Anna Preto
# AstroDeep Project
# Any dp2-ml command that needs no GPU: mmu-mirror (internet: prepost), probe-euclid, d1, d2 (from the MMU mirror),
# d1-merge, d2-merge, quality, blends. The command and its options come after the script name:
#   sbatch --partition=prepost --account=<projet>@cpu scripts/jeanzay/ml/cpu.slurm d1 --index '$SLURM_ARRAY_TASK_ID'
# Inside an array, the literal $SLURM_ARRAY_TASK_ID in the arguments is replaced by the task id.
#SBATCH --job-name=dp2-ml-cpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/ml_cpu_%x_%A_%a.out

# Slurm runs a copy of this file from its spool folder: env.sh is found from DP2_ML_SCRIPTS or the submit folder.
source "${DP2_ML_SCRIPTS:-${SLURM_SUBMIT_DIR:-.}/scripts/jeanzay/ml}/env.sh"
ARGS=()
for a in "$@"; do
    ARGS+=("${a//\$SLURM_ARRAY_TASK_ID/${SLURM_ARRAY_TASK_ID:-}}")
done
rc=0
dp2-ml "$DP2_ML_CONFIG" "${ARGS[@]}" || rc=$?
# A job of the deblending campaign (deblend-stats) advances it when it ends (deblend/campaign.py).
if [[ -n "${DEBLEND_CAMPAIGN:-}" ]]; then
    # shellcheck disable=SC2086
    dp2-ml "$DP2_ML_CONFIG" deblend-advance ${DEBLEND_SET:-} || echo "deblend-advance failed" >&2
fi
exit "$rc"
