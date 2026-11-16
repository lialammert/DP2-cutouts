#!/bin/bash
# plan.slurm
# Anna Preto
# AstroDeep Project
# Phase plan on Jean Zay: Rubin catalogue (TAP), Euclid catalogue (HuggingFace), copaings, cuts 1-2, one patch per
# array task. Needs internet: prepost partition. Submit with scripts/jeanzay/submit.sh, which sets the array size.
#SBATCH --job-name=dp2-plan
#SBATCH --partition=prepost
#SBATCH --account=CHANGE_ME@cpu
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/plan_%A_%a.out

set -euo pipefail
source "${DP2_VENV:?set DP2_VENV to the virtual environment}/bin/activate"
dp2-cutouts "${DP2_CONFIG:?set DP2_CONFIG}" plan --index "${SLURM_ARRAY_TASK_ID}"
