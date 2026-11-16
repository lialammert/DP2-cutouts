#!/bin/bash
# process.slurm
# Anna Preto
# AstroDeep Project
# Phase process on Jean Zay: pixel cuts and masking, one patch per array task. No network needed: compute nodes.
#SBATCH --job-name=dp2-process
#SBATCH --partition=cpu_p1
#SBATCH --account=CHANGE_ME@cpu
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/process_%A_%a.out

set -euo pipefail
source "${DP2_VENV:?set DP2_VENV to the virtual environment}/bin/activate"
dp2-cutouts "${DP2_CONFIG:?set DP2_CONFIG}" process --index "${SLURM_ARRAY_TASK_ID}"
