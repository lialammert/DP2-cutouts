#!/bin/bash
# fetch.slurm
# Anna Preto
# AstroDeep Project
# Phase fetch on Jean Zay: the six images of each patch that has targets (SIA + SODA). Needs internet: prepost
# partition. The number of simultaneous tasks is capped by submit.sh (%N) so as not to overload the Rubin services.
#SBATCH --job-name=dp2-fetch
#SBATCH --partition=prepost
#SBATCH --account=CHANGE_ME@cpu
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/fetch_%A_%a.out

set -euo pipefail
source "${DP2_VENV:?set DP2_VENV to the virtual environment}/bin/activate"
dp2-cutouts "${DP2_CONFIG:?set DP2_CONFIG}" fetch --index "${SLURM_ARRAY_TASK_ID}"
