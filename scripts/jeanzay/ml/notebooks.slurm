#!/bin/bash
# notebooks.slurm
# Anna Preto
# AstroDeep Project
# Executes the inspection notebooks of notebooks/ml on the products under DP2_ML_ROOT and writes the executed copies
# (with their figures) to $DP2_ML_ROOT/notebooks. No GPU, no network. The notebooks read DP2_ML_CONFIG and, for the
# run-specific ones, NB_STAGE1_RUN / NB_STAGE2_RUN. Arguments: the notebooks to run (default: all of them).
#SBATCH --job-name=dp2-ml-nb
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/ml_nb_%j.out

source "${DP2_ML_SCRIPTS:-${SLURM_SUBMIT_DIR:-.}/scripts/jeanzay/ml}/env.sh"
REPO="${DP2_REPO:-${SLURM_SUBMIT_DIR:-.}}"
OUT="$DP2_ML_ROOT/notebooks"
mkdir -p "$OUT"
NBS=("$@")
if [[ ${#NBS[@]} -eq 0 ]]; then
    NBS=("$REPO"/notebooks/ml/*.ipynb)
fi
status=0
for nb in "${NBS[@]}"; do
    echo "=== $nb"
    if ! jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
            --output-dir "$OUT" "$nb"; then
        echo "FAILED: $nb" >&2
        status=1
    fi
done
exit $status
