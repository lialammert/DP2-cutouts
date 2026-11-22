#!/bin/bash
# datasets_aside.sh
# Anna Preto
# AstroDeep Project
# Moves $DP2_ML_ROOT/datasets aside to the first free datasets_v<i>, before a rebuild (submit_ml.sh rebuild), or with
# an argument one dataset only: datasets_aside.sh d2 -> datasets/d2_v<i> (submit_ml.sh d2). Nothing is deleted.
# Without that folder, does nothing.
#SBATCH --output=logs/ml_cpu_%x_%j.out

set -euo pipefail
: "${DP2_ML_ROOT:?set DP2_ML_ROOT}"
d="$DP2_ML_ROOT/datasets${1:+/$1}"
if [[ ! -d "$d" ]]; then
    echo "no $d: nothing to move"
    exit 0
fi
i=1
while [[ -e "${d}_v$i" ]]; do
    i=$((i + 1))
done
mv "$d" "${d}_v$i"
echo "moved $d to ${d}_v$i"
