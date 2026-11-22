#!/bin/bash
# cpu_run.sh
# Anna Preto
# AstroDeep Project
# Any command of this repository as a short job on cpu_p1 (development QoS: starts fast, 2 h at most), with the ML
# environment (scripts/jeanzay/ml/env.sh). The log goes to logs/<name>_<jobid>.out.
# Usage: bash scripts/jeanzay/cpu_run.sh <name> <hh:mm:ss> <command ...>
#   e.g. bash scripts/jeanzay/cpu_run.sh overfit 02:00:00 python scripts/inspect/vae_overfit_test.py --run $RUN --out $OUT

set -euo pipefail
: "${DP2_CPU_ACCOUNT:?set DP2_CPU_ACCOUNT}"
name="$1"
walltime="$2"
shift 2
mkdir -p logs
sbatch --account="$DP2_CPU_ACCOUNT" --partition=cpu_p1 --qos=qos_cpu-dev --cpus-per-task=10 --hint=nomultithread \
    --time="$walltime" --job-name="$name" --output="logs/${name}_%j.out" \
    --wrap="source scripts/jeanzay/ml/env.sh && $*"
