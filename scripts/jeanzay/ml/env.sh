#!/bin/bash
# env.sh
# Anna Preto
# AstroDeep Project
# Environment shared by every cross-instrument job, sourced at the top of each .slurm file.
# Required (exported by submit_ml.sh, or by hand before sbatch):
#   DP2_ML_VENV     virtual environment holding dp2-cutouts[ml] AND debvader-keras[jax-cuda]
#   DP2_CONFIG      configuration of the stamp pipeline (patch by patch) that produced the final dataset
#   DP2_ML_CONFIG   config/ml_jeanzay.yaml (or a copy)
#   DP2_ML_ROOT     where datasets, runs and evaluations go (under $WORK or $SCRATCH, never $HOME)
# Required by the jobs that use the VAE:
#   DEBVADER_RUN    folder of the paper_baseline run (config.json, checkpoints/stage1_best.weights.h5, ...)
# Optional:
#   DEBVADER_STATS  stats.json of the dataset paper_baseline was trained on, if the run folder has none

set -euo pipefail
: "${DP2_ML_VENV:?set DP2_ML_VENV}" "${DP2_CONFIG:?set DP2_CONFIG}" "${DP2_ML_CONFIG:?set DP2_ML_CONFIG}"
: "${DP2_ML_ROOT:?set DP2_ML_ROOT}"
# Absolute paths: the notebook kernels run from notebooks/ml, not from the submit folder.
DP2_CONFIG="$(realpath "$DP2_CONFIG")"
DP2_ML_CONFIG="$(realpath "$DP2_ML_CONFIG")"
DP2_ML_ROOT="$(realpath -m "$DP2_ML_ROOT")"
export DP2_CONFIG DP2_ML_CONFIG DP2_ML_ROOT

module purge
# A100 nodes are AMD and need arch/a100 before any other module (as in debvader-keras/slurm/jean_zay).
if lscpu 2>/dev/null | grep -qi 'AMD'; then
    module load arch/a100
fi
if [[ ! -f "$DP2_ML_VENV/bin/activate" ]]; then
    echo "ERROR: no virtual environment at $DP2_ML_VENV" >&2
    exit 2
fi
source "$DP2_ML_VENV/bin/activate"
python -c "import numpy, dp2_cutouts" || { echo "ERROR: dp2_cutouts not importable from $DP2_ML_VENV" >&2; exit 2; }

# Python output written to the Slurm log as it comes (a file is block-buffered: a training looked silent for its
# whole first epoch).
export PYTHONUNBUFFERED=1
export KERAS_BACKEND="${KERAS_BACKEND:-jax}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
# V100 (Volta, sm_70): XLA's default cuDNN convolution autotuning (level 4) profiles engines that fail on this GPU
# (CUDNN_STATUS_EXECUTION_FAILED, even with cuDNN 9.6 and jax 0.4.38; gpu_probe.py, job 899355). Level 0 (no
# autotuning) always runs but picks slow kernels; levels 2 and 3 ran C1 and K1 twice as fast (speed_probe.py, job
# 902315: C1 279 -> 567, K1 358 -> 743 examples per second; level 1 failed on K1; float16 fails). Level 3 here;
# dp2-ml restarts itself at level 0 if a cuDNN engine still fails (mlcli.py, DP2_AUTOTUNE_FALLBACK).
# Set on V100 nodes only.
if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "V100"; then
    case " ${XLA_FLAGS:-} " in
        *xla_gpu_autotune_level*) ;;
        *) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=${DP2_V100_AUTOTUNE:-3}" ;;
    esac
    echo "[env] V100: XLA_FLAGS=$XLA_FLAGS"
fi
# JAX_PLATFORMS=cpu set in the submitting shell (for checks on a login node) is copied into the job by sbatch, and
# from it into every job the campaign submits: JAX then sees cpu:0 only (pilot of 2026-11-28: K1 and C1 stopped on
# "deblend.train.gpus = 4 but only 1 devices"). On a node with a GPU it is dropped.
if [[ -n "${JAX_PLATFORMS:-}" ]] && nvidia-smi -L >/dev/null 2>&1; then
    echo "[env] JAX_PLATFORMS=$JAX_PLATFORMS inherited from the submitting shell: unset on this GPU node"
    unset JAX_PLATFORMS
fi
# Caches out of $HOME (3 GiB quota on Jean Zay) and out of $WORK (shared by the lab, its file count is the tight
# limit: it filled up on 2026-11-27): on $SCRATCH.
_cache="${SCRATCH:-${DP2_ML_ROOT:-$WORK}}/.cache"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$_cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export KERAS_HOME="${KERAS_HOME:-$XDG_CACHE_HOME/keras}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTHONHASHSEED=0

DEBVADER_ARGS=()
if [[ -n "${DEBVADER_RUN:-}" ]]; then
    DEBVADER_ARGS+=(--debvader-run "$DEBVADER_RUN")
fi
if [[ -n "${DEBVADER_STATS:-}" ]]; then
    DEBVADER_ARGS+=(--debvader-stats "$DEBVADER_STATS")
fi
echo "[env] $(command -v python) | backend $KERAS_BACKEND | ml_root $DP2_ML_ROOT | debvader run ${DEBVADER_RUN:-<unset>}"
