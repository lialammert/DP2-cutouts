#!/bin/bash
# submit_ml.sh
# Anna Preto
# AstroDeep Project
# Submits the cross-instrument work on Jean Zay, step by step or all at once, with Slurm dependencies.
# Run from the repository root, on a login node, after the stamp pipeline has assembled its final dataset.
#
#   export DP2_ML_VENV=$WORK/venvs/dp2-ml DP2_CONFIG=config/<pipeline>.yaml DP2_ML_CONFIG=config/ml_jeanzay.yaml
#   export DP2_ML_ROOT=$WORK/dp2-ml DP2_CPU_ACCOUNT=<projet>@cpu DP2_GPU_ACCOUNT=<projet>@a100
#   export DEBVADER_RUN=<path of the paper_baseline run>          # given here, never written in a file
#   scripts/jeanzay/ml/submit_ml.sh probe <tract_patch>            # 1. which Euclid image source works
#   scripts/jeanzay/ml/submit_ml.sh phase0                         # 2. reference VAE on DP2 stamps
#   scripts/jeanzay/ml/submit_ml.sh datasets                       # 3. D1, D2, quality, controlled blends
#   scripts/jeanzay/ml/submit_ml.sh stage1 [ablation ...]          # 4. Stage 1 runs (default: every ablation)
#   STAGE1_RUN=s1_latent_cross scripts/jeanzay/ml/submit_ml.sh stage2 [ablation ...]   # 5. Stage 2 runs
#   scripts/jeanzay/ml/submit_ml.sh notebooks                      # 6. executed notebooks with the figures
#   scripts/jeanzay/ml/submit_ml.sh all                            # 2 to 6 chained (STAGE1_RUN: teacher of 5)
# FETCH_MAX (default 4) caps the simultaneous Euclid download tasks. Reruns skip shards and runs already done.

set -euo pipefail
: "${DP2_ML_VENV:?}" "${DP2_CONFIG:?}" "${DP2_ML_CONFIG:?}" "${DP2_ML_ROOT:?}"
: "${DP2_CPU_ACCOUNT:?set DP2_CPU_ACCOUNT=<projet>@cpu}"
DP2_CONFIG="$(realpath "$DP2_CONFIG")"
DP2_ML_CONFIG="$(realpath "$DP2_ML_CONFIG")"
DP2_ML_ROOT="$(realpath -m "$DP2_ML_ROOT")"
export DP2_ML_VENV DP2_CONFIG DP2_ML_CONFIG DP2_ML_ROOT
export DP2_ML_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
export DP2_REPO="$(cd "$DP2_ML_SCRIPTS/../../.." && pwd)"
export DEBVADER_RUN="${DEBVADER_RUN:-}" DEBVADER_STATS="${DEBVADER_STATS:-}"
FETCH_MAX="${FETCH_MAX:-4}"
mkdir -p logs
STEP="${1:?step: probe | phase0 | datasets | stage1 | stage2 | notebooks | all}"
shift || true

CPU=(sbatch --parsable --account="$DP2_CPU_ACCOUNT")
gpu() {
    : "${DP2_GPU_ACCOUNT:?set DP2_GPU_ACCOUNT=<projet>@a100}"
    : "${DEBVADER_RUN:?set DEBVADER_RUN to the paper_baseline run folder}"
    sbatch --parsable --account="$DP2_GPU_ACCOUNT" "$@"
}
dep() { [[ -n "${1:-}" ]] && echo "--dependency=afterok:$1" || true; }

source "$DP2_ML_VENV/bin/activate"
n_patches() {
    python -c "
from dp2_cutouts.config import Config
from dp2_cutouts.pipeline.patches import load_patches
print(len(load_patches(Config.load('$DP2_CONFIG'))))"
}
ablations() {
    python -c "
import yaml
print(' '.join(yaml.safe_load(open('$DP2_ML_CONFIG'))['$1']['ablations']))"
}

submit_probe() {
    local patch="${1:?probe needs a patch name, e.g. 5063_34}"
    "${CPU[@]}" --job-name=dp2-probe --time=00:30:00 "$DP2_ML_SCRIPTS/cpu.slurm" probe-euclid --patch "$patch"
}

submit_phase0() {
    gpu --job-name=dp2-phase0 --qos=qos_gpu_a100-dev --time=01:00:00 "$DP2_ML_SCRIPTS/gpu.slurm" phase0
}

submit_datasets() {
    local n last d1 d1m q1 bl d2 d2m q2
    n=$(n_patches); last=$((n - 1))
    d1=$("${CPU[@]}" --job-name=dp2-d1 --array=0-"$last"%"$FETCH_MAX" "$DP2_ML_SCRIPTS/cpu.slurm" \
         d1 --index '$SLURM_ARRAY_TASK_ID')
    d1m=$("${CPU[@]}" --job-name=dp2-d1-merge --dependency=afterany:"$d1" --time=10:00:00 \
          "$DP2_ML_SCRIPTS/cpu.slurm" d1-merge)
    q1=$("${CPU[@]}" --job-name=dp2-q1 --dependency=afterok:"$d1m" "$DP2_ML_SCRIPTS/cpu.slurm" quality d1)
    bl=$("${CPU[@]}" --job-name=dp2-blends --dependency=afterok:"$d1m" "$DP2_ML_SCRIPTS/cpu.slurm" blends)
    d2=$("${CPU[@]}" --job-name=dp2-d2 --array=0-"$last"%"$FETCH_MAX" "$DP2_ML_SCRIPTS/cpu.slurm" \
         d2 --index '$SLURM_ARRAY_TASK_ID')
    # D2 takes its normalisation statistics from D1: merged after both.
    d2m=$("${CPU[@]}" --job-name=dp2-d2-merge --dependency=afterany:"$d2",afterok:"$d1m" --time=10:00:00 \
          "$DP2_ML_SCRIPTS/cpu.slurm" d2-merge)
    q2=$("${CPU[@]}" --job-name=dp2-q2 --dependency=afterok:"$d2m" "$DP2_ML_SCRIPTS/cpu.slurm" quality d2)
    echo "datasets: d1 $d1, d1-merge $d1m, quality-d1 $q1, blends $bl, d2 $d2, d2-merge $d2m, quality-d2 $q2" >&2
    # Stage 1 needs D1 merged with a quality report that did not fail; Stage 2 also needs D2 and the blends.
    # Set as globals (not printed): the caller must not run this function in a subshell.
    AFTER_STAGE1="$q1"
    AFTER_STAGE2="$q1:$bl:$q2"
}

submit_stage1() {
    local after="${1:-}"; shift || true
    local names=("$@") ids=()
    [[ ${#names[@]} -eq 0 ]] && read -r -a names <<< "$(ablations stage1)"
    for a in "${names[@]}"; do
        ids+=("$(gpu --job-name="dp2-$a" $(dep "$after") --export=ALL,AFTER_TRAIN=eval-stage1 \
                 "$DP2_ML_SCRIPTS/gpu.slurm" train-stage1 --ablation "$a")")
    done
    echo "stage 1: ${names[*]} -> ${ids[*]}" >&2
    (IFS=:; echo "${ids[*]}")
}

submit_stage2() {
    local after="${1:-}"; shift || true
    : "${STAGE1_RUN:?set STAGE1_RUN to the Stage 1 run used as teacher (e.g. s1_latent_cross)}"
    local names=("$@") ids=()
    [[ ${#names[@]} -eq 0 ]] && read -r -a names <<< "$(ablations stage2)"
    for a in "${names[@]}"; do
        ids+=("$(gpu --job-name="dp2-$a" $(dep "$after") --export=ALL,AFTER_TRAIN=eval-stage2 \
                 "$DP2_ML_SCRIPTS/gpu.slurm" train-stage2 --ablation "$a" --stage1-run "$STAGE1_RUN")")
    done
    echo "stage 2 (teacher $STAGE1_RUN): ${names[*]} -> ${ids[*]}" >&2
    (IFS=:; echo "${ids[*]}")
}

submit_notebooks() {
    local after="${1:-}"
    sbatch --parsable --account="$DP2_CPU_ACCOUNT" $(dep "$after") \
        --export=ALL,NB_STAGE1_RUN="${STAGE1_RUN:-s1_latent_cross}",NB_STAGE2_RUN="${STAGE2_RUN:-s2_latent}" \
        "$DP2_ML_SCRIPTS/notebooks.slurm"
}

case "$STEP" in
    probe) submit_probe "$@" ;;
    phase0) submit_phase0 ;;
    datasets) submit_datasets ;;
    stage1) submit_stage1 "" "$@" >/dev/null ;;
    stage2) submit_stage2 "" "$@" >/dev/null ;;
    notebooks) submit_notebooks "" ;;
    all)
        export STAGE1_RUN="${STAGE1_RUN:-s1_latent_cross}"
        p0=$(submit_phase0)
        submit_datasets
        s1=$(submit_stage1 "$AFTER_STAGE1")
        # Stage 2 waits for every Stage 1 run (its teacher is one of them) and for the datasets.
        s2=$(submit_stage2 "$s1:$AFTER_STAGE2")
        nb=$(submit_notebooks "$p0:$s2")
        echo "phase0 $p0 | notebooks $nb"
        ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "logs: $(pwd)/logs; products: $DP2_ML_ROOT"
