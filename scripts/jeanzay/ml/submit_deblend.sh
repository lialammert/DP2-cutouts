#!/bin/bash
# submit_deblend.sh
# Anna Preto
# AstroDeep Project
# Starts the deblending campaign (Runs A, B, C) on Jean Zay. Run from the repository root, on a login node.
#
#   export DP2_ML_VENV=... DP2_CONFIG=... DP2_ML_CONFIG=config/ml_jeanzay.yaml DP2_ML_ROOT=$SCRATCH/dp2-ml-full
#   export DP2_CPU_ACCOUNT=<projet>@cpu DP2_GPU_ACCOUNT=<projet>@a100
#   AFTER=afterok:<d2-merge> scripts/jeanzay/ml/submit_deblend.sh pilot
#                       # every stage for PILOT_STEPS steps (run suffix _pilot), then every evaluation on 500 val
#                       # objects: stability, memory, speed
#   scripts/jeanzay/ml/submit_deblend.sh all       # the full campaign: stats, the 8 trainings, the evaluations
#   SEEDS="1 2" scripts/jeanzay/ml/submit_deblend.sh seeds   # replicates of A, B, C2, C3 (K1, K2, C1 shared)
#   scripts/jeanzay/ml/submit_deblend.sh final     # evaluations on the holdout split (final, once only)
#   scripts/jeanzay/ml/submit_deblend.sh status    # state of every campaign, nothing submitted
#
# The campaign advances itself (deblend/campaign.py): only the jobs whose dependencies are met are submitted, and
# every job submits what has become ready when it ends. A training is a sequence of 2 h dev-QoS jobs, each
# stopping on its time budget and resuming from the backup. At most deblend.campaign.max_queued jobs of the user
# sit in the dev QoS (it refuses more than about 10 per user). Graph:
#   stats -> rubin_vae, k1, c1;  k1 -> k2, A, B;  c1 -> c2 -> c3;  masking after k1; latent after k1, c1;
#   d1, d2, cost after every training.
# Evaluations run on the test split; the holdout is only read by `final`.
# AFTER=<dependency> applies to the first jobs only (e.g. afterok:<d2-merge> while the datasets are being built).

set -euo pipefail
: "${DP2_ML_VENV:?}" "${DP2_CONFIG:?}" "${DP2_ML_CONFIG:?}" "${DP2_ML_ROOT:?}"
: "${DP2_CPU_ACCOUNT:?set DP2_CPU_ACCOUNT=<projet>@cpu}" "${DP2_GPU_ACCOUNT:?set DP2_GPU_ACCOUNT=<projet>@a100}"
DP2_CONFIG="$(realpath "$DP2_CONFIG")"
DP2_ML_CONFIG="$(realpath "$DP2_ML_CONFIG")"
DP2_ML_ROOT="$(realpath -m "$DP2_ML_ROOT")"
export DP2_ML_VENV DP2_CONFIG DP2_ML_CONFIG DP2_ML_ROOT DP2_CPU_ACCOUNT DP2_GPU_ACCOUNT
export DP2_ML_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
export KERAS_BACKEND="${KERAS_BACKEND:-jax}"
mkdir -p logs
STEP="${1:?step: pilot | all | seeds | final | status}"
PILOT_STEPS="${PILOT_STEPS:-300}"
AFTER_OPT=()
[[ -n "${AFTER:-}" ]] && AFTER_OPT=(--after "$AFTER")

source "$DP2_ML_VENV/bin/activate"
ml() { dp2-ml "$DP2_ML_CONFIG" "$@"; }
case "$STEP" in
    pilot)
        ml deblend-campaign pilot --max-steps "$PILOT_STEPS" --split val ${AFTER_OPT[@]+"${AFTER_OPT[@]}"} \
            --set deblend.run_suffix=_pilot --set deblend.eval.max_examples=500 ;;
    all) ml deblend-campaign all --split test ${AFTER_OPT[@]+"${AFTER_OPT[@]}"} ;;
    seeds)
        for s in ${SEEDS:?set SEEDS, e.g. \"1 2\"}; do
            ml deblend-campaign seeds --seed "$s" --split test ${AFTER_OPT[@]+"${AFTER_OPT[@]}"}
        done ;;
    final) ml deblend-campaign final --split holdout ${AFTER_OPT[@]+"${AFTER_OPT[@]}"} ;;
    status) ml deblend-advance status ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "logs: $(pwd)/logs; products: $DP2_ML_ROOT/deblend; campaign state: $DP2_ML_ROOT/deblend/campaign"
