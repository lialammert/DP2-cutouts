#!/bin/bash
# submit_deblend.sh
# Anna Preto
# AstroDeep Project
# Submits the deblending campaign (Runs A, B, C) on Jean Zay as one graph of Slurm dependencies. Run from the
# repository root, on a login node, once D1 (built with dataset1.euclid_neighbour_mask) and D2 are merged.
#
#   export DP2_ML_VENV=... DP2_CONFIG=... DP2_ML_CONFIG=config/ml_jeanzay.yaml DP2_ML_ROOT=$SCRATCH/dp2-ml-full
#   export DP2_CPU_ACCOUNT=<projet>@cpu DP2_GPU_ACCOUNT=<projet>@a100
#   scripts/jeanzay/ml/submit_deblend.sh pilot     # every stage for PILOT_STEPS steps (run suffix _pilot), then
#                                                  # every evaluation on 500 val objects: stability, memory, speed
#   scripts/jeanzay/ml/submit_deblend.sh all       # the full campaign: stats, the 8 trainings, the evaluations
#   SEEDS="1 2" scripts/jeanzay/ml/submit_deblend.sh seeds   # replicates of A, B, C2, C3 (K1, K2, C1 shared)
#   scripts/jeanzay/ml/submit_deblend.sh final     # evaluations on the holdout split (final, once only)
#
# Graph of `all` (each training = DP2_GPU_CHAIN jobs of 2 h in a row, default 10, resuming from backups):
#   stats -> rubin_vae, k1, c1;  k1 -> k2, A, B;  c1 -> c2 -> c3;  evaluations: masking after k1; latent after
#   k1, c1; d1, d2, cost after every training. AFTER=<dependency list> delays the first jobs (e.g. afterok:<d2-merge>).
# Evaluations run on the test split; the holdout is only read by `final`.

set -euo pipefail
: "${DP2_ML_VENV:?}" "${DP2_CONFIG:?}" "${DP2_ML_CONFIG:?}" "${DP2_ML_ROOT:?}"
: "${DP2_CPU_ACCOUNT:?set DP2_CPU_ACCOUNT=<projet>@cpu}" "${DP2_GPU_ACCOUNT:?set DP2_GPU_ACCOUNT=<projet>@a100}"
DP2_CONFIG="$(realpath "$DP2_CONFIG")"
DP2_ML_CONFIG="$(realpath "$DP2_ML_CONFIG")"
DP2_ML_ROOT="$(realpath -m "$DP2_ML_ROOT")"
export DP2_ML_VENV DP2_CONFIG DP2_ML_CONFIG DP2_ML_ROOT
export DP2_ML_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
mkdir -p logs
STEP="${1:?step: pilot | all | seeds | final}"
CHAIN="${DP2_GPU_CHAIN:-10}"
PILOT_STEPS="${PILOT_STEPS:-300}"

NOGPU_ENV=(env -u SBATCH_QOS -u SBATCH_TIMELIMIT -u SBATCH_ACCOUNT)
CPU=("${NOGPU_ENV[@]}" sbatch --parsable --account="$DP2_CPU_ACCOUNT" --partition="${DP2_CPU_PARTITION:-cpu_p1}"
     --cpus-per-task="${DP2_CPU_CPUS:-10}")
gpu() {
    local extra=()
    [[ -n "${DP2_GPU_EXTRA:-}" ]] && read -r -a extra <<< "$DP2_GPU_EXTRA"
    sbatch --parsable --account="$DP2_GPU_ACCOUNT" ${extra[@]+"${extra[@]}"} "$@"
}
deps() {
    # deps id1 id2 ... -> --dependency=afterok:id1:id2 (empty ids skipped), plus $AFTER for the roots.
    local ids=() i
    for i in "$@"; do [[ -n "$i" ]] && ids+=("$i"); done
    if [[ ${#ids[@]} -gt 0 ]]; then (IFS=:; echo "--dependency=afterok:${ids[*]}"); fi
}
chain() {
    # chain <name> <n> <dependency option or ""> <dp2-ml args...>: n jobs in a row (afterany), prints the last id.
    local name="$1" n="$2" first="$3"; shift 3
    local id="" k d
    for ((k = 1; k <= n; k++)); do
        if [[ -z "$id" ]]; then d="$first"; else d="--dependency=afterany:$id"; fi
        id=$(gpu --job-name="$name" $d --export=ALL,DEBLEND_SET="${DEBLEND_SET:-}" \
             "$DP2_ML_SCRIPTS/deblend.slurm" "$@")
    done
    echo "$id"
}
root_dep() { [[ -n "${AFTER:-}" ]] && echo "--dependency=$AFTER" || true; }

stats_job() {
    "${CPU[@]}" --job-name=dp2-deblend-stats $(root_dep) --time=02:00:00 "$DP2_ML_SCRIPTS/cpu.slurm" deblend-stats
}

campaign() {
    # $1: steps per training ("" = full), $2: evaluation split, $3: seed option ("" or "--seed S"), $4: which stages
    local steps="$1" split="$2" seedopt="$3" which="$4" n="$CHAIN" ms=()
    [[ -n "$steps" ]] && { ms=(--max-steps "$steps"); n=1; }
    local st rv k1 k2 a b c1 c2 c3 ev
    st=$(stats_job); echo "stats $st" >&2
    if [[ "$which" == all ]]; then
        rv=$(chain dp2-rvae "$n" "$(deps "$st")" deblend-train rubin_vae "${ms[@]}")
        k1=$(chain dp2-k1 "$n" "$(deps "$st")" deblend-train k1 "${ms[@]}")
        c1=$(chain dp2-c1 "$n" "$(deps "$st")" deblend-train c1 "${ms[@]}")
        k2=$(chain dp2-k2 "$n" "$(deps "$k1")" deblend-train k2 "${ms[@]}")
    fi
    a=$(chain dp2-runA "$n" "$(deps "${k1:-}")" deblend-train A $seedopt "${ms[@]}")
    b=$(chain dp2-runB "$n" "$(deps "${k1:-}")" deblend-train B $seedopt "${ms[@]}")
    c2=$(chain dp2-c2 "$n" "$(deps "${c1:-}")" deblend-train c2 $seedopt "${ms[@]}")
    c3=$(chain dp2-c3 "$n" "$(deps "$c2")" deblend-train c3 $seedopt "${ms[@]}")
    echo "trainings: rubin_vae ${rv:-shared} k1 ${k1:-shared} k2 ${k2:-shared} c1 ${c1:-shared} A $a B $b c2 $c2 c3 $c3" >&2
    local all_done
    all_done=$(deps "${rv:-}" "${k2:-}" "$a" "$b" "$c3")
    for what in d1 d2 cost; do
        ev=$(gpu --job-name="dp2-eval-$what" $all_done --export=ALL,DEBLEND_SET="${DEBLEND_SET:-}" \
             "$DP2_ML_SCRIPTS/deblend.slurm" deblend-eval "$what" --split "$split" $seedopt)
        echo "eval $what $ev" >&2
    done
    if [[ "$which" == all ]]; then
        ev=$(gpu --job-name=dp2-eval-masking $(deps "$k1") --export=ALL,DEBLEND_SET="${DEBLEND_SET:-}" \
             "$DP2_ML_SCRIPTS/deblend.slurm" deblend-eval masking --split "$split")
        echo "eval masking $ev" >&2
        ev=$(gpu --job-name=dp2-eval-latent $(deps "$k1" "$c1") --export=ALL,DEBLEND_SET="${DEBLEND_SET:-}" \
             "$DP2_ML_SCRIPTS/deblend.slurm" deblend-eval latent --split "$split")
        echo "eval latent $ev" >&2
    fi
}

source "$DP2_ML_VENV/bin/activate"
case "$STEP" in
    pilot)
        export DEBLEND_SET="--set deblend.run_suffix=_pilot --set deblend.eval.max_examples=500"
        campaign "$PILOT_STEPS" val "" all ;;
    all) campaign "" test "" all ;;
    seeds)
        for s in ${SEEDS:?set SEEDS, e.g. \"1 2\"}; do campaign "" test "--seed $s" students; done ;;
    final)
        for what in d1 d2 cost; do
            gpu --job-name="dp2-final-$what" $(root_dep) "$DP2_ML_SCRIPTS/deblend.slurm" deblend-eval "$what" \
                --split holdout
        done ;;
    *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
echo "logs: $(pwd)/logs; products: $DP2_ML_ROOT/deblend"
