#!/bin/bash
# submit_ml.sh
# Anna Preto
# AstroDeep Project
# Submits the cross-instrument work on Jean Zay, step by step or all at once, with Slurm dependencies.
# Run from the repository root, on a login node, after the stamp pipeline has assembled its final dataset.
#
#   export DP2_ML_VENV=$WORK/venvs/dp2-ml DP2_CONFIG=config/<pipeline>.yaml DP2_ML_CONFIG=config/ml_jeanzay.yaml
#   export DP2_ML_ROOT=$WORK/dp2-ml DP2_CPU_ACCOUNT=<projet>@cpu DP2_GPU_ACCOUNT=<projet>@a100
#   export DEBVADER_RUN=<path of the DebVAder run>                 # given here, never written in a file
#   scripts/jeanzay/ml/submit_ml.sh mmu-mirror                     # 0. MMU partitions of all patches, downloaded
#        once into euclid.mmu_cache (~30 GB each; `dp2-ml $DP2_ML_CONFIG mmu-mirror --dry-run` lists them first)
#   scripts/jeanzay/ml/submit_ml.sh probe <tract_patch>            # 1. which Euclid image source works
#   scripts/jeanzay/ml/submit_ml.sh phase0                         # 2. reference VAE on DP2 stamps
#   scripts/jeanzay/ml/submit_ml.sh datasets                       # 3. D1, D2, quality, controlled blends
#   scripts/jeanzay/ml/submit_ml.sh rubin-vae                      # 3b. DebVAder stage 1 trained on D1 (Rubin),
#        DEBVADER_RUN = paper_baseline here (template of the architecture); afterwards DEBVADER_RUN is the new run
#        ($EXPERIMENTS_DIR/<rubin_vae.run_name>) for every step below
#   scripts/jeanzay/ml/submit_ml.sh d2                             # 3c. D2 alone, rebuilt (old one moved aside)
#   scripts/jeanzay/ml/submit_ml.sh stage1 [ablation ...]          # 4. Stage 1 runs (default: stage1.default_runs);
#        AFTER_JOB=<job id> makes them wait for that job (e.g. the reference VAE still training), also for stage2
#   STAGE1_RUN=s1_latent_cross scripts/jeanzay/ml/submit_ml.sh stage2 [ablation ...]   # 5. Stage 2 runs
#   scripts/jeanzay/ml/submit_ml.sh notebooks                      # 6. executed notebooks with the figures
#   AFTER=afterok:<assemble>,afterany:<job> scripts/jeanzay/ml/submit_ml.sh rebuild   # 3 + 3b after the given
#        Slurm dependencies: the current datasets folder is moved aside (datasets_v1, _v2, ...), D1, D2, quality,
#        blends are built, then the reference Rubin VAE is trained once the D1 quality report passes
#   scripts/jeanzay/ml/submit_ml.sh all                            # 2 to 6 chained (STAGE1_RUN: teacher of 5),
#        with DEBVADER_RUN already the reference VAE (3b done)
# FETCH_MAX (default 4) caps the simultaneous D1 / D2 tasks. Reruns skip shards and runs already done.
# CPU jobs run on DP2_CPU_PARTITION (default cpu_p1, no internet: the Euclid images come from the MMU mirror) with
# DP2_CPU_CPUS cores (default 10, i.e. ~40 GB on cpu_p1). Only mmu-mirror needs internet and always goes to prepost;
# with euclid.image_source irsa (or mmu_cache null), set DP2_CPU_PARTITION=prepost.

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
STEP="${1:?step: mmu-mirror | probe | phase0 | datasets | rebuild | d2 | rubin-vae | stage1 | stage2 | notebooks | all}"
shift || true

DP2_CPU_PARTITION="${DP2_CPU_PARTITION:-cpu_p1}"
DP2_CPU_CPUS="${DP2_CPU_CPUS:-10}"
# SBATCH_QOS / SBATCH_TIMELIMIT / SBATCH_ACCOUNT set for GPU work in the shell would override the CPU jobs' own
# headers (a GPU QoS is invalid on cpu_p1): the CPU and prepost jobs are submitted without them.
NOGPU_ENV=(env -u SBATCH_QOS -u SBATCH_TIMELIMIT -u SBATCH_ACCOUNT)
CPU=("${NOGPU_ENV[@]}" sbatch --parsable --account="$DP2_CPU_ACCOUNT" --partition="$DP2_CPU_PARTITION"
     --cpus-per-task="$DP2_CPU_CPUS")
NET=("${NOGPU_ENV[@]}" sbatch --parsable --account="$DP2_CPU_ACCOUNT" --partition=prepost)
# qos_cpu-dev (2 h, high priority) for the short probe, where the partition offers it.
DEV_QOS=()
[[ "$DP2_CPU_PARTITION" == cpu_p1 ]] && DEV_QOS=(--qos=qos_cpu-dev)
gpu() {
    # DP2_GPU_EXTRA: sbatch options added to (and overriding) the A100 headers of the GPU jobs, e.g. on the V100
    # when the A100 allocation is over-consumed: DP2_GPU_ACCOUNT=<projet>@v100
    # DP2_GPU_EXTRA="-C v100-32g --qos=qos_gpu-t3" (env.sh loads arch/a100 only on the AMD A100 nodes).
    : "${DP2_GPU_ACCOUNT:?set DP2_GPU_ACCOUNT=<projet>@a100}"
    : "${DEBVADER_RUN:?set DEBVADER_RUN to the paper_baseline run folder}"
    local extra=()
    [[ -n "${DP2_GPU_EXTRA:-}" ]] && read -r -a extra <<< "$DP2_GPU_EXTRA"
    sbatch --parsable --account="$DP2_GPU_ACCOUNT" ${extra[@]+"${extra[@]}"} "$@"
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
    # The runs launched when none is named: <stage>.default_runs, or else every ablation.
    python -c "
import yaml
s = yaml.safe_load(open('$DP2_ML_CONFIG'))['$1']
print(' '.join(s.get('default_runs') or s['ablations']))"
}

submit_mmu_mirror() {
    "${NET[@]}" --job-name=dp2-mmu-mirror --time=20:00:00 "$DP2_ML_SCRIPTS/cpu.slurm" mmu-mirror
}

submit_probe() {
    local patch="${1:?probe needs a patch name, e.g. 5063_34}"
    "${CPU[@]}" ${DEV_QOS[@]+"${DEV_QOS[@]}"} --job-name=dp2-probe --time=00:30:00 "$DP2_ML_SCRIPTS/cpu.slurm" probe-euclid --patch "$patch"
}

submit_phase0() {
    gpu --job-name=dp2-phase0 --qos=qos_gpu_a100-dev --time=01:00:00 "$DP2_ML_SCRIPTS/gpu.slurm" phase0
}

submit_datasets() {
    # $1: job(s) the shard builders wait for (optional).
    local after="${1:-}" n last d1 d1m q1 bl d2 d2m q2
    n=$(n_patches); last=$((n - 1))
    d1=$("${CPU[@]}" --job-name=dp2-d1 $(dep "$after") --array=0-"$last"%"$FETCH_MAX" "$DP2_ML_SCRIPTS/cpu.slurm" \
         d1 --index '$SLURM_ARRAY_TASK_ID')
    d1m=$("${CPU[@]}" --job-name=dp2-d1-merge --dependency=afterany:"$d1" --time=10:00:00 \
          "$DP2_ML_SCRIPTS/cpu.slurm" d1-merge)
    q1=$("${CPU[@]}" --job-name=dp2-q1 --dependency=afterok:"$d1m" "$DP2_ML_SCRIPTS/cpu.slurm" quality d1)
    bl=$("${CPU[@]}" --job-name=dp2-blends --dependency=afterok:"$d1m" "$DP2_ML_SCRIPTS/cpu.slurm" blends)
    d2=$("${CPU[@]}" --job-name=dp2-d2 $(dep "$after") --array=0-"$last"%"$FETCH_MAX" "$DP2_ML_SCRIPTS/cpu.slurm" \
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

submit_d2() {
    # D2 alone (D1 and its statistics already built): shards, merge, quality report.
    local after="${1:-}" n last d2 d2m q2
    n=$(n_patches); last=$((n - 1))
    d2=$("${CPU[@]}" --job-name=dp2-d2 $(dep "$after") --array=0-"$last"%"$FETCH_MAX" "$DP2_ML_SCRIPTS/cpu.slurm" \
         d2 --index '$SLURM_ARRAY_TASK_ID')
    d2m=$("${CPU[@]}" --job-name=dp2-d2-merge --dependency=afterany:"$d2" --time=10:00:00 \
          "$DP2_ML_SCRIPTS/cpu.slurm" d2-merge)
    q2=$("${CPU[@]}" --job-name=dp2-q2 --dependency=afterok:"$d2m" "$DP2_ML_SCRIPTS/cpu.slurm" quality d2)
    echo "d2 $d2, d2-merge $d2m, quality-d2 $q2" >&2
}

submit_rubin_vae() {
    # Export D1's Rubin stamps (CPU), then train DebVAder stage 1 on them and run phase 0 on the result (GPU).
    local after="${1:-}" ex tr
    : "${DEBVADER_RUN:?set DEBVADER_RUN to the template run (paper_baseline): its config.json gives the architecture}"
    ex=$("${CPU[@]}" --job-name=dp2-rvae-export $(dep "$after") "$DP2_ML_SCRIPTS/cpu.slurm" export-rubin-vae)
    tr=$(gpu --job-name=dp2-rubin-vae --dependency=afterok:"$ex" "$DP2_ML_SCRIPTS/rubin_vae.slurm")
    echo "rubin-vae: export $ex, train + phase0 $tr" >&2
    echo "$tr"
}

gpu_chain() {
    # One training as DP2_GPU_CHAIN (default 1) jobs in a row (afterany), for short QoS such as qos_gpu_a100-dev
    # (2 h): each job resumes from <run>/backup and a finished run (DONE) returns at once; only the last job runs
    # the evaluation ($2). $1: job(s) the first one waits for; $3: job name; then the dp2-ml command. Prints the
    # last job id.
    local after="$1" ev="$2" name="$3"; shift 3
    local n="${DP2_GPU_CHAIN:-1}" k id="" d ex
    for ((k = 1; k <= n; k++)); do
        if [[ -z "$id" ]]; then d=$(dep "$after"); else d="--dependency=afterany:$id"; fi
        ex="--export=ALL,AFTER_TRAIN="
        [[ $k -eq $n ]] && ex="--export=ALL,AFTER_TRAIN=$ev"
        id=$(gpu --job-name="$name" $d "$ex" "$DP2_ML_SCRIPTS/gpu.slurm" "$@")
    done
    echo "$id"
}

submit_stage1() {
    local after="${1:-}"; shift || true
    local names=("$@") ids=()
    [[ ${#names[@]} -eq 0 ]] && read -r -a names <<< "$(ablations stage1)"
    for a in "${names[@]}"; do
        ids+=("$(gpu_chain "$after" eval-stage1 "dp2-$a" train-stage1 --ablation "$a")")
    done
    echo "stage 1: ${names[*]} -> ${ids[*]} (last job of each chain of ${DP2_GPU_CHAIN:-1})" >&2
    (IFS=:; echo "${ids[*]}")
}

submit_stage2() {
    local after="${1:-}"; shift || true
    : "${STAGE1_RUN:?set STAGE1_RUN to the Stage 1 run used as teacher (e.g. s1_latent_cross)}"
    local names=("$@") ids=()
    [[ ${#names[@]} -eq 0 ]] && read -r -a names <<< "$(ablations stage2)"
    for a in "${names[@]}"; do
        ids+=("$(gpu_chain "$after" eval-stage2 "dp2-$a" train-stage2 --ablation "$a" --stage1-run "$STAGE1_RUN")")
    done
    echo "stage 2 (teacher $STAGE1_RUN): ${names[*]} -> ${ids[*]} (last job of each chain of ${DP2_GPU_CHAIN:-1})" >&2
    (IFS=:; echo "${ids[*]}")
}

submit_notebooks() {
    local after="${1:-}"
    "${NOGPU_ENV[@]}" sbatch --parsable --account="$DP2_CPU_ACCOUNT" $(dep "$after") \
        --export=ALL,NB_STAGE1_RUN="${STAGE1_RUN:-s1_latent_cross}",NB_STAGE2_RUN="${STAGE2_RUN:-s2_latent}" \
        "$DP2_ML_SCRIPTS/notebooks.slurm"
}

case "$STEP" in
    mmu-mirror) submit_mmu_mirror ;;
    probe) submit_probe "$@" ;;
    phase0) submit_phase0 ;;
    datasets) submit_datasets ;;
    rebuild)
        # Move the current datasets aside once the AFTER jobs are over (whatever their outcome: a test still reading
        # them only has to be finished), then rebuild everything and train the reference VAE on the new D1.
        # AFTER is a Slurm dependency list, e.g. afterok:<assemble>,afterany:<test reading the old datasets>.
        mv_job=$("${CPU[@]}" --job-name=dp2-datasets-aside --time=00:20:00 \
                 ${AFTER:+--dependency=$AFTER} "$DP2_ML_SCRIPTS/datasets_aside.sh")
        echo "datasets aside: $mv_job" >&2
        submit_datasets "$mv_job"
        submit_rubin_vae "$AFTER_STAGE1" >/dev/null
        ;;
    d2)
        # D2 rebuilt alone (e.g. new dataset2 settings): datasets/d2 moved aside (d2_v1, d2_v2, ...), then built.
        mv_job=$("${CPU[@]}" --job-name=dp2-d2-aside --time=00:20:00 "$DP2_ML_SCRIPTS/datasets_aside.sh" d2)
        echo "d2 aside: $mv_job" >&2
        submit_d2 "$mv_job"
        ;;
    rubin-vae) submit_rubin_vae "" >/dev/null ;;
    stage1) submit_stage1 "${AFTER_JOB:-}" "$@" >/dev/null ;;
    stage2) submit_stage2 "${AFTER_JOB:-}" "$@" >/dev/null ;;
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
