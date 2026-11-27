#!/bin/bash
# deblend.slurm
# Anna Preto
# AstroDeep Project
# One step of the deblending campaign on one A100: dp2-ml <ml config> <command> [args], e.g.
#   sbatch --account=<projet>@a100 scripts/jeanzay/ml/deblend.slurm deblend-train k1
#   sbatch --account=<projet>@a100 scripts/jeanzay/ml/deblend.slurm deblend-eval d2 --split test
# Normally submitted by the campaign (dp2-ml deblend-campaign / deblend-advance, see deblend/campaign.py), not by
# hand. Defaults to the 2 h A100 dev QoS; on the V100 (2026-11-27: the A100 allocation is over-consumed), from the
# login node before submitting:
#   export DP2_GPU_ACCOUNT=<projet>@v100 DP2_ML_VENV=<V100 venv>
#   export DP2_GPU_EXTRA="-C v100-32g --qos=qos_gpu-t3 --time=20:00:00 --cpus-per-task=10"
# - a training stops on its own after DEBLEND_TIME_BUDGET_MIN minutes (default: the job's time limit minus 20 min,
#   i.e. 100 min in a 2 h job, 1180 min in a 20 h one) with its backup written, without DONE; the campaign
#   resubmits it and it resumes from <run>/backup;
# - when the job ends (or 5 min before its walltime), and if it belongs to a campaign (DEBLEND_CAMPAIGN), it runs
#   dp2-ml deblend-advance, which submits whatever has become ready.
#SBATCH --job-name=dp2-deblend
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/deblend_%x_%j.out

source "${DP2_ML_SCRIPTS:-${SLURM_SUBMIT_DIR:-.}/scripts/jeanzay/ml}/env.sh"
# Time budget from the job's own time limit (D-HH:MM:SS, HH:MM:SS or MM:SS), 20 min kept for the backup and the
# validation pass.
limit_min() {
    local t d=0 h=0 m=0 sec=0
    t="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'TimeLimit=[^ ]*' | cut -d= -f2)"
    [[ -z "$t" || "$t" == UNLIMITED ]] && { echo 120; return; }
    [[ "$t" == *-* ]] && { d="${t%%-*}"; t="${t#*-}"; }
    IFS=: read -r -a p <<< "$t"
    if (( ${#p[@]} == 3 )); then h=${p[0]}; m=${p[1]}; sec=${p[2]}; else m=${p[0]}; sec=${p[1]:-0}; fi
    echo $(( 10#$d * 1440 + 10#$h * 60 + 10#$m ))
}
if [[ -z "${DEBLEND_TIME_BUDGET_MIN:-}" ]]; then
    lim="$(limit_min)"
    DEBLEND_TIME_BUDGET_MIN=$(( lim > 40 ? lim - 20 : lim / 2 ))
fi
export DEBLEND_TIME_BUDGET_MIN
echo "[deblend] time budget ${DEBLEND_TIME_BUDGET_MIN} min" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv || true

advance() {
    if [[ -n "${DEBLEND_CAMPAIGN:-}" ]]; then
        # shellcheck disable=SC2086
        dp2-ml "$DP2_ML_CONFIG" deblend-advance ${DEBLEND_SET:-} || echo "[deblend] deblend-advance failed" >&2
    fi
}
on_walltime() {
    echo "[deblend] walltime in 5 min: advancing the campaign (this step is resubmitted if it did not finish)" >&2
    advance
    exit 3
}
trap on_walltime USR1

# In the background so that the USR1 trap runs while waiting.
# shellcheck disable=SC2086
srun dp2-ml "$DP2_ML_CONFIG" "$@" ${DEBLEND_SET:-} &
rc=0
wait $! || rc=$?
echo "[deblend] dp2-ml $* ended with code $rc" >&2
advance
exit "$rc"
