# wandb_backfill.py
# Anna Preto
# AstroDeep Project
# Rebuilds the epoch curves of a training whose W&B tracking was lost, from its <run>/history.csv (every epoch's
# training and validation metrics, gradient-norm quantiles, lr), into the same W&B run (id in <run>/wandb_id), as
# the epoch/* curves the live tracking writes. 2026-11-28: the Rubin VAE and C1 lost their tracking after an
# autotuning restart (fixed since). No image (they are not in history.csv).
# Runs online: on a login node (internet), with wandb from DP2_WANDB_LIB and the account of `wandb login`.
# Usage: PYTHONPATH=$DP2_WANDB_LIB python scripts/inspect/wandb_backfill.py config/ml_jeanzay.yaml rubin_vae c1

import argparse
import json
import math
import os

import pandas as pd

from dp2_cutouts.deblend.campaign import run_dir
from dp2_cutouts.mlconfig import MLConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    cfg = MLConfig.load(a.config, [f"deblend.run_suffix={a.suffix}"] if a.suffix else [])
    import wandb
    for stage in a.stages:
        rd = run_dir(cfg, stage)
        if not (rd / "wandb_id").exists() or not (rd / "history.csv").exists():
            print(f"{stage}: no wandb_id or history.csv in {rd}: skipped")
            continue
        rid = (rd / "wandb_id").read_text().strip()
        spe = int(json.loads((rd / "run.json").read_text())["settings"]["schedule"]["updates_per_epoch"])
        h = pd.read_csv(rd / "history.csv").drop_duplicates("epoch", keep="last").sort_values("epoch")
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dp2-deblend"),
                         entity=os.environ.get("WANDB_ENTITY") or None, id=rid, resume="allow", name=rd.name,
                         mode="online")
        run.define_metric("epoch")
        run.define_metric("epoch/*", step_metric="epoch")
        for _, row in h.iterrows():
            e = int(row["epoch"]) + 1
            d = {f"epoch/{c}": float(v) for c, v in row.items()
                 if c != "epoch" and isinstance(v, (int, float)) and math.isfinite(float(v))}
            d["epoch"] = e
            run.log(d, step=e * spe)
        run.finish()
        print(f"{stage}: {len(h)} epochs written to W&B run {rid} ({rd.name})")


if __name__ == "__main__":
    main()
