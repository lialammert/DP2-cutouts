# wandb_vae_images.py
# Anna Preto
# AstroDeep Project
# Images of finished VAE stages (Rubin VAE, K1, K2, C1) from their best weights, added to their W&B runs (id in
# <run>/wandb_id) under images/: for the first 4 validation examples, observed | model | residual, Rubin i band
# and / or VIS, as the live tracking draws them. 2026-11-28: K1's images failed during training (cuDNN autotuning)
# and the Rubin VAE and C1 had lost their tracking. Computed on the CPU (no GPU needed) and uploaded online: run on a
# login node (internet), with wandb from DP2_WANDB_LIB and the account of `wandb login`. DP2 pixels go to the W&B
# project, as with DP2_WANDB_IMAGES=1.
# Usage (from the repository, environment as for the campaign):
#   JAX_PLATFORMS=cpu PYTHONPATH=$DP2_WANDB_LIB python scripts/inspect/wandb_vae_images.py config/ml_jeanzay.yaml \
#       rubin_vae k1 c1

import argparse
import json
import os

from dp2_cutouts.deblend import tracking
from dp2_cutouts.deblend.campaign import run_dir
from dp2_cutouts.mlconfig import MLConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("stages", nargs="+", choices=["rubin_vae", "k1", "k2", "c1"])
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    cfg = MLConfig.load(a.config, [])
    import wandb
    for stage in a.stages:
        rd = run_dir(cfg, stage)
        if not (rd / "DONE").exists() or not (rd / "wandb_id").exists():
            print(f"{stage}: not finished or no wandb_id in {rd}: skipped")
            continue
        imgs = tracking.finished_vae_panels(cfg, stage, a.n)
        if not imgs:
            print(f"{stage}: no image")
            continue
        sched = json.loads((rd / "run.json").read_text())["settings"].get("schedule", {})
        best = json.loads((rd / "best.json").read_text()) if (rd / "best.json").exists() else {}
        d = {}
        for title, img in imgs:
            d.setdefault(f"images/{title.split(' #')[0]}", []).append(
                wandb.Image(img, caption=f"{title}: observed | model | residual (best weights, epoch "
                                         f"{best.get('epoch', '?')})"))
        d["epoch"] = int(sched.get("epochs", 0) or 0)
        rid = (rd / "wandb_id").read_text().strip()
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dp2-deblend"),
                         entity=os.environ.get("WANDB_ENTITY") or None, id=rid, resume="allow", name=rd.name,
                         mode="online")
        tracking.define_axes(run)
        run.log(d)
        run.finish()
        print(f"{stage}: {len(imgs)} images added to W&B run {rid} ({rd.name})")


if __name__ == "__main__":
    main()
