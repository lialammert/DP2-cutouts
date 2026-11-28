# clip_thresholds.py
# Anna Preto
# AstroDeep Project
# Gradient-clipping thresholds of the eight deblending stages from the pilot (2026-11-28; deblend/optim.py): the
# pilot runs without clipping and every run writes the histogram of its gradient norms before clipping
# (<run>/grad_norms.json, every update). Prints, per stage, the quantiles 50 / 90 / 95 / 99 % of the steady norms
# (after the warm-up; C2: after lambda_D1 is fixed) and of all of them, the number of non-finite norms (history.csv),
# and the threshold to set: deblend.<stage>.train.global_clipnorm = the steady 95 % quantile.
# Usage: python scripts/inspect/clip_thresholds.py <ML config> [--suffix _pilot] [--out clip_thresholds.json]

import argparse
import json
from pathlib import Path

import pandas as pd

from dp2_cutouts.deblend.campaign import STAGES, run_dir
from dp2_cutouts.mlconfig import MLConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--suffix", default="_pilot")
    ap.add_argument("--out", default="clip_thresholds.json")
    a = ap.parse_args()
    cfg = MLConfig.load(a.config, [f"deblend.run_suffix={a.suffix}"])
    rows, out = [], {}
    for stage in STAGES:
        rd = run_dir(cfg, stage)
        p = rd / "grad_norms.json"
        if not p.exists():
            rows.append(dict(stage=stage, status=f"no {p}"))
            continue
        h = json.loads(p.read_text())
        nonfinite = None
        if (rd / "history.csv").exists():
            hist = pd.read_csv(rd / "history.csv")
            if "grad_norm_nonfinite" in hist:
                nonfinite = int(hist["grad_norm_nonfinite"].fillna(0).sum())
        st, al = h["steady"], h["all"]
        r = dict(stage=stage, updates=h.get("updates"), steady_n=st["n"], steady_from=h.get("stats_from"),
                 q50=st["q50"], q90=st["q90"], q95=st["q95"], q99=st["q99"], all_q95=al["q95"], all_q99=al["q99"],
                 nonfinite=nonfinite, clipped_in_pilot=h.get("threshold"))
        rows.append(r)
        out[stage] = r
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nThresholds (steady 95 % quantile):")
    for stage, r in out.items():
        if r["q95"] is not None:
            print(f"  deblend.{stage}.train.global_clipnorm: {r['q95']:.4g}")
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
