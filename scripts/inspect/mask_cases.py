# mask_cases.py
# Anna Preto
# AstroDeep Project
# One figure, three kinds of D2 targets, a few examples each (VIS, zoom on the target), to see what the before /
# after test and the lost-light indicator of mask_damage.py mean:
#   1. pass the before/after test: masking the neighbours does not change the measured shape;
#   2. fail it, but no target light under the masks (lost_sym < 0.005): the shape measured before masking was
#      pulled by the neighbour (blue ellipse), after masking it sits on the target alone (green ellipse);
#   3. fail it, with target light under the masks (lost_sym >= 0.05): the mask bites into the target.
# Each example: VIS before masking with the shape measured on it (blue), VIS after masking (masked pixels outlined
# in red) with the shape measured on it (green) and the blue one dashed. Ellipses: adaptive moments, drawn at 2 sigma.
# Needs mask_damage_d2.parquet (mask_damage.py) in the out folder. Targets drawn at random among VIS < 25 (visible).
# Usage: python scripts/inspect/mask_cases.py <ML config> <out dir> [--n 4] [--seed 0] [--max-mag 25]

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from d2_gallery import noise_sigma  # noqa: E402  (same folder)
from dp2_cutouts.mlconfig import MLConfig  # noqa: E402
from dp2_cutouts.pairs.common import moments  # noqa: E402
from dp2_cutouts.pairs.store import Store  # noqa: E402

HALF = 25          # zoom: (2 HALF + 1) VIS pixels around the target
CASES = (("1. pass the before/after test\nmasking changes nothing", lambda t: t["old_test_ok"]),
         ("2. fail it, no target light under the mask\nthe neighbour had pulled the measurement",
          lambda t: ~t["old_test_ok"] & (t["lost_sym"] < 0.005)),
         ("3. fail it, target light under the mask\nthe mask bites into the target",
          lambda t: ~t["old_test_ok"] & (t["lost_sym"] >= 0.05)))


def ellipse(ax, m, c0, color, ls="-"):
    # Adaptive-moment ellipse at 2 sigma; e1, e2 are distortions (a^2 - b^2) / (a^2 + b^2), sigma = sqrt(a b).
    if not m["ok"]:
        return
    e = min(np.hypot(m["e1"], m["e2"]), 0.95)
    q = np.sqrt((1 - e) / (1 + e))
    a, b = m["sigma"] / np.sqrt(q), m["sigma"] * np.sqrt(q)
    th = 0.5 * np.degrees(np.arctan2(m["e2"], m["e1"]))
    ax.add_patch(Ellipse((m["x"] - c0, m["y"] - c0), 4 * a, 4 * b, angle=th, fill=False, color=color, lw=1.2, ls=ls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4, help="examples per case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-mag", type=float, default=25.0)
    args = ap.parse_args()
    cfg = MLConfig.load(args.config)
    out = Path(args.out)
    tab = pd.read_parquet(out / "mask_damage_d2.parquet")
    tab = tab[np.isfinite(tab["lost_sym"]) & (tab["euclid_vis_mag"] < args.max_mag)]
    vis = list(cfg["euclid"]["bands"]).index("VIS")
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(len(CASES), 2 * args.n, figsize=(2.0 * 2 * args.n, 2.35 * len(CASES)), squeeze=False)
    for ci, (title, sel) in enumerate(CASES):
        g = tab[sel(tab)]
        share = 100 * len(g) / max(len(tab), 1)
        g = g.iloc[rng.permutation(len(g))[:args.n]]
        for k in range(args.n):
            a0, a1 = axes[ci, 2 * k], axes[ci, 2 * k + 1]
            a0.axis("off")
            a1.axis("off")
            if k >= len(g):
                continue
            r = g.iloc[k]
            st = Store(r["store"])
            i = [int(r["row"])]
            raw, iv = st.take("euclid_flux_unmasked", i)[0, ..., vis], st.take("euclid_ivar", i)[0, ..., vis]
            ef, fill = st.take("euclid_flux", i)[0, ..., vis], st.take("euclid_fill", i)[0]
            mk = st.take("euclid_mask", i)[0, ..., vis]
            n = raw.shape[0]
            c = n // 2
            w = (~mk).astype(np.int32)                      # the weight of the D2 before / after measurement
            m0 = moments(raw, w, (c, c))
            m1 = moments(ef, w, (c, c))
            s = noise_sigma(raw, iv)
            z = np.s_[c - HALF:c + HALF + 1, c - HALF:c + HALF + 1]
            for ax, img in ((a0, raw), (a1, ef)):
                ax.imshow(np.arcsinh(np.nan_to_num(img[z]) / s), origin="lower", cmap="gray_r", vmin=-1, vmax=6,
                          interpolation="nearest")
                ax.plot([HALF], [HALF], "+", color="orange", ms=6, mew=1)
            a1.contour(fill[z].astype(float), levels=[0.5], colors="red", linewidths=0.8)
            ellipse(a0, m0, c - HALF, "tab:blue")
            ellipse(a1, m0, c - HALF, "tab:blue", ls="--")
            ellipse(a1, m1, c - HALF, "tab:green")
            a0.set_title("before", fontsize=7)
            a1.set_title("after", fontsize=7)
            for ax in (a0, a1):
                ax.set_xlim(-0.5, 2 * HALF + 0.5)
                ax.set_ylim(-0.5, 2 * HALF + 0.5)
        axes[ci, 0].text(-0.08, 0.5, f"{title}\n({share:.0f} % of D2, VIS < {args.max_mag:g})", fontsize=8,
                         transform=axes[ci, 0].transAxes, ha="right", va="center")
    fig.suptitle("D2 targets, VIS zoom (5\").  Blue: shape measured before masking.  Green: after.  "
                 "Red: masked pixels.  Orange +: target", fontsize=9)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.02, wspace=0.05, hspace=0.25)
    path = out / "mask_cases.png"
    fig.savefig(path, dpi=130)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
