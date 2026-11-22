# d2_blend_strength.py
# Anna Preto
# AstroDeep Project
# How blended in Rubin are the D2 targets, against the candidates that were rejected? D2 asks for a Rubin blend
# (sb_rubin > sb_rubin_min) and a Euclid separation (masking the other sources barely changes the target): the
# closer the neighbour, the stronger the Rubin blend but the more its mask touches the target, so the two cuts pull
# against each other. rejected.parquet does not keep sb_rubin, so the candidate table of each patch is built again
# from the catalogues (dataset2.candidate_targets: no image read, minutes) and joined with the D2 outcome.
# Writes d2_blend_strength.csv (one row per candidate), prints quantiles of sb_rubin, partner separation and VIS
# magnitude per outcome, and draws their histograms (kept vs euclid_separation vs euclid_overlap).
# Usage: python scripts/inspect/d2_blend_strength.py <ML config> <out dir> [--d2 DIR]

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dp2_cutouts.mlconfig import MLConfig  # noqa: E402
from dp2_cutouts.pairs import dataset2  # noqa: E402
from dp2_cutouts.pairs.store import Store  # noqa: E402
from dp2_cutouts.pipeline.patches import load_patches, patch_case  # noqa: E402

COLS = ("sb_rubin", "partner_sep_arcsec", "euclid_vis_mag", "euclid_overlap_frac")


def outcome_table(cfg, d2):
    st = Store(d2)
    kept = st.meta[["euclid_object_id", "group_id"]].assign(outcome="kept", detail="")
    rej = st.rejected()
    rej = rej[rej["group_id"].notna()][["euclid_object_id", "group_id", "reason", "detail"]]
    rej = rej.rename(columns={"reason": "outcome"})
    out = pd.concat([kept, rej], ignore_index=True)
    out["patch_name"] = out["group_id"].str.split(":").str[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--d2", help="D2 dataset folder (default: datasets/d2 under ml_root)")
    args = ap.parse_args()
    cfg = MLConfig.load(args.config)
    d2 = Path(args.d2) if args.d2 else cfg.dataset_dir("d2")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    status = outcome_table(cfg, d2)
    pcfg = cfg.pipeline()
    tabs = []
    for p in load_patches(pcfg, sorted(status["patch_name"].unique())):
        cand, _ = dataset2.candidate_targets(patch_case(pcfg, p), cfg, p)
        if len(cand):
            tabs.append(cand)
        print(f"{p['name']}: {len(cand)} candidates", flush=True)
    cand = pd.concat(tabs, ignore_index=True)
    t = cand.merge(status[["euclid_object_id", "group_id", "outcome", "detail"]], on=["euclid_object_id", "group_id"],
                   how="left")
    t["outcome"] = t["outcome"].fillna("unknown")
    t.to_csv(out / "d2_blend_strength.csv", index=False)

    sb_min = float(cfg["dataset2"]["sb_rubin_min"])
    print(f"\n{len(t)} candidates; sb_rubin_min = {sb_min}")
    print(t["outcome"].value_counts().to_string())
    for c in COLS:
        q = t.groupby("outcome")[c].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack().round(3)
        q.insert(0, "n_finite", t.groupby("outcome")[c].count())
        print(f"\n{c} (quantiles 10/25/50/75/90 %)\n{q.to_string()}")
    # Fraction kept per sb_rubin bin, among the Rubin-blended candidates.
    b = t[t["sb_rubin"] > sb_min].copy()
    edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, np.inf])
    b["sb_bin"] = pd.cut(b["sb_rubin"], edges, include_lowest=True)
    f = b.groupby("sb_bin", observed=True)["outcome"].value_counts().unstack(fill_value=0)
    f["frac_kept"] = (f.get("kept", 0) / f.sum(axis=1)).round(3)
    print(f"\noutcome per sb_rubin bin (Rubin-blended candidates)\n{f.to_string()}")

    show = ["kept", "euclid_separation", "euclid_overlap"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, c, rng_ in zip(axes, ("sb_rubin", "partner_sep_arcsec", "euclid_vis_mag"), ((0, 1.2), (0, 6), (18, 28))):
        bins = np.linspace(*rng_, 40)
        for o in show:
            v = t.loc[t["outcome"] == o, c].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v):
                ax.hist(np.clip(v, *rng_), bins, histtype="step", density=True, lw=1.4, label=f"{o} ({len(v)})")
        ax.set_xlabel(c)
        ax.set_yticks([])
    axes[0].axvline(sb_min, color="k", ls=":", lw=0.8)
    axes[0].legend(fontsize=7)
    fig.suptitle("D2 candidates: kept vs rejected (normalised histograms)")
    fig.tight_layout()
    fig.savefig(out / "d2_blend_strength.png", dpi=130)
    print(f"\n-> {out / 'd2_blend_strength.csv'}, {out / 'd2_blend_strength.png'}")


if __name__ == "__main__":
    main()
