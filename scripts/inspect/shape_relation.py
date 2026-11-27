# shape_relation.py
# Anna Preto
# AstroDeep Project
# Is the Rubin / VIS shape relation the D2 evaluation uses as reference (deblend/evaluate.py d1_relation) sound?
# On D1 validation pairs, the same galaxy is measured in both instruments with the evaluation's own functions:
#   moments  adaptive moments, PSF-convolved (e1, e2)
#   rg       REGAUSS PSF-corrected distortion, Gaussian PSF model from the catalogue (e1_rg, e2_rg): the reference
# For each estimate, component and S/N_i bin: the least-squares fit e_Rubin = a e_VIS + b (what d1_relation does),
# a robust one (Theil-Sen on |e| < 1), the correlation, the share of |e| > 1 (failed corrections), and the robust slope
# after flipping the sign of e1, of e2 or of both (a flipped axis gives a negative e2 slope, a 90 degree rotation a
# negative e1 slope, a swap of x and y negative e1 only). Writes a figure of e_Rubin against e_VIS.
# Reads D1 arrays through memory maps and only the metadata columns it needs (fits a login node).
# Usage: python scripts/inspect/shape_relation.py <ML config> <out.png> [--n 3000] [--seed 0]

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import theilslopes  # noqa: E402

from dp2_cutouts.mlconfig import MLConfig  # noqa: E402

BINS = [10, 20, 50, 100, 1e9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = MLConfig.load(args.config)
    from dp2_cutouts.deblend.evaluate import psf_sigma_arcsec_from_fwhm, shapes
    from dp2_cutouts.pairs import common
    root = Path(cfg.dataset_dir("d1"))
    cols = [c for c in ("split", "snr_i", "psf_sigma_i", "psf_e1_i", "psf_e2_i", "resolution_i", "e1", "e2",
                        "e1_vis", "e2_vis") if c in pd.read_parquet(root / "meta.parquet", columns=["split"]).columns
            or True]
    import pyarrow.parquet as pq
    have = set(pq.read_schema(root / "meta.parquet").names)
    meta = pd.read_parquet(root / "meta.parquet", columns=[c for c in cols if c in have])
    val = np.flatnonzero((meta["split"] == "val").to_numpy() & (meta["snr_i"].to_numpy(float) >= BINS[0]))
    # Spread over S/N: up to n / 4 rows per S/N bin.
    rng = np.random.default_rng(args.seed)
    snr = meta["snr_i"].to_numpy(float)[val]
    pick = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        r = val[(snr >= lo) & (snr < hi)]
        pick.append(rng.choice(r, min(len(r), args.n // 4), replace=False) if len(r) else r)
    rows = np.sort(np.concatenate(pick))
    arr = {k: np.load(root / f"{k}.npy", mmap_mode="r") for k in
           ("rubin_flux", "rubin_ivar", "rubin_mask", "euclid_flux", "euclid_ivar", "euclid_mask", "euclid_fill")
           if (root / f"{k}.npy").exists()}
    rb, eb = list(cfg["rubin"]["bands"]), list(cfg["euclid"]["bands"])
    ib, iv = rb.index("i"), eb.index("VIS")
    s_vis = psf_sigma_arcsec_from_fwhm(cfg["euclid"].get("vis_psf_fwhm", 0.16)) / float(cfg["euclid"]["pixel_scale"])
    recs = []
    for k, i in enumerate(rows):
        r = meta.iloc[i]
        rf, ri, rm = arr["rubin_flux"][i][..., ib], arr["rubin_ivar"][i][..., ib], arr["rubin_mask"][i][..., ib]
        ef, ei, em = arr["euclid_flux"][i][..., iv], arr["euclid_ivar"][i][..., iv], arr["euclid_mask"][i][..., iv]
        fill = arr["euclid_fill"][i] if "euclid_fill" in arr else np.zeros(ef.shape, bool)
        rv = ~rm & (ri > 0) & np.isfinite(ri) & np.isfinite(rf)
        ev = ~em & ~fill & (ei > 0) & np.isfinite(ei) & np.isfinite(ef)
        sr = shapes(np.nan_to_num(rf), rv, float(r["psf_sigma_i"]) / common.RUBIN_SCALE,
                    float(r.get("psf_e1_i", 0.0) or 0.0), float(r.get("psf_e2_i", 0.0) or 0.0))
        sv = shapes(np.nan_to_num(ef), ev, s_vis)
        recs.append(dict(snr_i=float(r["snr_i"]), **{f"r_{q}": v for q, v in sr.items()},
                         **{f"v_{q}": v for q, v in sv.items()}))
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{len(rows)} measured", flush=True)
    df = pd.DataFrame(recs)
    out = Path(args.out)
    df.to_parquet(out.with_suffix(".parquet"), index=False)

    def fit(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 20:
            return dict(n=int(len(x)))
        a, b = np.polyfit(x, y, 1)
        good = (np.abs(x) < 1) & (np.abs(y) < 1)
        res = dict(n=int(len(x)), frac_abs_gt1=float(1 - good.mean()), ls_a=float(a), ls_b=float(b),
                   r=float(np.corrcoef(x[good], y[good])[0, 1]) if good.sum() > 10 else np.nan)
        if good.sum() > 10:
            for name, sx in (("robust_a", 1.0), ("robust_a_flipped", -1.0)):
                res[name] = float(theilslopes(y[good], sx * x[good])[0])
        return res

    lines = []
    for est, suf in (("moments", ""), ("rg", "_rg")):
        ok = df["r_ok" + ("_rg" if suf else "")].astype(bool) & df["v_ok" + ("_rg" if suf else "")].astype(bool)
        for comp in ("e1", "e2"):
            for lo, hi in [(BINS[0], 1e9)] + list(zip(BINS[:-1], BINS[1:])):
                sel = ok & (df["snr_i"] >= lo) & (df["snr_i"] < hi)
                f = fit(df.loc[sel, f"v_{comp}{suf}"].to_numpy(float), df.loc[sel, f"r_{comp}{suf}"].to_numpy(float))
                lines.append(dict(estimate=est, comp=comp, snr_i=f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}", **f))
    t = pd.DataFrame(lines)
    pd.set_option("display.width", 170)
    print(t.round(3).to_string(index=False))
    print("\nexpected for a sound relation: r > 0, robust_a > 0 (below 1: different PSF and band), robust_a_flipped < 0;"
          "\nrobust_a < 0 with robust_a_flipped > 0 means that component's sign differs between the Rubin and VIS frames.")

    fig, axs = plt.subplots(2, 2, figsize=(9, 8))
    for i, (est, suf) in enumerate((("moments", ""), ("REGAUSS (the reference)", "_rg"))):
        for j, comp in enumerate(("e1", "e2")):
            ax = axs[i, j]
            x, y = df[f"v_{comp}{suf}"].to_numpy(float), df[f"r_{comp}{suf}"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1.5) & (np.abs(y) < 1.5)
            ax.scatter(x[ok], y[ok], s=3, c=np.log10(df["snr_i"].to_numpy(float)[ok]), cmap="viridis", alpha=0.6)
            ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
            ax.set_xlim(-1, 1), ax.set_ylim(-1, 1)
            ax.set_xlabel(f"VIS {comp}"), ax.set_ylabel(f"Rubin i {comp}")
            ax.set_title(f"{est}, {comp} (colour: log S/N_i)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\n-> {out}, {out.with_suffix('.parquet')} ({len(df)} D1 val pairs)")


if __name__ == "__main__":
    main()
