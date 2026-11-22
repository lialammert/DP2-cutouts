# phase0_flux_bias.py
# Anna Preto
# AstroDeep Project
# Why does the phase 0 report a large flux loss in some bands (median of rec / obs - 1 in an aperture)? Two
# readings: (A) a ratio artefact: at aperture S/N ~ 1 the observed flux is often near zero or negative and the median
# of the ratio goes negative even for a perfect reconstruction; (B) a real shrinkage: the VAE, a denoiser, pulls
# low-S/N flux towards its prior. Per band, from phase0_sample.parquet: the aperture S/N (aperture noise from the
# median sky sigma of the report and the aperture area), the fraction of non-positive observed fluxes, the ratio of
# sums (noise averages out: insensitive to A), the median ratio overall and in bins of aperture S/N.
# Reads a few MB: runs on the login node.
# Usage: python scripts/inspect/phase0_flux_bias.py <phase0 evaluation folder> [--aperture-px 8]

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = "ugrizy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase0_dir")
    ap.add_argument("--aperture-px", type=float, default=8.0, help="stage1_acceptance.flux_aperture_px")
    args = ap.parse_args()
    d = Path(args.phase0_dir)
    df = pd.read_parquet(d / "phase0_sample.parquet")
    rep = None
    for p in sorted(glob.glob(str(d / "*.json"))):
        r = json.load(open(p))
        if "normalisation" in r:
            rep = r
            break
    if rep is None:
        raise SystemExit(f"no phase 0 report with a 'normalisation' block in {d}")
    sky = rep["normalisation"]["dp2_median_sky_sigma_vae_units"]
    scale = rep["normalisation"].get("scale_per_band")
    n_ap = np.pi * args.aperture_px ** 2
    print(f"{len(df)} stamps; aperture radius {args.aperture_px} px (~{n_ap:.0f} px)")
    if scale:
        print("scale_per_band:", dict(zip(BANDS, np.round(scale, 2))))
    rows = []
    for b in BANDS:
        obs, rec = df[f"obs_flux_{b}"].to_numpy(float), df[f"rec_flux_{b}"].to_numpy(float)
        ok = np.isfinite(obs) & np.isfinite(rec)
        obs, rec = obs[ok], rec[ok]
        noise = float(sky[b]) * np.sqrt(n_ap)
        snr = obs / noise
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = rec / obs - 1.0
        row = dict(band=b, sky_sigma=float(sky[b]), aperture_noise=noise, median_snr_ap=float(np.median(snr)),
                   frac_obs_le_0=float(np.mean(obs <= 0)), ratio_of_sums=float(rec.sum() / obs.sum() - 1.0),
                   median_ratio=float(np.nanmedian(ratio)))
        for lo, hi in ((-np.inf, 1), (1, 3), (3, 10), (10, 30), (30, np.inf)):
            s = (snr >= lo) & (snr < hi)
            row[f"n_snr_{lo}_{hi}"] = int(s.sum())
            row[f"med_ratio_snr_{lo}_{hi}"] = float(np.nanmedian(ratio[s])) if s.sum() >= 10 else np.nan
            row[f"sums_snr_{lo}_{hi}"] = float(rec[s].sum() / obs[s].sum() - 1.0) if s.sum() >= 10 else np.nan
        rows.append(row)
    t = pd.DataFrame(rows).set_index("band")
    pd.set_option("display.width", 200)
    print("\nper band (ratio = rec / obs - 1; ratio_of_sums = sum(rec) / sum(obs) - 1)")
    print(t[["sky_sigma", "aperture_noise", "median_snr_ap", "frac_obs_le_0", "median_ratio",
             "ratio_of_sums"]].round(3).to_string())
    bins = [c for c in t.columns if c.startswith("n_snr_")]
    print("\nnumber of stamps per aperture S/N bin")
    print(t[bins].rename(columns=lambda c: c[6:]).to_string())
    print("\nmedian ratio per aperture S/N bin (A: biased only at low S/N; B: still biased at high S/N)")
    print(t[[c for c in t.columns if c.startswith("med_ratio_")]].rename(columns=lambda c: c[15:]).round(3)
          .to_string())
    print("\nratio of sums per aperture S/N bin")
    print(t[[c for c in t.columns if c.startswith("sums_")]].rename(columns=lambda c: c[9:]).round(3).to_string())


if __name__ == "__main__":
    main()
