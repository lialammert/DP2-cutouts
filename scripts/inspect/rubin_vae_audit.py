# rubin_vae_audit.py
# Anna Preto
# AstroDeep Project
# Audit of a trained Rubin VAE (debvader-keras run), no training: is the decoder's y output dead because of the
# final ReLU, or because there is nothing to reconstruct? Reads the run's checkpoint, the data set it was trained on
# (debvader real build: images.npy in nJy, meta.npz with d1_row) and D1 (catalogue columns), and writes into <out>:
#   summary.json                   every number below
#   1_preactivations.png           distribution of the mean pre-activations per band, galaxy region (before ReLU)
#   2_bands.png                    per band: pre-activations <= 0, gradient that reaches the mean, sigma, NLL,
#                                  galaxy region against sky region
#   3_bright_flux.png              reconstructed against observed aperture flux, targets bright in y, i, z by
#                                  CATALOGUE photometry (independent of the VAE)
#   4_examples_y.png               the brightest-in-y targets: observed y, mean y, sigma y, residual, then i
#   0_data.png                     the data seen by the VAE: per band S/N of the galaxy aperture
# Galaxy region: reduced radius <= 2 in the catalogue ellipse of the target (i-band moments, half-light scale);
# sky region: > 4. Catalogue S/N in a band: 5 * 10^(-0.4 (cModel mag - depth5)), depth5 the local 5-sigma point
# source depth from the variance plane.
# Usage: python scripts/inspect/rubin_vae_audit.py --run <debvader run dir> --out <dir>
#            [--data <images.npy folder, default: the run's dataset_path>] [--d1 <D1 dir, default: <data>/../d1>]
#            [--weights <.weights.h5>] [--n 3000] [--split val] [--n-bright 300]

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dp2_cutouts.ml import vae_audit as A  # noqa: E402  (sets KERAS_BACKEND before keras)
from dp2_cutouts.pairs.store import Store  # noqa: E402

SPLITS = {"train": 0, "val": 1, "test": 2}
BANDS = list("ugrizy")


def infer(vae, x_norm, batch=128):
    # Decoder output after the final ReLU, and the same with the final activation switched off (pre-activations).
    import keras
    from keras import ops
    fc = vae.decoder.final_conv
    act = fc.activation
    outs, pres = [], []
    for s in range(0, len(x_norm), batch):
        xb = ops.convert_to_tensor(x_norm[s:s + batch])
        mu = vae.mu(vae.encode_params(xb))
        outs.append(ops.convert_to_numpy(vae.decoder(mu, training=False)))
        try:
            fc.activation = keras.activations.linear
            pres.append(ops.convert_to_numpy(vae.decoder(mu, training=False)))
        finally:
            fc.activation = act
    out, pre = np.concatenate(outs), np.concatenate(pres)
    gap = float(np.abs(np.maximum(pre, 0) - out).max())
    if gap > 1e-4:
        raise RuntimeError(f"ReLU(pre-activation) differs from the decoder output by {gap}: the final activation "
                           "is not the ReLU this audit assumes")
    C = vae.n_bands
    return out[..., :C], 1e-4 + out[..., C:], pre[..., :C], pre[..., C:]


def geometry(m, n):
    c = (n - 1) / 2.0
    dx = m["rubin_subpixel_dx"].to_numpy(float) if "rubin_subpixel_dx" in m else np.zeros(len(m))
    dy = m["rubin_subpixel_dy"].to_numpy(float) if "rubin_subpixel_dy" in m else np.zeros(len(m))
    centres = np.column_stack([c + np.nan_to_num(dx), c + np.nan_to_num(dy)])
    ell = m[["ellipse_a", "ellipse_b", "ellipse_theta"]].to_numpy(float)
    return centres, ell


def evaluate(vae, imgs, m):
    n = imgs.shape[1]
    x = vae.normalise(imgs).astype(np.float32)
    mu, sigma, pre_mu, pre_sc = infer(vae, x)
    gal, sky = A.regions((n, n), *geometry(m, n))
    rec = vae.denormalise_np(mu)
    return dict(x=x, mu=mu, sigma=sigma, pre_mu=pre_mu, pre_sc=pre_sc, gal=gal, sky=sky,
                flux_obs=A.aperture_flux(imgs, gal), flux_rec=A.aperture_flux(rec, gal),
                sky_sigma=A.sky_sigma(imgs, sky), npix=gal.sum(axis=(1, 2)), rec=rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data")
    ap.add_argument("--d1")
    ap.add_argument("--weights")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-bright", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    run = Path(args.run)
    conf = json.loads((run / "config.json").read_text())
    data = Path(args.data or conf["data"]["dataset_path"])
    if not (data / "images.npy").exists():
        raise SystemExit(f"no images.npy in {data}: pass --data (the rubin_vae folder the run was trained on, "
                         "e.g. under datasets_v0/ if the datasets were moved)")
    d1 = Store(Path(args.d1) if args.d1 else data.parent / "d1")
    from dp2_cutouts.ml.debvader_bridge import FrozenRubinVAE
    vae = FrozenRubinVAE(run, "stage1", stats_path=data / "stats.json", weights_path=args.weights)
    print(f"VAE {vae.weights_path} | data {data} | D1 {d1.path}  ({time.time() - t0:.0f} s)")

    images = np.load(data / "images.npy", mmap_mode="r")
    meta = dict(np.load(data / "meta.npz"))
    cat = d1.meta.iloc[meta["d1_row"]].reset_index(drop=True)
    for c in ("object_id",):
        if c in cat and not np.array_equal(cat[c].to_numpy(np.int64), meta["object_id"]):
            raise SystemExit("D1 rows do not match the VAE data set (object_id): pass the D1 it was exported from")
    snr = {b: A.catalogue_snr(cat[f"{b}_cModel_mag"], cat[f"depth5_{b}"]) for b in BANDS
           if f"{b}_cModel_mag" in cat and f"depth5_{b}" in cat}

    # Representative sample of the split, and the targets brightest in y by catalogue photometry (val + test).
    rng = np.random.default_rng(0)
    rows = np.flatnonzero(meta["split"] == SPLITS[args.split])
    samp = np.sort(rng.choice(rows, min(args.n, len(rows)), replace=False))
    unseen = np.flatnonzero(meta["split"] != SPLITS["train"])
    order = unseen[np.argsort(-np.nan_to_num(snr["y"][unseen], nan=-1))]
    bright = np.sort(order[:args.n_bright])
    print(f"sample {len(samp)} ({args.split}); bright in y: {len(bright)} unseen targets, catalogue S/N_y "
          f"{np.nanmin(snr['y'][bright]):.1f} - {np.nanmax(snr['y'][bright]):.1f}")

    S = evaluate(vae, np.asarray(images[samp], np.float32), cat.iloc[samp])
    B = evaluate(vae, np.asarray(images[bright], np.float32), cat.iloc[bright])
    print(f"inference done ({time.time() - t0:.0f} s)")

    summary = dict(run=str(run), weights=str(vae.weights_path), data=str(data), split=args.split,
                   n_sample=int(len(samp)), n_bright=int(len(bright)), scale_per_band=vae.scale_per_band.tolist(),
                   bands=BANDS)
    for name, R in (("sample", S), ("bright_y", B)):
        summary[name] = dict(galaxy=dict(zip(BANDS, A.band_stats(R["x"], R["mu"], R["sigma"], R["pre_mu"],
                                                                 R["pre_sc"], R["gal"]))),
                             sky=dict(zip(BANDS, A.band_stats(R["x"], R["mu"], R["sigma"], R["pre_mu"],
                                                              R["pre_sc"], R["sky"]))))
        ratio = R["flux_rec"] / np.where(R["flux_obs"] != 0, R["flux_obs"], np.nan)
        data_snr = R["flux_obs"] / (R["sky_sigma"] * np.sqrt(R["npix"])[:, None])
        summary[name]["aperture"] = {b: dict(median_flux_ratio_rec_obs=float(np.nanmedian(ratio[:, c])),
                                             median_data_snr=float(np.nanmedian(data_snr[:, c])),
                                             frac_rec_flux_zero=float((np.abs(R["flux_rec"][:, c]) < 1e-6).mean()))
                                     for c, b in enumerate(BANDS)}
    summary["catalogue_snr_y_bright"] = dict(min=float(np.nanmin(snr["y"][bright])),
                                             median=float(np.nanmedian(snr["y"][bright])))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Printed table: galaxy region, sample and bright-in-y set.
    rows_ = []
    for name in ("sample", "bright_y"):
        for b in BANDS:
            g, a_ = summary[name]["galaxy"][b], summary[name]["aperture"][b]
            rows_.append(dict(set=name, band=b, pre_mu_neg=g["frac_pre_mu_negative"], mu_zero=g["frac_mu_zero"],
                              grad_blocked=g["frac_gradient_blocked_mu"], sigma=g["median_sigma"], nll=g["mean_nll"],
                              flux_rec_over_obs=a_["median_flux_ratio_rec_obs"], data_snr=a_["median_data_snr"]))
    table = pd.DataFrame(rows_).set_index(["set", "band"]).round(4)
    print(table.to_string())
    table.to_csv(out / "galaxy_region_table.csv")

    # 0. What the VAE was given: S/N of the galaxy aperture per band, sample and bright set.
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for a_, (name, R) in zip(ax, (("sample", S), ("bright in y (catalogue)", B))):
        d = R["flux_obs"] / (R["sky_sigma"] * np.sqrt(R["npix"])[:, None])
        a_.boxplot([d[:, c][np.isfinite(d[:, c])] for c in range(6)], showfliers=False)
        a_.set_xticks(range(1, 7), BANDS)
        a_.axhline(0, c="0.5", lw=0.5)
        a_.set_title(f"data: aperture S/N per band, {name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "0_data.png", dpi=120)

    # 1. Pre-activations of the means, galaxy region.
    fig, axes = plt.subplots(2, 6, figsize=(16, 5), sharex="col")
    for r_, (name, R) in enumerate((("sample", S), ("bright in y", B))):
        for c, b in enumerate(BANDS):
            v = R["pre_mu"][..., c][R["gal"]]
            axes[r_, c].hist(np.clip(v, -1, 1), bins=80, color="0.4")
            axes[r_, c].axvline(0, c="C3", lw=1)
            axes[r_, c].set_title(f"{b}, {name}: {(v <= 0).mean():.0%} <= 0", fontsize=8)
    fig.suptitle("pre-activation of the decoder mean (before the final ReLU), galaxy region", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "1_preactivations.png", dpi=120)

    # 2. Per band, galaxy vs sky region.
    keys = [("frac_pre_mu_negative", "pre-activation <= 0"), ("frac_gradient_blocked_mu", "gradient blocked by ReLU"),
            ("median_sigma", "median sigma (norm. units)"), ("mean_nll", "mean NLL per pixel")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for r_, name in enumerate(("sample", "bright_y")):
        for k_, (key, label) in enumerate(keys):
            a_ = axes[r_, k_]
            xs = np.arange(6)
            a_.bar(xs - 0.2, [summary[name]["galaxy"][b][key] for b in BANDS], 0.4, label="galaxy")
            a_.bar(xs + 0.2, [summary[name]["sky"][b][key] for b in BANDS], 0.4, label="sky")
            a_.set_xticks(xs, BANDS)
            a_.set_title(f"{label} ({name})", fontsize=8)
            if r_ == 0 and k_ == 0:
                a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "2_bands.png", dpi=120)

    # 3. Bright targets (catalogue): reconstructed vs observed aperture flux, y with i and z for comparison.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for a_, b in zip(axes, ("y", "z", "i")):
        c = BANDS.index(b)
        top = unseen[np.argsort(-np.nan_to_num(snr[b][unseen], nan=-1))][:args.n_bright]
        R = B if b == "y" else evaluate(vae, np.asarray(images[np.sort(top)], np.float32), cat.iloc[np.sort(top)])
        a_.scatter(R["flux_obs"][:, c], R["flux_rec"][:, c], s=4, alpha=0.5)
        lim = np.nanpercentile(np.abs(R["flux_obs"][:, c]), 99)
        a_.plot([-lim, lim], [-lim, lim], c="C3", lw=0.8)
        a_.set_xlim(-0.1 * lim, lim)
        a_.set_ylim(-0.1 * lim, lim)
        a_.set_xlabel(f"observed {b} aperture flux [nJy]")
        a_.set_ylabel(f"reconstructed {b} [nJy]")
        a_.set_title(f"{len(top)} unseen targets brightest in {b} (catalogue)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "3_bright_flux.png", dpi=120)

    # 4. Examples: brightest in y.
    k = np.argsort(-np.nan_to_num(snr["y"][bright], nan=-1))[:8]
    cy, ci = BANDS.index("y"), BANDS.index("i")
    fig, axes = plt.subplots(len(k), 6, figsize=(11, 1.9 * len(k)), squeeze=False)
    for r_, j in enumerate(k):
        x, mu, sg = B["x"][j], B["mu"][j], B["sigma"][j]
        vmax = np.nanpercentile(np.abs(x[..., cy]), 99.5) or 1.0
        vmi = np.nanpercentile(np.abs(x[..., ci]), 99.5) or 1.0
        panels = [(x[..., cy], "obs y", (-0.2 * vmax, vmax), "gray_r"), (mu[..., cy], "mean y", (-0.2 * vmax, vmax), "gray_r"),
                  (sg[..., cy], "sigma y", None, "viridis"),
                  ((x[..., cy] - mu[..., cy]) / sg[..., cy], "(obs-mean)/sigma y", (-3, 3), "RdBu_r"),
                  (x[..., ci], "obs i", (-0.2 * vmi, vmi), "gray_r"), (mu[..., ci], "mean i", (-0.2 * vmi, vmi), "gray_r")]
        for a_, (img, title, lims, cmap) in zip(axes[r_], panels):
            kw = dict(vmin=lims[0], vmax=lims[1]) if lims else {}
            a_.imshow(img, origin="lower", cmap=cmap, **kw)
            a_.contour(B["gal"][j], levels=[0.5], colors="C1", linewidths=0.5)
            a_.set_xticks([])
            a_.set_yticks([])
            if r_ == 0:
                a_.set_title(title, fontsize=8)
        axes[r_, 0].set_ylabel(f"S/N_y cat {snr['y'][bright][j]:.0f}", fontsize=7)
    fig.suptitle("brightest-in-y unseen targets (catalogue); orange: galaxy region; normalised units", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "4_examples_y.png", dpi=120)
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
