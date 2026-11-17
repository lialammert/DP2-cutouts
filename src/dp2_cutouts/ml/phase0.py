# phase0.py
# Anna Preto
# AstroDeep Project
# Phase 0: the reference VAE (paper_baseline) before anything is built on it.
#   1 loading       checkpoints of stage 1 and stage 2 load, change the weights, and the stage 2 decoder is the
#                   stage 1 one (DebVAder freezes it in stage 2: if not, Stage 1 and Stage 2 here would not share
#                   a latent space)
#   2 determinism   D_R(mu(E_R(x))) is identical on two passes (the bridge never samples)
#   3 reference     optional: predictions saved by the campaign (npz with images and mu) are reproduced
#   4 units         nJy -> VAE units factors, and where DP2 stamps fall in the normalisation the VAE was trained with
#                   (per-stamp peak in normalised units, fraction of pixels above the round-trip saturation)
#   5 domain gap    reconstruction of DP2 isolated galaxies: chi^2 per pixel with the DP2 variance, ellipticity and
#                   flux of the reconstruction against the observed stamp, binned in S/N
# Reads the final stamp dataset of the pipeline directly: it does not need Euclid images or D1.

import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from keras import ops

from ..pairs import common
from .debvader_bridge import FrozenRubinVAE, weights_fingerprint
from .evaluate import measure, robust_scatter
from .training import environment
from .units import UnitConverter


def sample_final(pcfg, bands, n, seed):
    root = common.final_dir(pcfg) / "data"
    d = ds.dataset(root, format="parquet", partitioning="hive")
    total = d.count_rows()
    take = np.sort(np.random.default_rng(seed).choice(total, size=min(n, total), replace=False))
    t = d.take(take)
    meta = common.table_to_pandas(t)
    flux, ivar, mask = common.unpack_images(t.column("image"), bands)
    return meta, flux, ivar, mask, int(total)


def run(cfg, debvader_run, stats_path=None, n=3000, reference_npz=None, out_name="phase0"):
    out_dir = cfg.eval_dir(out_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = dict(environment=environment())

    # 1 loading
    v1 = FrozenRubinVAE(debvader_run, stage="stage1", stats_path=stats_path)
    rep["vae_stage1"] = v1.describe()
    try:
        v2 = FrozenRubinVAE(debvader_run, stage="stage2", stats_path=stats_path)
        rep["vae_stage2"] = v2.describe()
        rep["decoder_shared"] = weights_fingerprint(v1.decoder) == weights_fingerprint(v2.decoder)
        rep["encoder_differs"] = weights_fingerprint(v1.encoder) != weights_fingerprint(v2.encoder)
    except FileNotFoundError as e:
        rep["vae_stage2"] = f"not available: {e}"
        rep["decoder_shared"] = None

    # 2 determinism, 4 units, 5 domain gap: on a sample of DP2 stamps
    bands = list(cfg["rubin"]["bands"])
    meta, flux, ivar, mask, total = sample_final(cfg.pipeline(), bands, int(n), int(cfg["seed"]))
    rep["final_dataset_rows"] = total
    units = UnitConverter.for_vae(cfg, v1)
    rep["units"] = units.describe()
    f = units.flux(flux)
    iv = units.ivar(ivar)
    x = v1.normalise(f).astype(np.float32)
    x[mask] = float(cfg["rubin"]["masked_value_normalised"])
    w = (~mask).astype(np.float32)
    mu1, rec1 = v1.reconstruct(x)
    mu2, rec2 = v1.reconstruct(x)
    rep["deterministic"] = bool(np.array_equal(mu1, mu2) and np.array_equal(rec1, rec2))

    sat = v1.max_recoverable_flux()
    peak_norm = x.reshape(len(x), -1, len(bands)).max(axis=1)
    rep["normalisation"] = dict(
        scale_per_band=[float(v) for v in v1.scale_per_band],
        # A training stamp at the mean peak sits at tanh(asinh(2.5)) = 0.917 in normalised units.
        training_mean_peak_normalised=float(np.tanh(np.arcsinh(2.5))),
        dp2_median_peak_normalised={b: float(np.median(peak_norm[:, k])) for k, b in enumerate(bands)},
        dp2_p95_peak_normalised={b: float(np.percentile(peak_norm[:, k], 95)) for k, b in enumerate(bands)},
        frac_pixels_above_roundtrip_saturation={b: float((f[..., k] > sat[k]).mean()) for k, b in enumerate(bands)},
        dp2_median_sky_sigma_vae_units={b: float(np.median(1 / np.sqrt(iv[..., k][(iv[..., k] > 0)])))
                                        for k, b in enumerate(bands)})

    rec_phys = v1.denormalise_np(rec1)
    obs = measure(f, w, cfg, "obs")
    rec = measure(rec_phys, np.ones_like(w), cfg, "rec")
    df = pd.concat([meta.reset_index(drop=True), obs, rec], axis=1)
    df["chi2_rec_obs"] = ((rec_phys - f) ** 2 * iv * w).sum((1, 2, 3)) / np.maximum(w.sum((1, 2, 3)), 1)
    for q in ("e1", "e2"):
        df[f"d{q}_rec_obs"] = df[f"rec_{q}"] - df[f"obs_{q}"]
    for b in bands:
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"rflux_{b}_rec_obs"] = df[f"rec_flux_{b}"] / df[f"obs_flux_{b}"] - 1.0
    snr = df["cut6_value"].astype(float) if "cut6_value" in df else pd.Series(np.nan, index=df.index)
    df["snr_i"] = snr
    df.to_parquet(out_dir / "phase0_sample.parquet", index=False)
    by = []
    for lo, hi in zip(cfg["evaluation"]["bins"]["snr_i"][:-1], cfg["evaluation"]["bins"]["snr_i"][1:]):
        sub = df[(snr >= lo) & (snr < hi)]
        by.append(dict(snr_lo=lo, snr_hi=hi, n=int(len(sub)), median_chi2=float(sub["chi2_rec_obs"].median()),
                       mean_de1=float(np.nanmean(sub["de1_rec_obs"])), scatter_de1=robust_scatter(sub["de1_rec_obs"]),
                       median_rflux_i=float(np.nanmedian(sub["rflux_i_rec_obs"])),
                       hsm_failure_rec=float(1 - sub["rec_ok"].mean()) if len(sub) else np.nan))
    rep["domain_gap_by_snr"] = by
    rep["domain_gap"] = dict(median_chi2=float(df["chi2_rec_obs"].median()),
                             mean_de1=float(np.nanmean(df["de1_rec_obs"])),
                             mean_de2=float(np.nanmean(df["de2_rec_obs"])),
                             median_rflux={b: float(np.nanmedian(df[f"rflux_{b}_rec_obs"])) for b in bands})

    # 3 reference predictions of the campaign, if given: {"images": (N,59,59,6) in VAE units, "mu": (N, D)}
    if reference_npz:
        r = np.load(reference_npz)
        xr = v1.normalise(r["images"]).astype(np.float32)
        mu_ref, _ = v1.reconstruct(xr)
        rep["reference"] = dict(file=str(reference_npz), max_abs_mu_diff=float(np.abs(mu_ref - r["mu"]).max()),
                                reproduced=bool(np.allclose(mu_ref, r["mu"], atol=1e-4, rtol=1e-4)))

    k = min(int(cfg["evaluation"]["n_examples_figures"]), len(x))
    np.savez_compressed(out_dir / "phase0_examples.npz", obs=f[:k], rec=rec_phys[:k], weight=w[:k],
                        object_id=meta["object_id"].to_numpy()[:k])
    (out_dir / "phase0_report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps({k: rep[k] for k in ("decoder_shared", "deterministic", "normalisation", "domain_gap")},
                     indent=2, default=float))
    print(f"phase 0 -> {out_dir}")
    return rep
