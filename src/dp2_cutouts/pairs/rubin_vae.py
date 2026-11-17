# rubin_vae.py
# Anna Preto
# AstroDeep Project
# Training set of the reference Rubin VAE: the Rubin stamps of D1, written as a debvader-keras "real build", so
# that DebVAder stage 1 is trained from scratch on DP2 by debvader-keras' own trainer (scripts/train.py --stage 1).
# paper_baseline, trained on noiseless BTK simulations, reads the DP2 sky noise as diffuse light (phase 0): its
# latent space cannot serve as the reference for real stamps.
#
# Output folder (cfg.dataset_dir("rubin_vae")), in the format DebvaderDataLoader._fetch_real reads:
#   images.npy            (N, 59, 59, 6) float32, nJy, bands u g r i z y; masked pixels replaced by sky noise
#   meta.npz              split (0 train, 1 val, 2 test: the D1 split, by patch), d1_row, object_id, tract, patch,
#                         metadata_* (e1, e2 of the shape band, i magnitude, S/N), filled_fraction
#   stats.json            nonlinear scale_per_band (99.9th percentile of |flux| on the train split, as
#                         debvader-keras' recompute_stats.py), units "nJy"
#   debvader_config.json  the template run's config.json with dataset_path pointing here (rubin_vae overrides)
#   info.json             counts, provenance, configuration fingerprint
# No hsm_ok in meta.npz: the loader's S/N cut on shape labels does not apply (and real_min_snr is set to 0).

import json
import shutil
import warnings
from pathlib import Path

import numpy as np

from .store import Store

SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def fill_masked(flux, ivar, mask, rng):
    # flux, ivar, mask (N, H, W, C). Bad pixels (masked, non-finite flux, or non-positive / non-finite ivar) get
    # Gaussian noise with the median sky sigma of the good pixels of their stamp and band (0 if the band has none).
    # Returns the filled flux and the fraction of bad pixels per stamp.
    flux = np.array(flux, dtype=np.float32, copy=True)
    ivar = np.asarray(ivar, dtype=np.float32)
    good_iv = np.isfinite(ivar) & (ivar > 0)
    bad = np.asarray(mask, bool) | ~np.isfinite(flux) | ~good_iv
    sigma = np.where(good_iv & ~bad, 1.0 / np.sqrt(np.where(good_iv, ivar, 1.0)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # all-NaN band: no good pixel
        sky = np.nanmedian(sigma, axis=(1, 2))                       # (N, C)
    sky = np.nan_to_num(sky, nan=0.0).astype(np.float32)
    noise = rng.standard_normal(flux.shape).astype(np.float32) * sky[:, None, None, :]
    flux[bad] = noise[bad]
    return flux, bad.mean(axis=(1, 2, 3))


def scale_per_band(images, rows, max_stamps, seed):
    # debvader-keras' definition (scripts/recompute_stats.py): per-band 99.9th percentile of |flux| over the train
    # stamps, drawn at random when there are more than max_stamps.
    rows = np.asarray(rows)
    if len(rows) > max_stamps:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_stamps, replace=False))
    x = np.asarray(images[rows])
    return np.array([float(np.percentile(np.abs(x[..., b]).ravel(), 99.9)) for b in range(x.shape[-1])])


def set_dotted(conf, key, value):
    # "data.real_min_snr" -> conf["data"]["real_min_snr"]; the section must exist, the key may be new (fields
    # added to debvader-keras after the template run was saved).
    section, _, name = key.partition(".")
    if section not in conf or not isinstance(conf[section], dict) or not name:
        raise KeyError(f"rubin_vae.debvader_overrides: {key!r} is not section.key of the template config.json")
    added = name not in conf[section]
    conf[section][name] = value
    return added


def debvader_config(template_run, out_dir, overrides):
    conf = json.loads((Path(template_run) / "config.json").read_text())
    if list(conf["data"].get("normalization", [])) != ["nonlinear"]:
        raise ValueError(f"template run normalised with {conf['data'].get('normalization')}, expected ['nonlinear']")
    conf["data"]["dataset_path"] = str(out_dir)
    added = [k for k, v in dict(overrides or {}).items() if set_dotted(conf, k, v)]
    return conf, added


def export(cfg, template_run, force=False, chunk=512):
    out = cfg.dataset_dir("rubin_vae")
    if (out / "info.json").exists() and not force:
        print(f"Rubin VAE training set already written: {out} (--force to rewrite)")
        return json.loads((out / "info.json").read_text())
    rv = cfg["rubin_vae"]
    d1 = Store(cfg.dataset_dir("d1"))
    meta = d1.meta
    bands = list(cfg["rubin"]["bands"])
    if d1.info.get("rubin_bands", bands) != bands:
        raise ValueError(f"D1 bands {d1.info.get('rubin_bands')} differ from rubin.bands {bands}")

    # Stamps kept: enough unmasked pixels, a known split.
    valid = meta["rubin_valid_fraction"].to_numpy(float) if "rubin_valid_fraction" in meta else np.ones(len(meta))
    split = meta["split"].map(SPLIT_CODES).to_numpy()
    keep = (valid >= float(rv["min_valid_fraction"])) & np.isfinite(split.astype(float))
    rows = np.flatnonzero(keep)
    codes = split[rows].astype(np.int8)
    counts = {s: int((codes == c).sum()) for s, c in SPLIT_CODES.items()}
    if counts["train"] == 0 or counts["val"] == 0:
        raise RuntimeError(f"Rubin VAE training set needs train and val stamps, got {counts}: see splits.fixed")

    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n = len(rows)
    shape = d1.array("rubin_flux").shape[1:]
    images = np.lib.format.open_memmap(tmp / "images.npy", mode="w+", dtype=np.float32, shape=(n, *shape))
    filled = np.zeros(n, np.float32)
    rng = np.random.default_rng(int(cfg["seed"]))
    for s in range(0, n, chunk):
        r = rows[s:s + chunk]
        f, frac = fill_masked(d1.take("rubin_flux", r), d1.take("rubin_ivar", r), d1.take("rubin_mask", r), rng)
        images[s:s + len(r)] = f
        filled[s:s + len(r)] = frac
    images.flush()

    m = meta.iloc[rows].reset_index(drop=True)
    sb = cfg["rubin"]["shape_band"]

    def col(name, dtype=np.float32, fill=np.nan):
        return m[name].to_numpy(dtype) if name in m else np.full(n, fill, dtype)

    np.savez(tmp / "meta.npz", split=codes, d1_row=rows.astype(np.int64), object_id=col("object_id", np.int64, -1),
             tract=col("tract", np.int64, -1), patch=col("patch", np.int64, -1), filled_fraction=filled,
             metadata_id=col("object_id", np.int64, -1), metadata_e1=col("e1_rubin_check"),
             metadata_e2=col("e2_rubin_check"), metadata_magnitude=col(f"{sb}_cModel_mag"),
             metadata_snr=col("snr_i"))

    train_rows = np.flatnonzero(codes == SPLIT_CODES["train"])
    scale = scale_per_band(images, train_rows, int(rv["stats_max_stamps"]), int(cfg["seed"]))
    del images                                     # close the memory map before the folder is renamed
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise RuntimeError(f"invalid scale_per_band {scale}: empty or blank train stamps?")
    stats = dict(nonlinear=dict(scale_per_band=scale.tolist()), normalization_list=["nonlinear"], units="nJy",
                 bands=bands, definition="99.9th percentile of |flux| per band, train split (recompute_stats.py)",
                 n_stamps=int(min(len(train_rows), int(rv["stats_max_stamps"]))))
    (tmp / "stats.json").write_text(json.dumps(stats, indent=2))

    conf, added = debvader_config(template_run, out, rv.get("debvader_overrides"))
    if tuple(conf["data"]["input_shape"]) != tuple(shape):
        raise ValueError(f"template input_shape {conf['data']['input_shape']} differs from the stamps {shape}")
    (tmp / "debvader_config.json").write_text(json.dumps(conf, indent=2))

    info = dict(name="rubin_vae", n=n, counts=counts, bands=bands, units="nJy", stamp_shape=list(shape),
                source=str(d1.path), d1_n=len(meta), dropped_low_valid_fraction=int((~keep).sum()),
                min_valid_fraction=float(rv["min_valid_fraction"]),
                filled_fraction=dict(mean=float(filled.mean()), p95=float(np.percentile(filled, 95))),
                scale_per_band=scale.tolist(), template_run=str(template_run), run_name=rv["run_name"],
                overrides_added_to_template=added,
                config_fingerprint=cfg.fingerprint(["rubin", "dataset1", "splits", "seed", "rubin_vae"]),
                d1_config_fingerprint=d1.info.get("config_fingerprint"))
    (tmp / "info.json").write_text(json.dumps(info, indent=2))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    print(json.dumps({k: info[k] for k in ("n", "counts", "filled_fraction", "scale_per_band")}, indent=2))
    print(f"Rubin VAE training set -> {out}")
    return info
