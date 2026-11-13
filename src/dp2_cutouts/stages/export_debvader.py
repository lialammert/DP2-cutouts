# export_debvader.py
# AstroDeep Project
# Writes the products of the stamps and empty_sky steps as datasets in the debvader format, the one shared by
# DP2-cutouts (writer), debvader-keras (loader, blend builder) and the dbv-pipeline stages

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import masks
from ..config import EXPORT_DIR, TABLES_DIR
from .package import selection
from .stamps import split_of

SCALE_PERCENTILE = 99.9     # as debvader-keras scripts/recompute_stats.py

# meta.npz entry -> catalogue column; the names on the left are the ones debvader-keras reads.
META = {
    "metadata_id": "object_id",
    "ra": "ra", "dec": "dec", "tract": "tract", "patch": "patch",
    "x_tract": "x_tract", "y_tract": "y_tract",
    "offset_x": "offset_x", "offset_y": "offset_y",
    "metadata_magnitude": "i_cModel_mag",
    "metadata_e1": "e1", "metadata_e2": "e2",          # observed adaptive moments in the shape band, after masking
    "hsm_resolution": "resolution_i",
    "metadata_redshift": "euclid_phz_median",
    "masked_fraction": "masked_fraction",
    "sb_euclid": "sb_euclid",
    "has_euclid_counterpart": "has_euclid_counterpart",
    "rubin_deblend_pass": "rubin_deblend_pass",
    "star_galaxy_value": "star_galaxy_value",
}


def patch_files_dir(cfg):
    return cfg.data(TABLES_DIR, "patch_files", "x").parent


def _memmap(path, shape, dtype):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _to_hwc(flat, n_band, size):
    """A flattened (C, size, size) array -> (size, size, C)."""
    return np.moveaxis(np.asarray(flat).reshape(n_band, size, size), 0, -1)


def bad_mask(maskbits, variance, plane_bits, mask_planes):
    """True where a pixel must not be used: listed mask planes set, or variance not finite and positive."""
    reject = masks.bitmask(plane_bits, list(mask_planes))
    with np.errstate(invalid="ignore"):
        return ((maskbits & reject) != 0) | ~np.isfinite(variance) | (variance <= 0)


def write_stats(out_dir, images, split):
    """stats.json with the per-band scale of the non-linear normalisation, from the training split only."""
    train = np.flatnonzero(split == 0)
    if len(train) == 0:
        raise RuntimeError(f"{out_dir}: no object in the training split, the normalisation cannot be computed")
    take = train if len(train) <= 5000 else np.sort(np.random.default_rng(0).choice(train, 5000, replace=False))
    scale = np.percentile(np.asarray(images[take]), SCALE_PERCENTILE, axis=(0, 1, 2))
    stats = {"nonlinear": {"scale_per_band": [float(v) for v in scale]}, "normalization_list": ["nonlinear"],
             "computed_on": f"split 0, {len(take)} stamps, {SCALE_PERCENTILE}th percentile per band"}
    with open(Path(out_dir) / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return scale


def write_dataset_json(out_dir, cfg, kind, n, split, size, psf_px):
    info = dict(
        instrument="rubin", kind=kind, bands=list(cfg["bands"]), pixel_scale=float(cfg["pixel_scale"]),
        stamp_px=int(size), stamp_arcsec=round(size * float(cfg["pixel_scale"]), 6), psf_px=int(psf_px),
        unit="nJy", n_objects=int(n),
        n_per_split={s: int((split == k).sum()) for k, s in enumerate(("train", "val", "test"))},
        split=dict(cfg["split"]), mask_planes=list(cfg["export"]["mask_planes"]),
        cuts_applied=list(cfg["export"]["apply"]) if kind == "isolated" else None,
        source=dict(cfg.raw.get("butler", {})), name=cfg.name, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(Path(out_dir) / "dataset.json", "w") as f:
        json.dump(info, f, indent=2)


def export_isolated(cfg, out_dir):
    """Isolated-galaxy dataset from the stamps patch files. Returns the number of objects written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bands, size, p = list(cfg["bands"]), int(cfg["cuts"]["stamp_px"]), int(cfg["psf_px"])
    C = len(bands)
    tables = sorted(patch_files_dir(cfg).glob("table_*.parquet"))
    if not tables:
        raise FileNotFoundError(f"no stamps patch file in {patch_files_dir(cfg)}: run the stamps step first")

    # Which objects: every cut of export.apply (cut*_pass columns, in_region, mask_pass).
    table = pd.concat([pd.read_parquet(f) for f in tables], ignore_index=True)
    final = table[selection(table, cfg["export"]["apply"])].reset_index(drop=True)
    N = len(final)
    row_of = {int(o): k for k, o in enumerate(final["object_id"])}
    print(f"export isolated: {N:,} objects pass {cfg['export']['apply']} (of {len(table):,} rows in the patch files)")

    images = _memmap(out_dir / "images.npy", (N, size, size, C), np.float32)
    variance = _memmap(out_dir / "variance.npy", (N, size, size, C), np.float32)
    mask = _memmap(out_dir / "mask.npy", (N, size, size, C), bool)
    psf = _memmap(out_dir / "psf.npy", (N, p, p, C), np.float32)
    filled = np.zeros(N, bool)
    for f in sorted(patch_files_dir(cfg).glob("stamps_*.parquet")):
        info = json.load(open(f.with_suffix(".json")))
        if info["psf_px"] is None:
            raise RuntimeError(f"{f.name} has no PSF stamps (SODA path): the debvader format needs them")
        st = pd.read_parquet(f, columns=["object_id", "flux", "variance", "maskbits", "psf"])
        for r in st.itertuples(index=False):
            k = row_of.get(int(r.object_id))
            if k is None:
                continue
            v = _to_hwc(r.variance, C, size)
            images[k] = _to_hwc(r.flux, C, size)
            variance[k] = v
            mask[k] = bad_mask(_to_hwc(r.maskbits, C, size), v, info["plane_bits"], cfg["export"]["mask_planes"])
            psf[k] = _to_hwc(r.psf, C, p)
            filled[k] = True
    if not filled.all():
        raise RuntimeError(f"{int((~filled).sum())} selected objects have no stamp in the patch files")
    for a in (images, variance, mask, psf):
        a.flush()

    split = np.array([split_of(t, q, cfg["split"], x, y) for t, q, x, y in
                      zip(final["tract"], final["patch"], final["x_tract"], final["y_tract"])], np.int8)
    meta = {k: final[c].to_numpy() for k, c in META.items() if c in final}
    meta["hsm_sigma"] = final["sigma_hsm"].to_numpy(float) * float(cfg["pixel_scale"])   # pixels -> arcsec
    meta["psf_fwhm"] = np.stack([final[f"psf_fwhm_{b}"].to_numpy(float) for b in bands], axis=-1)
    for c in final.columns:                          # every cut, value and pass, for later reweighting
        if c.startswith("cut") and (c.endswith("_pass") or c.endswith("_value")):
            meta[c] = final[c].to_numpy()
    # Nullable or object columns -> plain floats (NaN), so that meta.npz loads without pickle.
    meta = {k: (pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float) if v.dtype == object else v)
            for k, v in meta.items()}
    np.savez(out_dir / "meta.npz", split=split, bands=np.array(bands), units=np.array("nJy"), **meta)

    write_stats(out_dir, images, split)
    write_dataset_json(out_dir, cfg, "isolated", N, split, size, p)
    print(f"  -> {out_dir}  (train {int((split == 0).sum())}, val {int((split == 1).sum())}, "
          f"test {int((split == 2).sum())})")
    return N


def export_empty(cfg, out_dir, isolated_dir):
    """Empty-sky dataset from the empty_sky patch files. stats.json is copied from the isolated dataset."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bands, size, p = list(cfg["bands"]), int(cfg["cuts"]["stamp_px"]), int(cfg["psf_px"])
    patch_files = sorted(patch_files_dir(cfg).glob("empty_*.npz"))
    if not patch_files:
        raise FileNotFoundError(f"no empty-sky patch file in {patch_files_dir(cfg)}: run the empty_sky step first")
    parts = [dict(np.load(f)) for f in patch_files]
    N = sum(len(d["ra"]) for d in parts)
    print(f"export empty: {N:,} empty-sky stamps from {len(patch_files)} patches "
          f"(clean windows per patch: {[int(d['n_clean']) for d in parts]})")

    images = _memmap(out_dir / "images.npy", (N, size, size, len(bands)), np.float32)
    variance = _memmap(out_dir / "variance.npy", (N, size, size, len(bands)), np.float32)
    mask = _memmap(out_dir / "mask.npy", (N, size, size, len(bands)), bool)
    psf = _memmap(out_dir / "psf.npy", (N, p, p, len(bands)), np.float32)
    at = 0
    for d in parts:
        n = len(d["ra"])
        if n == 0:
            continue
        if "psf" not in d:
            raise RuntimeError("empty-sky patch file without PSF stamps (SODA path): the debvader format needs them")
        images[at:at + n] = d["flux"]
        variance[at:at + n] = d["variance"]
        mask[at:at + n] = bad_mask(d["maskbits"], d["variance"], json.loads(str(d["plane_bits"])),
                                   cfg["export"]["mask_planes"])
        psf[at:at + n] = d["psf"]
        at += n
    for a in (images, variance, mask, psf):
        a.flush()

    def column(k):
        return np.concatenate([d[k] for d in parts])

    split = np.array([split_of(t, q, cfg["split"], x, y) for t, q, x, y in
                      zip(column("tract"), column("patch"), column("x_tract"), column("y_tract"))], np.int8)
    np.savez(out_dir / "meta.npz", split=split, ra=column("ra"), dec=column("dec"), tract=column("tract"),
             patch=column("patch"), x_tract=column("x_tract"), y_tract=column("y_tract"),
             bands=np.array(bands), units=np.array("nJy"))
    # The normalisation constants belong to the isolated training set; every other dataset copies them.
    (out_dir / "stats.json").write_text((Path(isolated_dir) / "stats.json").read_text())
    write_dataset_json(out_dir, cfg, "empty", N, split, size, p)
    print(f"  -> {out_dir}")
    return N


def run(cfg, isolated_dir=None, empty_dir=None):
    """CLI step: write both datasets, by default under <working folder>/debvader_datasets/{isolated,empty}.
    The dbv-pipeline MakeCutouts stage will call export_isolated and export_empty with its own output paths."""
    isolated_dir = Path(isolated_dir or cfg.data(EXPORT_DIR, "isolated", "x").parent)
    empty_dir = Path(empty_dir or cfg.data(EXPORT_DIR, "empty", "x").parent)
    export_isolated(cfg, isolated_dir)
    export_empty(cfg, empty_dir, isolated_dir)
