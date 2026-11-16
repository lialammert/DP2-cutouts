# common.py
# Anna Preto
# AstroDeep Project
# Pieces shared by the D1 and D2 builders: reading the final stamp dataset of one patch, unpacking its image
# column, moments with centroid, aperture signal-to-noise, sky offsets between the two surveys.

from pathlib import Path

import galsim
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

RUBIN_SCALE = 0.2


def final_dir(pipeline_cfg):
    return pipeline_cfg.data("final", "x", makedirs=False).parent


def read_final_patch(pipeline_cfg, tract, patch, columns=None):
    # Rows of the assembled final dataset owned by one patch (the image column included unless columns says no).
    root = Path(final_dir(pipeline_cfg)) / "data"
    if not root.exists():
        raise FileNotFoundError(f"{root} not found: run the stamp pipeline (assemble) first")
    d = ds.dataset(root, format="parquet", partitioning="hive")
    flt = (pc.field("tract") == int(tract)) & (pc.field("patch") == int(patch))
    return d.to_table(filter=flt, columns=columns)


def unpack_images(image_column, bands):
    # MultimodalUniverse image column (list of {band, flux, ivar, mask, scale} per row) -> (N, n, n, B) arrays.
    rows = image_column.to_pylist()
    n_rows = len(rows)
    if n_rows == 0:
        return (np.zeros((0, 0, 0, len(bands)), np.float32),) * 2 + (np.zeros((0, 0, 0, len(bands)), bool),)
    n = len(rows[0][0]["flux"])
    flux = np.zeros((n_rows, n, n, len(bands)), np.float32)
    ivar = np.zeros_like(flux)
    mask = np.ones(flux.shape, bool)
    for i, r in enumerate(rows):
        by_band = {s["band"]: s for s in r}
        missing = [b for b in bands if b not in by_band]
        if missing:
            raise KeyError(f"row {i}: bands {missing} missing from the image column")
        for k, b in enumerate(bands):
            s = by_band[b]
            flux[i, :, :, k] = np.asarray(s["flux"], np.float32)
            ivar[i, :, :, k] = np.asarray(s["ivar"], np.float32)
            mask[i, :, :, k] = np.asarray(s["mask"], bool)
    return flux, ivar, mask


def moments(image, weight=None, guess=None):
    # HSM adaptive moments with centroid. image [row, col]; weight 0/1; guess (x, y) 0-based. NaN on failure.
    out = dict(e1=np.nan, e2=np.nan, sigma=np.nan, x=np.nan, y=np.nan, flux=np.nan, ok=False)
    if weight is not None and not np.any(weight):
        return out
    kw = {}
    if weight is not None:
        kw["weight"] = galsim.Image(np.ascontiguousarray(weight, dtype=np.int32))
    if guess is not None:
        kw["guess_centroid"] = galsim.PositionD(guess[0] + 1.0, guess[1] + 1.0)
    try:
        r = galsim.hsm.FindAdaptiveMom(galsim.Image(np.ascontiguousarray(image, dtype=np.float64)), strict=False, **kw)
    except galsim.GalSimError:
        return out
    if r.moments_status != 0:
        return out
    return dict(e1=float(r.observed_shape.e1), e2=float(r.observed_shape.e2), sigma=float(r.moments_sigma),
                x=float(r.moments_centroid.x - 1.0), y=float(r.moments_centroid.y - 1.0),
                flux=float(r.moments_amp), ok=True)


def aperture(image, ivar, xc, yc, radius):
    # Flux and S/N in a circular aperture; pixels with ivar 0 are skipped (their count is returned).
    rows, cols = np.indices(image.shape)
    ap = (cols - xc) ** 2 + (rows - yc) ** 2 <= radius ** 2
    good = ap & (ivar > 0)
    f = float(image[good].sum())
    var = float((1.0 / ivar[good]).sum()) if good.any() else np.nan
    return f, (f / np.sqrt(var) if var and var > 0 else np.nan), int((ap & ~good).sum())


def sky_offset_arcsec(ra0, dec0, ra1, dec1):
    # Offset of (ra1, dec1) from (ra0, dec0) in the frame of the Rubin stamps: x towards west, y towards north.
    dra = (np.asarray(ra1) - np.asarray(ra0) + 180.0) % 360.0 - 180.0
    west = -dra * np.cos(np.radians(dec0)) * 3600.0
    north = (np.asarray(dec1) - np.asarray(dec0)) * 3600.0
    return west, north


def table_to_pandas(t, drop=("image", "image_maskbits")):
    keep = [c for c in t.column_names if c not in drop]
    return t.select(keep).to_pandas()


def as_int64(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")
