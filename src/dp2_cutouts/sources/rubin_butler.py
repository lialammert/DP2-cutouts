# rubin_butler.py
# AstroDeep Project
# Reads the Rubin DP2 Object catalogue through the Butler (at NERSC: repo "dp2"), within a sky box. Replaces
# rubin_tap where the Butler is available: no token, no TAP job, same output file and same columns, so the
# following steps (copaings, catalog_table) do not know which source was used.
#
# Configuration:
#   butler: {repo: dp2, collections: dp2, skymap: lsst_cells_v2}
#   rubin:  {source: butler, tracts: [5063]}
#
# The lsst imports are inside the functions: this module must stay importable without the LSST stack.

import numpy as np
import pandas as pd

from .rubin_tap import COPAINGS_COLUMNS, EXTRA_COLUMNS
from ..common import star_galaxy
from ..config import RUBIN_DIR


def get_butler(cfg):
    from lsst.daf.butler import Butler
    b = cfg["butler"]
    return Butler(b["repo"], collections=b["collections"])


def available_columns(butler, skymap, tract):
    return list(butler.get("object.columns", skymap=skymap, tract=int(tract)))


def fetch_objects(cfg):
    # Every row of the object table inside the sky box, for the tracts of the configuration, without any cut:
    # the pre-cut catalogue is used for reweighting. Rows are read tract by tract (the table is per tract).
    butler = get_butler(cfg)
    skymap = cfg["butler"]["skymap"]
    tracts = [int(t) for t in cfg["rubin"]["tracts"]]
    columns = list(dict.fromkeys(COPAINGS_COLUMNS + EXTRA_COLUMNS + star_galaxy.required_columns(cfg)))
    ra0, ra1, dec0, dec1 = cfg.sky_box

    parts = []
    for tract in tracts:
        have = set(available_columns(butler, skymap, tract))
        missing = [c for c in columns if c not in have]
        if missing:
            raise KeyError(f"tract {tract}: columns missing from the butler object table: {missing}")
        t = butler.get("object", skymap=skymap, tract=tract, parameters={"columns": columns})
        df = t.to_pandas()
        if df.index.name in columns:      # the table may come back indexed by objectId
            df = df.reset_index()
        n_all = len(df)
        keep = ((df["coord_ra"] >= ra0) & (df["coord_ra"] <= ra1)
                & (df["coord_dec"] >= dec0) & (df["coord_dec"] <= dec1))
        df = df[keep].reset_index(drop=True)
        print(f"  tract {tract}: {n_all:,} objects, {len(df):,} in the box")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    # Masked astropy columns come back as pandas nullable or object columns; the following steps expect plain
    # numpy floats with NaN, as TAP returns them.
    for c in df.columns:
        if c in ("objectId", "parentObjectId", "tract", "patch"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
        elif df[c].dtype == object or str(df[c].dtype) in ("boolean", "Float64", "Float32", "Int64"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Objects in the overlap of two tracts appear in both tables: the same check as on the TAP response.
    if df["objectId"].duplicated().any():
        n = int(df["objectId"].duplicated().sum())
        raise RuntimeError(f"{n} duplicate objectId in the butler object table (tract overlap?)")
    return df


def run(cfg):
    df = fetch_objects(cfg)
    out = cfg.data(RUBIN_DIR, "catalog", "object_box.parquet")
    df.to_parquet(out, index=False)
    print(f"Rubin (butler): {len(df):,} objects, {df.shape[1]} columns -> {out}")
    return df


# ----------------------------------------------------------------------------- pixels
def load_patch(cfg, tract, patch):
    """The deep coadds of one patch, all bands, in the dict layout that stamps.py works with.

    Keys (same as the SODA path, plus `psf`):
      flux, var   (C, H, W) float32, nJy and nJy^2
      mask        (C, H, W) int32, bit k set when mask plane k is set (bit numbers in `bits`)
      detected    (H, W) bool, DETECTED in at least one band
      wcs         list of C astropy WCS, in LOCAL array coordinates: pixel (0, 0) is array element [0, 0]
      x0, y0      tract-pixel position of array element [0, 0] (the patch's outer bounding box)
      bits        mask plane name -> bit value
      meta        tract, patch, patch_bbox (outer bounding box) and inner_bbox (the region whose objects belong
                  to this patch), tract pixels, inclusive
      psf         list of C CellPointSpreadFunction; evaluate with psf_stamp()

    The arrays cover the patch's OUTER bounding box (the inner region plus a 150 px overlap with the
    neighbours), so a stamp of an object assigned to this patch (inner region) never runs off the image.
    """
    if tract is None or patch is None:
        raise ValueError("load_patch needs a tract and a patch")
    butler = get_butler(cfg)
    skymap = cfg["butler"]["skymap"]
    flux, var, mask, psfs, wcs = [], [], [], [], []
    bits, bbox = None, None
    for band in cfg["bands"]:
        c = butler.get("deep_coadd", band=band, skymap=skymap, tract=int(tract), patch=int(patch))
        if str(c.unit) != "nJy":
            raise RuntimeError(f"{tract}/{patch}/{band}: unexpected unit {c.unit}, expected nJy")
        planes = [p.name for p in c.mask.schema]
        if bits is None:
            bits = {name: 1 << k for k, name in enumerate(planes)}
            bbox = c.bbox
        elif list(bits) != planes or c.bbox != bbox:
            raise RuntimeError(f"{tract}/{patch}/{band}: mask planes or bounding box differ between bands")
        m = np.zeros(c.image.array.shape, np.int32)
        for name, bit in bits.items():
            m[c.mask.get(name)] |= bit
        flux.append(c.image.array.astype(np.float32))
        var.append(c.variance.array.astype(np.float32))
        mask.append(m)
        psfs.append(c.psf)
        wcs.append(c.fits_wcs)
        del c
    mask = np.stack(mask)
    detected = np.any((mask & bits["DETECTED"]) != 0, axis=0)
    x0, y0 = int(bbox.x.start), int(bbox.y.start)
    h, w = flux[0].shape
    inner = butler.get("skyMap", skymap=skymap, collections="skymaps")[int(tract)][int(patch)].getInnerBBox()
    meta = dict(tract=int(tract), patch=int(patch),
                patch_bbox=dict(x_min=x0, x_max=x0 + w - 1, y_min=y0, y_max=y0 + h - 1),
                inner_bbox=dict(x_min=int(inner.getMinX()), x_max=int(inner.getMaxX()),
                                y_min=int(inner.getMinY()), y_max=int(inner.getMaxY())))
    return dict(flux=np.stack(flux), var=np.stack(var), mask=mask, detected=detected, wcs=wcs,
                x0=x0, y0=y0, bits=bits, meta=meta, psf=psfs)


def psf_stamp(img, band_index, x_tract, y_tract, size):
    """PSF kernel image at a tract-pixel position, size x size (odd), centred on size // 2, unit sum, float32.

    The coadd PSF is the effective PSF (pixel response included) and is piecewise constant over 150 px cells.
    Some cells have no PSF (seen on tract 5063, patch 34): then None is returned, and the caller records it.
    """
    if size % 2 != 1:
        raise ValueError("PSF stamps must have an odd size")
    try:
        image = img["psf"][band_index].compute_kernel_image(x=float(x_tract), y=float(y_tract))
    except Exception as e:                       # lsst.images BoundsError: "Cell ... is missing for this PSF"
        if type(e).__name__ == "BoundsError":
            return None
        raise
    k = np.asarray(image.array, np.float64)
    n = k.shape[0]
    if k.shape != (n, n) or n % 2 != 1:
        raise RuntimeError(f"unexpected PSF kernel shape {k.shape}")
    out = np.zeros((size, size))
    w = min(n, size)
    src, dst = (n - w) // 2, (size - w) // 2       # both centres stay on index n // 2 -> size // 2
    out[dst:dst + w, dst:dst + w] = k[src:src + w, src:src + w]
    return (out / out.sum()).astype(np.float32)
