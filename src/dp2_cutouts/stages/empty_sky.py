# empty_sky.py
# AstroDeep Project
# Empty-sky stamps: windows of the coadds with no detected, unexposed or damaged pixel in any band.
#
# Output: one file per patch, <working folder>/tables_and_stamps/patch_files/empty_<tract>_<patch>.npz, holding
#   flux, variance   (N, H, W, C) float32   nJy, nJy^2 (channel-last, like the debvader format)
#   maskbits         (N, H, W, C) int32     mask planes (bits in plane_bits); no rejecting plane is set, by selection
#   psf              (N, P, P, C) float32   PSF at the window centre, unit sum (Butler path only)
#   ra, dec          (N,)                   sky position of the window centre, degrees
#   x_tract, y_tract (N,)                   tract-pixel position of the window centre
#   tract, patch     (N,)
#   plane_bits       JSON string, mask plane name -> bit
#   n_windows, n_clean   number of candidate windows in the patch and number fully clean (a measure of the field's
#                        crowding: in a deep field a fully clean 59 x 59 window is not common)
# export_debvader turns the patch files into the empty_cutouts dataset.
#
# Windows are on a grid with a stride of one stamp, so they do not overlap, and their centres are restricted to the
# patch's inner region, so that two neighbouring patches never contribute the same piece of sky.

import json

import numpy as np

from ..common import masks
from ..config import TABLES_DIR
from .stamps import SkyBank, load_images, patch_list, patch_files_path


def clean_windows(img, cfg, size):
    """Corners (row, col) of the non-overlapping windows with no rejecting pixel, centred in the inner region."""
    reject = masks.bitmask(img["bits"], ["DETECTED", "NO_DATA"] + list(cfg["cuts"]["bad_planes"]))
    bank = SkyBank(img, reject, size, stride=size)
    corners = bank.corners
    half = size // 2
    inner = img["meta"].get("inner_bbox")
    if inner is not None:
        cx, cy = corners[:, 1] + half + img["x0"], corners[:, 0] + half + img["y0"]
        keep = (cx >= inner["x_min"]) & (cx <= inner["x_max"]) & (cy >= inner["y_min"]) & (cy <= inner["y_max"])
        corners = corners[keep]
    clean = np.array([not bank.unusable[r:r + size, c:c + size].any() for r, c in corners], bool)
    return corners[clean], len(corners)


def run(cfg):
    import pandas as pd

    n_want = int(cfg["export"]["n_empty_per_patch"])
    size = int(cfg["cuts"]["stamp_px"])
    half = size // 2
    bands = list(cfg["bands"])
    cat = pd.read_parquet(cfg.data(TABLES_DIR, "catalog.parquet"), columns=["tract", "patch"])

    for tract, patch in patch_list(cfg, cat):
        out = patch_files_path(cfg, "empty", tract, patch, "npz")
        if out.exists():
            print(f"empty sky {tract}/{patch}: already done, skipped")
            continue
        img = load_images(cfg, tract, patch)
        corners, n_windows = clean_windows(img, cfg, size)
        rng = np.random.default_rng([int(cfg["cuts"]["seed"]), int(tract), int(patch)])
        order = rng.permutation(len(corners))

        windows = {k: [] for k in ("flux", "variance", "maskbits", "psf", "ra", "dec", "x_tract", "y_tract")}
        wcs = img["wcs"][bands.index("i") if "i" in bands else 0]
        n_no_psf = 0
        for r, c in corners[order]:
            if len(windows["ra"]) >= n_want:
                break
            col, row = c + half, r + half                         # window centre, local array coordinates
            if "psf" in img:                                      # a window without a PSF is skipped
                from ..sources.rubin_butler import psf_stamp
                psfs = [psf_stamp(img, k, img["x0"] + col, img["y0"] + row, int(cfg["psf_px"]))
                        for k in range(len(bands))]
                if any(p is None for p in psfs):
                    n_no_psf += 1
                    continue
                windows["psf"].append(np.stack(psfs, axis=-1))
            sl = np.s_[:, r:r + size, c:c + size]
            windows["flux"].append(np.moveaxis(img["flux"][sl], 0, -1))
            windows["variance"].append(np.moveaxis(img["var"][sl], 0, -1))
            windows["maskbits"].append(np.moveaxis(img["mask"][sl], 0, -1))
            ra, dec = wcs.pixel_to_world_values(col, row)
            windows["ra"].append(float(ra))
            windows["dec"].append(float(dec))
            windows["x_tract"].append(img["x0"] + col)
            windows["y_tract"].append(img["y0"] + row)

        n, p = len(windows["ra"]), int(cfg["psf_px"])
        print(f"empty sky {tract}/{patch}: {len(corners)} clean windows of {n_windows}; kept {n} "
              f"({n_no_psf} skipped for a missing PSF)")
        arrays = dict(
            flux=np.asarray(windows["flux"], np.float32).reshape(n, size, size, len(bands)),
            variance=np.asarray(windows["variance"], np.float32).reshape(n, size, size, len(bands)),
            maskbits=np.asarray(windows["maskbits"], np.int32).reshape(n, size, size, len(bands)),
            ra=np.asarray(windows["ra"]), dec=np.asarray(windows["dec"]),
            x_tract=np.asarray(windows["x_tract"], np.int64), y_tract=np.asarray(windows["y_tract"], np.int64),
            tract=np.full(n, tract, np.int32), patch=np.full(n, patch, np.int32),
            plane_bits=np.array(json.dumps(img["bits"])), n_windows=np.array(n_windows),
            n_clean=np.array(len(corners)), n_no_psf=np.array(n_no_psf))
        if windows["psf"] or n == 0:
            arrays["psf"] = np.asarray(windows["psf"], np.float32).reshape(n, p, p, len(bands))
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(out)                                          # the file appears only when complete
        del img
