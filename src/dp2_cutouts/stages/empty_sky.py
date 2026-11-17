# empty_sky.py
# AstroDeep Project
# Empty-sky stamps: windows of the coadds with no unexposed or damaged pixel in any band, as empty of detected
# sources as the patch allows. Detected pixels (DETECTED plane, in any band) are replaced by real empty sky taken
# elsewhere in the patch, as the companions of the isolated stamps are (stamps.SkyBank / stamps.fill), and get the
# FILLED_SKY bit.
#
# Why relax "no detected pixel at all": in a deep coadd about 30% of the pixels are DETECTED, and a fully clean
# 59 x 59 window is rare (3 of 2,500 in patch 5063/34). Those few are the emptiest corners of the patch and not a
# representative sample of the sky noise; there are also far too few of them for MeasureNoise and the
# injected-central blends.
#
# Why fill with real sky and not with Gaussian noise from the variance plane: MeasureNoise measures f_b on these
# stamps. Synthetic noise drawn from the variance would return f_b = 1 by construction and carry no pixel
# correlation. Copied sky pixels are real undetected sky with their own variance, so they are valid noise samples.
# Pixels that could not be filled from clean sky keep their DETECTED bit; export_empty masks them.
#
# Selection: candidate windows are on a grid with a stride of one stamp (no overlap), centred in the patch's inner
# region (two patches never contribute the same piece of sky), with no NO_DATA or bad-plane pixel, and a detected
# fraction <= empty_max_detected_fraction. The emptiest candidates are taken first (lowest detected fraction, ties
# in random order), up to n_empty_per_patch.
#
# Output: one file per patch, <working folder>/tables_and_stamps/patch_files/empty_<tract>_<patch>.npz, holding
#   flux, variance   (N, H, W, C) float32   nJy, nJy^2 (channel-last, like the debvader format), after the fill
#   maskbits         (N, H, W, C) int32     mask planes (bits in plane_bits), FILLED_SKY on filled pixels
#   psf              (N, P, P, C) float32   PSF at the window centre, unit sum (Butler path only)
#   ra, dec          (N,)                   sky position of the window centre, degrees
#   x_tract, y_tract (N,)                   tract-pixel position of the window centre
#   tract, patch     (N,)
#   detected_fraction (N,)                  fraction of the window's pixels DETECTED in some band, before the fill
#   fill_bad_fraction (N,)                  fraction of those that could not be filled from clean sky (masked)
#   plane_bits       JSON string, mask plane name -> bit
#   selection        JSON string, the parameters of the selection (a file made with others is recomputed)
#   n_windows, n_candidates, n_clean   candidate windows in the patch's inner region; those passing the hard
#                    criteria and the detected-fraction cap; those with no detected pixel at all
# export_debvader turns the patch files into the empty_cutouts dataset.

import json

import numpy as np

from ..common import masks
from ..common.progress import Progress, log
from ..config import TABLES_DIR
from .stamps import SkyBank, fill, load_images, patch_list, patch_files_path

DEFAULT_MAX_DETECTED_FRACTION = 0.3


def selection_params(cfg):
    """The parameters that decide which windows are written, and how they are filled."""
    ex, cc = cfg["export"], cfg["cuts"]
    return dict(n_empty_per_patch=int(ex["n_empty_per_patch"]),
                max_detected_fraction=float(ex.get("empty_max_detected_fraction", DEFAULT_MAX_DETECTED_FRACTION)),
                stamp_px=int(cc["stamp_px"]), bad_planes=list(cc["bad_planes"]), sky_tries=int(cc["sky_tries"]),
                seed=int(cc["seed"]))


def rank_windows(img, bad_planes, size, max_detected_fraction, rng):
    """
    Candidate empty-sky windows of a patch, emptiest first.

    Returns (corners, detected_fraction, counts):
      corners            (M, 2) int, (row, col) of the selected windows' first pixel, ordered by increasing
                         detected fraction (ties in random order)
      detected_fraction  (M,) float, fraction of the window's pixels DETECTED in at least one band
      counts             dict n_windows (grid windows centred in the inner region), n_candidates, n_clean
    """
    half = size // 2
    hard = np.any((img["mask"] & masks.bitmask(img["bits"], ["NO_DATA"] + list(bad_planes))) != 0, axis=0)
    detected = img["detected"]
    H, W = detected.shape
    rr, cc = np.meshgrid(np.arange(0, H - size + 1, size), np.arange(0, W - size + 1, size), indexing="ij")
    corners = np.column_stack([rr.ravel(), cc.ravel()])
    inner = img["meta"].get("inner_bbox")
    if inner is not None:
        cx, cy = corners[:, 1] + half + img["x0"], corners[:, 0] + half + img["y0"]
        keep = (cx >= inner["x_min"]) & (cx <= inner["x_max"]) & (cy >= inner["y_min"]) & (cy <= inner["y_max"])
        corners = corners[keep]
    n_windows = len(corners)

    has_bad = np.array([hard[r:r + size, c:c + size].any() for r, c in corners], bool)
    frac = np.array([detected[r:r + size, c:c + size].mean() for r, c in corners], float)
    ok = ~has_bad & (frac <= max_detected_fraction)
    corners, frac = corners[ok], frac[ok]
    order = np.lexsort((rng.random(len(frac)), frac))        # primary key: detected fraction; then random
    counts = dict(n_windows=int(n_windows), n_candidates=int(len(corners)), n_clean=int((frac == 0).sum()))
    return corners[order], frac[order], counts


def fill_window(img, bank, r, c, size, rng, tries):
    """
    Cut the window at (r, c) and replace its detected pixels with real empty sky from the patch.

    Returns flux, variance, maskbits as (C, size, size) and the fraction of the detected pixels that could not be
    filled from clean sky (they keep their DETECTED bit and are masked at export).
    """
    need = img["detected"][r:r + size, c:c + size]
    if not need.any():
        sl = np.s_[:, r:r + size, c:c + size]
        return img["flux"][sl].copy(), img["var"][sl].copy(), img["mask"][sl].copy(), 0.0
    window, nbad = bank.find(need, rng, tries)
    flux, var, mbits = fill(img, c, r, size, need, window)
    return flux, var, mbits, float(nbad / need.sum())


def _up_to_date(path, params):
    if not path.exists():
        return False
    with np.load(path) as d:
        return "selection" in d and json.loads(str(d["selection"])) == params


def run(cfg):
    import pandas as pd

    params = selection_params(cfg)
    n_want, size = params["n_empty_per_patch"], params["stamp_px"]
    half = size // 2
    bands = list(cfg["bands"])
    cat = pd.read_parquet(cfg.data(TABLES_DIR, "catalog.parquet"), columns=["tract", "patch"])

    for tract, patch in patch_list(cfg, cat):
        out = patch_files_path(cfg, "empty", tract, patch, "npz")
        if _up_to_date(out, params):
            log(f"empty sky {tract}/{patch}: already done with the same selection, skipped")
            continue
        if out.exists():
            log(f"empty sky {tract}/{patch}: existing file made with another selection, recomputing")
        log(f"empty sky {tract}/{patch}: loading the coadds")
        img = load_images(cfg, tract, patch)
        rng = np.random.default_rng([params["seed"], int(tract), int(patch)])
        corners, frac, counts = rank_windows(img, params["bad_planes"], size, params["max_detected_fraction"], rng)
        log(f"empty sky {tract}/{patch}: {counts['n_candidates']} candidate windows of {counts['n_windows']} "
            f"(detected fraction <= {params['max_detected_fraction']}; {counts['n_clean']} fully clean)")
        # Sky to fill from: every pixel neither detected, unexposed nor damaged (as for the isolated stamps).
        unusable = masks.bitmask(img["bits"], ["DETECTED", "NO_DATA"] + params["bad_planes"])
        bank = SkyBank(img, unusable, size)

        keys = ("flux", "variance", "maskbits", "psf", "ra", "dec", "x_tract", "y_tract", "detected_fraction",
                "fill_bad_fraction")
        windows = {k: [] for k in keys}
        wcs = img["wcs"][bands.index("i") if "i" in bands else 0]
        n_no_psf = 0
        progress = Progress(min(n_want, len(corners)), f"empty sky {tract}/{patch} windows kept")
        for (r, c), f_det in zip(corners, frac):
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
            flux, var, mbits, bad_frac = fill_window(img, bank, r, c, size, rng, params["sky_tries"])
            windows["flux"].append(np.moveaxis(flux, 0, -1))
            windows["variance"].append(np.moveaxis(var, 0, -1))
            windows["maskbits"].append(np.moveaxis(mbits, 0, -1))
            ra, dec = wcs.pixel_to_world_values(col, row)
            windows["ra"].append(float(ra))
            windows["dec"].append(float(dec))
            windows["x_tract"].append(img["x0"] + col)
            windows["y_tract"].append(img["y0"] + row)
            windows["detected_fraction"].append(float(f_det))
            windows["fill_bad_fraction"].append(bad_frac)
            progress.update()

        n, p = len(windows["ra"]), int(cfg["psf_px"])
        df = np.asarray(windows["detected_fraction"])
        log(f"empty sky {tract}/{patch}: kept {n} ({n_no_psf} skipped for a missing PSF); detected fraction "
            f"median {np.median(df) if n else float('nan'):.3f}, max {df.max() if n else float('nan'):.3f}; "
            f"unfilled pixels {np.mean(windows['fill_bad_fraction']) if n else float('nan'):.4f} of the detected")
        if n < n_want:
            log(f"empty sky {tract}/{patch}: WARNING only {n} windows of the {n_want} requested")
        arrays = dict(
            flux=np.asarray(windows["flux"], np.float32).reshape(n, size, size, len(bands)),
            variance=np.asarray(windows["variance"], np.float32).reshape(n, size, size, len(bands)),
            maskbits=np.asarray(windows["maskbits"], np.int32).reshape(n, size, size, len(bands)),
            ra=np.asarray(windows["ra"]), dec=np.asarray(windows["dec"]),
            x_tract=np.asarray(windows["x_tract"], np.int64), y_tract=np.asarray(windows["y_tract"], np.int64),
            tract=np.full(n, tract, np.int32), patch=np.full(n, patch, np.int32),
            detected_fraction=df.astype(np.float32),
            fill_bad_fraction=np.asarray(windows["fill_bad_fraction"], np.float32),
            plane_bits=np.array(json.dumps(img["bits"])), selection=np.array(json.dumps(params)),
            n_windows=np.array(counts["n_windows"]), n_candidates=np.array(counts["n_candidates"]),
            n_clean=np.array(counts["n_clean"]), n_no_psf=np.array(n_no_psf))
        if windows["psf"] or n == 0:
            arrays["psf"] = np.asarray(windows["psf"], np.float32).reshape(n, p, p, len(bands))
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(out)                                          # the file appears only when complete
        del img
