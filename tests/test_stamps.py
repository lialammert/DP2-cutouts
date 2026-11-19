# test_stamps.py
# Anna Preto
# AstroDeep Project
# The two pitfalls of the recipe, on a synthetic image: the (column, row) order of the WCS against the
# [row, column] indexing of arrays, and the FITS origin 1 against the array origin 0.

import numpy as np
from astropy.wcs import WCS

from dp2_cutouts.common.shapes import adaptive_moments


def tract_like_wcs():
    # Same form as the deep_coadd WCS: TAN, x towards west (PC1_1 < 0), y towards north, 0.2 arcsec.
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [53.1, -27.8]
    w.wcs.crpix = [101.0, 81.0]  # FITS: reference pixel as 1-based indices
    s = 0.2 / 3600
    w.wcs.cd = [[-s, 0.0], [0.0, s]]
    return w


def gaussian(shape, xc, yc, sx, sy):
    rows, cols = np.indices(shape)
    return np.exp(-0.5 * (((cols - xc) / sx) ** 2 + ((rows - yc) / sy) ** 2))


def test_world_to_pixel_is_column_row_zero_based():
    w = tract_like_wcs()
    x, y = w.world_to_pixel_values(53.1, -27.8)
    # The FITS reference pixel (101, 81) is array index [row 80, column 100].
    assert np.isclose(x, 100.0) and np.isclose(y, 80.0)


def test_stamp_centered_and_not_transposed():
    w = tract_like_wcs()
    shape = (160, 200)  # row, column: not square so that a transposition shows
    ra, dec = w.pixel_to_world_values(130.3, 60.7)
    img = gaussian(shape, 130.3, 60.7, 4.0, 1.5)  # elongated along x (columns): e1 > 0
    x, y = w.world_to_pixel_values(ra, dec)
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    stamp = img[iy - 29:iy + 30, ix - 29:ix + 30]
    peak = np.unravel_index(np.argmax(stamp), stamp.shape)
    assert peak == (29, 29)
    m = adaptive_moments(stamp, center=(x - (ix - 29), y - (iy - 29)))
    assert m["ok"] and m["e1"] > 0.5 and abs(m["e2"]) < 0.01
    # The same galaxy, stamp transposed by mistake: e1 changes sign.
    mt = adaptive_moments(stamp.T, center=(29, 29))
    assert mt["e1"] < -0.5


def test_shared_footprint_companion_wings_are_measured():
    # A target and a companion in one detection footprint: the companion's core (shared_k times its ellipse) is
    # masked, its wings stay. shared_light_frac counts the kept light where the companion is brighter than the
    # target: large for a bright companion, near 0 for a faint one (a clump, a faint neighbour), 0 when alone.
    import pandas as pd

    from dp2_cutouts.stages.stamps import companion_mask_footprint, shared_light_fraction

    shape = (41, 41)
    usable = np.ones(shape, bool)
    target_img = gaussian(shape, 14.0, 20.0, 2.0, 2.0)
    f_t = target_img.sum()
    target = (14.0, 20.0, 2.0, 2.0, 0.0, f_t)

    def scene(amp, xc=26.0):
        comp_img = amp * gaussian(shape, xc, 20.0, 4.0, 4.0)
        flux = target_img + comp_img
        det = flux > 0.02 * target_img.max()
        comp = pd.DataFrame(dict(x=[xc], y=[20.0], a=[2.0 * 4.0 / 2.0], b=[4.0], theta=[0.0],
                                 flux=[comp_img.sum()]))
        need, info = companion_mask_footprint(det, shape, 0, 0, target, comp, 10.0, 2, 2.0)
        return flux, need, info

    flux, need, info = scene(20.0)                                # 80 times the target's flux
    assert info["n_shared"] == 1 and info["flux_ratio_max"] > 50
    assert info["target_fp"][20, 14] and info["target_fp"][20, 26]
    assert need[20, 26] and not need[20, 14]                      # companion core masked, target kept
    assert info["shared"][20, 36] and not info["shared"][20, 14]  # its wings are the companion's
    assert shared_light_fraction(flux, need, info, usable) > 0.3

    flux, need, info = scene(0.01, 19.0)                          # a faint companion in the target's wings
    assert info["n_shared"] == 1 and info["flux_ratio_max"] < 0.1
    assert shared_light_fraction(flux, need, info, usable) < 0.02

    alone = target_img
    need, info = companion_mask_footprint(alone > 0.05 * alone.max(), shape, 0, 0, target,
                                          pd.DataFrame(columns=["x", "y", "a", "b", "theta", "flux"]), 10.0, 2, 2.0)
    assert info["n_shared"] == 0 and not need.any()
    assert shared_light_fraction(alone, need, info, usable) == 0.0
