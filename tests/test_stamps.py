# test_stamps.py
# Anna Preto
# AstroDeep Project
# The two pitfalls of the recipe, on a synthetic image: the (column, row) order of the WCS against the
# [row, column] indexing of arrays, and the FITS origin 1 against the array origin 0.

import numpy as np
import pytest
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


def test_companion_light_left_on_the_target():
    # Footprint masking keeps the wings of a companion sharing the target footprint (only shared_k times its
    # ellipse is filled). companion_light_ratio predicts that light on the kept pixels around the target: large for
    # a bright neighbour, small for a faint one, 0 when alone; a Euclid duplicate of a Rubin companion counts once.
    import pandas as pd

    from dp2_cutouts.stages.stamps import companion_light_ratio, companion_mask_footprint

    shape = (41, 41)
    usable = np.ones(shape, bool)
    target = (14.0, 20.0, 2.0, 2.0, 0.0, 25.0)

    def comp_at(xc, flux, survey="rubin"):
        return pd.DataFrame(dict(x=[xc], y=[20.0], a=[4.0], b=[4.0], theta=[0.0], flux=[flux], survey=[survey]))

    flux = gaussian(shape, 14.0, 20.0, 2.0, 2.0) + 20.0 * gaussian(shape, 26.0, 20.0, 4.0, 4.0)
    bright = comp_at(26.0, 2000.0)
    need, info = companion_mask_footprint(flux > 0.02, shape, 0, 0, target, bright, 10.0, 2, 2.0)
    assert info["n_shared"] == 1 and info["flux_ratio_max"] == 80.0
    assert need[20, 26] and not need[20, 14]                      # companion core masked, target kept
    r_bright = companion_light_ratio(shape, target, bright, ~need & usable)
    assert r_bright > 0.3

    faint = comp_at(26.0, 2.0)
    r_faint = companion_light_ratio(shape, target, faint, ~need & usable)
    assert r_faint < 0.01 and r_faint == pytest.approx(r_bright / 1000.0)

    twice = pd.concat([bright, comp_at(26.5, 2000.0, "euclid")], ignore_index=True)
    assert companion_light_ratio(shape, target, twice, ~need & usable) == pytest.approx(r_bright)

    alone = comp_at(0.0, 0.0).iloc[:0]
    assert companion_light_ratio(shape, target, alone, usable) == 0.0
    assert np.isnan(companion_light_ratio(shape, (*target[:5], np.nan), bright, usable))
