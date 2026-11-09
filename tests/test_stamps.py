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
