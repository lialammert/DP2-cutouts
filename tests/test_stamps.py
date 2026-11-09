# test_stamps.py
# Anna Preto
# AstroDeep Project
# Les deux pièges de la recette, sur une image synthétique : l'ordre (colonne, ligne) de la solution astrométrique
# contre l'indexation [ligne, colonne] des tableaux, et l'origine 1 de FITS contre l'origine 0 des tableaux.

import numpy as np
from astropy.wcs import WCS

from res.common.shapes import adaptive_moments


def tract_like_wcs():
    # Même forme que le WCS des deep_coadd : TAN, x vers l'ouest (PC1_1 < 0), y vers le nord, 0,2 seconde d'arc.
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [53.1, -27.8]
    w.wcs.crpix = [101.0, 81.0]  # FITS : pixel de référence en indices 1
    s = 0.2 / 3600
    w.wcs.cd = [[-s, 0.0], [0.0, s]]
    return w


def gaussian(shape, xc, yc, sx, sy):
    rows, cols = np.indices(shape)
    return np.exp(-0.5 * (((cols - xc) / sx) ** 2 + ((rows - yc) / sy) ** 2))


def test_world_to_pixel_is_column_row_zero_based():
    w = tract_like_wcs()
    x, y = w.world_to_pixel_values(53.1, -27.8)
    # Le pixel de référence FITS (101, 81) est l'indice tableau [ligne 80, colonne 100].
    assert np.isclose(x, 100.0) and np.isclose(y, 80.0)


def test_stamp_centered_and_not_transposed():
    w = tract_like_wcs()
    shape = (160, 200)  # ligne, colonne : non carré pour qu'une transposition se voie
    ra, dec = w.pixel_to_world_values(130.3, 60.7)
    img = gaussian(shape, 130.3, 60.7, 4.0, 1.5)  # allongée selon x (colonnes) : e1 > 0
    x, y = w.world_to_pixel_values(ra, dec)
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    stamp = img[iy - 29:iy + 30, ix - 29:ix + 30]
    peak = np.unravel_index(np.argmax(stamp), stamp.shape)
    assert peak == (29, 29)
    m = adaptive_moments(stamp, center=(x - (ix - 29), y - (iy - 29)))
    assert m["ok"] and m["e1"] > 0.5 and abs(m["e2"]) < 0.01
    # La même galaxie, vignette transposée par erreur : e1 change de signe.
    mt = adaptive_moments(stamp.T, center=(29, 29))
    assert mt["e1"] < -0.5
