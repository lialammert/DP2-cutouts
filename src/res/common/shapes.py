# shapes.py
# Anna Preto
# AstroDeep Project
# Mesure de forme par moments adaptatifs (galsim HSM). L'ellipticité rendue est la distorsion
# e = (Ixx - Iyy, 2 Ixy) / (Ixx + Iyy), la même définition que les colonnes e1, e2 de copaings.
# Une mesure ratée rend NaN : la mesure de forme ne sert jamais à sélectionner.

import galsim
import numpy as np


def adaptive_moments(image, weight=None, center=None):
    # image : tableau [ligne, colonne] ; weight : 0 sur les pixels à ignorer ; center : (colonne, ligne) en indices 0.
    img = galsim.Image(np.ascontiguousarray(image, dtype=np.float64))
    kw = {}
    if weight is not None:
        kw["weight"] = galsim.Image(np.ascontiguousarray(weight, dtype=np.int32))
    if center is not None:
        # galsim indexe les pixels à partir de 1.
        kw["guess_centroid"] = galsim.PositionD(center[0] + 1.0, center[1] + 1.0)
    res = galsim.hsm.FindAdaptiveMom(img, strict=False, **kw)
    if res.moments_status != 0:
        return dict(e1=np.nan, e2=np.nan, sigma=np.nan, ok=False)
    return dict(e1=float(res.observed_shape.e1), e2=float(res.observed_shape.e2),
                sigma=float(res.moments_sigma), ok=True)


def delta_e(m0, m1):
    return float(np.hypot(m1["e1"] - m0["e1"], m1["e2"] - m0["e2"]))


def circular_weight(shape, xc, yc, radius):
    rows, cols = np.indices(shape)
    return ((cols - xc) ** 2 + (rows - yc) ** 2 <= radius ** 2).astype(np.int32)
