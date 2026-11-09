# ellipse.py
# Anna Preto
# AstroDeep Project
# L'ellipse unique de la chaîne, celle du test de recouvrement de copaings : moments d'ordre deux en bande i avec le
# facteur 2 ln 2 (demi-lumière) pour Rubin, demi-axes du catalogue MER élargis par la tache pour Euclid.
# Elle sert à la coupure 5 (k fois l'ellipse) et au masquage des compagnes.
#
# Repère : celui des pixels du tract Rubin, x = colonne (vers l'ouest), y = ligne (vers le nord). Un tableau
# s'indexe [ligne, colonne] = [y, x] ; theta est compté depuis l'axe x, en degrés.

import numpy as np

FWHM2SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
LSST_PIXEL_SCALE = 0.2  # seconde d'arc par pixel


def moments_to_ellipse(ixx, iyy, ixy):
    # Mêmes demi-axes et angle que friendly.ellipses.moments2ab, écrits par valeurs propres pour rester définis
    # quand l'angle vaut 45 degrés (cos 2theta = 0 dans la formule de friendly).
    ixx, iyy, ixy = (np.asarray(v, dtype=float) for v in (ixx, iyy, ixy))
    half_tr = 0.5 * (ixx + iyy)
    root = np.sqrt((0.5 * (ixx - iyy)) ** 2 + ixy ** 2)
    with np.errstate(invalid="ignore"):
        a = np.sqrt(2.0 * np.log(2.0) * (half_tr + root))
        b = np.sqrt(2.0 * np.log(2.0) * (half_tr - root))
    theta = np.degrees(0.5 * np.arctan2(2.0 * ixy, ixx - iyy)) % 180.0
    return a, b, theta


def psf_ellipse(ixx_psf, iyy_psf, pixel_scale=LSST_PIXEL_SCALE):
    # Repli pour un objet Rubin sans moments : cercle de la tache, au même facteur 2 ln 2.
    sigma2 = 0.5 * (np.asarray(ixx_psf, float) + np.asarray(iyy_psf, float))
    r = np.sqrt(2.0 * np.log(2.0) * sigma2)
    return r, r, np.zeros_like(r)


def euclid_ellipse(size_true, size_minor_true, position_angle_true, psf_fwhm_i, pixel_scale=LSST_PIXEL_SCALE):
    # Recopie de copaings.eot_fast.EllipseCache.true_params (demi-axes élargis par la tache, en pixels Rubin,
    # et theta = 270 - position_angle_true ramené dans [0, 180]) : la même ellipse que celle du test de recouvrement.
    psf = np.sqrt(2.0) * np.asarray(psf_fwhm_i, float) / FWHM2SIGMA  # seconde d'arc
    a = np.sqrt(np.asarray(size_true, float) ** 2 + psf ** 2) / pixel_scale
    b = np.sqrt(np.asarray(size_minor_true, float) ** 2 + psf ** 2) / pixel_scale
    theta = 270.0 - np.asarray(position_angle_true, float)
    theta = np.where(theta > 180.0, theta - 180.0, theta)
    return a, b, theta


def elliptical_radius(dx, dy, a, b, theta_deg):
    # Rayon réduit : 1 sur le bord de l'ellipse, k sur le bord de k fois l'ellipse.
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt((u / a) ** 2 + (v / b) ** 2)


def radius_map(shape, xc, yc, a, b, theta_deg):
    # Rayon réduit de chaque pixel d'une boîte ; xc, yc dans le repère de la boîte (colonne, ligne).
    rows, cols = np.indices(shape)
    return elliptical_radius(cols - xc, rows - yc, a, b, theta_deg)
