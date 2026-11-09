# ellipse.py
# Anna Preto
# AstroDeep Project
# The pipeline's single ellipse, the one used by the copaings overlap test: i-band second moments with the
# 2 ln 2 factor (half light) for Rubin, MER catalogue semi-axes broadened by the PSF for Euclid.
# It is used by cut 5 (k times the ellipse) and by companion masking.
#
# Frame: Rubin tract pixels, x = column (towards west), y = row (towards north). An array is indexed
# [row, column] = [y, x]; theta is measured from the x axis, in degrees.

import numpy as np

FWHM2SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
LSST_PIXEL_SCALE = 0.2  # arcsec per pixel


def moments_to_ellipse(ixx, iyy, ixy):
    # Same semi-axes and angle as friendly.ellipses.moments2ab, written with eigenvalues so that they stay defined
    # when the angle is 45 degrees (cos 2theta = 0 in friendly's formula).
    ixx, iyy, ixy = (np.asarray(v, dtype=float) for v in (ixx, iyy, ixy))
    half_tr = 0.5 * (ixx + iyy)
    root = np.sqrt((0.5 * (ixx - iyy)) ** 2 + ixy ** 2)
    with np.errstate(invalid="ignore"):
        a = np.sqrt(2.0 * np.log(2.0) * (half_tr + root))
        b = np.sqrt(2.0 * np.log(2.0) * (half_tr - root))
    theta = np.degrees(0.5 * np.arctan2(2.0 * ixy, ixx - iyy)) % 180.0
    return a, b, theta


def psf_ellipse(ixx_psf, iyy_psf, pixel_scale=LSST_PIXEL_SCALE):
    # Fallback for a Rubin object without moments: the PSF circle, with the same 2 ln 2 factor.
    sigma2 = 0.5 * (np.asarray(ixx_psf, float) + np.asarray(iyy_psf, float))
    r = np.sqrt(2.0 * np.log(2.0) * sigma2)
    return r, r, np.zeros_like(r)


def euclid_ellipse(size_true, size_minor_true, position_angle_true, psf_fwhm_i, pixel_scale=LSST_PIXEL_SCALE):
    # Copy of copaings.eot_fast.EllipseCache.true_params (semi-axes broadened by the PSF, in Rubin pixels, and
    # theta = 270 - position_angle_true folded into [0, 180]): the same ellipse as in the overlap test.
    psf = np.sqrt(2.0) * np.asarray(psf_fwhm_i, float) / FWHM2SIGMA  # arcsec
    a = np.sqrt(np.asarray(size_true, float) ** 2 + psf ** 2) / pixel_scale
    b = np.sqrt(np.asarray(size_minor_true, float) ** 2 + psf ** 2) / pixel_scale
    theta = 270.0 - np.asarray(position_angle_true, float)
    theta = np.where(theta > 180.0, theta - 180.0, theta)
    return a, b, theta


def elliptical_radius(dx, dy, a, b, theta_deg):
    # Reduced radius: 1 on the edge of the ellipse, k on the edge of k times the ellipse.
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt((u / a) ** 2 + (v / b) ** 2)


def radius_map(shape, xc, yc, a, b, theta_deg):
    # Reduced radius of every pixel of a box; xc, yc in the box frame (column, row).
    rows, cols = np.indices(shape)
    return elliptical_radius(cols - xc, rows - yc, a, b, theta_deg)
