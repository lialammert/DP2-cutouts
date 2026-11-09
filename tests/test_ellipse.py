# test_ellipse.py
# Anna Preto
# AstroDeep Project
# L'ellipse de la chaîne doit être exactement celle de copaings et friendly (test de recouvrement), sans quoi la
# coupure 5 et le masquage n'utiliseraient pas la même ellipse que la coupure 2.

import numpy as np
import pytest

from res.common.ellipse import elliptical_radius, euclid_ellipse, moments_to_ellipse


def test_moments_match_friendly():
    ellipses = pytest.importorskip("friendly.ellipses")
    rng = np.random.default_rng(0)
    ixx, iyy = rng.uniform(1, 20, 500), rng.uniform(1, 20, 500)
    ixy = rng.uniform(-0.9, 0.9, 500) * np.sqrt(ixx * iyy)
    a, b, th = moments_to_ellipse(ixx, iyy, ixy)
    fa, fb, fth = ellipses.moments2ab(ixx.copy(), iyy.copy(), ixy.copy())
    # friendly ordonne les axes selon theta ; on compare le couple (grand axe, petit axe) et l'angle du grand axe.
    np.testing.assert_allclose(a, np.maximum(fa, fb), rtol=1e-10)
    np.testing.assert_allclose(b, np.minimum(fa, fb), rtol=1e-10)
    fmajor = np.where(fa >= fb, fth, fth + 90.0) % 180.0
    d = (th - fmajor + 90.0) % 180.0 - 90.0
    np.testing.assert_allclose(d, 0.0, atol=1e-8)


def test_half_light_factor():
    # Gaussienne ronde de sigma 2 px : demi-axe sqrt(2 ln 2) sigma, le contour à mi-hauteur.
    a, b, _ = moments_to_ellipse(4.0, 4.0, 0.0)
    assert np.isclose(a, np.sqrt(2 * np.log(2)) * 2.0) and np.isclose(b, a)


def test_euclid_ellipse_matches_copaings():
    pytest.importorskip("copaings")
    import pandas as pd
    from copaings.eot_fast import EllipseCache

    rng = np.random.default_rng(1)
    n = 50
    lsst = pd.DataFrame(dict(ra=[53.1], dec=[-27.8], x=[0.0], y=[0.0], Ixx_pixel_i=[4.0], Iyy_pixel_i=[3.0],
                             Ixy_pixel_i=[0.5], psf_fwhm_i=[0.8]))
    true = pd.DataFrame(dict(ra=53.1 + rng.normal(0, 1e-4, n), dec=-27.8 + rng.normal(0, 1e-4, n),
                             size_true=rng.uniform(0.1, 1.0, n), position_angle_true=rng.uniform(-90, 270, n)))
    true["size_minor_true"] = true["size_true"] * rng.uniform(0.3, 1.0, n)
    cache = EllipseCache(lsst, true)
    params, _, _, amax = cache.true_params(np.arange(n), [0])
    a, b, th = euclid_ellipse(true["size_true"], true["size_minor_true"], true["position_angle_true"], 0.8)
    np.testing.assert_allclose(np.maximum(a, b), amax, rtol=1e-12)


def test_elliptical_radius_orientation():
    # Ellipse allongée le long de x : un point à (a, 0) est sur le bord, un point à (0, a) en est loin.
    assert np.isclose(elliptical_radius(4.0, 0.0, 4.0, 1.0, 0.0), 1.0)
    assert elliptical_radius(0.0, 4.0, 4.0, 1.0, 0.0) > 3.9
    assert np.isclose(elliptical_radius(0.0, 4.0, 4.0, 1.0, 90.0), 1.0)
