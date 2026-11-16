# test_ml.py
# Anna Preto
# AstroDeep Project
# Pieces of the learning code that can be checked without the DebVAder weights: augmentation geometry (checked
# against measured moments, not against itself), the scene index maps, the loss terms, the unit conversion.
# Skipped where keras / surveycodex are not installed.

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from dp2_cutouts.ml import data, losses  # noqa: E402
from dp2_cutouts.pairs import blends, common  # noqa: E402


def gaussian(n, xc, yc, sx, sy, rho=0.0):
    rows, cols = np.indices((n, n))
    u, v = (cols - xc) / sx, (rows - yc) / sy
    return np.exp(-0.5 * (u * u + v * v - 2 * rho * u * v) / (1 - rho * rho))


@pytest.mark.parametrize("k,flip", [(0, True), (1, False), (1, True), (2, False), (3, True)])
def test_augmented_ellipticity_matches_measurement(k, flip):
    n = 41
    img = gaussian(n, 20, 20, 4.0, 2.0, 0.4)
    m0 = common.moments(img)
    aug = data.aug_images(img[..., None], k, flip)[..., 0]
    m1 = common.moments(aug)
    e1, e2 = data.aug_ellipticity(m0["e1"], m0["e2"], k, flip)
    assert m1["e1"] == pytest.approx(e1, abs=1e-4) and m1["e2"] == pytest.approx(e2, abs=1e-4)


@pytest.mark.parametrize("k,flip", [(1, False), (2, True), (3, False), (0, True)])
def test_augmented_shift_matches_images(k, flip):
    n, dx, dy = 15, 3, -2
    a = np.zeros((n, n, 1))
    c = n // 2
    a[c, c, 0] = 1.0
    a[c + dy, c + dx, 0] = 2.0
    b = data.aug_images(a, k, flip)[..., 0]
    r, cc = np.argwhere(b == 2.0)[0]
    ndx, ndy = data.aug_shift(dx, dy, k, flip)
    assert (cc - c, r - c) == (ndx, ndy)


def test_index_map_equals_shift():
    n = 11
    img = np.random.default_rng(0).normal(size=(n, n, 2))
    for dx, dy in [(0, 0), (3, -2), (-4, 5)]:
        idx, valid = data.index_map(n, dx, dy)
        moved = img.reshape(n * n, 2)[idx] * valid[:, None]
        assert np.allclose(moved.reshape(n, n, 2), blends.shift_zero(img, dx, dy))


def test_cond_indices_pairs_bands():
    cols = ["psf_sigma_r", "psf_e1_r", "psf_e1_i", "psf_e2_i", "psf_e2_r"]
    e1, e2 = data.cond_indices(cols)
    assert list(e1) == [1, 2] and list(e2) == [4, 3]


def test_holes_never_touch_centre():
    x = np.ones((61, 61, 1), np.float32)
    x[25:36, 25:36] = 5.0
    rng = np.random.default_rng(1)
    for _ in range(50):
        y = data.elliptical_holes(x, rng)
        assert np.all(y[28:33, 28:33] == 5.0)


def _np(t):
    return np.asarray(keras.ops.convert_to_numpy(t))


def test_latent_losses():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=(8, 4)).astype(np.float32)
    assert _np(losses.latent_mu_mse(mu, mu)) == pytest.approx(0.0)
    assert _np(losses.latent_mu_whitened(mu, mu + 1, np.eye(4, dtype=np.float32))) == pytest.approx(1.0)
    L = np.tile(np.diag([1.0, 2.0, 0.5, 1.5]).astype(np.float32), (8, 1, 1))
    assert _np(losses.gaussian_kl(mu, L, mu, L)) == pytest.approx(0.0, abs=1e-5)
    assert _np(losses.gaussian_kl(mu, L, mu + 0.5, L)) > 0


def test_whitening_from_latents():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=(5000, 3)) @ np.array([[2.0, 0, 0], [0.5, 1.0, 0], [0, 0, 0.1]])
    W, mean, cov = losses.whitening_from_latents(mu, ridge=0.0)
    white = (mu - mean) @ W.T
    assert np.allclose(np.cov(white, rowvar=False), np.eye(3), atol=0.05)
    # With the ridge, a near-inactive direction (variance 1e-6) is not blown up to unit variance.
    dead = np.c_[mu[:, :2], 1e-3 * rng.normal(size=len(mu))]
    W, mean, _ = losses.whitening_from_latents(dead)
    assert np.abs(W).max() < 20.0


def test_info_nce_and_twins():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(16, 8)).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    q = rng.normal(size=(16, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    aligned = _np(losses.info_nce(p, p, 10.0))
    random = _np(losses.info_nce(p, q, 10.0))
    assert aligned < random
    mu = np.zeros((3, 2), np.float32)
    mu[2] = 5.0
    m = _np(losses.twin_negative_mask(mu, 1.0))
    assert m[0, 1] == 0 and m[0, 2] == 1
    assert losses.retrieval_accuracy(p, p, 1) == 1.0


def test_units_lsst_electrons():
    pytest.importorskip("surveycodex")
    from dp2_cutouts.ml.units import UnitConverter
    u = UnitConverter("lsst_electrons", ["u", "g", "r", "i", "z", "y"])
    assert np.all(u.factor > 0)
    # Deeper bands collect more electrons per nJy for the same exposure: u is the least efficient.
    assert u.factor[0] == u.factor.min()
    f = np.ones((2, 3, 3, 6))
    assert np.allclose(u.flux_back(u.flux(f)), f, rtol=1e-6)
