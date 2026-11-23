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


def test_units_follow_the_vae(tmp_path):
    pytest.importorskip("surveycodex")
    from types import SimpleNamespace
    from dp2_cutouts.ml.debvader_bridge import training_units
    from dp2_cutouts.ml.units import UnitConverter
    real = tmp_path / "real"
    real.mkdir()
    (real / "images.npy").write_bytes(b"")
    (real / "meta.npz").write_bytes(b"")
    conf = lambda p: SimpleNamespace(data=SimpleNamespace(dataset_path=str(p)))  # noqa: E731
    assert training_units(conf(real), {}) == "nJy"
    assert training_units(conf("paper_v1"), {}) == "lsst_electrons"
    assert training_units(conf("paper_v1"), {"units": "nJy"}) == "nJy"
    cfg = dict(rubin=dict(vae_units="auto", bands=list("ugrizy")))
    vae = SimpleNamespace(units_mode="nJy", run_dir=tmp_path)
    assert np.all(UnitConverter.for_vae(cfg, vae).factor == 1.0)
    cfg["rubin"]["vae_units"] = "lsst_electrons"
    with pytest.raises(ValueError):
        UnitConverter.for_vae(cfg, vae)


def test_ablation_settings_override_latent_loss_and_dropout():
    # An ablation entry gives loss weights, and may set latent_loss and the Euclid encoder dropout; those two never
    # end up among the weights.
    from dp2_cutouts.ml import stage1, stage2

    cfg = dict(stage1=dict(latent_loss="mu_mse", euclid_encoder=dict(dropout=0.0), weights=dict(latent=1.0),
                           ablations=dict(a=dict(latent=1.0, cross=1.0, infonce=0.1, dropout=0.2),
                                          b=dict(latent=1.0, cross=1.0, infonce=0.0, latent_loss="mu_whitened"))),
               stage2=dict(latent_loss="mu_mse", scene=dict(enabled=False, weight=1.0), weights=dict(latent=1.0),
                           ablations=dict(w=dict(latent=1.0, infonce=0.0, scene=0.0, latent_loss="mu_whitened"))))
    s = stage1.settings(cfg, "a")
    assert s["euclid_encoder"]["dropout"] == 0.2 and s["latent_loss"] == "mu_mse"
    assert s["weights"] == dict(latent=1.0, cross=1.0, infonce=0.1)
    s = stage1.settings(cfg, "b")
    assert s["latent_loss"] == "mu_whitened" and s["euclid_encoder"]["dropout"] == 0.0 and "latent_loss" not in s["weights"]
    assert stage1.settings(cfg)["latent_loss"] == "mu_mse"
    s = stage2.settings(cfg, "w")
    assert s["latent_loss"] == "mu_whitened" and "latent_loss" not in s["weights"] and not s["scene"]["enabled"]


def test_euclid_encoder_dropout_acts_in_training_only():
    pytest.importorskip("debvader")
    from dp2_cutouts.ml.models import EuclidEncoder

    keras.utils.set_random_seed(0)
    enc = EuclidEncoder(4, filters=(4, 8), cond_embedding=4, dense=16, dropout=0.5)
    x = [np.random.default_rng(0).normal(size=(2, 16, 16, 1)).astype("float32"), np.ones((2, 3), "float32")]
    a, b = (np.asarray(enc(x, training=False)) for _ in range(2))
    assert np.allclose(a, b)                                   # inference: deterministic
    c, d = (np.asarray(enc(x, training=True)) for _ in range(2))
    assert not np.allclose(c, d)                               # training: dropout draws differ
    assert EuclidEncoder.from_config(enc.get_config()).dropout_rate == 0.5


def test_phase0_draws_from_one_split_and_reports_per_field(tmp_path):
    # The phase 0 sample is taken from the patches of one split (the assignment D1 carries), read in the order
    # Dataset.take indexes; fields are told apart by declination; the ratio of sums is reported next to the median.
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as pds
    from dp2_cutouts.ml import phase0
    from dp2_cutouts.pairs import splits

    tract = np.repeat([5063, 2395], 200)
    patch = np.tile(np.repeat(np.arange(20), 10), 2)
    t = pa.table(dict(tract=tract, patch=patch, healpix5=np.repeat([1, 2], 200), row=np.arange(400)))
    pds.write_dataset(t, tmp_path / "data", format="parquet", partitioning=["healpix5"], partitioning_flavor="hive")
    d = pds.dataset(tmp_path / "data", format="parquet", partitioning="hive")
    cfg = dict(seed=0, splits=dict(fractions=dict(train=0.5, val=0.0, test=0.5, holdout=0.0)))
    rows = phase0.split_rows(d, cfg, "test")
    got = d.take(np.sort(rows)).to_pandas()
    want = splits.split_table(pd.DataFrame(dict(tract=got["tract"], patch=got["patch"])), cfg)
    assert len(rows) and (want == "test").all()
    assert len(phase0.split_rows(d, cfg, "all")) == 400
    with pytest.raises(ValueError):
        phase0.split_rows(d, dict(seed=0, splits=dict(fractions=dict(train=1.0))), "test")
    assert list(phase0.field_of([-27.8, -48.4])) == ["EDF-F", "EDF-S"]
    df = pd.DataFrame(dict(obs_flux_i=[1.0, -1.0, 3.0], rec_flux_i=[1.0, 1.0, 2.0],
                           rflux_i_rec_obs=[0.0, -2.0, -1 / 3]))
    s = phase0.flux_summary(df, ["i"])
    assert np.isclose(s["sums_rflux"]["i"], 4 / 3 - 1) and np.isclose(s["median_rflux"]["i"], -1 / 3)


def test_training_schedule_in_steps_backup_and_validation_subset(tmp_path):
    # train_steps sets the number of epochs from the size of the training split; backup_every_steps makes the
    # resumable backup step-based; val_max_examples takes a fixed random subset of the val split.
    import pandas as pd
    from dp2_cutouts.ml import training
    assert training.schedule(dict(epochs=300), 130) == 300
    assert training.schedule(dict(epochs=300, train_steps=None), 130) == 300
    assert training.schedule(dict(epochs=300, train_steps=150000), 5000) == 30
    assert training.schedule(dict(epochs=300, train_steps=10), 5000) == 1
    meta = pd.DataFrame(dict(split=["train"] * 50 + ["val"] * 40 + ["test"] * 10))
    assert training.validation_indices(meta, "val", None, 0) is None
    assert training.validation_indices(meta, "val", 100, 0) is None
    v = training.validation_indices(meta, "val", 15, 0)
    assert len(v) == 15 and set(v) <= set(range(50, 90))
    assert np.array_equal(v, training.validation_indices(meta, "val", 15, 0))
    cb = training.callbacks(tmp_path, {}, 5, True, backup_every_steps=500)
    assert cb[0].save_freq == 500
    assert training.callbacks(tmp_path, {}, 5, True)[0].save_freq == "epoch"
