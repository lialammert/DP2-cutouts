# test_pairs.py
# Anna Preto
# AstroDeep Project
# Paired datasets without the network or the VAE: storage and merge, splits, Euclid cutout handling, Euclid-side
# overlap, controlled blends, configuration overrides, orientation check.

import json

import numpy as np
import pandas as pd
import pytest

from dp2_cutouts.pairs import blends, splits, store
from dp2_cutouts.pairs.dataset2 import overlap_fraction
from dp2_cutouts.pairs.quality import angle_diff
from dp2_cutouts.sources import euclid_images


def _shard(tmp, name, n, start):
    meta = pd.DataFrame(dict(object_id=np.arange(start, start + n), tract=1, patch=start))
    arrays = dict(a=np.arange(start, start + n, dtype=np.float32)[:, None, None] * np.ones((n, 3, 3), np.float32),
                  m=np.zeros((n, 3, 3), bool))
    store.write_shard(tmp / "shards" / name, meta, arrays, pd.DataFrame([dict(reason="x")]))


def test_store_merge_and_take(tmp_path):
    _shard(tmp_path, "p0", 4, 0)
    _shard(tmp_path, "p1", 0, 100)  # empty patch: must not break the merge
    _shard(tmp_path, "p2", 3, 10)
    n = store.merge_shards(sorted((tmp_path / "shards").glob("*")), tmp_path, dict(name="t"))
    assert n == 7
    st = store.Store(tmp_path)
    assert list(st.meta["object_id"]) == [0, 1, 2, 3, 10, 11, 12]
    got = st.take("a", [5, 0, 3])[:, 0, 0]
    assert np.array_equal(got, [11, 0, 3])  # any order in, same order out
    assert len(st.rejected()) == 3
    assert not (tmp_path / ".merge.lock").exists()


def test_store_rejects_mismatched_rows(tmp_path):
    with pytest.raises(ValueError):
        store.write_shard(tmp_path / "s", pd.DataFrame(dict(a=[1, 2])), dict(x=np.zeros((3, 2))))


def test_splits_by_patch_are_stable():
    fr = dict(train=0.8, val=0.1, test=0.1)
    tracts, patches = np.repeat([5063, 5064], 50), np.tile(np.arange(50), 2)
    s1 = splits.assign(tracts, patches, fr, seed=1)
    order = np.random.default_rng(0).permutation(100)
    s2 = splits.assign(tracts[order], patches[order], fr, seed=1)
    assert np.array_equal(s1[order], s2)  # does not depend on order or on the other patches
    assert set(s1) <= set(splits.SPLITS)
    many = splits.assign(np.zeros(20000, int), np.arange(20000), fr, seed=3)
    assert abs((many == "train").mean() - 0.8) < 0.02


def test_cross_split_neighbours():
    ra = np.array([10.0, 10.0 + 1 / 3600, 20.0])
    dec = np.zeros(3)
    bad = splits.cross_split_neighbours(ra, dec, np.array(["train", "test", "train"]), 2.0)
    assert list(bad) == [True, True, False]


def test_crop_center_odd_and_border():
    a = np.arange(25.0).reshape(5, 5)
    out, dx, dy, (x0, y0) = euclid_images.crop_center(a, 2.3, 1.6, 3, -1.0)
    assert out.shape == (3, 3) and out[1, 1] == a[2, 2]
    assert np.isclose(dx, 0.3) and np.isclose(dy, -0.4)
    edge, *_ = euclid_images.crop_center(a, 0.0, 0.0, 3, -1.0)
    assert edge[0, 0] == -1.0 and edge[1, 1] == a[0, 0]


def test_parse_mmu_image_and_band_array():
    n = 9
    rec = [dict(band=b, flux=np.full((n, n), k + 1.0).tolist(), ivar=np.ones((n, n)).tolist(),
                mask=np.zeros((n, n), bool).tolist(), scale=0.1) for k, b in enumerate(["VIS", "NIR_J"])]
    cut = euclid_images.parse_mmu_image(rec, ["VIS", "Y", "J", "H"], 5, 0.1)
    assert cut["bands"] == ["VIS", "J"] and cut["flux"].shape == (5, 5, 2) and cut["scale_ok"]
    f, iv, m, present = euclid_images.to_band_array(cut, ["VIS", "Y", "J", "H"])
    assert list(present) == [True, False, True, False]
    assert m[..., 1].all() and not m[..., 0].any() and np.all(f[..., 2] == 2.0)


def test_overlap_fraction():
    t = (0.0, 0.0, 3.0, 2.0, 0.0)
    assert overlap_fraction(t, [], 0.1) == 0.0
    assert overlap_fraction(t, [(20.0, 0.0, 3.0, 2.0, 0.0)], 0.1) == 0.0
    assert overlap_fraction(t, [t], 0.1) == pytest.approx(1.0)
    half = overlap_fraction((0.0, 0.0, 3.0, 3.0, 0.0), [(100.0, 0.0, 100.0, 100.0, 0.0)], 0.05)
    assert half == pytest.approx(0.5, abs=0.03)


def test_shift_zero_and_blend_noise():
    a = np.zeros((7, 7, 1), np.float32)
    a[3, 3, 0] = 1.0
    s = blends.shift_zero(a, 2, -1)
    assert s[2, 5, 0] == 1.0 and s.sum() == 1.0
    t = dict(flux=a, ivar=np.full_like(a, 4.0), mask=np.zeros_like(a, bool))
    f, iv, m = blends.make_one(t, t, 0, 0, 0.5)
    assert f[3, 3, 0] == pytest.approx(1.5)
    assert iv[0, 0, 0] == pytest.approx(1.0 / (0.25 + 0.25 / 4.0))


def test_angle_diff():
    assert angle_diff(10.0, 170.0) == pytest.approx(20.0)
    assert angle_diff(0.0, 90.0) == pytest.approx(90.0)


def test_mlconfig_overrides_and_env(tmp_path, monkeypatch):
    from dp2_cutouts.mlconfig import MLConfig
    p = tmp_path / "c.yaml"
    p.write_text("ml_root: ${ROOT_X}/out\nstage1:\n  epochs: 3\n  weights: {latent: 1.0}\n")
    monkeypatch.setenv("ROOT_X", str(tmp_path))
    cfg = MLConfig.load(p, ["stage1.epochs=7", "stage1.weights.latent=0.5"])
    assert cfg["stage1"]["epochs"] == 7 and cfg["stage1"]["weights"]["latent"] == 0.5
    assert cfg.root == tmp_path / "out"
    with pytest.raises(KeyError):
        MLConfig.load(p, ["stage1.epoch=7"])  # typo must not pass silently
    monkeypatch.delenv("ROOT_X")
    with pytest.raises(KeyError):
        MLConfig.load(p)


def test_repo_ml_config_parses(monkeypatch):
    from pathlib import Path
    from dp2_cutouts.mlconfig import MLConfig
    monkeypatch.setenv("DP2_CONFIG", "x.yaml")
    monkeypatch.setenv("DP2_ML_ROOT", "/tmp/ml")
    cfg = MLConfig.load(Path(__file__).parents[1] / "config" / "ml_jeanzay.yaml")
    fr = cfg["splits"]["fractions"]
    assert abs(sum(fr.values()) - 1.0) < 1e-9
    assert len(cfg["stage1"]["band_weights"]) == len(cfg["rubin"]["bands"])
    assert set(cfg["euclid"]["encoder_bands"]) <= set(cfg["euclid"]["bands"])
    json.dumps(cfg.raw)
