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


def test_fixed_splits_override_the_draw():
    meta = pd.DataFrame(dict(tract=[5063] * 9, patch=[23, 24, 25, 33, 34, 35, 43, 44, 45]))
    cfg = dict(seed=20261116, splits=dict(fractions=dict(train=0.8, val=0.1, test=0.1)))
    drawn = splits.split_table(meta, cfg)
    cfg["splits"]["fixed"] = {"5063_45": "test"}
    s = splits.split_table(meta, cfg)
    assert s.iloc[-1] == "test" and (s.iloc[:-1] == drawn.iloc[:-1]).all()
    cfg["splits"]["fixed"] = {"5063_45": "tset"}
    with pytest.raises(ValueError):
        splits.split_table(meta, cfg)


def test_repo_split_of_the_first_scale_run_has_every_split(monkeypatch):
    # The 9 patches of the first Jean Zay scale run must give train, val and test with the repository config.
    from pathlib import Path
    from dp2_cutouts.mlconfig import MLConfig
    monkeypatch.setenv("DP2_CONFIG", "x.yaml")
    monkeypatch.setenv("DP2_ML_ROOT", "/tmp/ml")
    cfg = MLConfig.load(Path(__file__).parents[1] / "config" / "ml_jeanzay.yaml")
    meta = pd.DataFrame(dict(tract=[5063] * 9, patch=[23, 24, 25, 33, 34, 35, 43, 44, 45]))
    assert set(splits.split_table(meta, cfg)) == set(splits.SPLITS)


def test_fill_masked_replaces_only_bad_pixels():
    from dp2_cutouts.pairs.rubin_vae import fill_masked
    rng = np.random.default_rng(0)
    n, h, c = 3, 41, 2
    flux = rng.normal(size=(n, h, h, c)).astype(np.float32) * 2.0
    ivar = np.full(flux.shape, 0.25, np.float32)        # sigma 2
    mask = np.zeros(flux.shape, bool)
    mask[:, :20] = True                                  # half the stamp masked
    flux[0, 30, 30, 0] = np.nan                          # non-finite pixel, not masked
    ivar[1, 35, 35, 1] = 0.0                             # no variance, not masked
    out, frac = fill_masked(flux, ivar, mask, rng)
    good = ~mask
    good[0, 30, 30, 0] = good[1, 35, 35, 1] = False
    assert np.array_equal(out[good], flux[good])
    assert np.all(np.isfinite(out))
    assert abs(out[mask].std() - 2.0) < 0.05             # sky noise of the stamp, not zeros
    assert frac[2] == pytest.approx(20 / 41)


def test_export_rubin_vae_real_build(tmp_path):
    from dp2_cutouts.pairs import rubin_vae
    rng = np.random.default_rng(1)
    n, h = 12, 59
    d1 = tmp_path / "datasets" / "d1"
    d1.mkdir(parents=True)
    split = np.array(["train"] * 8 + ["val"] * 2 + ["test"] * 2, dtype=object)
    meta = pd.DataFrame(dict(object_id=np.arange(n), tract=5063, patch=np.repeat([23, 33, 45], 4), split=split,
                             rubin_valid_fraction=np.r_[np.full(n - 1, 0.95), 0.2], e1_rubin_check=0.1,
                             e2_rubin_check=-0.1, i_cModel_mag=23.0, snr_i=30.0))
    meta.to_parquet(d1 / "meta.parquet", index=False)
    flux = rng.normal(size=(n, h, h, 6)).astype(np.float32)
    mask = np.zeros(flux.shape, bool)
    mask[:, :5] = True
    for name, arr in dict(rubin_flux=flux, rubin_ivar=np.ones_like(flux), rubin_mask=mask).items():
        np.save(d1 / f"{name}.npy", arr)
    (d1 / "info.json").write_text(json.dumps(dict(n=n, rubin_bands=list("ugrizy"), config_fingerprint="x",
                                                  arrays={})))
    template = tmp_path / "paper_baseline"
    template.mkdir()
    (template / "config.json").write_text(json.dumps(dict(
        data=dict(dataset_path="paper_v1", normalization=["nonlinear"], input_shape=[59, 59, 6], train_size=100000),
        model=dict(latent_dim=32), training=dict(epochs=2000))))

    class Cfg(dict):
        def dataset_dir(self, name):
            return tmp_path / "datasets" / name

        def fingerprint(self, sections):
            return "f"

    cfg = Cfg(seed=3, rubin=dict(bands=list("ugrizy"), shape_band="i"),
              rubin_vae=dict(min_valid_fraction=0.7, stats_max_stamps=100, run_name="r",
                             debvader_overrides={"data.train_size": 0, "data.real_min_snr": 0.0}))
    info = rubin_vae.export(cfg, template)
    out = tmp_path / "datasets" / "rubin_vae"
    images = np.load(out / "images.npy")
    m = np.load(out / "meta.npz")
    assert images.shape == (n - 1, h, h, 6) and info["dropped_low_valid_fraction"] == 1
    assert list(m["split"]) == [0] * 8 + [1] * 2 + [2]
    assert np.array_equal(images[:, 5:], flux[: n - 1, 5:])           # unmasked pixels untouched
    assert np.all(images[:, :5] != 0)
    conf = json.loads((out / "debvader_config.json").read_text())
    assert conf["data"]["dataset_path"] == str(out) and conf["data"]["train_size"] == 0
    assert info["overrides_added_to_template"] == ["data.real_min_snr"]
    stats = json.loads((out / "stats.json").read_text())
    assert stats["units"] == "nJy" and len(stats["nonlinear"]["scale_per_band"]) == 6
    assert rubin_vae.export(cfg, template)["n"] == n - 1                # already written: skipped
    with pytest.raises(KeyError):
        rubin_vae.debvader_config(template, out, {"datta.x": 1})


def test_read_mmu_rows_by_row_group(tmp_path, monkeypatch):
    # MMU-like file: string object_id, nested image column, several row groups; only target rows come back.
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = 50
    img = [[dict(band="VIS", flux=[[float(i)] * 3] * 3, scale=0.1)] for i in range(n)]
    t = pa.table(dict(object_id=pa.array([str(-1000 - i) for i in range(n)]), image=img))
    f = tmp_path / "Npix=1.parquet"
    pq.write_table(t, f, row_group_size=10)
    monkeypatch.setattr(euclid_images, "mmu_partition_files", lambda url, box=None: [f])
    df = euclid_images.read_mmu_rows("x", [-1003, -1042, 7], ["object_id", "image"])
    assert sorted(df["object_id"]) == ["-1003", "-1042"]
    row = df.set_index("object_id").loc["-1042", "image"]
    assert euclid_images._as_records(row)[0]["flux"][0][0] == 42.0
    lay = euclid_images.mmu_layout([f])[0]
    assert lay["row_groups"] == 5 and lay["max_rows_per_group"] == 10
    with pytest.raises(MemoryError):
        euclid_images.read_mmu_rows("x", [-1003], ["object_id", "image"], max_group_mb=0.0)
