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
    # MMU convention: mask True = valid pixel.
    rec = [dict(band=b, flux=np.full((n, n), k + 1.0).tolist(), ivar=np.ones((n, n)).tolist(),
                mask=np.ones((n, n), bool).tolist(), scale=0.1) for k, b in enumerate(["VIS", "NIR_J"])]
    cut = euclid_images.parse_mmu_image(rec, ["VIS", "Y", "J", "H"], 5, 0.1)
    assert cut["bands"] == ["VIS", "J"] and cut["flux"].shape == (5, 5, 2) and cut["scale_ok"]
    f, iv, m, present = euclid_images.to_band_array(cut, ["VIS", "Y", "J", "H"])
    assert list(present) == [True, False, True, False]
    assert m[..., 1].all() and not m[..., 0].any() and np.all(f[..., 2] == 2.0)
    # An MMU-invalid pixel (False), a zero-ivar pixel and a NaN flux are ours to mask; the flag can be flipped.
    vis = rec[0]
    vis["mask"][4][4] = False
    vis["ivar"][4][5] = 0.0
    vis["flux"][5][4] = float("nan")
    m = euclid_images.parse_mmu_image([vis], ["VIS"], n, 0.1)["mask"][..., 0]
    assert m[4, 4] and m[4, 5] and m[5, 4] and m.sum() == 3
    m = euclid_images.parse_mmu_image([vis], ["VIS"], n, 0.1, mask_true_is_valid=False)["mask"][..., 0]
    assert m.sum() == n * n - 1


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
    # Batches smaller than the row groups and not aligned on them: hits on both sides of a batch boundary.
    df = euclid_images.read_mmu_rows("x", [-1006, -1007, -1008, -1049], ["object_id", "image"], batch_rows=3)
    assert sorted(df["object_id"]) == ["-1006", "-1007", "-1008", "-1049"]
    got = {k: euclid_images._as_records(v)[0]["flux"][0][0] for k, v in zip(df["object_id"], df["image"])}
    assert got == {"-1006": 6.0, "-1007": 7.0, "-1008": 8.0, "-1049": 49.0}


def test_mmu_retries_transient_network_errors_only():
    class ProxyError(Exception):
        pass

    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("download failed") from ProxyError("500 Internal Server Error")
        return "ok"

    assert euclid_images.with_retries(flaky, "test", sleep=waits.append) == "ok"
    assert len(calls) == 3 and waits == [10.0, 20.0]
    assert euclid_images.is_transient(OSError("HTTP status=503 from proxy"))

    def broken():
        calls.append(1)
        raise KeyError("object_id")

    calls.clear()
    with pytest.raises(KeyError):
        euclid_images.with_retries(broken, "test", sleep=waits.append)
    assert len(calls) == 1

    def always():
        raise ProxyError("500 Internal Server Error")

    with pytest.raises(ProxyError):
        euclid_images.with_retries(always, "test", tries=3, sleep=lambda s: None)


def test_mmu_mirror_manifest_drives_the_reader(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    url = "hf://datasets/org/repo/euclid"
    assert euclid_images.hf_repo_file(url + "/dataset/Norder=7/Dir=0/Npix=5.parquet") == \
        ("org/repo", None, "euclid/dataset/Norder=7/Dir=0/Npix=5.parquet")
    assert euclid_images.hf_repo_file("hf://datasets/org/repo@v1/a/b.parquet") == ("org/repo", "v1", "a/b.parquet")
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scr"))
    assert euclid_images.mmu_cache_dir({"mmu_cache": "auto"}) == tmp_path / "scr" / "mmu_euclid_q1"
    assert euclid_images.mmu_cache_dir({"mmu_cache": None}) is None

    cache = tmp_path / "mirror"
    files = []
    for k, ids in enumerate([range(0, 20), range(20, 40)]):
        p = cache / "euclid" / f"Npix={k}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        img = [[dict(band="VIS", flux=[[float(i)]], scale=0.1)] for i in ids]
        pq.write_table(pa.table(dict(object_id=pa.array([str(i) for i in ids]), image=img)), p)
        files.append(dict(remote=f"{url}/Npix={k}.parquet", local=str(p), size=p.stat().st_size,
                          patches=["A"] if k == 0 else ["A", "B"]))
    box_a, box_b = (1.0, 2.0, 3.0, 4.0), (2.0, 3.0, 3.0, 4.0)
    man = dict(hats_url=url, boxes={"A": list(box_a), "B": list(box_b)}, files=files)
    with pytest.raises(FileNotFoundError):
        euclid_images.mirrored_files(url, cache, box_a)
    (cache / euclid_images.MIRROR_MANIFEST).write_text(json.dumps(man))
    assert len(euclid_images.mirrored_files(url, cache, box_a)) == 2
    assert [p.name for p in euclid_images.mirrored_files(url, cache, box_b)] == ["Npix=1.parquet"]
    with pytest.raises(ValueError):
        euclid_images.mirrored_files(url, cache, (0.0, 1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        euclid_images.mirrored_files("hf://datasets/other/cat", cache, box_a)
    df = euclid_images.read_mmu_rows(url, [3, 25, 99], ["image"], box_a, cache=cache, batch_rows=4)
    assert sorted(df["object_id"]) == ["25", "3"]
    # A purged file is reported, not silently read as "no image".
    (cache / "euclid" / "Npix=1.parquet").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        euclid_images.mirrored_files(url, cache, box_b)


def test_pa_check_reports_the_angle_the_code_uses():
    from dp2_cutouts.common.ellipse import euclid_ellipse
    from dp2_cutouts.pairs import quality
    rng = np.random.default_rng(0)
    n = 400
    theta_true = rng.uniform(0, 180, n)                      # orientation on the VIS stamp
    e = 0.4
    m = pd.DataFrame(dict(e1_vis=e * np.cos(np.radians(2 * theta_true)),
                          e2_vis=e * np.sin(np.radians(2 * theta_true))))
    # "measured" tiles: the code's theta = 270 - PA must equal the orientation -> PA = 270 - theta.
    m["euclid_position_angle"] = (270.0 - theta_true) % 360.0
    _, _, th = euclid_ellipse(np.ones(n), np.ones(n), m["euclid_position_angle"], 0.7)
    assert np.allclose(quality.angle_diff(th, theta_true), 0.0, atol=1e-9)
    r = quality.pa_check(m)
    assert r["used_by_code_median_abs_diff_deg"] < 1.0
    # "copaings" tiles of the scale run: the catalogue angle IS the orientation, the code's 270 - PA is unrelated.
    m["euclid_position_angle"] = theta_true
    r = quality.pa_check(m)
    assert r["median_abs_diff_deg"]["PA"] < 1.0 and r["used_by_code_median_abs_diff_deg"] > 30.0


def test_pa_convention_must_be_explicit():
    from dp2_cutouts.sources import euclid_hf

    class Cfg(dict):
        pass

    with pytest.raises(ValueError):
        euclid_hf.pa_convention(Cfg(euclid={}))
    assert euclid_hf.pa_convention(Cfg(euclid=dict(pa_convention="measured"))) == "measured"


def test_rubin_to_euclid_px_follows_the_pixel_scales():
    from dp2_cutouts.pairs import dataset2
    assert dataset2.rubin_to_euclid_px({"euclid": {"pixel_scale": 0.1}}) == pytest.approx(2.0)
    assert dataset2.rubin_to_euclid_px({"euclid": {"pixel_scale": 0.05}}) == pytest.approx(4.0)


def test_d1_rejects_targets_in_a_shared_footprint():
    from dp2_cutouts.pairs.dataset1 import alone_in_footprint

    meta = pd.DataFrame(dict(shared_light_frac=[0.0, 0.1, 0.5, np.nan]))
    assert alone_in_footprint(meta, dict(max_shared_light_frac=0.2)).tolist() == [True, True, False, True]
    assert alone_in_footprint(meta, dict(max_shared_light_frac=None)).all()
    with pytest.raises(RuntimeError, match="rerun the pipeline"):
        alone_in_footprint(meta.drop(columns="shared_light_frac"), dict(max_shared_light_frac=0.2))


def test_blend_flux_ratio_is_companion_over_target():
    # flux_ratio is F_companion / F_target in the blend: the companion stamp is scaled down to it, never up.
    rng = np.random.default_rng(0)
    n = 40
    mag = rng.uniform(20.0, 26.0, n)
    meta = pd.DataFrame(dict(split="test", tract=1, patch=0, i_cModel_mag=mag))
    cb = dict(same_patch=True, separation_px=[2.0, 6.0], flux_ratio=[0.1, 1.0])
    plan = blends.draw(meta, "test", 200, cb, rng)
    f = 10.0 ** (-0.4 * mag)
    ft, fc = f[plan["d1_row"]], f[plan["companion_row"]]
    assert np.allclose(plan["companion_scale"] * fc / ft, plan["flux_ratio"])
    assert (plan["companion_scale"] <= 1.0 + 1e-12).all()
    assert (plan["d1_row"] != plan["companion_row"]).all()
    with pytest.raises(RuntimeError, match="only"):
        blends.draw(meta.iloc[:2], "test", 5, dict(cb, flux_ratio=[50.0, 60.0]), rng, max_tries_per_blend=10)
