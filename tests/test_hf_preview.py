# test_hf_preview.py
# Anna Preto
# AstroDeep Project
# The preview export for the private Hugging Face viewer (scripts/hf): fake D1 / D2 shards before any merge,
# holdout left out, D2 selection flag, static site records decoded back to the exported images.

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NR, NE = 59, 119


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stores(tmp_path):
    from dp2_cutouts.pairs import store
    rng = np.random.default_rng(0)
    for name in ("d1", "d2"):
        for k in range(2):
            n = 8
            rf = rng.normal(0, 3, (n, NR, NR, 6)).astype(np.float32) + 50
            ef = rng.normal(0, 0.05, (n, NE, NE, 4)).astype(np.float32)
            fill = np.zeros((n, NE, NE), bool)
            fill[:, 10:20, 10:20] = True
            meta = dict(split=["train", "val", "test", "holdout"] * 2, euclid_vis_mag=np.linspace(22, 27, n),
                        euclid_object_id=np.arange(n) + 100 * k, euclid_dec=-48.0 if k else -28.0)
            arrs = dict(euclid_flux=ef, euclid_ivar=np.full_like(ef, 400.0), euclid_mask=np.zeros(ef.shape, bool),
                        euclid_fill=fill)
            if name == "d2":
                arrs.update(euclid_flux_unmasked=ef + 1, rubin_flux_rc=rf, rubin_ivar_rc=np.full_like(rf, 1 / 9.0),
                            rubin_mask_rc=np.zeros(rf.shape, bool))
                meta.update(group_id=[f"p{k}:{i}" for i in range(n)], sep_delta_e=[0.0, 0.5] * 4,
                            sep_centroid_shift_arcsec=0.0, sep_rel_size_change=0.0, sep_rel_flux_change=0.0,
                            rc_offset_x_px=-2, rc_offset_y_px=1)
            else:
                arrs.update(rubin_flux=rf, rubin_ivar=np.full_like(rf, 1 / 9.0), rubin_mask=np.zeros(rf.shape, bool))
            store.write_shard(tmp_path / "datasets" / name / "shards" / f"p{k}", pd.DataFrame(meta), arrs,
                              pd.DataFrame(), info={})
    from dp2_cutouts.mlconfig import MLConfig
    raw = dict(ml_root=str(tmp_path), rubin=dict(bands=list("ugrizy"), stamp_px=NR),
               euclid=dict(bands=["VIS", "Y", "J", "H"], stamp_px=NE, pixel_scale=0.1),
               deblend=dict(d2_selection=dict(before_after=dict(max_delta_e=0.02), vis_mag_max=None)))
    return MLConfig(raw=raw, path=tmp_path / "cfg.yaml")


def test_preview_export_and_viewer_figures(stores, tmp_path):
    ex = _module(ROOT / "scripts" / "hf" / "export_preview.py", "export_preview")
    rng = np.random.default_rng(1)
    d1, i1 = ex.export(stores, "d1", 100, rng)
    d1_meta = pd.concat([ex.Store(p).meta for p in ex.sources(stores, "d1")])
    d2, i2 = ex.export(stores, "d2", 100, rng, d1_meta)
    assert len(d1) == len(d2) == 12 and not (d1["split"] == "holdout").any()
    assert d2["selected"].tolist() == (d2["sep_delta_e"] < 0.02).tolist()
    assert set(d2["field"]) == {"EDF-F", "EDF-S"} and (d2["target_rubin_x"] == 2.0).all()

    # Static site: records decoded the way index.html does (fixed-size records, float16, big-endian packbits).
    site_mod = _module(ROOT / "scripts" / "hf" / "static_site.py", "static_site")
    exp = tmp_path / "export"
    exp.mkdir()
    d1.to_parquet(exp / "d1.parquet", index=False)
    d2.to_parquet(exp / "d2.parquet", index=False)
    (exp / "info.json").write_text(json.dumps(dict(d1=i1, d2=i2), default=str))
    site = site_mod.build_site(exp, tmp_path / "site")
    assert (site / "index.html").exists() and "sdk: static" in (site / "README.md").read_text(encoding="utf-8")
    for name, t, info in (("d1", d1, i1), ("d2", d2, i2)):
        js = json.loads((site / "data" / f"{name}.json").read_text())
        assert len(js["rows"]) == len(t) and js["record_bytes"] % 8 == 0
        assert all(isinstance(r["euclid_object_id"], str) for r in js["rows"])
        assert all("rubin" not in r and "fill" not in r for r in js["rows"])
        k = len(t) - 1
        buf = (site / "data" / f"{name}_{k // js['chunk']:03d}.bin").read_bytes()
        base = (k % js["chunk"]) * js["record_bytes"]
        pos, n = js["layout"]["rubin"]
        rub = np.frombuffer(buf[base + pos:base + pos + n], np.float16).reshape(NR, NR, 6)
        assert np.array_equal(rub, np.frombuffer(t.iloc[k]["rubin"], np.float16).reshape(NR, NR, 6))
        pos, n = js["layout"]["fill"]
        fill = np.unpackbits(np.frombuffer(buf[base + pos:base + pos + n], np.uint8), count=NE * NE).reshape(NE, NE)
        assert fill[15, 15] and not fill[60, 60]
        assert ("vis_masked" in js["layout"]) == (name == "d2")
        assert js["rows"][k]["split"] == t.iloc[k]["split"] and js["rows"][k]["field"] == t.iloc[k]["field"]
    assert (site / "data" / "d2.json").read_text().count('"selected":true') == int(d2["selected"].sum())
