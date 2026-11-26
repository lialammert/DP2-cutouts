# test_hf_preview.py
# Anna Preto
# AstroDeep Project
# The preview export for the private Hugging Face viewer (scripts/hf): fake D1 / D2 shards before any merge,
# holdout left out, D2 selection flag, images decoded back to the stored values, figures drawn.

import importlib.util
import json
import sys
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

    sys.path.insert(0, str(ROOT / "scripts" / "hf" / "space"))
    try:
        core = _module(ROOT / "scripts" / "hf" / "space" / "viewer_core.py", "viewer_core")
    finally:
        sys.path.pop(0)
    row = d2.iloc[0].to_dict()
    im = core.decode(row, i2)
    sig = np.asarray(json.loads(row["rubin_sigma"]))
    assert np.allclose(sig, 3.0, rtol=1e-3) and im["rubin"].shape == (NR, NR, 6)
    assert np.allclose(im["vis"] * row["vis_sigma"], im["vis_masked"] * row["vis_sigma"] + 1, atol=0.02)
    assert im["fill"][15, 15] and not im["fill"][60, 60]
    for kind, t, info in (("d1", d1, i1), ("d2", d2, i2)):
        rows = core.select(t, kind, ["train"], ["EDF-F", "EDF-S"], (18, 30), "selected")
        assert len(rows) and (t.iloc[rows]["split"] == "train").all()
        fig = core.figure(t.iloc[rows[0]].to_dict(), info, kind)
        assert len(fig.axes) == 4
        assert "rubin" not in core.describe(t.iloc[rows[0]].to_dict(), kind)
