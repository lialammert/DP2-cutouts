# test_pipeline.py
# Anna Preto
# AstroDeep Project
# Patch-by-patch bookkeeping, without data: the sky box of a patch case, object ownership, and the manifest rules
# (a phase done with the same configuration is skipped, a change of configuration reruns it, a rerun phase
# invalidates the later ones, a failure is recorded and does not stop the other patches).

import numpy as np
import pandas as pd
import pytest

from dp2_cutouts.config import Config
from dp2_cutouts.pipeline import manifest
from dp2_cutouts.pipeline.patches import owned, patch_case

PATCH = dict(tract=5063, patch=34, name="5063_34", ra_min=53.0781, ra_max=53.2855, dec_min=-27.8623,
             dec_max=-27.6788, s_ra=53.18, s_dec=-27.77, s_xel1=3300, s_xel2=3300, bands="ugrizy", complete=True)


@pytest.fixture
def cfg(tmp_path):
    raw = dict(name="t", data_root=str(tmp_path), sky_box=[0, 1, 0, 1],
               sources=dict(rubin_catalog="tap", euclid_catalog="hf", rubin_pixels="soda"),
               patches=dict(catalog_margin_arcsec=10.0), rubin=dict(tracts=[1]), euclid={}, copaings={},
               cuts=dict(snr_min=5.0, k_ellipse=3.0), masking=dict(method="footprint"))
    return Config(raw=raw, path=tmp_path / "c.yaml")


def test_patch_case_box_and_folder(cfg):
    case = patch_case(cfg, PATCH)
    m = 10.0 / 3600
    ra0, ra1, d0, d1 = case.sky_box
    assert np.isclose(d0, PATCH["dec_min"] - m) and np.isclose(d1, PATCH["dec_max"] + m)
    assert ra0 < PATCH["ra_min"] - m and ra1 > PATCH["ra_max"] + m  # RA margin widened by 1 / cos(dec)
    assert case.name == "5063_34" and case.data_root.name == "5063_34"
    assert case["rubin"]["tracts"] == [5063] and case["rubin"]["tap_tract_filter"] is False
    assert cfg["rubin"]["tracts"] == [1]  # the parent configuration is untouched


def test_owned():
    t = pd.DataFrame(dict(tract=[5063, 5063, 5064], patch=[34, 35, 34]))
    assert owned(t, PATCH).tolist() == [True, False, False]


def test_manifest_skip_rerun_and_invalidate(cfg):
    calls = []

    def ok(c, p):
        calls.append(1)
        return "done", dict(n=1)

    manifest.run_phase(cfg, PATCH, "plan", ok)
    manifest.run_phase(cfg, PATCH, "fetch", ok)
    manifest.run_phase(cfg, PATCH, "plan", ok)            # same configuration: skipped
    assert len(calls) == 2
    assert manifest.load(cfg, PATCH)["phases"]["fetch"]["status"] == "done"

    cfg.raw["copaings"]["linking_length"] = 2.0            # matters to plan: rerun, fetch invalidated
    manifest.run_phase(cfg, PATCH, "plan", ok)
    assert len(calls) == 3
    assert "fetch" not in manifest.load(cfg, PATCH)["phases"]

    cfg.raw["masking"]["grow_px"] = 3                       # matters to process only: plan still done
    assert manifest.is_done(cfg, manifest.load(cfg, PATCH), "plan")


def test_manifest_records_failure(cfg):
    def boom(c, p):
        raise ValueError("network down")

    manifest.run_phase(cfg, PATCH, "plan", lambda c, p: ("done", {}))
    entry = manifest.run_phase(cfg, PATCH, "fetch", boom)
    assert entry["status"] == "failed" and "network down" in entry["error"]
    assert not manifest.is_done(cfg, manifest.load(cfg, PATCH), "fetch")
    with pytest.raises(RuntimeError):                       # process refuses to run before fetch is done
        manifest.run_phase(cfg, PATCH, "process", lambda c, p: ("done", {}))
