# test_regression.py
# Anna Preto
# AstroDeep Project
# Checks that the products of a case, when present on this machine, give the reference fingerprints: same final
# identifiers and same decisions (decisions_sha256), same stamp arrays (pixels_sha256). The fingerprints reveal no
# DP2 data. Skipped when the products have not been computed here (run "dp2-cutouts <config> all" first).

import json
from pathlib import Path

import pytest

from dp2_cutouts.config import Config
from dp2_cutouts.regression import fingerprints, load_current

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((Path(__file__).parent / "reference_fingerprints.json").read_text())


@pytest.mark.parametrize("case", sorted(REFERENCE))
def test_fingerprints(case):
    ref = REFERENCE[case]
    cfg = Config.load(ROOT / ref["config"])
    if not cfg.data("40_final", "dataset_info.json", makedirs=False).exists():
        pytest.skip(f"products of {case} not computed on this machine")
    fp = fingerprints(*load_current(cfg), columns=ref["decision_columns"])
    assert (fp["n_catalog"], fp["n_final"]) == (ref["n_catalog"], ref["n_final"])
    assert fp["decisions_sha256"] == ref["decisions_sha256"], "different decisions: run 'regress' for details"
    assert fp["pixels_sha256"] == ref["pixels_sha256"], "same decisions, different stamp pixels"
