import numpy as np
import pandas as pd
import pytest

from dp2_cutouts.common import star_galaxy
from dp2_cutouts.config import Config


def cfg(**sg):
    return Config(raw={"rubin": {"star_galaxy": sg} if sg else {}}, path=None)


RAW = pd.DataFrame({"i_extendedness": [1.0, 0.0, np.nan, 1.0],
                    "griz_model_extendedness": [0.9, 0.2, np.nan, 0.4]})


def test_default_is_i_extendedness():
    assert star_galaxy.required_columns(cfg()) == ["i_extendedness"]
    assert list(star_galaxy.passes(RAW, cfg())) == [True, False, False, True]


def test_configured_column_and_threshold():
    c = cfg(column="griz_model_extendedness", threshold=0.5)
    assert list(star_galaxy.passes(RAW, c)) == [True, False, False, False]
    out = star_galaxy.apply(RAW, c)
    assert list(out["i_extendedness"]) == [1.0, 0.0, 0.0, 0.0]
    assert list(out["i_extendedness_rubin"].fillna(-1)) == [1.0, 0.0, -1.0, 1.0]
    assert list(RAW["i_extendedness"].fillna(-1)) == [1.0, 0.0, -1.0, 1.0]       # input not modified


def test_keep_nan_and_none():
    assert list(star_galaxy.passes(RAW, cfg(column="griz_model_extendedness", keep_nan=True))) == [True, False, True, False]
    assert star_galaxy.required_columns(cfg(column="none")) == []
    assert star_galaxy.passes(RAW, cfg(column="none")).all()
    assert star_galaxy.describe(cfg(column="none")) == "no Rubin star-galaxy cut"


def test_errors():
    with pytest.raises(ValueError, match="unknown"):
        star_galaxy.settings(cfg(colum="x"))
    with pytest.raises(KeyError, match="not in the Rubin catalogue"):
        star_galaxy.values(RAW, cfg(column="missing_column"))
