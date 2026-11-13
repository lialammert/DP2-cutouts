# star_galaxy.py
# Star-galaxy separation on the Rubin object table, chosen in the configuration:
#
#   rubin:
#     star_galaxy:
#       column: griz_model_extendedness   # any numeric column of the Rubin object table, or "none"
#       threshold: 0.5                    # an object is kept as a galaxy if column > threshold
#       keep_nan: false                   # objects where the column is NaN: dropped (false) or kept (true)
#
# Without a `star_galaxy` block the previous behaviour is used: i_extendedness > 0.5, NaN dropped.
# `column: none` applies no Rubin-side star-galaxy cut (the Euclid cut 1 still removes point-like counterparts).
#
# copaings keeps only the Rubin objects whose `i_extendedness` is 1 (copaings.catalogs.build_lsst). `apply` makes
# that filter follow the configured criterion: it returns a copy of the table in which `i_extendedness` is 1 for
# the objects that pass and 0 for the others, and the original column is kept as `i_extendedness_rubin`.
# The table written to disk by the catalogue step is not modified.

import numpy as np

DEFAULT = {"column": "i_extendedness", "threshold": 0.5, "keep_nan": False}


def settings(cfg):
    sg = dict(DEFAULT)
    sg.update(cfg["rubin"].get("star_galaxy") or {})
    unknown = sorted(set(sg) - set(DEFAULT))
    if unknown:
        raise ValueError(f"rubin.star_galaxy: unknown option(s) {unknown}; allowed: {sorted(DEFAULT)}")
    if str(sg["column"]).lower() == "none":
        sg["column"] = None
    return sg


def required_columns(cfg):
    """Columns of the Rubin object table that the criterion needs, to be fetched with the catalogue."""
    col = settings(cfg)["column"]
    return [col] if col else []


def describe(cfg):
    sg = settings(cfg)
    if sg["column"] is None:
        return "no Rubin star-galaxy cut"
    nan = ", NaN kept" if sg["keep_nan"] else ""
    return f"{sg['column']} > {sg['threshold']:g}{nan}"


def values(raw, cfg):
    """The configured column as floats (NaN everywhere when no column is configured)."""
    col = settings(cfg)["column"]
    if col is None:
        return np.full(len(raw), np.nan)
    if col not in raw.columns:
        raise KeyError(f"rubin.star_galaxy.column '{col}' is not in the Rubin catalogue; "
                       "re-run the rubin_catalog step so that it is fetched")
    return raw[col].to_numpy(float)


def passes(raw, cfg):
    """Boolean array: True for the objects kept as galaxies."""
    sg = settings(cfg)
    v = values(raw, cfg)
    if sg["column"] is None:
        return np.ones(len(raw), bool)
    keep = v > float(sg["threshold"])
    return keep | np.isnan(v) if sg["keep_nan"] else keep


def apply(raw, cfg):
    """Copy of `raw` in which `i_extendedness` is 1 where the criterion passes and 0 elsewhere (see the header)."""
    out = raw.copy()
    out["i_extendedness_rubin"] = raw["i_extendedness"]
    out["i_extendedness"] = np.where(passes(raw, cfg), 1.0, 0.0)
    return out
