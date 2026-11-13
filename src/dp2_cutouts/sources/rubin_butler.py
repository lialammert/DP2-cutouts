# rubin_butler.py
# AstroDeep Project
# Reads the Rubin DP2 Object catalogue through the Butler (at NERSC: repo "dp2"), within a sky box. Replaces
# rubin_tap where the Butler is available: no token, no TAP job, same output file and same columns, so the
# following steps (copaings, catalog_table) do not know which source was used.
#
# Configuration:
#   butler: {repo: dp2, collections: dp2, skymap: lsst_cells_v2}
#   rubin:  {source: butler, tracts: [5063]}
#
# The lsst imports are inside the functions: this module must stay importable without the LSST stack.

import pandas as pd

from .rubin_tap import COPAINGS_COLUMNS, EXTRA_COLUMNS
from ..common import star_galaxy
from ..config import RUBIN_DIR


def get_butler(cfg):
    from lsst.daf.butler import Butler
    b = cfg["butler"]
    return Butler(b["repo"], collections=b["collections"])


def available_columns(butler, skymap, tract):
    return list(butler.get("object.columns", skymap=skymap, tract=int(tract)))


def fetch_objects(cfg):
    # Every row of the object table inside the sky box, for the tracts of the configuration, without any cut:
    # the pre-cut catalogue is used for reweighting. Rows are read tract by tract (the table is per tract).
    butler = get_butler(cfg)
    skymap = cfg["butler"]["skymap"]
    tracts = [int(t) for t in cfg["rubin"]["tracts"]]
    columns = list(dict.fromkeys(COPAINGS_COLUMNS + EXTRA_COLUMNS + star_galaxy.required_columns(cfg)))
    ra0, ra1, dec0, dec1 = cfg.sky_box

    parts = []
    for tract in tracts:
        have = set(available_columns(butler, skymap, tract))
        missing = [c for c in columns if c not in have]
        if missing:
            raise KeyError(f"tract {tract}: columns missing from the butler object table: {missing}")
        t = butler.get("object", skymap=skymap, tract=tract, parameters={"columns": columns})
        df = t.to_pandas()
        if df.index.name in columns:      # the table may come back indexed by objectId
            df = df.reset_index()
        n_all = len(df)
        keep = ((df["coord_ra"] >= ra0) & (df["coord_ra"] <= ra1)
                & (df["coord_dec"] >= dec0) & (df["coord_dec"] <= dec1))
        df = df[keep].reset_index(drop=True)
        print(f"  tract {tract}: {n_all:,} objects, {len(df):,} in the box")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    # Masked astropy columns come back as pandas nullable or object columns; the following steps expect plain
    # numpy floats with NaN, as TAP returns them.
    for c in df.columns:
        if c in ("objectId", "parentObjectId", "tract", "patch"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("int64")
        elif df[c].dtype == object or str(df[c].dtype) in ("boolean", "Float64", "Float32", "Int64"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Objects in the overlap of two tracts appear in both tables: the same check as on the TAP response.
    if df["objectId"].duplicated().any():
        n = int(df["objectId"].duplicated().sum())
        raise RuntimeError(f"{n} duplicate objectId in the butler object table (tract overlap?)")
    return df


def run(cfg):
    df = fetch_objects(cfg)
    out = cfg.data(RUBIN_DIR, "catalog", "object_box.parquet")
    df.to_parquet(out, index=False)
    print(f"Rubin (butler): {len(df):,} objects, {df.shape[1]} columns -> {out}")
    return df
