# euclid_flags.py
# Anna Preto
# AstroDeep Project
# Cut 1 of D1 (the Euclid object is neither point-like nor spurious) for the D2 targets. copaings keeps only extended
# Rubin objects (i_extendedness == 1) but no Euclid star cut: two stars, or a star on a galaxy, make an extended Rubin
# blend that Euclid splits into D2 targets, while D1 drops point-like counterparts (stages/catalog_table.py). The
# MER flags of every D2 target are read once from the patches' Euclid catalogues (00_euclid/euclid_box.parquet,
# already on disk: no network) and written next to D2:
#   d2/euclid_flags.parquet   euclid_object_id, euclid_point_like_flag, euclid_spurious_flag, euclid_point_like_prob,
#                             euclid_spurious_prob, euclid_cut1_pass
# D2 itself is not changed: deblend.d2_selection.euclid_cut1 applies the cut when D2 is read (deblend/data.py).

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FILE = "euclid_flags.parquet"
COLUMNS = {"mer_catalogue_point_like_flag": "euclid_point_like_flag",
           "mer_catalogue_spurious_flag": "euclid_spurious_flag",
           "mer_catalogue_point_like_prob": "euclid_point_like_prob",
           "mer_catalogue_spurious_prob": "euclid_spurious_prob"}


def cut1_pass(point_like_flag, spurious_flag):
    # Same rule as catalog_table cut 1: a missing flag counts as not set.
    point = pd.to_numeric(pd.Series(point_like_flag), errors="coerce").fillna(0).astype(int).to_numpy() == 1
    spurious = pd.to_numeric(pd.Series(spurious_flag), errors="coerce").fillna(0).astype(int).to_numpy() == 1
    return ~(point | spurious)


def patch_flags(case):
    t = pq.read_table(case.data("00_euclid", "euclid_box.parquet"), columns=["object_id", *COLUMNS]).to_pandas()
    t = t.dropna(subset=["object_id"]).drop_duplicates("object_id")
    t["object_id"] = t["object_id"].astype("int64")
    return t.rename(columns=COLUMNS).rename(columns={"object_id": "euclid_object_id"})


def build(cfg):
    # One pass over the patches holding D2 targets; refuses to write unless every target found its flags.
    from ..pipeline.patches import load_patches, patch_case
    root = cfg.dataset_dir("d2")
    meta = pd.read_parquet(root / "meta.parquet", columns=["euclid_object_id", "shard"])
    pcfg = cfg.pipeline()
    patches = {p["name"]: p for p in load_patches(pcfg)}
    parts = []
    for name, rows in meta.groupby("shard", sort=False):
        f = patch_flags(patch_case(pcfg, patches[name]))
        want = rows["euclid_object_id"].astype("int64")
        parts.append(f[f["euclid_object_id"].isin(set(want))])
    flags = pd.concat(parts, ignore_index=True).drop_duplicates("euclid_object_id")
    out = meta[["euclid_object_id"]].astype("int64").merge(flags, on="euclid_object_id", how="left")
    missing = ~out["euclid_object_id"].isin(flags["euclid_object_id"])
    if missing.any():
        raise RuntimeError(f"{int(missing.sum())} D2 targets not found in their patch's Euclid catalogue")
    out["euclid_cut1_pass"] = cut1_pass(out["euclid_point_like_flag"], out["euclid_spurious_flag"])
    for c in ("euclid_point_like_flag", "euclid_spurious_flag"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    tmp = root / (FILE + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(root / FILE)
    return out


def cut1_for(cfg, euclid_object_id):
    # euclid_cut1_pass of the given D2 targets (any subset, any order), from d2/euclid_flags.parquet.
    path = cfg.dataset_dir("d2") / FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `dp2-ml <config> d2-flags` (deblend.d2_selection.euclid_cut1)")
    f = pd.read_parquet(path, columns=["euclid_object_id", "euclid_cut1_pass"]).set_index("euclid_object_id")
    ids = pd.Series(euclid_object_id).astype("int64")
    got = f["euclid_cut1_pass"].reindex(ids.to_numpy())
    if got.isna().any():
        raise RuntimeError(f"{int(got.isna().sum())} D2 targets missing from {path}: run d2-flags again")
    return got.to_numpy(bool)
