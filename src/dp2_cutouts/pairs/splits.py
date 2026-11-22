# splits.py
# Anna Preto
# AstroDeep Project
# Train / validation / test / holdout assignment by Rubin patch (tract, patch), shared by every dataset of the project.
# train fits the networks, val chooses (early stopping, settings), test validates each stage before the next one
# (Rubin VAE, Stage 1, Stage 2), holdout is read by no training nor choice: the final evaluation only.
# A patch is a contiguous 0.2 deg area with its own seeing, depth and sky: splitting by galaxy would put neighbours,
# which share all three, on both sides. The draw is a hash of (seed, tract, patch), so it does not depend on the
# order or the number of patches processed so far: adding patches never moves an existing one to another split.

import hashlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SPLITS = ("train", "val", "test", "holdout")


def patch_uniform(seed, tract, patch):
    h = hashlib.sha256(f"{int(seed)}:{int(tract)}:{int(patch)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def assign(tracts, patches, fractions, seed):
    # A split missing from the fractions has none (holdout appended last: without it, the former three-split
    # assignment is unchanged).
    edges = np.cumsum([float(fractions.get(s, 0.0)) for s in SPLITS])
    if not np.isclose(edges[-1], 1.0):
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    u = np.array([patch_uniform(seed, t, p) for t, p in zip(tracts, patches)])
    return np.array(SPLITS, dtype=object)[np.searchsorted(edges, u, side="right").clip(0, len(SPLITS) - 1)]


def unit_vectors(ra, dec):
    r, d = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


def cross_split_neighbours(ra, dec, split, radius_arcsec):
    # Number of objects that have a neighbour of another split within radius (patch borders).
    xyz = unit_vectors(ra, dec)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(2 * np.sin(np.radians(radius_arcsec / 3600.0) / 2))
    split = np.asarray(split)
    bad = np.zeros(len(split), bool)
    for i, j in pairs:
        if split[i] != split[j]:
            bad[i] = bad[j] = True
    return bad


def split_table(meta, cfg):
    sp = cfg["splits"]
    tracts, patches = meta["tract"].to_numpy(), meta["patch"].to_numpy()
    s = assign(tracts, patches, sp["fractions"], cfg["seed"])
    # splits.fixed {"tract_patch": split} overrides the draw: with a handful of patches the hash can leave a split
    # empty (9 patches gave no test patch), and the patches that matter are then chosen by hand.
    fixed = dict(sp.get("fixed") or {})
    bad = {k: v for k, v in fixed.items() if v not in SPLITS}
    if bad:
        raise ValueError(f"splits.fixed: unknown split in {bad}; expected one of {SPLITS}")
    if fixed:
        names = np.array([f"{int(t)}_{int(p)}" for t, p in zip(tracts, patches)], dtype=object)
        for name, split in fixed.items():
            s[names == name] = split
    return pd.Series(s, index=meta.index, name="split")
