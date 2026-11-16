# splits.py
# Anna Preto
# AstroDeep Project
# Train / validation / test assignment by Rubin patch (tract, patch), shared by every dataset of the project.
# A patch is a contiguous 0.2 deg area with its own seeing, depth and sky: splitting by galaxy would put neighbours,
# which share all three, on both sides. The draw is a hash of (seed, tract, patch), so it does not depend on the
# order or the number of patches processed so far: adding patches never moves an existing one to another split.

import hashlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SPLITS = ("train", "val", "test")


def patch_uniform(seed, tract, patch):
    h = hashlib.sha256(f"{int(seed)}:{int(tract)}:{int(patch)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def assign(tracts, patches, fractions, seed):
    edges = np.cumsum([fractions[s] for s in SPLITS])
    if not np.isclose(edges[-1], 1.0):
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    u = np.array([patch_uniform(seed, t, p) for t, p in zip(tracts, patches)])
    return np.array(SPLITS, dtype=object)[np.searchsorted(edges, u, side="right").clip(0, 2)]


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
    s = assign(meta["tract"].to_numpy(), meta["patch"].to_numpy(), sp["fractions"], cfg["seed"])
    return pd.Series(s, index=meta.index, name="split")
