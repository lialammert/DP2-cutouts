# invariance.py
# Anna Preto
# AstroDeep Project
# Invariance of the catalogue decisions to the patch partition. Reference: copaings run once on the whole area of
# the planned patches. Partition: copaings run patch by patch with the catalogue margin (the plan phase). For every
# object owned by a planned patch, compares: the composition of its group after the ellipse overlap test (Rubin
# identifiers and Euclid positions of every member), the Euclid counterpart, S_b, cuts 1, 2 and 6.
# Catalogues only: no pixel is needed.

import pickle

import numpy as np
import pandas as pd

from .. import adapters
from ..stages import catalog_table, run_copaings
from . import manifest
from .patches import load_patches, owned, patch_case

COMPARED = ("has_euclid_counterpart", "euclid_object_id", "cut1_pass", "cut2_pass", "cut6_pass",
            "has_valid_i_moments")


def reference_case(cfg, patches):
    m = float(cfg["patches"]["catalog_margin_arcsec"]) / 3600.0
    ra0, ra1 = min(p["ra_min"] for p in patches), max(p["ra_max"] for p in patches)
    d0, d1 = min(p["dec_min"] for p in patches), max(p["dec_max"] for p in patches)
    mra = m / np.cos(np.radians(0.5 * (d0 + d1)))
    tracts = sorted({int(p["tract"]) for p in patches})
    return cfg.derive("invariance_reference", cfg.data_root / "invariance", [ra0 - mra, ra1 + mra, d0 - m, d1 + m],
                      rubin={"tracts": tracts, "tap_tract_filter": False})


def _same(x, y):
    # Equality with missing values (None, NaN, pd.NA) equal to each other only.
    if pd.isna(x) or pd.isna(y):
        return bool(pd.isna(x) and pd.isna(y))
    return bool(x == y)


def group_keys(case):
    # objectId -> identity of its group after the overlap test (sorted Rubin ids, sorted Euclid positions).
    cdir = catalog_table.copaings_dir(case)
    lsst = pd.read_parquet(cdir / "lsst_sb.parquet")
    eu = pd.read_parquet(cdir / "catalogs" / "euclid.parquet")
    groups = pickle.load(open(cdir / "groups" / "E_no_overlap.pkl", "rb"))["data"]
    keys = {}
    for g in groups:
        rid = tuple(sorted(int(v) for v in lsst.loc[list(g[1]), "objectId"]))
        epos = tuple(sorted(zip(eu.loc[list(g[0]), "ra"].round(9), eu.loc[list(g[0]), "dec"].round(9))))
        for r in rid:
            keys[r] = (rid, epos)
    return keys


def run(cfg, refetch=False):
    patches = [p for p in load_patches(cfg) if manifest.is_done(cfg, manifest.load(cfg, p), "plan")]
    if not patches:
        raise RuntimeError("no planned patch: run the plan phase first")
    ref = reference_case(cfg, patches)
    if refetch or not ref.data("30_processed", "catalog.parquet", makedirs=False).exists():
        adapters.fetch_rubin_catalog(ref)
        adapters.fetch_euclid_catalog(ref)
        run_copaings.run(ref)
        catalog_table.run(ref)
    rt = pd.read_parquet(ref.data("30_processed", "catalog.parquet")).set_index("object_id")
    rkeys = group_keys(ref)

    rows = []
    for p in patches:
        case = patch_case(cfg, p)
        t = pd.read_parquet(case.data("30_processed", "catalog.parquet"))
        t = t[owned(t, p)].set_index("object_id")
        keys = group_keys(case)
        missing = t.index.difference(rt.index)
        extra = rt.index[owned(rt.reset_index(), p).to_numpy()].difference(t.index)
        common = t.index.intersection(rt.index)
        row = dict(patch=p["name"], n_owned=len(t), missing_in_reference=len(missing), missing_in_patch=len(extra))
        for c in COMPARED:
            a, b = t.loc[common, c].astype(object), rt.loc[common, c].astype(object)
            row[c] = int(sum(not _same(x, y) for x, y in zip(a, b)))
        sa, sb = t.loc[common, "sb_euclid"].to_numpy(float), rt.loc[common, "sb_euclid"].to_numpy(float)
        row["sb_euclid"] = int((~np.isclose(sa, sb, rtol=1e-6, atol=1e-9, equal_nan=True)).sum())
        row["group"] = int(sum(keys.get(i) != rkeys.get(i) for i in common))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(cfg.data("invariance", "report.csv"), index=False)
    print(out.to_string(index=False))
    decisive = ["missing_in_reference", "missing_in_patch", "group"] + list(COMPARED)
    n_bad = int(out[decisive].to_numpy().sum())
    print(f"\n{'INVARIANT' if n_bad == 0 else 'NOT INVARIANT'}: {n_bad} disagreements on groups and decisions "
          f"over {int(out['n_owned'].sum()):,} owned objects; S_b values differ for {int(out['sb_euclid'].sum())} "
          "objects (see report)")
    return n_bad == 0
