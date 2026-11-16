# regression.py
# Anna Preto
# AstroDeep Project
# Non-regression reference of a case. "snapshot" copies the current processed outputs under
# data_root/regression/; "regress" compares the current outputs with that copy, column by column. Only two
# fingerprints (hashes of the decisions and of the pixels) go into the repository: they reveal no DP2 data and let
# anyone check that they reproduce exactly the same dataset.

import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# Columns that carry a decision: identical on every machine, otherwise the dataset differs.
DECISION_SUFFIXES = ("_pass",)
DECISION_EXTRA = ("in_region", "pixel_target", "has_euclid_counterpart", "cut8_evaluated", "shape_ok")
ARRAY_COLUMNS = ("flux_raw", "flux", "ivar", "maskbits", "companion_mask", "diag_flux", "diag_need")
FLOAT_RTOL, FLOAT_ATOL = 1e-9, 1e-12


def reference_dir(cfg):
    return cfg.data("regression", "x").parent


def load_current(cfg):
    table = pd.read_parquet(cfg.data("30_processed", "table.parquet")).set_index("object_id").sort_index()
    stamps = pd.read_parquet(cfg.data("30_processed", "stamps.parquet")).set_index("object_id").sort_index()
    final = ds.dataset(cfg.data("40_final", "data", "x").parent, format="parquet", partitioning="hive")
    final_ids = np.sort(final.to_table(columns=["object_id"]).column("object_id").to_numpy())
    return table, stamps, final_ids


def decision_columns(table):
    return sorted(c for c in table.columns if c.endswith(DECISION_SUFFIXES) or c in DECISION_EXTRA)


def array_hashes(stamps):
    # One SHA-256 per object and per array column (missing array = empty string).
    out = pd.DataFrame(index=stamps.index)
    for c in ARRAY_COLUMNS:
        if c in stamps.columns:
            out[c] = [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() if isinstance(v, np.ndarray)
                      else "" for v in stamps[c]]
    return out


def fingerprints(table, stamps, final_ids, columns=None):
    # decisions: final identifiers + every decision column, objects sorted by identifier.
    # pixels: hashes of every stamp array. A different machine may differ on pixels at an ellipse boundary
    # (floating point) without the decisions changing; both are reported separately.
    columns = decision_columns(table) if columns is None else columns
    h = hashlib.sha256()
    h.update(np.asarray(final_ids, np.int64).tobytes())
    for c in columns:
        h.update(c.encode())
        h.update(table[c].astype("boolean").fillna(False).to_numpy(dtype=bool).tobytes())
    hp = hashlib.sha256()
    ah = array_hashes(stamps)
    for c in ah.columns:
        hp.update(c.encode())
        hp.update("".join(ah[c]).encode())
    return dict(n_catalog=int(len(table)), n_final=int(len(final_ids)), decision_columns=columns,
                decisions_sha256=h.hexdigest(), pixels_sha256=hp.hexdigest())


def snapshot(cfg):
    table, stamps, final_ids = load_current(cfg)
    out = reference_dir(cfg)
    table.reset_index().to_parquet(out / "table.parquet", index=False)
    array_hashes(stamps).reset_index().to_parquet(out / "stamp_hashes.parquet", index=False)
    np.save(out / "final_ids.npy", final_ids)
    fp = fingerprints(table, stamps, final_ids)
    with open(out / "fingerprints.json", "w") as f:
        json.dump(fp, f, indent=2)
    print(f"reference written to {out}: {fp['n_final']} final stamps out of {fp['n_catalog']} objects")
    print(json.dumps({k: fp[k] for k in ("decisions_sha256", "pixels_sha256")}, indent=2))
    return fp


def _compare_column(a, b):
    # Number of objects whose value differs, with floating-point tolerance and NaN == NaN.
    if a.dtype == object or b.dtype == object:
        # Element by element: pandas treats None != None as True.
        def key(v):
            return None if v is None or (np.isscalar(v) and pd.isna(v)) else np.asarray(v).tobytes()
        return sum(key(x) != key(y) for x, y in zip(a, b))
    if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
        x, y = a.to_numpy(float, na_value=np.nan), b.to_numpy(float, na_value=np.nan)
        same = np.isclose(x, y, rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True) | (np.isinf(x) & (x == y))
        return int((~same).sum())
    x, y = a.astype(object), b.astype(object)
    both_na = x.isna() & y.isna()
    return int((~both_na & (x != y)).sum())


def regress(cfg):
    ref = reference_dir(cfg)
    rt = pd.read_parquet(ref / "table.parquet").set_index("object_id").sort_index()
    rh = pd.read_parquet(ref / "stamp_hashes.parquet").set_index("object_id").sort_index()
    rf = np.load(ref / "final_ids.npy")
    rfp = json.load(open(ref / "fingerprints.json"))
    table, stamps, final_ids = load_current(cfg)
    problems = []

    if not table.index.equals(rt.index):
        problems.append(f"catalogue objects: {len(rt)} in reference, {len(table)} now, "
                        f"{len(rt.index.symmetric_difference(table.index))} differ")
    if not np.array_equal(rf, final_ids):
        problems.append(f"final dataset: {len(rf)} in reference, {len(final_ids)} now, "
                        f"{len(np.setxor1d(rf, final_ids))} identifiers differ")

    missing = [c for c in rt.columns if c not in table.columns]
    added = [c for c in table.columns if c not in rt.columns]
    if missing:
        problems.append(f"columns removed: {missing}")
    common = rt.index.intersection(table.index)
    for c in rt.columns:
        if c in table.columns:
            n = _compare_column(rt.loc[common, c], table.loc[common, c])
            if n:
                problems.append(f"column {c}: {n} objects differ")

    ch = array_hashes(stamps)
    if not ch.index.equals(rh.index):
        problems.append(f"stamps: {len(rh)} in reference, {len(ch)} now")
    else:
        for c in rh.columns:
            n = int((rh[c] != ch.get(c, "")).sum())
            if n:
                problems.append(f"stamp array {c}: {n} objects differ")

    fp = fingerprints(table, stamps, final_ids, columns=rfp["decision_columns"])
    for k in ("decisions_sha256", "pixels_sha256"):
        if fp[k] != rfp[k]:
            problems.append(f"{k} differs")

    if added:
        print(f"new columns (not compared): {added}")
    if problems:
        print("REGRESSION:")
        for p in problems:
            print("  -", p)
    else:
        print(f"identical to the reference: {len(final_ids)} final stamps, {len(table)} objects, "
              f"{len(rt.columns)} columns, {len(rh.columns)} stamp arrays")
    return not problems
