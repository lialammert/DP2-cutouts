# split_leakage.py
# Anna Preto
# AstroDeep Project
# Leaks between the train / val / test / holdout splits of D1 and D2 (2026-11-28), read-only, metadata only. The
# splits are drawn per Rubin patch (pairs/splits.py), shared by D1 and D2; what could still leak:
#   1  a patch in two splits (within D1, within D2, or between them);
#   2  one Euclid object in two splits: D1 pairs, D2 targets and their partners (neighbours) together; and one Rubin
#      object in two splits (D1 objects, D2 Rubin owners). Patches overlap at their borders, so one galaxy can be
#      cut out of two patches that fell in different splits;
#   3  positions closer than --dup-radius (0.5") in different splits, with their identifiers: candidates only, two
#      distinct close objects also land here (the separation and whether the ids differ are reported);
#   4  objects with an object of another split within --stamp-radius (half a Rubin stamp, 5.9"): the pixels of one
#      galaxy can then be seen in a stamp of another split (border exposure), per split.
# Usage (CPU, a few minutes; login node or cpu_p1):
#   python scripts/inspect/split_leakage.py <ML config> [--out leakage.json] [--dup-radius 0.5] [--stamp-radius 5.9]

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from dp2_cutouts.mlconfig import MLConfig

D1_COLS = ["split", "tract", "patch", "object_id", "euclid_object_id", "euclid_ra", "euclid_dec"]
D2_COLS = ["split", "tract", "patch", "group_id", "euclid_object_id", "partner_euclid_object_id", "rubin_object_id",
           "euclid_ra", "euclid_dec"]


def read(path, cols):
    have = set(pq.read_schema(path).names)
    use = [c for c in cols if c in have]
    missing = [c for c in cols if c not in have]
    return pd.read_parquet(path, columns=use), missing


def unit(ra, dec):
    r, d = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


def chord(arcsec):
    return 2 * np.sin(np.radians(arcsec / 3600.0) / 2)


def patches_in_two_splits(frames):
    t = pd.concat([f[["tract", "patch", "split"]].assign(ds=n) for n, f in frames.items()
                   if {"tract", "patch"} <= set(f.columns)], ignore_index=True)
    g = t.groupby(["tract", "patch"])["split"].nunique()
    bad = g[g > 1].index.tolist()
    return dict(n_patches=int(len(g)), in_two_splits=len(bad), examples=[list(map(int, b)) for b in bad[:10]])


def ids_in_two_splits(rows):
    # rows: DataFrame(id, split, role); ids in more than one split.
    rows = rows.dropna(subset=["id"])
    rows = rows.assign(id=rows["id"].astype("int64"))
    g = rows.groupby("id")["split"].nunique()
    bad = g[g > 1].index
    sub = rows[rows["id"].isin(bad)]
    pairs = sub.groupby("id")["split"].apply(lambda s: "/".join(sorted(set(s)))).value_counts().to_dict()
    roles = sub.groupby("id")["role"].apply(lambda s: "+".join(sorted(set(s)))).value_counts().to_dict()
    return dict(n_ids=int(len(g)), in_two_splits=int(len(bad)), by_split_pair=pairs, by_role=roles,
                examples=[int(i) for i in bad[:10]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="split_leakage.json")
    ap.add_argument("--dup-radius", type=float, default=0.5)
    ap.add_argument("--stamp-radius", type=float, default=5.9)
    a = ap.parse_args()
    cfg = MLConfig.load(a.config, [])
    d1, miss1 = read(cfg.dataset_dir("d1") / "meta.parquet", D1_COLS)
    d2, miss2 = read(cfg.dataset_dir("d2") / "meta.parquet", D2_COLS)
    out = dict(rows=dict(d1=int(len(d1)), d2=int(len(d2))), missing_columns=dict(d1=miss1, d2=miss2),
               splits=dict(d1=d1["split"].value_counts().to_dict(), d2=d2["split"].value_counts().to_dict()))
    out["1_patches"] = patches_in_two_splits(dict(d1=d1, d2=d2))

    eu = [d1[["euclid_object_id", "split"]].rename(columns={"euclid_object_id": "id"}).assign(role="d1"),
          d2[["euclid_object_id", "split"]].rename(columns={"euclid_object_id": "id"}).assign(role="d2_target")]
    if "partner_euclid_object_id" in d2:
        eu.append(d2[["partner_euclid_object_id", "split"]].rename(columns={"partner_euclid_object_id": "id"})
                  .assign(role="d2_partner"))
    out["2_euclid_ids"] = ids_in_two_splits(pd.concat(eu, ignore_index=True))
    ru = [d1[["object_id", "split"]].rename(columns={"object_id": "id"}).assign(role="d1")]
    if "rubin_object_id" in d2:
        ru.append(d2[["rubin_object_id", "split"]].rename(columns={"rubin_object_id": "id"}).assign(role="d2_owner"))
    out["2_rubin_ids"] = ids_in_two_splits(pd.concat(ru, ignore_index=True))

    pos = pd.concat([d1[["euclid_ra", "euclid_dec", "split", "euclid_object_id"]].assign(ds="d1"),
                     d2[["euclid_ra", "euclid_dec", "split", "euclid_object_id"]].assign(ds="d2")],
                    ignore_index=True)
    # One position per Euclid object and split (D2 repeats a target in no other row; D1 and D2 may share one).
    pos = pos.dropna(subset=["euclid_ra", "euclid_dec"]).drop_duplicates(["euclid_object_id", "split"])
    tree = cKDTree(unit(pos["euclid_ra"], pos["euclid_dec"]))
    split = pos["split"].to_numpy()
    ids = pos["euclid_object_id"].to_numpy()
    xyz = tree.data
    p = tree.query_pairs(chord(a.dup_radius), output_type="ndarray")
    cross = p[split[p[:, 0]] != split[p[:, 1]]] if len(p) else p
    sep = np.degrees(2 * np.arcsin(np.linalg.norm(xyz[cross[:, 0]] - xyz[cross[:, 1]], axis=1) / 2)) * 3600 \
        if len(cross) else np.zeros(0)
    same = ids[cross[:, 0]] == ids[cross[:, 1]] if len(cross) else np.zeros(0, bool)
    out["3_close_pairs_across_splits"] = dict(
        radius_arcsec=a.dup_radius, n_pairs=int(len(cross)), same_euclid_id=int(same.sum()),
        separation_arcsec_quantiles=np.quantile(sep, [0, 0.5, 0.9, 1]).round(3).tolist() if len(sep) else [],
        note="candidates: two distinct close objects also count; same_euclid_id are true duplicates")
    p = tree.query_pairs(chord(a.stamp_radius), output_type="ndarray")
    cross = p[split[p[:, 0]] != split[p[:, 1]]] if len(p) else p
    exposed = np.zeros(len(pos), bool)
    if len(cross):
        exposed[cross[:, 0]] = exposed[cross[:, 1]] = True
    out["4_border_exposure"] = dict(
        radius_arcsec=a.stamp_radius,
        fraction_with_other_split_neighbour={s: float(exposed[split == s].mean()) for s in np.unique(split)},
        note="an object of another split within half a Rubin stamp: its pixels can appear in that split's stamps")
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
