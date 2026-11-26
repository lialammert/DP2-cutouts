# store.py
# Anna Preto
# AstroDeep Project
# On-disk format of a paired dataset: one folder with
#   info.json                 shapes, bands, units, configuration fingerprint, counts
#   meta.parquet              one row per example, row i <-> index i of every array
#   <name>.npy                one array per quantity, read through memory maps (never loaded whole)
#   rejected.parquet          examples dropped, with the reason
# Built shard by shard (one shard per Rubin patch, written by independent jobs), then merged sequentially into
# memory-mapped arrays. Two merges must never run at the same time on the same folder: the merge takes a lock file.

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def write_shard(shard_dir, meta, arrays, rejected=None, info=None):
    # meta: DataFrame (n rows); arrays: dict name -> ndarray with len n; written atomically (tmp folder, rename).
    shard_dir = Path(shard_dir)
    tmp = shard_dir.with_name(shard_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n = len(meta)
    for name, arr in arrays.items():
        if len(arr) != n:
            raise ValueError(f"array {name} has {len(arr)} rows, meta has {n}")
        np.save(tmp / f"{name}.npy", np.ascontiguousarray(arr))
    meta.reset_index(drop=True).to_parquet(tmp / "meta.parquet", index=False)
    (rejected if rejected is not None else pd.DataFrame(columns=["reason"])).to_parquet(
        tmp / "rejected.parquet", index=False)
    with open(tmp / "info.json", "w") as f:
        json.dump(dict(info or {}, n=n, arrays={k: [list(v.shape[1:]), str(v.dtype)] for k, v in arrays.items()}),
                  f, indent=2)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    tmp.rename(shard_dir)


def shard_done(shard_dir):
    return (Path(shard_dir) / "info.json").exists()


def merge_shards(shard_dirs, out_dir, info):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".merge.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"{lock} exists: another merge is running (or crashed; remove the file if so)")
    os.close(fd)
    try:
        shards = [Path(s) for s in shard_dirs if shard_done(s)]
        if not shards:
            raise RuntimeError("no finished shard to merge")
        infos = [json.loads((s / "info.json").read_text()) for s in shards]
        ref = next((inf for inf in infos if inf["n"] > 0), infos[0])
        names = sorted(ref["arrays"])
        for s, inf in zip(shards, infos):
            if sorted(inf["arrays"]) != names:
                raise RuntimeError(f"shard {s} does not hold the same arrays as the others")
            for k in names:
                if inf["arrays"][k] != ref["arrays"][k] and inf["n"] > 0:
                    raise RuntimeError(f"shard {s}: array {k} has shape/dtype {inf['arrays'][k]}, "
                                       f"expected {ref['arrays'][k]}")
        counts = [inf["n"] for inf in infos]
        total = int(sum(counts))
        meta = merged_meta(shards, infos)
        rejs = []
        for s in shards:
            r = pd.read_parquet(s / "rejected.parquet")
            if len(r):
                r["shard"] = s.name
                rejs.append(r)
        if len(meta) != total:
            raise RuntimeError("meta rows and array rows disagree")
        for k in names:
            shape, dtype = ref["arrays"][k]
            out = np.lib.format.open_memmap(out_dir / f"{k}.npy.tmp", mode="w+", dtype=np.dtype(dtype),
                                            shape=(total, *shape))
            pos = 0
            for s, n in zip(shards, counts):
                if n:
                    out[pos:pos + n] = np.load(s / f"{k}.npy", mmap_mode="r")
                pos += n
            out.flush()
            del out
            os.replace(out_dir / f"{k}.npy.tmp", out_dir / f"{k}.npy")
        meta.to_parquet(out_dir / "meta.parquet", index=False)
        (pd.concat(rejs, ignore_index=True) if rejs else pd.DataFrame(columns=["reason"])).to_parquet(
            out_dir / "rejected.parquet", index=False)
        with open(out_dir / "info.json", "w") as f:
            json.dump(dict(info, n=total, n_shards=len(shards),
                           arrays={k: ref["arrays"][k] for k in names}), f, indent=2)
        return total
    finally:
        lock.unlink(missing_ok=True)


def merged_meta(shards, infos):
    # Metadata of the shards in merge order. Empty shards (patches without pairs, written with a column-less
    # DataFrame) are left out of the concatenation: pandas would otherwise turn every int64 column into float64,
    # which rounds the 64-bit Rubin / Euclid identifiers (2^59 and more) and makes thousands of them collide.
    metas = []
    for s, inf in zip(shards, infos):
        if inf["n"] == 0:
            continue
        m = pd.read_parquet(s / "meta.parquet")
        m["shard"] = s.name
        metas.append(m)
    return pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()


def repair_meta(shard_dirs, out_dir):
    # Rewrites meta.parquet of a merged dataset from its shards, arrays untouched (for datasets merged before the
    # empty-shard fix). Refuses unless the rows line up with the merged arrays: same shards, same count, same
    # positions and same float columns as the current meta.parquet.
    out_dir = Path(out_dir)
    info = json.loads((out_dir / "info.json").read_text())
    shards = [Path(s) for s in shard_dirs if shard_done(s)]
    infos = [json.loads((s / "info.json").read_text()) for s in shards]
    if len(shards) != info["n_shards"] or sum(inf["n"] for inf in infos) != info["n"]:
        raise RuntimeError(f"shards ({len(shards)}, {sum(inf['n'] for inf in infos)} rows) do not match the merged "
                           f"dataset ({info['n_shards']}, {info['n']} rows): merge again instead")
    new = merged_meta(shards, infos)
    old = pd.read_parquet(out_dir / "meta.parquet")
    if len(new) != len(old) or list(new.columns) != list(old.columns):
        raise RuntimeError("rebuilt metadata does not have the rows / columns of the merged one")
    if not (old["shard"].astype(str).to_numpy() == new["shard"].astype(str).to_numpy()).all():
        raise RuntimeError("shard order differs from the merged dataset")
    fixed = []
    for c in new.columns:
        a, b = old[c], new[c]
        if a.dtype == b.dtype:
            if pd.api.types.is_float_dtype(a) and not np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True):
                raise RuntimeError(f"column {c} differs: rows do not line up")
            continue
        fixed.append(f"{c}: {a.dtype} -> {b.dtype}")
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=2.0 ** -52, atol=0):
                raise RuntimeError(f"column {c} differs beyond the float rounding: rows do not line up")
    tmp = out_dir / "meta.parquet.tmp"
    new.to_parquet(tmp, index=False)
    os.replace(tmp, out_dir / "meta.parquet")
    return fixed


class Store:
    # Read side: memory-mapped arrays, metadata in RAM (one row of scalars per example).
    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "info.json").exists():
            raise FileNotFoundError(f"{self.path} is not a finished dataset (no info.json)")
        self.info = json.loads((self.path / "info.json").read_text())
        self.meta = pd.read_parquet(self.path / "meta.parquet")
        self._arrays = {}

    def __len__(self):
        return len(self.meta)

    def names(self):
        return sorted(self.info["arrays"])

    def array(self, name):
        if name not in self._arrays:
            self._arrays[name] = np.load(self.path / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def take(self, name, idx):
        # Rows idx (any order) read in sorted order, which is what memory maps like, then put back in place.
        idx = np.asarray(idx)
        order = np.argsort(idx, kind="stable")
        out = np.asarray(self.array(name)[idx[order]])
        back = np.empty_like(order)
        back[order] = np.arange(len(order))
        return out[back]

    def rejected(self):
        p = self.path / "rejected.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["reason"])
