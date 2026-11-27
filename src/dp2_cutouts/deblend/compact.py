# compact.py
# Anna Preto
# AstroDeep Project
# Compact copies of D1 and D2 for the deblending loaders (dp2-ml <cfg> deblend-compact d1 | d2).
#
# Why: the loaders read, per example, every Euclid band (VIS, Y, J, H: flux, ivar, MMU mask, neighbour fill) and the
# Rubin flux, ivar and mask as separate arrays, while the campaign uses VIS only and needs, per pixel, a flux and a
# likelihood weight. On Lustre the reads, not the GPU, set the pace (2-3 s per step after the block reading).
#
# What: per example and side, two quantities only, in one float32 array (..., [flux channels, weight channels]):
#   flux  the stored flux with NaN / inf set to 0 (as the loaders already returned it)
#   w     the likelihood weight: ivar on the pixels that are observations of the target, 0 elsewhere (MMU mask, ivar
#         <= 0 or not finite, flux not finite, and for Euclid the neighbour fill)
# This loses nothing the loaders use: they derive every input from (flux, w) (data.Prepare.rubin_w / euclid_w,
# tests/test_deblend.py checks that the batches are identical, bit for bit).
#   d1   <d1>/compact/euclid.npy (n, 119, 119, 2), every D1 row in D1's order (the selection can change without a new
#        copy); optional rubin.npy (n, 59, 59, 12) with deblend.data.compact_d1: [euclid, rubin]
#   d2   <d2>/compact/rows.npy (the D2 rows: targets of deblend.d2_selection in every split, for the centring of
#        deblend.d2_centering), rubin.npy (m, 59, 59, 12) from the stamps of that centring, euclid.npy (m, 119, 119, 2)
# info.json is written last: a copy without it is unfinished. It records the source (rows, size and date of
# meta.parquet) so that a copy of an older dataset is refused. The copy runs in chunks over `workers` processes and
# resumes after a walltime cut (<name>.done lists the finished chunks).
# Sizes: D1 Euclid 113 kB per row (168 GB for 1.49 M rows; the full Euclid arrays hold 524 kB per row), Rubin
# 167 kB per row (248 GB); D2 280 kB per target (~40 GB for ~144 000 targets).

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

CHUNK = 2048
VERSION = 1


def rubin_fw(f, iv, m):
    # Stored Rubin flux, ivar, mask (..., H, W, C) -> (flux, w) float32.
    bad = m | ~(iv > 0) | ~np.isfinite(iv) | ~np.isfinite(f)
    return np.nan_to_num(f).astype(np.float32), np.where(bad, 0.0, iv).astype(np.float32)


def euclid_fw(f, iv, m, fill, band):
    # Stored Euclid flux, ivar, mask (..., H, W, bands), fill (..., H, W) or None -> (flux, w) of `band`, (..., H, W, 1).
    f, iv, m = f[..., band:band + 1], iv[..., band:band + 1], m[..., band:band + 1]
    bad = m | ~(iv > 0) | ~np.isfinite(iv) | ~np.isfinite(f)
    if fill is not None:
        bad = bad | fill[..., None]
    return np.nan_to_num(f).astype(np.float32), np.where(bad, 0.0, iv).astype(np.float32)


def compact_dir(cfg, which):
    return Path(cfg.dataset_dir(which)) / "compact"


def source_stamp(root):
    p = Path(root) / "meta.parquet"
    st = p.stat()
    return dict(meta_size=int(st.st_size), meta_mtime_ns=int(st.st_mtime_ns),
                n=int(json.loads((Path(root) / "info.json").read_text())["n"]))


def read_info(cfg, which):
    # info.json of a finished copy that matches the current dataset, else None.
    d = compact_dir(cfg, which)
    p = d / "info.json"
    if not p.exists():
        return None
    info = json.loads(p.read_text())
    if info.get("version") != VERSION or info.get("source") != source_stamp(cfg.dataset_dir(which)):
        return None
    if info.get("euclid_band") != cfg["deblend"]["euclid_band"]:
        return None
    if which == "d2" and info.get("centering") != cfg["deblend"]["d2_centering"]:
        return None
    return info


# ----------------------------------------------------------------------------- copy
def _chunk(job):
    # One chunk: reads the source rows through memory maps (never the Store: it would load the whole metadata in
    # every process) and writes positions [p0, p0 + len(rows)) of the outputs.
    k, src, out, rows, p0, sides, band, rkeys = job
    src, out = Path(src), Path(out)
    load = {}

    def a(name):
        if name not in load:
            load[name] = np.load(src / f"{name}.npy", mmap_mode="r")
        return load[name]

    if "euclid" in sides:
        has_fill = (src / "euclid_fill.npy").exists()
        f, iv, m = (np.asarray(a(n)[rows]) for n in ("euclid_flux", "euclid_ivar", "euclid_mask"))
        fill = np.asarray(a("euclid_fill")[rows]) if has_fill else None
        fl, w = euclid_fw(f, iv, m, fill, band)
        o = np.load(out / "euclid.npy.tmp", mmap_mode="r+")
        o[p0:p0 + len(rows)] = np.concatenate([fl, w], axis=-1)
        o.flush()
        del o
    if "rubin" in sides:
        f, iv, m = (np.asarray(a(n)[rows]) for n in rkeys)
        fl, w = rubin_fw(f, iv, m)
        o = np.load(out / "rubin.npy.tmp", mmap_mode="r+")
        o[p0:p0 + len(rows)] = np.concatenate([fl, w], axis=-1)
        o.flush()
        del o
    return k


def copy(src, out, rows, sides, band, rkeys, shapes, workers, extra_info):
    # rows: sorted source rows; sides: ("euclid",) / ("euclid", "rubin"); shapes: side -> per-row output shape.
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "info.json").unlink(missing_ok=True)
    rows = np.asarray(rows, np.int64)
    m = len(rows)
    n_chunks = -(-m // CHUNK)
    tag = f"{len(rows)}_{int(rows[0]) if m else 0}_{int(rows[-1]) if m else 0}_{'-'.join(sides)}"
    done_file = out / f"progress_{tag}.done"
    done = set()
    for side in sides:
        p = out / f"{side}.npy.tmp"
        if not p.exists() or not done_file.exists():
            np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(m, *shapes[side]))
            done_file.unlink(missing_ok=True)
    if done_file.exists():
        done = {int(s) for s in done_file.read_text().split()}
        print(f"compact: resuming, {len(done)}/{n_chunks} chunks already copied", flush=True)
    np.save(out / "rows.npy.tmp.npy", rows)
    jobs = [(k, str(src), str(out), rows[k * CHUNK:(k + 1) * CHUNK], k * CHUNK, tuple(sides), band, tuple(rkeys))
            for k in range(n_chunks) if k not in done]
    t0 = time.time()
    step = max(n_chunks // 50, 1)
    with open(done_file, "a") as log:
        if workers <= 1:
            it = map(_chunk, jobs)
            pool = None
        else:
            # spawn: the parent may have imported JAX, which must not be forked.
            pool = mp.get_context("spawn").Pool(int(workers))
            it = pool.imap_unordered(_chunk, jobs)
        try:
            for n, k in enumerate(it, 1):
                log.write(f"{k}\n")
                log.flush()
                if n % step == 0 or n == len(jobs):
                    el = time.time() - t0
                    print(f"compact: {len(done) + n}/{n_chunks} chunks, {el / 60:.1f} min, "
                          f"~{el / n * (len(jobs) - n) / 60:.0f} min left", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    for side in sides:
        os.replace(out / f"{side}.npy.tmp", out / f"{side}.npy")
    os.replace(out / "rows.npy.tmp.npy", out / "rows.npy")
    done_file.unlink(missing_ok=True)
    info = dict(version=VERSION, source=source_stamp(src), rows=int(m), sides=list(sides),
                arrays={s: [m, *shapes[s]] for s in sides},
                bytes_per_row=int(sum(4 * np.prod(shapes[s]) for s in sides)),
                content="float32 (..., [flux channels, weight channels]); weight = ivar on observed target pixels, "
                        "0 elsewhere", **extra_info)
    (out / "info.json").write_text(json.dumps(info, indent=2))
    return info


def build(cfg, which, workers=8):
    root = Path(cfg.dataset_dir(which))
    band = list(cfg["euclid"]["bands"]).index(cfg["deblend"]["euclid_band"])
    n_r, b_r, n_e = int(cfg["rubin"]["stamp_px"]), len(cfg["rubin"]["bands"]), int(cfg["euclid"]["stamp_px"])
    shapes = dict(euclid=(n_e, n_e, 2), rubin=(n_r, n_r, 2 * b_r))
    extra = dict(euclid_band=cfg["deblend"]["euclid_band"])
    if which == "d1":
        sides = tuple((cfg["deblend"].get("data") or {}).get("compact_d1", ["euclid"]))
        rows = np.arange(source_stamp(root)["n"], dtype=np.int64)
        rkeys = ("rubin_flux", "rubin_ivar", "rubin_mask")
    elif which == "d2":
        import pandas as pd

        from . import data
        meta = pd.read_parquet(root / "meta.parquet")
        keep, sel = data.d2_keep(cfg, meta)
        centering = cfg["deblend"]["d2_centering"]
        rows = np.unique(np.concatenate([data.d2_rows(meta, s, centering, keep) for s in meta["split"].unique()]))
        sides = ("euclid", "rubin")
        sfx = "_rc" if centering == "rubin" else ""
        rkeys = tuple(f"rubin_{q}{sfx}" for q in ("flux", "ivar", "mask"))
        extra.update(centering=centering, d2_selection=sel,
                     rows_per_split={str(s): int((meta["split"].to_numpy()[rows] == s).sum())
                                     for s in meta["split"].unique()})
        del meta
    else:
        raise ValueError(which)
    gb = len(rows) * sum(4 * np.prod(shapes[s]) for s in sides) / 1e9
    print(f"compact {which}: {len(rows):,} rows, sides {list(sides)}, {gb:.0f} GB -> {compact_dir(cfg, which)}",
          flush=True)
    return copy(root, compact_dir(cfg, which), rows, sides, band, rkeys, shapes, workers, extra)


# ----------------------------------------------------------------------------- read side
def memory_limit_bytes():
    # Memory this job may use: the cgroup limit (Slurm), else the node's available memory.
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1 << 60:
                return int(v)
        except (OSError, ValueError):
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class Side:
    # Rows of one compact array: memory-mapped, or the rows a loader needs copied into memory.
    def __init__(self, path, rows_of_copy, need_rows, in_memory, label):
        arr = np.load(path, mmap_mode="r")
        self.pos_of = None
        if rows_of_copy is None:                     # D1: position = row
            pos = np.asarray(need_rows, np.int64)
        else:
            pos = np.searchsorted(rows_of_copy, need_rows)
        self.rows_copy = rows_of_copy
        if in_memory:
            order = np.sort(np.unique(pos))
            t0 = time.time()
            self.arr = np.empty((len(order), *arr.shape[1:]), arr.dtype)
            for s in range(0, len(order), 4096):
                self.arr[s:s + 4096] = arr[order[s:s + 4096]]
            self.local = order                       # position in the copy -> index in self.arr (searchsorted)
            print(f"{label}: {len(order):,} rows loaded in memory ({self.arr.nbytes / 1e9:.1f} GB, "
                  f"{time.time() - t0:.0f} s)", flush=True)
        else:
            self.arr, self.local = arr, None

    def take(self, rows):
        rows = np.asarray(rows, np.int64)
        pos = rows if self.rows_copy is None else np.searchsorted(self.rows_copy, rows)
        if self.local is not None:
            return self.arr[np.searchsorted(self.local, pos)]
        order = np.argsort(pos, kind="stable")
        out = np.asarray(self.arr[pos[order]])
        back = np.empty_like(order)
        back[order] = np.arange(len(order))
        return out[back]


def open_sides(cfg, which, need_rows, train):
    # The compact sides usable by a loader of `which` needing `need_rows`: {side: Side}; {} when the copy is off,
    # missing, stale or (D2) does not hold every needed row (the loader then reads the original arrays).
    opt = cfg["deblend"].get("data") or {}
    if not opt.get("compact", False):
        return {}
    info = read_info(cfg, which)
    d = compact_dir(cfg, which)
    if info is None:
        if opt.get("require_compact", True):
            raise RuntimeError(f"deblend.data.compact is on but {d} holds no finished copy of the current {which}: "
                               f"run dp2-ml <cfg> deblend-compact {which} (or set deblend.data.compact: false)")
        print(f"{which}: no up-to-date compact copy in {d}, reading the original arrays", flush=True)
        return {}
    need = np.asarray(need_rows, np.int64)
    rows_copy = None
    if which == "d2":
        rows_copy = np.load(d / "rows.npy")
        pos = np.searchsorted(rows_copy, need)
        ok = (pos < len(rows_copy)) & (rows_copy[np.minimum(pos, len(rows_copy) - 1)] == need)
        if not ok.all():
            print(f"d2: {int((~ok).sum()):,} of the {len(need):,} rows are not in the compact copy (another selection?):"
                  " reading the original arrays", flush=True)
            return {}
    mem = train and which in (opt.get("in_memory") or [])
    if mem:
        size = len(np.unique(need)) * int(info["bytes_per_row"])
        lim = memory_limit_bytes()
        frac = float(opt.get("memory_fraction", 0.6))
        if lim is not None and size > frac * lim:
            print(f"{which}: {size / 1e9:.1f} GB to load > {frac:.0%} of the {lim / 1e9:.0f} GB this job may use: "
                  "kept memory-mapped", flush=True)
            mem = False
    return {s: Side(d / f"{s}.npy", rows_copy, need, mem, f"{which} compact {s}") for s in info["sides"]}
