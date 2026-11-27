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
#   flux  the stored flux on the pixels of weight > 0; on the others (masked, neighbours) a sky-noise draw at the
#         median sky sigma of that image and band, made once here (fixed seed) instead of at every batch: the
#         network input is then asinh(flux) only
#   w     the likelihood weight: ivar on the pixels that are observations of the target, 0 elsewhere (MMU mask, ivar
#         <= 0 or not finite, flux not finite, and for Euclid the neighbour fill)
# The likelihood ignores the pixels of weight 0, so the loaders lose nothing: same weights, same flux where w > 0,
# only the noise draw of the masked pixels is fixed instead of new at every epoch (tests/test_deblend.py).
#   d1   <d1>/compact/euclid.npy (n, 119, 119, 2) and rubin.npy (n, 59, 59, 12) (deblend.data.compact_d1), every D1
#        row (the selection can change without a new copy)
#   d2   euclid.npy (m, 119, 119, 2) and rubin.npy (m, 59, 59, 12) of the targets of deblend.d2_selection in every
#        split, the Rubin stamp of the centring of deblend.d2_centering
#   rows.npy  the dataset row held at each position of the copy.
# Layout (version 3, 2026-11-27): the rows are grouped by split (train, val, test, holdout, ...) and shuffled once
# within each split (fixed seed). Consecutive positions are then unrelated galaxies, so that a loader reads its
# training batches as a few long contiguous runs, mixed: deblend-bench showed that reading through memory maps,
# page by page, row by row, capped the reading at ~90 MB/s (325 examples per second with 10 threads); the read side
# here (Side) reads runs of rows with one pread each.
# info.json is written last: a copy without it is unfinished. It records the source (rows, size and date of
# meta.parquet) so that a copy of an older dataset is refused. The copy runs in chunks over `workers` processes (each
# reads consecutive dataset rows and writes them to their shuffled positions) and resumes after a walltime cut
# (progress_*.done lists the finished chunks).
# Sizes: D1 Euclid 113 kB per row (168 GB for 1.49 M rows; the full Euclid arrays hold 524 kB per row), Rubin
# 167 kB per row (248 GB); D2 280 kB per target (~40 GB for ~144 000 targets).

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

CHUNK = 2048
VERSION = 3          # 2: weight-0 pixels filled with a fixed sky-noise draw; 3: rows grouped by split and shuffled
FILL_SEED = 20261127
LAYOUT_SEED = 20261128
SPLIT_ORDER = ("train", "val", "test", "holdout")


def sky_sigma(w, bad):
    # Median sky sigma 1 / sqrt(w) of the valid pixels, per image and band: (..., H, W, C) -> (..., C); 0 if none.
    ok = ~bad & (w > 0) & np.isfinite(w)
    s = np.where(ok, 1.0 / np.sqrt(np.where(ok, w, 1.0)), np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sig = np.nanmedian(s, axis=(-3, -2))
    return np.nan_to_num(sig, nan=0.0).astype(np.float32)


def noise_fill(flux, ivar, bad, rng):
    # flux, ivar (..., H, W, C), bad (..., H, W, C) bool -> copy of flux with the bad pixels drawn from N(0, sigma),
    # sigma = median sky sigma of the valid pixels of the same image and band (0 if none). Vectorised.
    noise = rng.standard_normal(flux.shape).astype(np.float32) * sky_sigma(ivar, bad)[..., None, None, :]
    f = np.where(bad, noise, flux).astype(np.float32)
    f[~np.isfinite(f)] = 0.0
    return f


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


def npy_layout(path):
    # (offset of the data, shape, dtype) of a .npy file, to read or write rows by offset.
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            shape, _, dtype = np.lib.format._read_array_header(f, version)
        return f.tell(), tuple(shape), np.dtype(dtype)


def split_layout(rows, splits, seed=LAYOUT_SEED):
    # Dataset rows -> the order of the copy: grouped by split (SPLIT_ORDER first, others after, by name), shuffled
    # within each. Returns (rows in copy order, {split: [first position, end)}).
    rows, splits = np.asarray(rows, np.int64), np.asarray(splits).astype(str)
    names = [n for n in SPLIT_ORDER if n in set(splits)] + sorted(set(splits) - set(SPLIT_ORDER))
    rng = np.random.default_rng(seed)
    parts, ranges, p0 = [], {}, 0
    for n in names:
        r = rows[splits == n]
        parts.append(rng.permutation(r))
        ranges[n] = [p0, p0 + len(r)]
        p0 += len(r)
    return (np.concatenate(parts) if parts else rows[:0]), ranges


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
def _write_rows(path, rec, pos):
    # rec (k, ...) float32 written at positions pos of the .npy file `path` (one pwrite per row).
    off, shape, _ = npy_layout(path)
    rb = int(np.prod(shape[1:])) * 4
    fd = os.open(path, os.O_WRONLY)
    try:
        for j, q in enumerate(pos):
            buf = np.ascontiguousarray(rec[j], np.float32).tobytes()
            done = 0
            while done < len(buf):
                done += os.pwrite(fd, buf[done:], off + int(q) * rb + done)
    finally:
        os.close(fd)


def _chunk(job):
    # One chunk: reads consecutive source rows through memory maps (never the Store: it would load the whole
    # metadata in every process) and writes each at its position `pos` of the outputs.
    k, src, out, rows, pos, sides, band, rkeys = job
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
        fl = noise_fill(fl, w, ~(w > 0), np.random.default_rng([FILL_SEED, k, 0]))
        _write_rows(out / "euclid.npy.tmp", np.concatenate([fl, w], axis=-1), pos)
    if "rubin" in sides:
        f, iv, m = (np.asarray(a(n)[rows]) for n in rkeys)
        fl, w = rubin_fw(f, iv, m)
        fl = noise_fill(fl, w, ~(w > 0), np.random.default_rng([FILL_SEED, k, 1]))
        _write_rows(out / "rubin.npy.tmp", np.concatenate([fl, w], axis=-1), pos)
    return k


def copy(src, out, layout, sides, band, rkeys, shapes, workers, extra_info):
    # layout: the source row of each position of the copy (split_layout); sides: ("euclid", "rubin"); shapes: side ->
    # per-row output shape. Chunks run over the source rows in increasing order (consecutive reads).
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "info.json").unlink(missing_ok=True)
    layout = np.asarray(layout, np.int64)
    m = len(layout)
    order = np.argsort(layout, kind="stable")
    rows, pos_sorted = layout[order], order            # source rows increasing, and their positions
    n_chunks = -(-m // CHUNK)
    tag = f"v{VERSION}_{m}_{int(layout[0]) if m else 0}_{int(layout[-1]) if m else 0}_{'-'.join(sides)}"
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
    np.save(out / "rows.npy.tmp.npy", layout)
    jobs = [(k, str(src), str(out), rows[k * CHUNK:(k + 1) * CHUNK], pos_sorted[k * CHUNK:(k + 1) * CHUNK],
             tuple(sides), band, tuple(rkeys)) for k in range(n_chunks) if k not in done]
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
                filled=True, content="float32 (..., [flux channels, weight channels]); weight = ivar on observed target pixels, "
                        "0 elsewhere, where the flux is a fixed sky-noise draw", **extra_info)
    (out / "info.json").write_text(json.dumps(info, indent=2))
    return info


def build(cfg, which, workers=8):
    root = Path(cfg.dataset_dir(which))
    band = list(cfg["euclid"]["bands"]).index(cfg["deblend"]["euclid_band"])
    n_r, b_r, n_e = int(cfg["rubin"]["stamp_px"]), len(cfg["rubin"]["bands"]), int(cfg["euclid"]["stamp_px"])
    shapes = dict(euclid=(n_e, n_e, 2), rubin=(n_r, n_r, 2 * b_r))
    extra = dict(euclid_band=cfg["deblend"]["euclid_band"])
    import pandas as pd
    if which == "d1":
        sides = tuple((cfg["deblend"].get("data") or {}).get("compact_d1", ["euclid", "rubin"]))
        split = pd.read_parquet(root / "meta.parquet", columns=["split"])["split"].to_numpy()
        rows = np.arange(len(split), dtype=np.int64)
        rkeys = ("rubin_flux", "rubin_ivar", "rubin_mask")
    elif which == "d2":
        from . import data
        from ..pairs.store import read_meta
        meta = read_meta(root, data.D2_TRAIN_COLUMNS)              # the columns present
        keep, sel = data.d2_keep(cfg, meta)
        centering = cfg["deblend"]["d2_centering"]
        rows = np.unique(np.concatenate([data.d2_rows(meta, s, centering, keep) for s in meta["split"].unique()]))
        sides = ("euclid", "rubin")
        sfx = "_rc" if centering == "rubin" else ""
        rkeys = tuple(f"rubin_{q}{sfx}" for q in ("flux", "ivar", "mask"))
        extra.update(centering=centering, d2_selection=sel,
                     rows_per_split={str(s): int((meta["split"].to_numpy()[rows] == s).sum())
                                     for s in meta["split"].unique()})
        split = meta["split"].to_numpy()
        del meta
    else:
        raise ValueError(which)
    layout, ranges = split_layout(rows, split[rows])
    extra["split_ranges"] = ranges
    gb = len(rows) * sum(4 * np.prod(shapes[s]) for s in sides) / 1e9
    print(f"compact {which}: {len(rows):,} rows, sides {list(sides)}, {gb:.0f} GB, splits {ranges} -> "
          f"{compact_dir(cfg, which)}", flush=True)
    return copy(root, compact_dir(cfg, which), layout, sides, band, rkeys, shapes, workers, extra)


# ----------------------------------------------------------------------------- read side
def memory_limit_bytes():
    # Memory this job may use: Slurm's allocation (SLURM_MEM_PER_NODE, or SLURM_MEM_PER_CPU x cpus, in MB), the
    # job's own cgroup limit, else the node's available memory. (On Jean Zay the root cgroup files are not the job's:
    # the first version read the node's 136 GB on a V100 job of 10 cores.)
    env = os.environ
    try:
        if env.get("SLURM_MEM_PER_NODE"):
            return int(env["SLURM_MEM_PER_NODE"]) * 1024 ** 2
        if env.get("SLURM_MEM_PER_CPU"):
            cpus = int(env.get("SLURM_CPUS_PER_TASK") or env.get("SLURM_CPUS_ON_NODE") or 1)
            return int(env["SLURM_MEM_PER_CPU"]) * cpus * 1024 ** 2
    except ValueError:
        pass
    paths = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, ctrl, rel = line.split(":", 2)
            if ctrl == "":
                paths.append(f"/sys/fs/cgroup{rel}/memory.max")
            elif "memory" in ctrl.split(","):
                paths.append(f"/sys/fs/cgroup/memory{rel}/memory.limit_in_bytes")
    except (OSError, ValueError):
        pass
    for p in paths + ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
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
    # Rows of one compact array, read with one pread per run of close positions (a gap of at most GAP unneeded rows is
    # read through), each read at most MAX_READ bytes; or the rows a loader needs, loaded in memory the same way.
    # Thread-safe (pread takes its offset; no shared file position).
    GAP = 32
    MAX_READ = 128 << 20

    def __init__(self, path, pos_of_row, need_rows, in_memory, label):
        self.path = str(path)
        self.offset, shape, self.dtype = npy_layout(path)
        self.row_shape = tuple(shape[1:])
        self.row_bytes = int(np.prod(self.row_shape)) * self.dtype.itemsize
        self.fd = os.open(self.path, os.O_RDONLY)
        self.pos_of = pos_of_row
        self.local = None
        if in_memory:
            pos = np.unique(pos_of_row[np.asarray(need_rows, np.int64)])
            t0 = time.time()
            self.arr = self.read_positions(pos)
            self.local = pos                         # position in the copy -> index in self.arr (searchsorted)
            print(f"{label}: {len(pos):,} rows loaded in memory ({self.arr.nbytes / 1e9:.1f} GB, "
                  f"{time.time() - t0:.0f} s)", flush=True)

    def __del__(self):
        try:
            os.close(self.fd)
        except (OSError, AttributeError):
            pass

    def _pread_into(self, out, off):
        # Fills the bytes of the C-contiguous array `out` from file offset `off`.
        mv = memoryview(out).cast("B")
        done = 0
        while done < len(mv):
            if hasattr(os, "preadv"):
                n = os.preadv(self.fd, [mv[done:]], off + done)
            else:
                b = os.pread(self.fd, len(mv) - done, off + done)
                mv[done:done + len(b)] = b
                n = len(b)
            if n <= 0:
                raise IOError(f"{self.path}: short read at offset {off + done}")
            done += n

    def read_positions(self, pos):
        # pos: sorted unique positions -> (len(pos), *row_shape).
        pos = np.asarray(pos, np.int64)
        out = np.empty((len(pos), *self.row_shape), self.dtype)
        if not len(pos):
            return out
        cut = np.flatnonzero(np.diff(pos) > self.GAP + 1) + 1
        max_rows = max(1, self.MAX_READ // self.row_bytes)
        for s0, e0 in zip(np.concatenate([[0], cut]), np.concatenate([cut, [len(pos)]])):
            i = int(s0)
            while i < e0:
                j = i + int(np.searchsorted(pos[i:e0], pos[i] + max_rows))
                a, b = int(pos[i]), int(pos[j - 1])
                off = self.offset + a * self.row_bytes
                if b - a == j - 1 - i:                   # dense run: straight into the output
                    self._pread_into(out[i:j], off)
                else:
                    tmp = np.empty((b - a + 1, *self.row_shape), self.dtype)
                    self._pread_into(tmp, off)
                    out[i:j] = tmp[pos[i:j] - a]
                i = j
        return out

    def take(self, rows):
        pos = self.pos_of[np.asarray(rows, np.int64)]
        if self.local is not None:
            return self.arr[np.searchsorted(self.local, pos)]
        u, inv = np.unique(pos, return_inverse=True)
        return self.read_positions(u)[inv.reshape(-1)]


_POS_CACHE = {}


def position_key(cfg, which):
    # Position in the compact copy of every dataset row (-1: not in it), or None when the copy is off or unusable.
    # Shared by the loaders of a process; their training order follows it so that blocks are contiguous runs.
    if not (cfg["deblend"].get("data") or {}).get("compact", False):
        return None
    info = read_info(cfg, which)
    if info is None:
        return None
    p = compact_dir(cfg, which) / "rows.npy"
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _POS_CACHE:
        layout = np.load(p)
        pos = np.full(int(info["source"]["n"]), -1, np.int64)
        pos[layout] = np.arange(len(layout))
        _POS_CACHE.clear()
        _POS_CACHE[key] = pos
    return _POS_CACHE[key]


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
    pos_of = position_key(cfg, which)
    ok = pos_of[need] >= 0
    if not ok.all():
        print(f"{which}: {int((~ok).sum()):,} of the {len(need):,} rows are not in the compact copy (another "
              "selection?): reading the original arrays", flush=True)
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
    return {s: Side(d / f"{s}.npy", pos_of, need, mem, f"{which} compact {s}") for s in info["sides"]}
