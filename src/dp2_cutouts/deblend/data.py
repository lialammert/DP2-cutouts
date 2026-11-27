# data.py
# Anna Preto
# AstroDeep Project
# Streaming batches of D1 and D2 (memory-mapped stores of the pairs package, rows read per batch, never the whole
# split), as keras PyDatasets (backend agnostic).
#
# What each image carries:
#   *_in    network input: tanh(arcsinh(2.5 f / s_b)) per band (s_b: norm_stats.json, D1 train split), pixels that
#           are not observations of the target replaced by a noise draw at their own ivar before normalisation;
#           the Euclid input has a second channel = validity (1 observed target pixel, 0 otherwise): the encoder is
#           mask-aware, so a masked neighbour is never mistaken for sky;
#   *_flux  observed flux in the dataset units (Rubin nJy, Euclid as delivered: MER microJy per pixel);
#   *_w     likelihood weight = ivar on pixels that are observations of the target, 0 elsewhere.
# Not observations of the target: Rubin pixels flagged by the MMU mask (NO_DATA, SATURATED, CR, INTERPOLATED,
# DETECTION_EDGE) or with ivar <= 0; Euclid pixels flagged by the MMU mask, with ivar <= 0, or belonging to another
# source (euclid_fill: D2 always, D1 when built with dataset1.euclid_neighbour_mask). The Rubin pixels that the
# stamp pipeline filled with sky are real sky pixels of the same region and count as observations.
#
# D2 input centring (deblend.d2_centering):
#   rubin   (default) the Rubin blend stamp centred on the Rubin object of the group (rubin_*_rc), i.e. what a
#           Rubin-only user has; the target is the Euclid member of the group nearest to that centre, one example
#           per group, so that one input never has two targets;
#   euclid  the stamp centred on the Euclid target (rubin_*): the centring then carries Euclid information that is
#           not available at inference (kept for diagnostics only).
# Augmentation (train only): the same rotation by k x 90 degrees and flip on every image of an example. Optional
# masking augmentation of D1 Euclid inputs with real D2 neighbour masks (deblend.k1.mask_augment_prob).
# Reading (deblend.data): from the compact copies of deblend/compact.py when deblend.data.compact is on (flux and
# weight per side, VIS only; identical batches), the training rows of the datasets listed in deblend.data.in_memory
# copied into memory when they fit.

import json
import time

import keras
import numpy as np
import pandas as pd

from ..ml.data import aug_images, aug_mask2d
from ..pairs.store import Store
from .compact import euclid_fw, noise_fill, open_sides, position_key, rubin_fw  # noqa: F401 (noise_fill: callers)


def asinh_norm(f, s):
    return np.tanh(np.arcsinh(2.5 * f / s)).astype(np.float32)


def compute_norm_stats(cfg, max_examples=20000, seed=0):
    # Per-band scales s_b on the D1 train split, DebVAder's definition (mean over stamps of the maximum in the
    # central 21 x 21 pixels; 41 x 41 Euclid pixels for the same sky area), and the median sky sigma per band.
    st = Store(cfg.dataset_dir("d1"))
    rows = np.flatnonzero((st.meta["split"] == "train").to_numpy())
    if len(rows) > max_examples:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_examples, replace=False))
    e_bands = list(cfg["euclid"]["bands"])
    vis = e_bands.index(cfg["deblend"]["euclid_band"])
    out = {}
    for side, key, h, ch in (("rubin", "rubin", 10, None), ("euclid", "euclid", 20, vis)):
        mx, sky = [], []
        for s in range(0, len(rows), 1024):
            r = rows[s:s + 1024]
            f = st.take(f"{key}_flux", r)
            iv = st.take(f"{key}_ivar", r)
            m = st.take(f"{key}_mask", r)
            if ch is not None:
                f, iv, m = f[..., ch:ch + 1], iv[..., ch:ch + 1], m[..., ch:ch + 1]
            n = f.shape[1]
            c = n // 2
            h = min(h, c)
            mx.append(f[:, c - h:c + h + 1, c - h:c + h + 1].reshape(len(r), -1, f.shape[-1]).max(axis=1))
            ok = ~m & (iv > 0) & np.isfinite(iv)
            sig = np.where(ok, 1.0 / np.sqrt(np.where(ok, iv, 1.0)), np.nan)
            sky.append(np.nanmedian(sig.reshape(len(r), -1, f.shape[-1]), axis=1))
        out[side] = dict(scale=np.nanmean(np.concatenate(mx), axis=0).astype(float).tolist(),
                         sky_sigma=np.nanmedian(np.concatenate(sky), axis=0).astype(float).tolist())
    out["n_examples"] = int(len(rows))
    out["rubin_bands"] = list(cfg["rubin"]["bands"])
    out["euclid_band"] = cfg["deblend"]["euclid_band"]
    out["definition"] = "scale: mean of per-stamp max in the central 21x21 (Rubin) / 41x41 (Euclid) px, D1 train"
    return out


def deblend_dir(cfg):
    return cfg.root / "deblend"


def load_norm_stats(cfg):
    p = deblend_dir(cfg) / "norm_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `dp2-ml <cfg> deblend-stats` first")
    return json.loads(p.read_text())


class Norm:
    def __init__(self, stats):
        self.r_scale = np.asarray(stats["rubin"]["scale"], np.float32)
        self.e_scale = np.asarray(stats["euclid"]["scale"], np.float32)
        for name, s in (("rubin", self.r_scale), ("euclid", self.e_scale)):
            if not np.all(np.isfinite(s) & (s > 0)):
                raise ValueError(f"invalid {name} normalisation scale {s}")


def d2_rows(meta, split, centering, keep=None):
    # Rows of D2 used for a split; with Rubin centring, one target per group: the Euclid member nearest to the
    # centre of the Rubin stamp (smallest |rc_offset|; ties by euclid_object_id). keep: boolean array over meta (the
    # selection of d2_keep), applied after the choice of the target, so that a group whose nearest member fails is
    # dropped rather than given its farther member as target.
    sel = meta[meta["split"] == split]
    if centering == "euclid":
        rows = np.sort(sel.index.to_numpy())
    elif centering == "rubin":
        d = np.hypot(sel["rc_offset_x_px"].astype(float), sel["rc_offset_y_px"].astype(float))
        order = sel.assign(_d=d.to_numpy()).sort_values(["group_id", "_d", "euclid_object_id"])
        rows = np.sort(order.groupby("group_id", sort=False).head(1).index.to_numpy())
    else:
        raise ValueError(f"deblend.d2_centering must be rubin or euclid, not {centering!r}")
    return rows if keep is None else rows[np.asarray(keep, bool)[rows]]


def d1_vis_mag_limit(cfg, quantile):
    # VIS magnitude below which a fraction `quantile` of the D1 train galaxies lie (D1 is selected on Rubin S/N).
    m = pd.read_parquet(cfg.dataset_dir("d1") / "meta.parquet", columns=["split", "euclid_vis_mag"])
    v = m.loc[m["split"] == "train", "euclid_vis_mag"].to_numpy(float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise RuntimeError("no finite euclid_vis_mag in the D1 train split")
    return float(np.quantile(v, float(quantile)))


def d2_keep(cfg, meta):
    # Selection of the D2 targets used for training and evaluation (deblend.d2_selection; absent or null: all).
    # D2 itself is not changed. Returns (boolean array over meta, description).
    sel = cfg["deblend"].get("d2_selection")
    keep = np.ones(len(meta), bool)
    info = {}
    if not sel:
        return keep, info
    ba = sel.get("before_after")
    if ba:
        # The target measured on VIS before and after the neighbours are masked: the masking must leave it
        # unchanged (a failed measurement, NaN, fails).
        for col, key in (("sep_delta_e", "max_delta_e"), ("sep_centroid_shift_arcsec", "max_centroid_shift_arcsec"),
                         ("sep_rel_size_change", "max_rel_size_change"),
                         ("sep_rel_flux_change", "max_rel_flux_change")):
            if ba.get(key) is not None:
                ok = meta[col].to_numpy(float) < float(ba[key])
                info[f"pass_{key}"] = float(ok.mean())
                keep &= ok
    lim = sel.get("vis_mag_max")
    if lim is not None:
        if lim == "auto":
            lim = d1_vis_mag_limit(cfg, sel.get("vis_mag_quantile", 0.95))
        ok = meta["euclid_vis_mag"].to_numpy(float) <= float(lim)
        info.update(vis_mag_max=float(lim), pass_vis_mag=float(ok.mean()))
        keep &= ok
    if sel.get("euclid_cut1"):
        # D1's cut 1 (Euclid object neither point-like nor spurious), from d2/euclid_flags.parquet (dp2-ml d2-flags).
        from ..pairs import euclid_flags
        ok = euclid_flags.cut1_for(cfg, meta["euclid_object_id"])
        info["pass_euclid_cut1"] = float(ok.mean())
        keep &= ok
    if sel.get("max_lost_flux_frac") is not None:
        # Target VIS light hidden by the neighbour masks (pairs/mask_loss.py, dp2-ml mask-loss d2): a target losing
        # more than max_lost_flux_frac, by more than lost_sigma uncertainties, is left out. Targets not measured
        # (outside the rows mask-loss d2 was run on) are kept.
        from ..pairs import mask_loss
        t = mask_loss.read(cfg, "d2").set_index("euclid_object_id")
        ids = meta["euclid_object_id"].astype("int64").to_numpy()
        frac = t["lost_flux_frac"].reindex(ids).to_numpy(float)
        sig = t["lost_flux_sigma"].reindex(ids).to_numpy(float)
        ok = mask_loss.cut_pass(frac, sig, sel["max_lost_flux_frac"], sel.get("lost_sigma", 3.0))
        info["pass_lost_flux"] = float(ok[keep].mean()) if keep.any() else 1.0
        keep &= ok
    info["kept"] = float(keep.mean())
    return keep, info


def d1_keep(cfg, meta):
    # Selection of the D1 pairs used for training and evaluation (deblend.d1_selection; absent or null: all), on
    # the Rubin light the neighbour masks hide (pairs/mask_loss.py, dp2-ml mask-loss d1). D1 itself is not changed.
    # Returns (boolean array over meta, description).
    sel = cfg["deblend"].get("d1_selection") or {}
    keep = np.ones(len(meta), bool)
    if sel.get("max_lost_flux_frac") is None:
        return keep, {}
    from ..pairs import mask_loss
    t = mask_loss.read(cfg, "d1")
    if len(t) != len(meta) or not (t["object_id"].to_numpy() == meta["object_id"].astype("int64").to_numpy()).all():
        raise RuntimeError("d1/mask_loss.parquet does not match D1's rows: run dp2-ml mask-loss d1 again")
    keep = mask_loss.cut_pass(t["lost_flux_frac"], t["lost_flux_sigma"], sel["max_lost_flux_frac"],
                              sel.get("lost_sigma", 3.0))
    return keep, dict(max_lost_flux_frac=float(sel["max_lost_flux_frac"]), lost_sigma=float(sel.get("lost_sigma", 3.0)),
                      kept=float(keep.mean()))


class Batches(keras.utils.PyDataset):
    # Common machinery: row order, epoch shuffling, augmentation draw.
    # block (deblend.train.read_block): training batches are made of blocks of `block` consecutive rows of the
    # store, the blocks shuffled. Rows drawn one by one meant hundreds of scattered reads per batch in arrays of
    # ~1 TB on Lustre (the pilot of 2026-11-26 ran at 5-9 s per step, the GPU waiting); consecutive rows are read
    # in one go. A batch of 128 in blocks of 16 still mixes 8 places of the store (patches).
    # key: position of every dataset row in the compact copy (compact.position_key): training rows are then ordered
    # by position, so that a block is one contiguous run of an already shuffled copy. block "auto": batch_size / 4
    # (a batch mixes 4 runs; with the compact copy only, consecutive rows of the original arrays are one patch).
    def __init__(self, rows, batch_size, train, seed, block=1, key=None, **kw):
        super().__init__(**kw)
        rows = np.asarray(rows, np.int64)
        if not train:
            self.rows_all = rows
        elif key is not None:
            self.rows_all = rows[np.argsort(key[rows], kind="stable")]
        else:
            self.rows_all = np.sort(rows)
        self.batch_size = int(batch_size)
        if block == "auto":
            block = max(self.batch_size // 4, 1) if key is not None else 16
        self.block = max(int(block or 1), 1)
        self.train = bool(train)
        self.seed = int(seed)
        self.epoch = 0
        self.timers = None                  # {phase: seconds} when profiling (deblend-bench, one thread)
        self._order()

    def tick(self, name, t0):
        # Adds the time since t0 to phase `name` when profiling; returns now.
        t = time.perf_counter()
        if self.timers is not None:
            self.timers[name] = self.timers.get(name, 0.0) + t - t0
        return t

    def _order(self):
        if not self.train:
            self.order = self.rows_all
            return
        rng = np.random.default_rng([self.seed, self.epoch])
        if self.block == 1:
            self.order = rng.permutation(self.rows_all)
            return
        n, k = len(self.rows_all), self.block
        idx = (rng.permutation(-(-n // k))[:, None] * k + np.arange(k)).ravel()
        self.order = self.rows_all[idx[idx < n]]

    def __len__(self):
        n = len(self.rows_all) // self.batch_size if self.train else -(-len(self.rows_all) // self.batch_size)
        return max(n, 1)

    def on_epoch_end(self):
        self.epoch += 1
        self._order()

    def rows(self, i):
        return self.order[i * self.batch_size:(i + 1) * self.batch_size]

    def rng(self, i):
        return np.random.default_rng([self.seed, self.epoch, int(i)])

    def augment(self, rng, n):
        if not self.train:
            return np.zeros(n, int), np.zeros(n, bool)
        return rng.integers(0, 4, n), rng.integers(0, 2, n).astype(bool)


def _apply_aug(a, ks, flips, two_d=False):
    # The same rotation / flip on every image of an example; applied per group of examples sharing it (8 groups at
    # most, square stamps) instead of example by example.
    f = aug_mask2d if two_d else aug_images
    a = np.asarray(a)
    out = np.empty_like(a)
    ks, flips = np.asarray(ks, int), np.asarray(flips, bool)
    for k in np.unique(ks):
        for fl in (False, True):
            sel = np.flatnonzero((ks == k) & (flips == fl))
            if len(sel):
                out[sel] = f(a[sel], int(k), bool(fl))
    return out


class Prepare:
    # Turns stored arrays into network inputs, fluxes and likelihood weights.
    def __init__(self, cfg, stats):
        self.norm = Norm(stats)
        e_bands = list(cfg["euclid"]["bands"])
        self.vis = e_bands.index(cfg["deblend"]["euclid_band"])
        # deblend.data.normalise_on: cpu (here, default) or gpu (in the encoders, layers.normalise_input).
        where = str((cfg["deblend"].get("data") or {}).get("normalise_on", "cpu"))
        if where not in ("cpu", "gpu"):
            raise ValueError(f"deblend.data.normalise_on must be cpu or gpu, not {where!r}")
        self.on_gpu = where == "gpu"
        from . import layers as _layers
        _layers.INPUT_NORM.clear()
        if self.on_gpu:
            _layers.INPUT_NORM.update(rubin=self.norm.r_scale, euclid=self.norm.e_scale)

    def _norm(self, f, scale):
        return f.astype(np.float32) if self.on_gpu else asinh_norm(f, scale)

    # Everything derives from (flux, w), w = likelihood weight (compact.py): the stored arrays and the compact copies
    # give the same batches.
    # filled: the pixels of weight 0 already hold a sky-noise draw (compact copies): the input is asinh(flux) only.
    def rubin_w(self, f, w, rng, filled=False):
        x = self._norm(f if filled else noise_fill(f, w, ~(w > 0), rng), self.norm.r_scale)
        return x, f, w

    def euclid_w(self, f, w, rng, filled=False):
        ok = w > 0
        x = self._norm(f if filled else noise_fill(f, w, ~ok, rng), self.norm.e_scale)
        return np.concatenate([x, ok.astype(np.float32)], axis=-1), f, w

    def rubin(self, f, iv, m, rng):
        return self.rubin_w(*rubin_fw(f, iv, m), rng)

    def euclid(self, f, iv, m, fill, rng):
        return self.euclid_w(*euclid_fw(f, iv, m, fill, self.vis), rng)


# Metadata columns a training needs (the evaluations read every column): split, the selections, the D2 targets.
D1_TRAIN_COLUMNS = ("split", "object_id")
D2_TRAIN_COLUMNS = ("split", "group_id", "euclid_object_id", "rc_offset_x_px", "rc_offset_y_px", "euclid_vis_mag",
                    "sep_delta_e", "sep_centroid_shift_arcsec", "sep_rel_size_change", "sep_rel_flux_change")


class D1Batches(Batches):
    # D1 examples: Rubin isolated R and Euclid isolated E. fields: which of "rubin", "euclid" to read.
    # meta_columns: metadata columns to load (None: all), shared between the loaders of one process.
    def __init__(self, cfg, split, batch_size, train, seed=0, indices=None, fields=("rubin", "euclid"),
                 mask_pool=None, mask_prob=0.0, meta_columns=None, **kw):
        self.st = Store(cfg.dataset_dir("d1"), meta_columns, shared=True)
        self.keep, self.selection = d1_keep(cfg, self.st.meta)
        rows = np.flatnonzero((self.st.meta["split"] == split).to_numpy() & self.keep) if indices is None else indices
        super().__init__(rows, batch_size, train, seed, key=position_key(cfg, "d1"), **kw)
        self.prep = Prepare(cfg, load_norm_stats(cfg))
        self.fields = tuple(fields)
        self.has_fill = "euclid_fill" in self.st.names()
        if "euclid" in self.fields and not self.has_fill and cfg["deblend"].get("require_d1_euclid_fill", True):
            raise RuntimeError("D1 has no euclid_fill array: rebuild D1 with dataset1.euclid_neighbour_mask, or set "
                               "deblend.require_d1_euclid_fill: false (Euclid neighbours then count as target)")
        self.mask_pool = mask_pool          # (M, n_e, n_e) bool: real D2 neighbour masks
        self.mask_prob = float(mask_prob) if train else 0.0
        self.sides = open_sides(cfg, "d1", self.rows_all, train)

    def __getitem__(self, i):
        rows = self.rows(i)
        rng = self.rng(i)
        ks, fl = self.augment(rng, len(rows))
        x = {}
        if "rubin" in self.fields:
            if "rubin" in self.sides:
                t = time.perf_counter()
                fw = self.sides["rubin"].take(rows)
                t = self.tick("read", t)
                fw = _apply_aug(fw, ks, fl)
                t = self.tick("augment", t)
                h = fw.shape[-1] // 2
                x["r_in"], x["r_flux"], x["r_w"] = self.prep.rubin_w(fw[..., :h], fw[..., h:], rng, filled=True)
                self.tick("prepare", t)
            else:
                f, iv, m = (_apply_aug(self.st.take(k, rows), ks, fl)
                            for k in ("rubin_flux", "rubin_ivar", "rubin_mask"))
                x["r_in"], x["r_flux"], x["r_w"] = self.prep.rubin(f, iv, m, rng)
        if "euclid" in self.fields:
            filled = "euclid" in self.sides
            if filled:
                t = time.perf_counter()
                fw = self.sides["euclid"].take(rows)
                t = self.tick("read", t)
                fw = _apply_aug(fw, ks, fl)
                self.tick("augment", t)
                f, w = fw[..., :1], fw[..., 1:]
            else:
                f, iv, m = (_apply_aug(self.st.take(k, rows), ks, fl) for k in ("euclid_flux", "euclid_ivar",
                                                                              "euclid_mask"))
                fill = _apply_aug(self.st.take("euclid_fill", rows), ks, fl, two_d=True) if self.has_fill \
                    else None
                f, w = euclid_fw(f, iv, m, fill, self.prep.vis)
            if self.mask_pool is not None and self.mask_prob > 0:
                use = rng.random(len(rows)) < self.mask_prob
                pick = rng.integers(0, len(self.mask_pool), len(rows))
                extra = np.where(use[:, None, None], self.mask_pool[pick], False)
                w = np.where(extra[..., None], 0.0, w).astype(np.float32)
                x["e_aug_masked"] = use.astype(np.float32)
                filled = filled and not use.any()        # the added masked pixels need their noise draw
            t = time.perf_counter()
            x["e_in"], x["e_flux"], x["e_w"] = self.prep.euclid_w(f, w, rng, filled=filled)
            self.tick("prepare", t)
        return x, np.zeros((len(rows),), np.float32)


class D2Batches(Batches):
    # D2 examples: Rubin blend X (input) and the masked Euclid central E_c (teacher input / likelihood target).
    def __init__(self, cfg, split, batch_size, train, seed=0, indices=None, fields=("blend", "euclid"),
                 meta_columns=None, **kw):
        self.st = Store(cfg.dataset_dir("d2"), meta_columns, shared=True)
        self.centering = cfg["deblend"]["d2_centering"]
        keep, self.selection = d2_keep(cfg, self.st.meta)
        rows = d2_rows(self.st.meta, split, self.centering, keep) if indices is None else indices
        super().__init__(rows, batch_size, train, seed, key=position_key(cfg, "d2"), **kw)
        self.prep = Prepare(cfg, load_norm_stats(cfg))
        self.fields = tuple(fields)
        sfx = "_rc" if self.centering == "rubin" else ""
        self.rkeys = tuple(f"rubin_{q}{sfx}" for q in ("flux", "ivar", "mask"))
        self.sides = open_sides(cfg, "d2", self.rows_all, train)

    def __getitem__(self, i):
        rows = self.rows(i)
        rng = self.rng(i)
        ks, fl = self.augment(rng, len(rows))
        x = {}
        if "blend" in self.fields:
            if "rubin" in self.sides:
                t = time.perf_counter()
                fw = self.sides["rubin"].take(rows)
                t = self.tick("read", t)
                fw = _apply_aug(fw, ks, fl)
                t = self.tick("augment", t)
                h = fw.shape[-1] // 2
                x["x_in"], x["x_flux"], x["x_w"] = self.prep.rubin_w(fw[..., :h], fw[..., h:], rng, filled=True)
                self.tick("prepare", t)
            else:
                f, iv, m = (_apply_aug(self.st.take(k, rows), ks, fl) for k in self.rkeys)
                x["x_in"], x["x_flux"], x["x_w"] = self.prep.rubin(f, iv, m, rng)
        if "euclid" in self.fields:
            if "euclid" in self.sides:
                t = time.perf_counter()
                fw = self.sides["euclid"].take(rows)
                t = self.tick("read", t)
                fw = _apply_aug(fw, ks, fl)
                t = self.tick("augment", t)
                x["e_in"], x["e_flux"], x["e_w"] = self.prep.euclid_w(fw[..., :1], fw[..., 1:], rng, filled=True)
                self.tick("prepare", t)
            else:
                f, iv, m = (_apply_aug(self.st.take(k, rows), ks, fl) for k in ("euclid_flux", "euclid_ivar",
                                                                              "euclid_mask"))
                fill = _apply_aug(self.st.take("euclid_fill", rows), ks, fl, two_d=True)
                x["e_in"], x["e_flux"], x["e_w"] = self.prep.euclid(f, iv, m, fill, rng)
        return x, np.zeros((len(rows),), np.float32)


class Mixed(keras.utils.PyDataset):
    # Interleaves two datasets batch by batch (D2 and the D1 replay): batch i of the mix = batch i of each, keys
    # prefixed d2_ / d1_. Length = that of the first.
    def __init__(self, first, second, **kw):
        super().__init__(**kw)
        self.a, self.b = first, second

    def __len__(self):
        return len(self.a)

    def on_epoch_end(self):
        self.a.on_epoch_end()
        self.b.on_epoch_end()

    def __getitem__(self, i):
        xa, ya = self.a[i]
        xb, _ = self.b[i % len(self.b)]
        x = {f"d2_{k}": v for k, v in xa.items()}
        x.update({f"d1_{k}": v for k, v in xb.items()})
        return x, ya


def d2_mask_pool(cfg, n=5000, seed=0):
    # Real neighbour masks of D2 train examples (euclid_fill), for the masking augmentation / robustness test.
    st = Store(cfg.dataset_dir("d2"))
    rows = np.flatnonzero((st.meta["split"] == "train").to_numpy())
    if len(rows) == 0:
        raise RuntimeError("no D2 train example for the mask pool")
    rows = np.sort(np.random.default_rng(seed).choice(rows, min(n, len(rows)), replace=False))
    return st.take("euclid_fill", rows)
