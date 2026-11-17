# data.py
# Anna Preto
# AstroDeep Project
# Batches for Stage 1 (D1) and Stage 2 (D2), read from the memory-mapped paired datasets, as keras PyDatasets
# (backend agnostic, as in debvader-keras).
#
# Normalisations:
#   Rubin   nJy -> VAE units (units.py) -> DebVAder nonlinear normalisation with paper_baseline's stats.json;
#           masked pixels set to rubin.masked_value_normalised and given zero weight.
#   Euclid  tanh(arcsinh(2.5 f / s_b)), s_b from euclid_stats.json (D1 train split); masked pixels -> 0.
#   PSF     conditioning features standardised with cond_stats.json (D1 train split).
# Augmentation (train only): the same random rotation by k x 90 degrees and flip on every image of an example
# (Euclid and Rubin, which share the north-up / east-left orientation), with the PSF ellipticity and the partner
# shift transformed accordingly; optional elliptical holes in the Euclid input, filled with its own sky.

import json

import keras
import numpy as np

from ..pairs.store import Store


# ----------------------------------------------------------------------------- geometry of the augmentation
def aug_images(a, k, flip):
    # a (..., H, W, C): np.rot90 k times in the (H, W) plane, then flip of the columns.
    a = np.rot90(a, k, axes=(-3, -2))
    return np.flip(a, axis=-2) if flip else a


def aug_mask2d(a, k, flip):
    a = np.rot90(a, k, axes=(-2, -1))
    return np.flip(a, axis=-1) if flip else a


def aug_ellipticity(e1, e2, k, flip):
    # Distortion of a shape measured with x = column, y = row: a quarter turn swaps the axes (e1 -> -e1) and
    # reverses the cross moment (e2 -> -e2); a flip of the columns reverses the cross moment only.
    s = -1.0 if k % 2 else 1.0
    return e1 * s, e2 * s * (-1.0 if flip else 1.0)


def aug_shift(dx, dy, k, flip):
    # Displacement (dx columns, dy rows) under the same operations. np.rot90 sends (row r, col c) to
    # (H - 1 - c, r): (dx, dy) -> (dy, -dx).
    for _ in range(k % 4):
        dx, dy = dy, -dx
    return (-dx if flip else dx), dy


def cond_indices(columns):
    # Positions of the e1 / e2 features in the conditioning vector, paired band by band.
    e1 = [i for i, c in enumerate(columns) if c.startswith("psf_e1_")]
    e2 = [columns.index("psf_e2_" + columns[i][len("psf_e1_"):]) for i in e1]
    return np.array(e1, int), np.array(e2, int)


# ----------------------------------------------------------------------------- normalisation
class Normaliser:
    def __init__(self, cfg, vae, dataset_dir):
        from .units import UnitConverter
        self.units = UnitConverter.for_vae(cfg, vae)
        self.vae = vae
        self.masked_value = float(cfg["rubin"]["masked_value_normalised"])
        es = json.loads((dataset_dir / "euclid_stats.json").read_text())["nonlinear"]
        self.euclid_bands = list(cfg["euclid"]["bands"])
        self.encoder_bands = list(cfg["euclid"]["encoder_bands"])
        self.e_idx = [self.euclid_bands.index(b) for b in self.encoder_bands]
        self.e_scale = np.asarray(es["scale_per_band"], np.float32)[self.e_idx]
        if not np.all(np.isfinite(self.e_scale) & (self.e_scale > 0)):
            raise ValueError(f"invalid Euclid normalisation scale {self.e_scale} for bands {self.encoder_bands}")
        cs = json.loads((dataset_dir / "cond_stats.json").read_text())
        self.cond_columns = list(cfg["dataset1"]["cond_features"])
        if cs["columns"] != self.cond_columns:
            raise ValueError("cond_stats.json was computed for other conditioning features")
        self.cond_mean = np.asarray(cs["mean"], np.float32)
        self.cond_std = np.asarray(cs["std"], np.float32)
        self.e1_idx, self.e2_idx = cond_indices(self.cond_columns)

    def rubin(self, flux_njy, ivar_njy, mask):
        f = self.units.flux(flux_njy)
        iv = self.units.ivar(ivar_njy)
        x = self.vae.normalise(f).astype(np.float32)
        x[mask] = self.masked_value
        w = (~mask).astype(np.float32)
        iv = np.where(mask, 0.0, iv).astype(np.float32)
        return x, w, f.astype(np.float32), iv

    def euclid(self, flux, mask):
        f = flux[..., self.e_idx]
        m = mask[..., self.e_idx]
        x = np.tanh(np.arcsinh(2.5 * f / self.e_scale)).astype(np.float32)
        x[m] = 0.0
        return x

    def cond(self, raw):
        raw = np.asarray(raw, np.float32)
        out = (raw - self.cond_mean) / self.cond_std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def elliptical_holes(x, rng, max_holes=2, rmin=3.0, rmax=15.0, keep_center=6.0):
    # Holes on x (H, W, C, normalised) filled with pixels of the stamp's outer region; never on the centre.
    H, W = x.shape[:2]
    rows, cols = np.indices((H, W))
    c = (H - 1) / 2.0
    outer = np.hypot(rows - c, cols - c) > 0.4 * H
    pool = np.flatnonzero(outer.ravel())
    x = x.copy()
    for _ in range(rng.integers(1, max_holes + 1)):
        a = rng.uniform(rmin, rmax)
        b = a * rng.uniform(0.3, 1.0)
        th = rng.uniform(0, np.pi)
        d = rng.uniform(keep_center + a, 0.5 * H)
        ang = rng.uniform(0, 2 * np.pi)
        x0, y0 = c + d * np.cos(ang), c + d * np.sin(ang)
        u = (cols - x0) * np.cos(th) + (rows - y0) * np.sin(th)
        v = -(cols - x0) * np.sin(th) + (rows - y0) * np.cos(th)
        hole = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        if not hole.any():
            continue
        pick = rng.choice(pool, size=int(hole.sum()))
        for ch in range(x.shape[-1]):
            x[..., ch][hole] = x[..., ch].ravel()[pick]
    return x


# ----------------------------------------------------------------------------- Stage 1 (D1)
class Stage1Batches(keras.utils.PyDataset):
    def __init__(self, cfg, vae, split, batch_size, train, seed=0, indices=None, **kwargs):
        super().__init__(**kwargs)
        self.st = Store(cfg.dataset_dir("d1"))
        self.norm = Normaliser(cfg, vae, self.st.path)
        idx = np.flatnonzero((self.st.meta["split"] == split).to_numpy()) if indices is None else np.asarray(indices)
        self.idx = idx
        self.batch_size = int(batch_size)
        self.train = bool(train)
        s1 = cfg["stage1"]
        self.flips = bool(s1["augment_flips"]) and self.train
        self.hole_prob = float(s1["mask_augment_prob"]) if self.train else 0.0
        self.cond_raw = self.st.meta[self.norm.cond_columns].to_numpy(np.float32)
        self.rng = np.random.default_rng(seed)
        self.order = self.idx.copy()
        if self.train:
            self.rng.shuffle(self.order)

    def __len__(self):
        n = len(self.order) // self.batch_size if self.train else int(np.ceil(len(self.order) / self.batch_size))
        return max(n, 1)

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.order)

    def rows(self, i):
        return self.order[i * self.batch_size:(i + 1) * self.batch_size]

    def __getitem__(self, i):
        rows = self.rows(i)
        rf, ri, rm = (self.st.take(k, rows) for k in ("rubin_flux", "rubin_ivar", "rubin_mask"))
        ef, em = self.st.take("euclid_flux", rows), self.st.take("euclid_mask", rows)
        cond = self.cond_raw[rows].copy()
        xr, wr, fr, ivr = self.norm.rubin(rf, ri, rm)
        xe = self.norm.euclid(ef, em)
        if self.flips or self.hole_prob > 0:
            # Per-example rng, seeded by epoch-independent draws of the shared generator.
            for j in range(len(rows)):
                if self.flips:
                    k, flip = int(self.rng.integers(4)), bool(self.rng.integers(2))
                    xr[j], wr[j], fr[j], ivr[j] = (np.ascontiguousarray(aug_images(a, k, flip))
                                                   for a in (xr[j], wr[j], fr[j], ivr[j]))
                    xe[j] = np.ascontiguousarray(aug_images(xe[j], k, flip))
                    e1, e2 = aug_ellipticity(cond[j, self.norm.e1_idx], cond[j, self.norm.e2_idx], k, flip)
                    cond[j, self.norm.e1_idx], cond[j, self.norm.e2_idx] = e1, e2
                if self.hole_prob > 0 and self.rng.random() < self.hole_prob:
                    xe[j] = elliptical_holes(xe[j], self.rng)
        x = dict(euclid=xe, cond=self.norm.cond(cond), rubin_norm=xr, rubin_weight=wr, rubin_flux=fr,
                 rubin_ivar=ivr)
        return x, np.zeros((len(rows),), np.float32)


# ----------------------------------------------------------------------------- Stage 2 (D2)
def index_map(n, dx, dy):
    # Flat indices that move an n x n image by (dx, dy): out[p] = img[idx[p]] where valid[p].
    rows, cols = np.indices((n, n))
    sr, sc = rows - dy, cols - dx
    valid = (sr >= 0) & (sr < n) & (sc >= 0) & (sc < n)
    return np.where(valid, sr * n + sc, 0).ravel().astype(np.int32), valid.ravel().astype(np.float32)


class Stage2Batches(keras.utils.PyDataset):
    # x: the Rubin blend (student input) and the Euclid view of the target (teacher input); for the scene loss,
    # the Euclid view of the partner and the map that places its reconstruction in the target's stamp.
    def __init__(self, cfg, vae, split, batch_size, train, seed=0, dataset="d2", indices=None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.st = Store(cfg.dataset_dir(dataset))
        self.norm = Normaliser(cfg, vae, cfg.dataset_dir("d1"))
        meta = self.st.meta
        self.idx = np.flatnonzero((meta["split"] == split).to_numpy()) if indices is None else np.asarray(indices)
        self.batch_size = int(batch_size)
        self.train = bool(train)
        self.flips = bool(cfg["stage1"]["augment_flips"]) and self.train
        suffix = "_rc" if cfg["stage2"]["centering"] == "rubin" else ""
        if suffix and "rubin_flux_rc" not in self.st.names():
            raise ValueError(f"{dataset} has no Rubin-centred stamps")
        self.keys = (f"rubin_flux{suffix}", f"rubin_ivar{suffix}", f"rubin_mask{suffix}")
        self.cond_raw = meta[self.norm.cond_columns].to_numpy(np.float32)
        self.scene = bool(cfg["stage2"]["scene"]["enabled"]) and "partner_kept" in meta
        if self.scene and suffix:
            raise ValueError("the scene loss uses partner offsets measured from the Euclid-centred stamp: it cannot "
                             "be combined with stage2.centering = rubin")
        if self.scene:
            pos = {int(v): i for i, v in enumerate(meta["euclid_object_id"])}
            self.partner = np.array([pos.get(int(p), -1) if k else -1 for p, k in
                                     zip(meta["partner_euclid_object_id"], meta["partner_kept"])], int)
            self.scene_ok = (meta["scene_complete"].to_numpy(bool) & (self.partner >= 0)
                             & (meta["split"].to_numpy() == meta["split"].to_numpy()[np.maximum(self.partner, 0)]))
            self.pdx = meta["partner_dx_px"].to_numpy(int)
            self.pdy = meta["partner_dy_px"].to_numpy(int)
        self.rng = np.random.default_rng(seed)
        self.order = self.idx.copy()
        if self.train:
            self.rng.shuffle(self.order)

    def __len__(self):
        n = len(self.order) // self.batch_size if self.train else int(np.ceil(len(self.order) / self.batch_size))
        return max(n, 1)

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.order)

    def __getitem__(self, i):
        rows = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        rf, ri, rm = (self.st.take(k, rows) for k in self.keys)
        xr, wr, fr, ivr = self.norm.rubin(rf, ri, rm)
        xe = self.norm.euclid(self.st.take("euclid_flux", rows), self.st.take("euclid_mask", rows))
        cond = self.cond_raw[rows].copy()
        B, n = len(rows), xr.shape[1]
        if self.scene:
            prow = self.partner[rows]
            ok = self.scene_ok[rows]
            safe = np.where(prow >= 0, prow, rows)
            xp = self.norm.euclid(self.st.take("euclid_flux", safe), self.st.take("euclid_mask", safe))
            pcond = self.cond_raw[safe].copy()
            pidx = np.zeros((B, n * n), np.int32)
            pval = np.zeros((B, n * n), np.float32)
            dxs, dys = self.pdx[rows].copy(), self.pdy[rows].copy()
        for j in range(B):
            k, flip = (int(self.rng.integers(4)), bool(self.rng.integers(2))) if self.flips else (0, False)
            if self.flips:
                xr[j], wr[j], fr[j], ivr[j] = (np.ascontiguousarray(aug_images(a, k, flip))
                                               for a in (xr[j], wr[j], fr[j], ivr[j]))
                xe[j] = np.ascontiguousarray(aug_images(xe[j], k, flip))
                cond[j, self.norm.e1_idx], cond[j, self.norm.e2_idx] = aug_ellipticity(
                    cond[j, self.norm.e1_idx], cond[j, self.norm.e2_idx], k, flip)
            if self.scene:
                if self.flips:
                    xp[j] = np.ascontiguousarray(aug_images(xp[j], k, flip))
                    pcond[j, self.norm.e1_idx], pcond[j, self.norm.e2_idx] = aug_ellipticity(
                        pcond[j, self.norm.e1_idx], pcond[j, self.norm.e2_idx], k, flip)
                    dxs[j], dys[j] = aug_shift(int(dxs[j]), int(dys[j]), k, flip)
                pidx[j], pval[j] = index_map(n, int(dxs[j]), int(dys[j]))
        x = dict(rubin_norm=xr, rubin_weight=wr, rubin_flux=fr, rubin_ivar=ivr, euclid=xe,
                 cond=self.norm.cond(cond))
        if self.scene:
            x.update(partner_euclid=xp, partner_cond=self.norm.cond(pcond), partner_index=pidx,
                     partner_valid=pval * ok[:, None].astype(np.float32), scene_ok=ok.astype(np.float32))
        return x, np.zeros((B,), np.float32)
