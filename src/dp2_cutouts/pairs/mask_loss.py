# mask_loss.py
# Anna Preto
# AstroDeep Project
# Target light hidden by the neighbour masks, and a selection on it (decided 2026-11-27): a pair whose masks hide
# more than max_lost_flux_frac of the target's own light, by more than lost_sigma times its uncertainty, is left out
# of the training and evaluation reads (D1 and D2 themselves unchanged).
#   D1   on the Rubin i stamp: the pipeline's neighbour mask (companion_mask of the patch's 30_processed/stamps.parquet;
#        the masked pixels are replaced by sky in D1, the estimate never reads them);
#   D2   on the VIS cutout of the target: the neighbour mask euclid_fill, raw image euclid_flux_unmasked.
# Estimate (as scripts/inspect/mask_damage.lost_sym, validated by injecting real D2 masks on isolated D1 galaxies):
# the visible target is the connected region of the smoothed image (masked pixels set to 0, Gaussian of SMOOTH_PX)
# above DETECT_SIGMA times the smoothed noise that reaches the stamp centre; a masked pixel near it whose mirror
# through the centre is visible target is given the light of its mirror; lost_flux_frac = lost / (visible + lost).
# lost_flux_sigma: first-order uncertainty from the pixel noise (sums of smoothed pixels, kernel autocorrelation);
# on noise draws it overestimates the scatter by 30-40 % for faint and intermediate targets (the cut is cautious
# there) and underestimates it for bright ones, where it is ~0.1 % against the 3 % threshold.
# Outputs: <dataset>/mask_loss.parquet with object_id (D1) / euclid_object_id (D2), lost_flux_frac, lost_flux_sigma,
# lost_unknown (share of the masked pixels near the target whose mirror is masked too: the loss is a lower bound).

import json
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import ndimage

FILE = "mask_loss.parquet"
SMOOTH_PX = 1.0
NOISE_GAIN = 1.0 / (2.0 * np.sqrt(np.pi) * SMOOTH_PX)
DETECT_SIGMA = 2.0


def support(img, valid, fill, sig):
    # Visible light of the target (stamps of odd size centred on it): None when nothing reaches the centre.
    n = img.shape[0]
    c = (n - 1) / 2
    sm = ndimage.gaussian_filter(np.where(valid & ~fill, np.nan_to_num(img), 0.0), SMOOTH_PX)
    lab, _ = ndimage.label(sm > DETECT_SIGMA * NOISE_GAIN * sig, structure=np.ones((3, 3)))
    yy, xx = np.indices(img.shape)
    r = np.hypot(xx - c, yy - c)
    near = (lab > 0) & (r <= 3)
    if not near.any():
        return None, sm
    t = (lab == lab.flat[np.argmin(np.where(near, r, np.inf))]) & ~fill
    return t, sm


def lost_light(img, valid, fill, sig):
    # (lost_flux_frac, lost_unknown, lost_flux_sigma); NaN without a visible target.
    t, sm = support(img, valid, fill, sig)
    if t is None:
        return np.nan, np.nan, np.nan
    near = fill & ndimage.binary_dilation(t, iterations=4)
    known = near & t[::-1, ::-1]
    pos = np.clip(sm, 0, None)
    lost = float(pos[::-1, ::-1][known].sum())
    vis = float(pos[t].sum())
    unknown = float((near & fill[::-1, ::-1]).sum()) / max(float(near.sum()), 1.0) if near.any() else 0.0

    def var_sum(sel):
        w = sel.astype(float)
        return float((ndimage.gaussian_filter(w, SMOOTH_PX * np.sqrt(2.0)) * w).sum()) * sig ** 2

    total = max(vis + lost, 1e-30)
    frac = lost / total
    var_f = ((1 - frac) ** 2 * var_sum(known[::-1, ::-1]) + frac ** 2 * var_sum(t)) / total ** 2
    return frac, unknown, float(np.sqrt(var_f))


def stamp_losses(flux, ivar, fill):
    # flux, ivar (B, n, n) of one band, fill (B, n, n) bool -> (B, 3) as lost_light; the noise of each stamp is the
    # median pixel noise (stamps of ~12": nearly uniform depth).
    out = np.full((len(flux), 3), np.nan)
    for k in range(len(flux)):
        iv = ivar[k]
        ok = (iv > 0) & np.isfinite(iv) & np.isfinite(flux[k])
        if not ok.any():
            continue
        sig = float(np.median(1.0 / np.sqrt(iv[ok])))
        out[k] = lost_light(flux[k], ok, fill[k].astype(bool), sig)
    return out


# ----------------------------------------------------------------------------- D1 (Rubin i)
def patch_masks(pcfg, patch, object_ids):
    # companion_mask (n, n) of the given objects from the patch's stamps.parquet: {object_id: mask}.
    import pyarrow.parquet as pq

    from ..pipeline.patches import patch_case
    case = patch_case(pcfg, patch)
    t = pq.read_table(case.data("30_processed", "stamps.parquet"), columns=["object_id", "companion_mask"],
                      filters=[("object_id", "in", [int(i) for i in object_ids])]).to_pandas()
    n = int(case["cuts"]["stamp_px"])
    return {int(r.object_id): np.asarray(r.companion_mask, bool).reshape(n, n) for r in t.itertuples(index=False)}


def _d1_shard(job):
    root, ib, rows, ids, pcfg, patch, masks_for = job
    masks = masks_for(pcfg, patch, ids)                  # read here: all of D1's masks would not fit in one process
    from .store import Store
    st = Store(root)
    sel = np.asarray(rows)
    flux = np.asarray(st.array("rubin_flux")[sel[0]:sel[-1] + 1])[sel - sel[0], ..., ib]
    ivar = np.asarray(st.array("rubin_ivar")[sel[0]:sel[-1] + 1])[sel - sel[0], ..., ib]
    n = flux.shape[1]
    fill = np.stack([masks.get(int(i), np.zeros((n, n), bool)) for i in ids])
    have = np.array([int(i) in masks for i in ids])
    out = stamp_losses(flux, ivar, fill)
    out[~have] = np.nan                                   # no mask record: unknown, kept by the selection
    return out


def build_d1(cfg, workers=8, masks_for=patch_masks):
    from ..pipeline.patches import load_patches
    root = cfg.dataset_dir("d1")
    meta = pd.read_parquet(root / "meta.parquet", columns=["object_id", "shard"])
    ib = list(cfg["rubin"]["bands"]).index(cfg["rubin"].get("shape_band", "i"))
    pcfg = cfg.pipeline()
    patches = {p["name"]: p for p in load_patches(pcfg)}
    jobs, order = [], []
    for name, g in meta.groupby("shard", sort=False):
        ids = g["object_id"].astype("int64").to_numpy()
        jobs.append((str(root), ib, g.index.to_numpy(), ids, pcfg, patches[name], masks_for))
        order.append(g.index.to_numpy())
    res = _run(_d1_shard, jobs, workers)
    out = np.full((len(meta), 3), np.nan)
    for rows, r in zip(order, res):
        out[rows] = r
    return _write(root, pd.DataFrame(dict(object_id=meta["object_id"].astype("int64"))), out)


# ----------------------------------------------------------------------------- D2 (VIS)
def _d2_chunk(job):
    root, vis, rows = job
    from .store import Store
    st = Store(root)
    sel = np.asarray(rows)
    lo, hi = int(sel[0]), int(sel[-1]) + 1
    flux = np.asarray(st.array("euclid_flux_unmasked")[lo:hi])[sel - lo, ..., vis]
    ivar = np.asarray(st.array("euclid_ivar")[lo:hi])[sel - lo, ..., vis]
    fill = np.asarray(st.array("euclid_fill")[lo:hi])[sel - lo]
    return stamp_losses(flux, ivar, fill)


def build_d2(cfg, rows, workers=8, chunk=2048):
    # rows: the D2 rows to measure (sorted), e.g. those the other criteria of deblend.d2_selection keep.
    root = cfg.dataset_dir("d2")
    meta = pd.read_parquet(root / "meta.parquet", columns=["euclid_object_id"])
    vis = list(cfg["euclid"]["bands"]).index("VIS")
    rows = np.sort(np.asarray(rows, np.int64))
    jobs = [(str(root), vis, rows[k:k + chunk]) for k in range(0, len(rows), chunk)]
    out = np.concatenate(_run(_d2_chunk, jobs, workers)) if jobs else np.zeros((0, 3))
    ids = pd.DataFrame(dict(euclid_object_id=meta["euclid_object_id"].iloc[rows].astype("int64").to_numpy()))
    return _write(root, ids, out)


def _run(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    res = [None] * len(jobs)
    # spawn: the parent may have imported JAX (deblend selection), which must not be forked.
    with mp.get_context("spawn").Pool(int(workers)) as pool:
        for k, r in enumerate(pool.imap(fn, jobs, chunksize=1)):
            res[k] = r
            if (k + 1) % max(len(jobs) // 20, 1) == 0:
                print(f"mask loss: {k + 1}/{len(jobs)}", flush=True)
    return res


def _write(root, ids, out):
    t = ids.assign(lost_flux_frac=out[:, 0], lost_unknown=out[:, 1], lost_flux_sigma=out[:, 2])
    tmp = root / (FILE + ".tmp")
    t.to_parquet(tmp, index=False)
    tmp.replace(root / FILE)
    return t


# ----------------------------------------------------------------------------- selection
def cut_pass(frac, sigma, max_frac, k_sigma):
    # True = kept: not (loss above max_frac and above k_sigma times its uncertainty). Unknown (NaN) is kept.
    frac, sigma = np.asarray(frac, float), np.asarray(sigma, float)
    bad = (frac > float(max_frac)) & (frac > float(k_sigma) * np.nan_to_num(sigma, nan=np.inf))
    return ~bad


def read(cfg, which):
    path = cfg.dataset_dir(which) / FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `dp2-ml <config> mask-loss {which}` (max_lost_flux_frac set)")
    return pd.read_parquet(path)


def summary(t, max_frac=0.03, k_sigma=3.0):
    keep = cut_pass(t["lost_flux_frac"], t["lost_flux_sigma"], max_frac, k_sigma)
    return dict(n=int(len(t)), measured=int(np.isfinite(t["lost_flux_frac"]).sum()),
                above_frac=float((t["lost_flux_frac"] > max_frac).mean()), cut=float((~keep).mean()))


def dumps(d):
    return json.dumps(d, default=float)
