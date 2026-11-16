# blends.py
# Anna Preto
# AstroDeep Project
# Controlled blends for evaluation: a D1 Rubin stamp (target) plus a scaled, shifted D1 Rubin stamp of another
# galaxy of the same patch (companion), in the validation and test splits only. Unlike D2, the isolated target is
# known pixel by pixel, so deblending can be scored against truth. The Euclid view of the target is read from D1
# through d1_row (not copied).
# The companion's noise is scaled with it and added to the target's: the blend is noisier than a real blend of the
# same two galaxies by the companion's noise; noise_excess_rms records by how much.

import json

import numpy as np
import pandas as pd

from . import store


def shift_zero(a, dx, dy):
    # a (..., H, W, C) moved by (dx columns, dy rows); what enters from the border is 0.
    out = np.zeros_like(a)
    H, W = a.shape[-3], a.shape[-2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[..., yd, xd, :] = a[..., ys, xs, :]
    return out


def make_one(t, c, dx, dy, ratio):
    # t, c: dicts of flux, ivar, mask (H, W, C). Returns blend flux, ivar, mask.
    cf = shift_zero(c["flux"], dx, dy) * ratio
    civ = shift_zero(c["ivar"], dx, dy)
    cm = shift_zero(c["mask"].astype(np.uint8), dx, dy).astype(bool)
    with np.errstate(divide="ignore"):
        var = np.where(t["ivar"] > 0, 1.0 / t["ivar"], np.inf) + np.where(civ > 0, ratio ** 2 / civ, 0.0)
    ivar = np.where(np.isfinite(var), 1.0 / var, 0.0).astype(np.float32)
    return (t["flux"] + cf).astype(np.float32), ivar, t["mask"] | cm


def draw(meta, split, n, cb, rng):
    # (target, companion) rows of D1 in the same patch, integer shift with radius in the range, flux ratio.
    rows = np.flatnonzero((meta["split"] == split).to_numpy())
    by_patch = meta.iloc[rows].groupby(["tract", "patch"]).indices
    groups = [rows[v] for v in by_patch.values() if len(v) >= 2] if cb["same_patch"] else [rows]
    if not groups:
        return pd.DataFrame()
    sizes = np.array([len(g) for g in groups], float)
    r0, r1 = cb["separation_px"]
    q0, q1 = cb["flux_ratio"]
    out = []
    while len(out) < n:
        g = groups[rng.choice(len(groups), p=sizes / sizes.sum())]
        t, c = rng.choice(g, size=2, replace=False)
        rad, ang = rng.uniform(r0, r1), rng.uniform(0, 2 * np.pi)
        dx, dy = int(np.rint(rad * np.cos(ang))), int(np.rint(rad * np.sin(ang)))
        if (dx, dy) == (0, 0):
            continue
        out.append(dict(d1_row=int(t), companion_row=int(c), dx=dx, dy=dy, separation_px=float(np.hypot(dx, dy)),
                        flux_ratio=float(rng.uniform(q0, q1)), split=split))
    return pd.DataFrame(out)


def build(cfg):
    d1 = store.Store(cfg.dataset_dir("d1"))
    cb = cfg["controlled_blends"]
    rng = np.random.default_rng(int(cfg["seed"]) + 7)
    parts = [draw(d1.meta, s, int(n), cb, rng) for s, n in cb["n_per_split"].items()]
    plan = pd.concat([p for p in parts if len(p)], ignore_index=True)
    root = cfg.dataset_dir("blends")
    shard = root / "shards" / "all"
    flux, ivar, mask, extra = [], [], [], []
    for s in range(0, len(plan), 1024):
        p = plan.iloc[s:s + 1024]
        T = {k: d1.take(f"rubin_{k}", p["d1_row"].to_numpy()) for k in ("flux", "ivar", "mask")}
        C = {k: d1.take(f"rubin_{k}", p["companion_row"].to_numpy()) for k in ("flux", "ivar", "mask")}
        for i, r in enumerate(p.itertuples(index=False)):
            f, iv, m = make_one({k: v[i] for k, v in T.items()}, {k: v[i] for k, v in C.items()}, r.dx, r.dy,
                                r.flux_ratio)
            flux.append(f)
            ivar.append(iv)
            mask.append(m)
            good = (T["ivar"][i] > 0) & (iv > 0)
            extra.append(float(np.sqrt(np.median((1 / iv[good]) - (1 / T["ivar"][i][good])))) if good.any() else np.nan)
    plan["noise_excess_rms"] = extra
    keep = ["object_id", "euclid_object_id", "tract", "patch", "ra", "dec", "i_cModel_mag", "snr_i", "sigma_hsm",
            "e1", "e2", "psf_sigma_i"]
    tm = d1.meta.iloc[plan["d1_row"].to_numpy()][[c for c in keep if c in d1.meta]].reset_index(drop=True)
    cm = d1.meta.iloc[plan["companion_row"].to_numpy()][["object_id", "i_cModel_mag"]].reset_index(drop=True)
    meta = pd.concat([plan.reset_index(drop=True), tm, cm.add_prefix("companion_")], axis=1)
    store.write_shard(shard, meta, dict(rubin_flux=np.stack(flux), rubin_ivar=np.stack(ivar), rubin_mask=np.stack(mask)))
    n = store.merge_shards([shard], root, dict(name="controlled_blends", source="d1",
                                               config_fingerprint=cfg.fingerprint(["controlled_blends", "seed"]),
                                               d1_fingerprint=d1.info.get("config_fingerprint")))
    (root / "plan.json").write_text(json.dumps(cb, indent=2))
    print(f"controlled blends: {n:,} -> {root}")
    return n
