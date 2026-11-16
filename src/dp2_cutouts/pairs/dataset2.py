# dataset2.py
# Anna Preto
# AstroDeep Project
# Dataset 2: Rubin blends resolved by Euclid. One example per Euclid target:
#   - its copaings group holds 2..max_euclid_in_group Euclid objects and at least one Rubin object, and the Rubin
#     side is blended (max blending entropy S_b of the Rubin members > sb_rubin_min);
#   - at Euclid resolution the target is alone: no other Euclid ellipse (VIS PSF) overlaps its own
#     (overlap fraction <= sb_euclid_max);
#   - on the Euclid image, masking every other source (k times its ellipse, holes filled with empty sky of the same
#     cutout) barely changes the target: shape, centroid, size and flux before / after within the accept limits.
# x_R is the Rubin blend stamp (never masked) centred on the Euclid target ("euclid" centering) and, for the
# Rubin-only setting, centred on the Rubin object of the group; x_E is the masked Euclid stamp of the target.
# Every quantity behind a decision is stored; failed targets go to rejected.parquet with their reason.

import json

import numpy as np
import pandas as pd

from ..common import masks
from ..common.ellipse import FWHM2SIGMA, euclid_ellipse, radius_map
from ..sources import euclid_images
from ..sources.rubin_soda import BANDS
from ..stages import stamps as stamp_stage
from ..stages.catalog_table import copaings_dir
from . import common, splits, store

RUBIN_TO_EUCLID_PX = 2.0  # 0.2" / 0.1"


def shard_dir(cfg, name):
    return cfg.dataset_dir("d2") / "shards" / name


def euclid_table(case):
    # copaings Euclid table (row order of the groups) with the MMU identifier, as in catalog_table.
    from ..sources import euclid_hf
    e_cop = pd.read_parquet(copaings_dir(case) / "catalogs" / "euclid.parquet")
    e_box = euclid_hf.read_box(case)
    key = ["mer_catalogue_right_ascension", "mer_catalogue_declination"]
    e_box = e_box.drop_duplicates(subset=key)[key + ["object_id"]]
    e = e_cop.reset_index().rename(columns={"index": "euclid_row"})
    e = e.merge(e_box, left_on=["ra", "dec"], right_on=key, how="left").set_index("euclid_row").sort_index()
    if e["object_id"].isna().any():
        raise RuntimeError("copaings Euclid objects without an MMU identifier")
    e["object_id"] = e["object_id"].astype("int64")
    return e


def euclid_ellipses(e, cfg, pixel_scale):
    # Euclid-resolution ellipses (VIS PSF), semi-axes in pixels of the given scale, theta in degrees.
    ec = cfg["euclid"]
    pa = e["position_angle_true"].to_numpy(float) + float(ec["pa_offset_deg"])
    a, b, th = euclid_ellipse(e["size_true"], e["size_minor_true"], pa, float(ec["vis_psf_fwhm"]),
                              pixel_scale=pixel_scale)
    return np.asarray(a, float), np.asarray(b, float), np.asarray(th, float)


def overlap_fraction(t, others, step):
    # Fraction of ellipse t = (x, y, a, b, theta) covered by the union of `others`, on a grid of the given step.
    x, y, a, b, th = t
    if not (np.isfinite(a) and np.isfinite(b) and b > 0) or not len(others):
        return 0.0
    r = a + 1.0
    gx, gy = np.meshgrid(np.arange(x - r, x + r + step, step), np.arange(y - r, y + r + step, step))
    from ..common.ellipse import elliptical_radius
    inside = elliptical_radius(gx - x, gy - y, a, b, th) <= 1.0
    if not inside.any():
        return 0.0
    cov = np.zeros_like(inside)
    for ox, oy, oa, ob, oth in others:
        if np.isfinite(oa) and np.isfinite(ob) and ob > 0 and np.hypot(ox - x, oy - y) < a + oa + 1:
            cov |= elliptical_radius(gx - ox, gy - oy, oa, ob, oth) <= 1.0
    return float((cov & inside).sum() / inside.sum())


def fill_with_sky(flux, ivar, need, sky, rng):
    # Pixels `need` replaced by values drawn from the sky pixels of the same band (noise level kept, correlations
    # not). Returns copies; ivar of filled pixels is the median sky ivar.
    f, iv = flux.copy(), ivar.copy()
    for k in range(f.shape[-1]):
        pool = np.flatnonzero(sky[..., k].ravel())
        if not need.any() or pool.size < 50:
            continue
        pick = rng.choice(pool, size=int(need.sum()), replace=True)
        f[..., k][need] = f[..., k].ravel()[pick]
        iv[..., k][need] = np.median(iv[..., k].ravel()[pool])
    return f, iv


def euclid_separation(cut, target, others, cfg, rng):
    # Masks the other Euclid sources on the target cutout and measures the target before / after (VIS).
    d2 = cfg["dataset2"]
    eb = list(cfg["euclid"]["bands"])
    f, iv, m, present = euclid_images.to_band_array(cut, eb)
    n = f.shape[0]
    vis = eb.index("VIS")
    w = cut["wcs"]
    k = float(d2["mask_k"])
    need = np.zeros((n, n), bool)
    src = np.zeros((n, n), bool)
    for o in others.itertuples(index=False):
        ox, oy = w.world_to_pixel_values(o.ra, o.dec)
        if not (-o.a_e * k <= ox <= n + o.a_e * k and -o.a_e * k <= oy <= n + o.a_e * k):
            continue
        rm = radius_map((n, n), float(ox), float(oy), o.a_e, o.b_e, o.theta)
        need |= rm <= k
        src |= rm <= 3.0
    tx, ty = w.world_to_pixel_values(target["ra"], target["dec"])
    rt = radius_map((n, n), float(tx), float(ty), target["a_e"], target["b_e"], target["theta"])
    src |= rt <= 3.0
    sky = (~src[..., None]) & ~m
    ff, ivf = fill_with_sky(f, iv, need, sky, rng)
    weight = (~m[..., vis]).astype(np.int32)
    m0 = common.moments(f[..., vis], weight, (float(tx), float(ty)))
    m1 = common.moments(ff[..., vis], weight, (float(tx), float(ty)))
    s = float(cfg["euclid"]["pixel_scale"])
    out = dict(euclid_masked_fraction=float(need.mean()),
               euclid_masked_fraction_in_target=float(need[rt <= 1.0].mean()) if (rt <= 1.0).any() else np.nan,
               sep_e1_before=m0["e1"], sep_e2_before=m0["e2"], e1_vis=m1["e1"], e2_vis=m1["e2"],
               sigma_vis_px=m1["sigma"], hsm_vis_ok=bool(m0["ok"] and m1["ok"]))
    if out["hsm_vis_ok"]:
        out["sep_delta_e"] = float(np.hypot(m1["e1"] - m0["e1"], m1["e2"] - m0["e2"]))
        out["sep_centroid_shift_arcsec"] = float(np.hypot(m1["x"] - m0["x"], m1["y"] - m0["y"]) * s)
        out["sep_rel_size_change"] = float(abs(m1["sigma"] / m0["sigma"] - 1.0))
        out["sep_rel_flux_change"] = float(abs(m1["flux"] / m0["flux"] - 1.0)) if m0["flux"] else np.nan
    else:
        out.update(sep_delta_e=np.nan, sep_centroid_shift_arcsec=np.nan, sep_rel_size_change=np.nan,
                   sep_rel_flux_change=np.nan)
    acc = d2["accept"]
    reasons = []
    if not out["hsm_vis_ok"]:
        reasons.append("hsm_failed")
    else:
        if out["sep_delta_e"] > acc["max_delta_e"]:
            reasons.append("delta_e")
        if out["sep_centroid_shift_arcsec"] > acc["max_centroid_shift_arcsec"]:
            reasons.append("centroid_shift")
        if out["sep_rel_size_change"] > acc["max_rel_size_change"]:
            reasons.append("size_change")
        if not np.isfinite(out["sep_rel_flux_change"]) or out["sep_rel_flux_change"] > acc["max_rel_flux_change"]:
            reasons.append("flux_change")
    if not np.isfinite(out["euclid_masked_fraction_in_target"]) or \
            out["euclid_masked_fraction_in_target"] > acc["max_masked_fraction_in_target"]:
        reasons.append("target_masked")
    return out, reasons, (ff, ivf, m, present, need, f)


def rubin_stamp(img, x, y, n):
    # n x n stamp of the six images centred on the integer pixel nearest to (x, y) (region pixels), or None.
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    h = n // 2
    H, W = img["flux"].shape[1:]
    if not (h <= ix < W - h and h <= iy < H - h):
        return None
    box = np.s_[:, iy - h:iy + h + 1, ix - h:ix + h + 1]
    return dict(flux=img["flux"][box], var=img["var"][box], mask=img["mask"][box], ix=ix, iy=iy,
                dx=float(x - ix), dy=float(y - iy))


def to_hwc(st, bits, mmu_planes):
    nodata = masks.bitmask(bits, ["NO_DATA"])
    with np.errstate(divide="ignore"):
        ivar = np.where((st["var"] > 0) & ((st["mask"] & nodata) == 0), 1.0 / st["var"], 0.0)
    bad = (st["mask"] & masks.bitmask(bits, mmu_planes)) != 0
    mv = lambda a: np.moveaxis(a, 0, -1)  # noqa: E731
    return mv(st["flux"]).astype(np.float32), mv(ivar).astype(np.float32), mv(bad)


def candidate_targets(case, cfg, patch):
    # Euclid targets of the D2 groups owned by the patch, with their group, Rubin members and Euclid-side overlap.
    import pickle
    from ..pipeline.patches import owned

    d2 = cfg["dataset2"]
    cdir = copaings_dir(case)
    groups = pickle.load(open(cdir / "groups" / "E_no_overlap.pkl", "rb"))["data"]
    lsst = pd.read_parquet(cdir / "lsst_sb.parquet")
    e = euclid_table(case)
    a_r, b_r, th_r = euclid_ellipses(e, cfg, common.RUBIN_SCALE)
    e["a_r"], e["b_r"], e["theta"] = a_r, b_r, th_r
    e["a_e"], e["b_e"] = a_r * RUBIN_TO_EUCLID_PX, b_r * RUBIN_TO_EUCLID_PX
    # Local tangent-plane coordinates in Rubin pixels (x towards west, y towards north), for the overlap test.
    ra0, dec0 = float(e["ra"].mean()), float(e["dec"].mean())
    wx, ny = common.sky_offset_arcsec(ra0, dec0, e["ra"].to_numpy(), e["dec"].to_numpy())
    e["px"], e["py"] = wx / common.RUBIN_SCALE, ny / common.RUBIN_SCALE
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[e["px"], e["py"]])

    rows = []
    for gid, g in enumerate(groups):
        e_rows, r_rows = list(g[0]), list(g[1])
        if not (2 <= len(e_rows) <= int(d2["max_euclid_in_group"])) or len(r_rows) < 1:
            continue
        rub = lsst.loc[r_rows]
        sb_rubin = float(np.nanmax(rub["blending_entropy_Euclid"].to_numpy(float)))
        owner = rub.iloc[int(np.argmin(rub["i_cModel_mag"].to_numpy(float)))]
        if not owned(pd.DataFrame([dict(tract=owner["tract_id"], patch=owner["patch"])]), patch).iloc[0]:
            continue
        for er in e_rows:
            t = e.loc[er]
            near = [j for j in tree.query_ball_point([t["px"], t["py"]], r=float(t["a_r"]) + 60.0) if j != er]
            others = [(e["px"].iat[j], e["py"].iat[j], e["a_r"].iat[j], e["b_r"].iat[j], e["theta"].iat[j])
                      for j in near]
            ov = overlap_fraction((t["px"], t["py"], t["a_r"], t["b_r"], t["theta"]), others,
                                  float(d2["overlap_grid_px"]))
            partners = [x for x in e_rows if x != er]
            rows.append(dict(group_id=f"{patch['name']}:{gid}", euclid_row=int(er),
                             euclid_object_id=int(t["object_id"]), euclid_ra=float(t["ra"]),
                             euclid_dec=float(t["dec"]), euclid_vis_mag=float(t["vis_mag"]),
                             n_euclid_in_group=len(e_rows), n_rubin_in_group=len(r_rows), sb_rubin=sb_rubin,
                             euclid_overlap_frac=ov, rubin_object_id=int(owner["objectId"]),
                             rubin_ra=float(owner["ra"]), rubin_dec=float(owner["dec"]),
                             rubin_i_cModel_mag=float(owner["i_cModel_mag"]), tract=int(owner["tract_id"]),
                             patch=int(owner["patch"]),
                             partner_euclid_object_id=int(e.loc[partners[0], "object_id"]) if len(partners) == 1 else -1,
                             partner_vis_mag=float(e.loc[partners[0], "vis_mag"]) if len(partners) == 1 else np.nan,
                             partner_sep_arcsec=float(np.hypot(e.loc[partners[0], "px"] - t["px"],
                                                               e.loc[partners[0], "py"] - t["py"]) * common.RUBIN_SCALE)
                             if len(partners) == 1 else np.nan,
                             group_euclid_rows=",".join(str(int(x)) for x in e_rows)))
    return pd.DataFrame(rows), e


def build_patch(cfg, patch):
    from ..pipeline.patches import patch_case

    pcfg = cfg.pipeline()
    case = patch_case(pcfg, patch)
    d2 = cfg["dataset2"]
    n_r, n_e = int(cfg["rubin"]["stamp_px"]), int(cfg["euclid"]["stamp_px"])
    rb, eb = list(cfg["rubin"]["bands"]), list(cfg["euclid"]["bands"])
    if rb != list(BANDS):
        raise ValueError(f"rubin.bands must be {BANDS} (the order of the pipeline images)")
    mmu_planes = pcfg["package"]["mmu_mask_planes"]

    if not (copaings_dir(case) / "lsst_sb.parquet").exists():
        # Patch not planned (no copaings output): an empty shard records that it was looked at.
        store.write_shard(shard_dir(cfg, patch["name"]), pd.DataFrame(), empty_arrays(n_r, len(rb), n_e, len(eb)),
                          pd.DataFrame([dict(reason="patch_not_planned")]), info=dict(patch=patch["name"]))
        return dict(n_targets=0, n_candidates=0, n_rejected=0)
    cand, e = candidate_targets(case, cfg, patch)
    rejected = []

    def reject(df, reason, detail=None):
        for r in df.itertuples(index=False):
            rejected.append(dict(euclid_object_id=int(r.euclid_object_id), group_id=r.group_id, reason=reason,
                                 detail=detail if detail is None else str(detail)))

    if len(cand):
        blended = cand["sb_rubin"] > float(d2["sb_rubin_min"])
        reject(cand[~blended], "rubin_not_blended")
        cand = cand[blended]
        alone = cand["euclid_overlap_frac"] <= float(d2["sb_euclid_max"])
        reject(cand[~alone], "euclid_overlap")
        cand = cand[alone].reset_index(drop=True)

    # The pipeline fetches pixels only for patches with isolated targets: a patch without images gives no D2 target.
    if len(cand) and not case.data("10_rubin", "images", "images.json", makedirs=False).exists():
        reject(cand, "no_rubin_images")
        cand = cand.iloc[0:0]
    cat = pd.read_parquet(case.data("30_processed", "catalog.parquet")).set_index("object_id")
    img = stamp_stage.load_images(case) if len(cand) else None
    cuts = euclid_images.fetch(cfg["euclid"], pd.DataFrame(dict(
        object_id=cand["euclid_object_id"], ra=cand["euclid_ra"], dec=cand["euclid_dec"])), case.sky_box) \
        if len(cand) else {}

    keep_meta, A = [], {k: [] for k in ("rubin_flux", "rubin_ivar", "rubin_mask", "rubin_flux_rc", "rubin_ivar_rc",
                                        "rubin_mask_rc", "euclid_flux", "euclid_ivar", "euclid_mask",
                                        "euclid_present", "euclid_flux_unmasked", "euclid_fill")}
    cc = pcfg["cuts"]
    for r in cand.itertuples(index=False):
        row = r._asdict()
        cut = cuts.get(int(r.euclid_object_id))
        if cut is None:
            reject(pd.DataFrame([row]), "no_euclid_image")
            continue
        if "VIS" not in cut["bands"]:
            reject(pd.DataFrame([row]), "no_vis_band")
            continue
        rng = np.random.default_rng([int(cfg["seed"]), int(r.euclid_object_id) & 0x7FFFFFFF])
        tgt = dict(ra=r.euclid_ra, dec=r.euclid_dec, a_e=e.loc[r.euclid_row, "a_e"], b_e=e.loc[r.euclid_row, "b_e"],
                   theta=e.loc[r.euclid_row, "theta"])
        others = e.drop(index=r.euclid_row)
        dra = np.abs(((others["ra"] - r.euclid_ra + 180) % 360) - 180) * np.cos(np.radians(r.euclid_dec)) * 3600
        others = others[(dra < 15) & (np.abs(others["dec"] - r.euclid_dec) * 3600 < 15)]
        sep, reasons, (ef, eiv, em, ep, need, f_raw) = euclid_separation(cut, tgt, others, cfg, rng)
        row.update(sep)
        row["euclid_theta_cat"] = float(tgt["theta"])
        if reasons:
            reject(pd.DataFrame([row]), "euclid_separation", "+".join(reasons))
            continue

        # Rubin blend stamp, centred on the Euclid target, then on the Rubin object (Rubin-only setting).
        w = img["wcs"][BANDS.index("i")]
        x, y = w.world_to_pixel_values(r.euclid_ra, r.euclid_dec)
        st = rubin_stamp(img, float(x), float(y), n_r)
        xr, yr = w.world_to_pixel_values(r.rubin_ra, r.rubin_dec)
        st_rc = rubin_stamp(img, float(xr), float(yr), n_r)
        if st is None or st_rc is None:
            reject(pd.DataFrame([row]), "rubin_stamp_outside_region")
            continue
        pb = img["meta"]["patch_bbox"]
        tx, ty = img["x0"] + st["ix"], img["y0"] + st["iy"]
        row["rubin_margin_px"] = int(min(tx - pb["x_min"], pb["x_max"] - tx, ty - pb["y_min"], pb["y_max"] - ty))
        nod = (st["mask"] & masks.bitmask(img["bits"], ["NO_DATA"])) != 0
        row["rubin_nodata_px"] = int(nod.any(axis=0).sum())
        a_r, b_r = e.loc[r.euclid_row, "a_r"], e.loc[r.euclid_row, "b_r"]
        rm = radius_map((n_r, n_r), n_r // 2 + st["dx"], n_r // 2 + st["dy"], a_r, b_r, tgt["theta"])
        badpix = np.any((st["mask"] & masks.bitmask(img["bits"], d2["bad_planes"])) != 0, axis=0)
        row["rubin_bad_in_k"] = int((badpix & (rm <= float(d2["k_ellipse_bad_pixels"]))).sum())
        if row["rubin_margin_px"] < int(cc["margin_px"]) or row["rubin_nodata_px"] > 0 or row["rubin_bad_in_k"] > 0:
            reject(pd.DataFrame([row]), "rubin_pixels",
                   f"margin={row['rubin_margin_px']} nodata={row['rubin_nodata_px']} bad={row['rubin_bad_in_k']}")
            continue
        row.update(rubin_subpixel_dx=st["dx"], rubin_subpixel_dy=st["dy"], x_tract=float(img["x0"] + x),
                   y_tract=float(img["y0"] + y), ix_tract=int(tx), iy_tract=int(ty),
                   rc_offset_x_px=int(st_rc["ix"] - st["ix"]), rc_offset_y_px=int(st_rc["iy"] - st["iy"]),
                   euclid_subpixel_dx=cut["dx"], euclid_subpixel_dy=cut["dy"],
                   euclid_wcs_assumed=bool(cut.get("wcs_assumed", False)))
        # Scene completeness: no other bright Euclid source (outside the group) in the Rubin stamp.
        half = n_r * common.RUBIN_SCALE / 2
        grp = {int(v) for v in r.group_euclid_rows.split(",")}
        o = others[~others.index.isin(grp)]
        dx_o, dy_o = common.sky_offset_arcsec(r.euclid_ra, r.euclid_dec, o["ra"].to_numpy(), o["dec"].to_numpy())
        inside = (np.abs(dx_o) < half) & (np.abs(dy_o) < half)
        bright = o["vis_mag"].to_numpy(float) < r.euclid_vis_mag + float(d2["scene_complete_delta_mag"])
        row["n_bright_outside_group"] = int((inside & bright).sum())
        row["scene_complete"] = row["n_bright_outside_group"] == 0
        # Rubin PSF at the target: that of the Rubin object of the group.
        crow = cat.loc[r.rubin_object_id]
        for b in BANDS:
            for q in ("sigma", "e1", "e2"):
                row[f"psf_{q}_{b}"] = float(crow[f"psf_{q}_{b}"])
        row["resolution_i"] = float(crow["resolution_i"])
        snr = crow["i_cModelFlux"] / crow["i_cModelFluxErr"] if crow["i_cModelFluxErr"] else np.nan
        row["snr_i"] = float(snr)

        fx, iv, mk = to_hwc(st, img["bits"], mmu_planes)
        fx_rc, iv_rc, mk_rc = to_hwc(st_rc, img["bits"], mmu_planes)
        for k, v in (("rubin_flux", fx), ("rubin_ivar", iv), ("rubin_mask", mk), ("rubin_flux_rc", fx_rc),
                     ("rubin_ivar_rc", iv_rc), ("rubin_mask_rc", mk_rc), ("euclid_flux", ef), ("euclid_ivar", eiv),
                     ("euclid_mask", em), ("euclid_present", ep), ("euclid_flux_unmasked", f_raw),
                     ("euclid_fill", need)):
            A[k].append(v)
        keep_meta.append(row)

    meta = pd.DataFrame(keep_meta)
    if len(meta):
        # Partner of a two-target group: integer shift between the two stamp centres (Rubin pixels), filled below
        # only if the partner was kept too (scene consistency needs both reconstructions).
        pos = meta.set_index("euclid_object_id")[["ix_tract", "iy_tract"]]
        has = meta["partner_euclid_object_id"].isin(pos.index)
        meta["partner_kept"] = has
        meta["partner_dx_px"] = [int(pos.loc[p, "ix_tract"] - ix) if h else 0 for p, ix, h in
                                 zip(meta["partner_euclid_object_id"], meta["ix_tract"], has)]
        meta["partner_dy_px"] = [int(pos.loc[p, "iy_tract"] - iy) if h else 0 for p, iy, h in
                                 zip(meta["partner_euclid_object_id"], meta["iy_tract"], has)]
        meta["split"] = splits.split_table(meta, cfg).values
        arrays = {k: np.stack(v) for k, v in A.items()}
    else:
        arrays = empty_arrays(n_r, len(rb), n_e, len(eb))
    store.write_shard(shard_dir(cfg, patch["name"]), meta, arrays, pd.DataFrame(rejected),
                      info=dict(patch=patch["name"]))
    return dict(n_targets=int(len(meta)), n_candidates=int(len(cand)), n_rejected=len(rejected))


def empty_arrays(n_r, b_r, n_e, b_e):
    z = lambda n, b, t: np.zeros((0, n, n, b), t)  # noqa: E731
    return dict(rubin_flux=z(n_r, b_r, np.float32), rubin_ivar=z(n_r, b_r, np.float32), rubin_mask=z(n_r, b_r, bool),
                rubin_flux_rc=z(n_r, b_r, np.float32), rubin_ivar_rc=z(n_r, b_r, np.float32),
                rubin_mask_rc=z(n_r, b_r, bool), euclid_flux=z(n_e, b_e, np.float32),
                euclid_ivar=z(n_e, b_e, np.float32), euclid_mask=z(n_e, b_e, bool),
                euclid_present=np.zeros((0, b_e), bool), euclid_flux_unmasked=z(n_e, b_e, np.float32),
                euclid_fill=np.zeros((0, n_e, n_e), bool))


def merge(cfg):
    root = cfg.dataset_dir("d2")
    shards = [s for s in sorted((root / "shards").glob("*")) if s.is_dir() and not s.name.endswith(".tmp")]
    info = dict(name="d2", rubin_bands=list(cfg["rubin"]["bands"]), euclid_bands=list(cfg["euclid"]["bands"]),
                rubin_units="nJy", centering="rubin_* centred on the Euclid target, rubin_*_rc on the Rubin object",
                euclid_image_source=cfg["euclid"]["image_source"],
                config_fingerprint=cfg.fingerprint(["rubin", "euclid", "dataset2", "splits", "seed"]),
                mask_convention="True = pixel not to be trusted; euclid_fill = pixel replaced by sky")
    n = store.merge_shards(shards, root, info)
    # Euclid normalisation and conditioning statistics are those of D1 (the teacher was trained with them).
    for name in ("euclid_stats.json", "cond_stats.json"):
        src = cfg.dataset_dir("d1") / name
        if src.exists():
            (root / name).write_text(src.read_text())
    st = store.Store(root)
    summary = dict(n=n, by_split=st.meta["split"].value_counts().to_dict(),
                   scene_complete=int(st.meta["scene_complete"].sum()),
                   pairs_with_partner=int(st.meta["partner_kept"].sum()),
                   rejected=st.rejected()["reason"].value_counts().to_dict())
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(f"D2: {n:,} targets from {len(shards)} shards -> {root}")
    return n
