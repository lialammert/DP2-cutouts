# dataset1.py
# Anna Preto
# AstroDeep Project
# Dataset 1: pairs (x_R, x_E) of the same galaxy, usable individually in both surveys.
# Rubin side: the final dataset of the stamp pipeline, unchanged (every cut of the recipe, companion masking
# included); nothing is re-selected. Euclid side: a cutout of the same galaxy around its Euclid position.
# Pairs are dropped only when unusable (no Euclid image, too few valid Euclid pixels, wrong pixel scale), each with
# its reason; every other check is recorded as a column, for the quality report and for later selections.
# One shard per Rubin patch (the job of one Slurm array task), merged afterwards.

import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..sources import euclid_images
from ..stages.catalog_table import copaings_dir
from . import common, splits, store

ARRAYS_RUBIN = ("rubin_flux", "rubin_ivar", "rubin_mask")
ARRAYS_EUCLID = ("euclid_flux", "euclid_ivar", "euclid_mask")


def shard_dir(cfg, name):
    return cfg.dataset_dir("d1") / "shards" / name


def euclid_neighbours(case):
    # Every Euclid source of the patch box with its VIS magnitude (copaings Euclid table).
    e = pd.read_parquet(copaings_dir(case) / "catalogs" / "euclid.parquet", columns=["ra", "dec", "vis_mag"])
    return e, cKDTree(splits.unit_vectors(e["ra"], e["dec"]))


def contamination(neigh, tree, ra, dec, vis_mag, radius_arcsec, dmag):
    # Other Euclid sources around the target: count within radius brighter than vis_mag + dmag, nearest distance.
    e = neigh
    xyz = splits.unit_vectors([ra], [dec])[0]
    chord = 2 * np.sin(np.radians(radius_arcsec / 3600.0) / 2)
    idx = [j for j in tree.query_ball_point(xyz, chord)]
    d = np.degrees(2 * np.arcsin(np.linalg.norm(splits.unit_vectors(e["ra"].values[idx], e["dec"].values[idx]) - xyz,
                                                axis=1) / 2)) * 3600.0 if idx else np.zeros(0)
    others = d > 0.05  # the target itself
    d, mags = d[others], e["vis_mag"].values[idx][others] if idx else np.zeros(0)
    bright = np.isfinite(mags) & (mags < vis_mag + dmag) if np.isfinite(vis_mag) else np.isfinite(mags)
    return int(bright.sum()), (float(d.min()) if d.size else np.inf)


def build_patch(cfg, patch):
    from ..pipeline.patches import patch_case

    pcfg = cfg.pipeline()
    case = patch_case(pcfg, patch)
    rb = list(cfg["rubin"]["bands"])
    ec = cfg["euclid"]
    eb = list(ec["bands"])
    d1 = cfg["dataset1"]
    n_r, n_e = int(cfg["rubin"]["stamp_px"]), int(ec["stamp_px"])

    t = common.read_final_patch(pcfg, patch["tract"], patch["patch"])
    meta = common.table_to_pandas(t)
    flux_r, ivar_r, mask_r = common.unpack_images(t.column("image"), rb)
    rejected = []

    has = meta["euclid_object_id"].notna().to_numpy()
    for i in np.flatnonzero(~has):
        rejected.append(dict(object_id=int(meta["object_id"].iloc[i]), reason="no_euclid_counterpart"))
    meta, flux_r, ivar_r, mask_r = meta[has].reset_index(drop=True), flux_r[has], ivar_r[has], mask_r[has]
    if len(meta) == 0:
        # Patch without final stamps (no target, not processed): an empty shard records that it was looked at.
        store.write_shard(shard_dir(cfg, patch["name"]), pd.DataFrame(), empty_arrays(n_r, len(rb), n_e, len(eb)),
                          pd.DataFrame(rejected), info=dict(patch=patch["name"]))
        return dict(n_pairs=0, n_rejected=len(rejected))

    objects = pd.DataFrame(dict(object_id=common.as_int64(meta["euclid_object_id"]).astype("int64"),
                                ra=meta["euclid_ra"].astype(float), dec=meta["euclid_dec"].astype(float)))
    box = case.sky_box if len(objects) else None
    cuts = euclid_images.fetch(ec, objects, box) if len(objects) else {}
    neigh, tree = euclid_neighbours(case)

    keep, rows = [], []
    ef = np.zeros((len(meta), n_e, n_e, len(eb)), np.float32)
    ei, em = np.zeros_like(ef), np.ones(ef.shape, bool)
    ep = np.zeros((len(meta), len(eb)), bool)
    vis = eb.index("VIS") if "VIS" in eb else None
    ib = rb.index(cfg["rubin"]["shape_band"])
    c_r, c_e = (n_r - 1) / 2.0, (n_e - 1) / 2.0
    for i, r in enumerate(meta.itertuples(index=False)):
        oid, eid = int(r.object_id), int(r.euclid_object_id)
        cut = cuts.get(eid)
        if cut is None:
            rejected.append(dict(object_id=oid, euclid_object_id=eid, reason="no_euclid_image"))
            continue
        f, iv, m, present = euclid_images.to_band_array(cut, eb)
        if vis is None or not present[vis]:
            rejected.append(dict(object_id=oid, euclid_object_id=eid, reason="no_vis_band"))
            continue
        if not cut.get("scale_ok", True):
            rejected.append(dict(object_id=oid, euclid_object_id=eid, reason="euclid_scale_mismatch",
                                 detail=str(cut.get("source_scale"))))
            continue
        valid_vis = float((~m[..., vis]).mean())
        if valid_vis < float(d1["min_valid_fraction_euclid"]):
            rejected.append(dict(object_id=oid, euclid_object_id=eid, reason="euclid_invalid_pixels",
                                 detail=f"{valid_vis:.3f}"))
            continue
        ef[i], ei[i], em[i], ep[i] = f, iv, m, present
        keep.append(i)

        # Alignment: catalogue offset Euclid - Rubin, sub-pixel centring of each stamp, measured centroids.
        xr = float(r.x_tract) - np.floor(float(r.x_tract) + 0.5)
        yr = float(r.y_tract) - np.floor(float(r.y_tract) + 0.5)
        west, north = common.sky_offset_arcsec(r.ra, r.dec, r.euclid_ra, r.euclid_dec)
        wr = (~mask_r[i][..., ib]).astype(np.int32)
        mr = common.moments(flux_r[i][..., ib], wr, (c_r + xr, c_r + yr))
        we = (~m[..., vis]).astype(np.int32)
        me = common.moments(f[..., vis], we, (c_e + cut["dx"], c_e + cut["dy"]))
        # Centroids as sky offsets from the Rubin catalogue position, x towards west, y towards north.
        rx, ry = ((mr["x"] - c_r - xr) * common.RUBIN_SCALE, (mr["y"] - c_r - yr) * common.RUBIN_SCALE)
        ex = west + (me["x"] - c_e - cut["dx"]) * float(ec["pixel_scale"])
        ey = north + (me["y"] - c_e - cut["dy"]) * float(ec["pixel_scale"])
        _, snr_vis, _ = common.aperture(f[..., vis], iv[..., vis], c_e + cut["dx"], c_e + cut["dy"], 5.0)
        n_cont, d_near = contamination(neigh, tree, r.euclid_ra, r.euclid_dec, float(r.euclid_vis_mag),
                                       float(d1["contamination_radius_arcsec"]), float(d1["contamination_delta_mag"]))
        rows.append(dict(
            astrometric_offset_arcsec=float(np.hypot(west, north)), astrometric_west_arcsec=float(west),
            astrometric_north_arcsec=float(north),
            astrometric_flag=bool(np.hypot(west, north) > float(d1["max_astrometric_offset_arcsec"])),
            rubin_subpixel_dx=xr, rubin_subpixel_dy=yr, euclid_subpixel_dx=cut["dx"], euclid_subpixel_dy=cut["dy"],
            centroid_rubin_x_arcsec=rx, centroid_rubin_y_arcsec=ry, centroid_euclid_x_arcsec=ex,
            centroid_euclid_y_arcsec=ey, centroid_offset_arcsec=float(np.hypot(ex - rx, ey - ry)),
            e1_vis=me["e1"], e2_vis=me["e2"], sigma_vis_px=me["sigma"], hsm_vis_ok=me["ok"],
            e1_rubin_check=mr["e1"], e2_rubin_check=mr["e2"], snr_vis_aper=snr_vis,
            euclid_valid_fraction_vis=valid_vis, euclid_wcs_assumed=bool(cut.get("wcs_assumed", False)),
            euclid_ivar_estimated=bool(cut.get("ivar_estimated", False)), euclid_source=cut.get("source"),
            euclid_source_px=int(cut.get("source_px", -1)), n_euclid_contaminants=n_cont,
            nearest_euclid_arcsec=d_near,
            rubin_valid_fraction=float((~mask_r[i]).mean()), rubin_bad_ivar=int((~np.isfinite(ivar_r[i])).sum()
                                                                                 + (ivar_r[i] < 0).sum())))

    keep = np.asarray(keep, int)
    meta = pd.concat([meta.iloc[keep].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    meta["snr_i"] = meta["cut6_value"].astype(float) if "cut6_value" in meta else np.nan
    meta["split"] = splits.split_table(meta, cfg).values if len(meta) else pd.Series(dtype=object)
    arrays = dict(rubin_flux=flux_r[keep], rubin_ivar=ivar_r[keep], rubin_mask=mask_r[keep],
                  euclid_flux=ef[keep], euclid_ivar=ei[keep], euclid_mask=em[keep], euclid_present=ep[keep])
    if len(keep) == 0:
        arrays = empty_arrays(n_r, len(rb), n_e, len(eb))
    store.write_shard(shard_dir(cfg, patch["name"]), meta, arrays, pd.DataFrame(rejected),
                      info=dict(patch=patch["name"]))
    return dict(n_pairs=int(len(keep)), n_rejected=len(rejected))


def empty_arrays(n_r, b_r, n_e, b_e):
    return dict(rubin_flux=np.zeros((0, n_r, n_r, b_r), np.float32), rubin_ivar=np.zeros((0, n_r, n_r, b_r), np.float32),
                rubin_mask=np.zeros((0, n_r, n_r, b_r), bool), euclid_flux=np.zeros((0, n_e, n_e, b_e), np.float32),
                euclid_ivar=np.zeros((0, n_e, n_e, b_e), np.float32), euclid_mask=np.zeros((0, n_e, n_e, b_e), bool),
                euclid_present=np.zeros((0, b_e), bool))


def info(cfg):
    # Pixels are stored in nJy whatever the VAE: the conversion to its units is done when a batch is read.
    return dict(name="d1", rubin_bands=list(cfg["rubin"]["bands"]), euclid_bands=list(cfg["euclid"]["bands"]),
                rubin_units="nJy", euclid_units="as delivered by the source (MER: microJy per pixel)",
                rubin_pixel_scale=common.RUBIN_SCALE, euclid_pixel_scale=float(cfg["euclid"]["pixel_scale"]),
                euclid_image_source=cfg["euclid"]["image_source"],
                config_fingerprint=cfg.fingerprint(["rubin", "euclid", "dataset1", "splits", "seed"]),
                mask_convention="True = pixel not to be trusted",
                stamp_centre="integer pixel nearest to the catalogue position; sub-pixel offsets in meta")


def euclid_norm_stats(st, bands, train):
    # Per-band scale of the nonlinear normalisation, defined as in DebVAder: the mean over training stamps of the
    # maximum pixel value, here inside the central 21 x 21 pixels so that a neighbour does not set it.
    n = st.info["arrays"]["euclid_flux"][0][0]
    c, h = n // 2, 10
    idx = np.flatnonzero(train)
    scale = np.zeros(len(bands))
    for k in range(len(bands)):
        vals = []
        for s in range(0, len(idx), 2048):
            part = st.take("euclid_flux", idx[s:s + 2048])[:, c - h:c + h + 1, c - h:c + h + 1, k]
            pres = st.take("euclid_present", idx[s:s + 2048])[:, k]
            vals.append(part.reshape(len(part), -1).max(axis=1)[pres])
        v = np.concatenate(vals) if vals else np.zeros(0)
        scale[k] = float(np.mean(v)) if v.size else np.nan
    return dict(nonlinear=dict(scale_per_band=[float(v) for v in scale], bands=list(bands), beta=2.5,
                               definition="mean of the per-stamp maximum in the central 21x21 px, train split"))


def cond_stats(meta, columns, train):
    sub = meta.loc[train, columns].astype(float)
    return dict(columns=list(columns), mean=sub.mean().tolist(), std=sub.std().replace(0, 1.0).fillna(1.0).tolist())


def merge(cfg):
    root = cfg.dataset_dir("d1")
    shards = sorted((root / "shards").glob("*"))
    shards = [s for s in shards if s.is_dir() and not s.name.endswith(".tmp")]
    n = store.merge_shards(shards, root, info(cfg))
    st = store.Store(root)
    train = (st.meta["split"] == "train").to_numpy()
    with open(root / "euclid_stats.json", "w") as f:
        json.dump(euclid_norm_stats(st, cfg["euclid"]["bands"], train), f, indent=2)
    with open(root / "cond_stats.json", "w") as f:
        json.dump(cond_stats(st.meta, cfg["dataset1"]["cond_features"], train), f, indent=2)
    print(f"D1: {n:,} pairs from {len(shards)} shards -> {root}")
    return n
