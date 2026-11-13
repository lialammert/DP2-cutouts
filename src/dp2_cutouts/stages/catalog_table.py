# catalog_table.py
# Anna Preto
# AstroDeep Project
# Catalogue table, one row per copaings Rubin object (galaxies in the Rubin sense, magnitude <= mag_cut), with no
# row removed: each catalogue cut writes its continuous quantity and its condition. Joins the Euclid counterpart of
# one-to-one groups, the cut 1 and redshift columns, the PSF in all six bands, the single ellipse and the
# resolution factor.

import pickle

import healpy as hp
import numpy as np
import pandas as pd

from ..common import star_galaxy
from ..common.ellipse import LSST_PIXEL_SCALE, moments_to_ellipse
from ..sources import euclid_hf
from ..sources.rubin_tap import BANDS
from ..config import COPAINGS_DIR, RUBIN_DIR, TABLES_DIR

PARTITION_ORDER = 5  # HEALPix NESTED partitions of order 5 (nside 32), like the HATS Norder folders


def copaings_dir(cfg):
    return cfg.data(COPAINGS_DIR, cfg.name, "x").parent


def one_to_one(groups):
    # One-to-one groups after the overlap test: Rubin row -> Euclid row in the copaings tables.
    pairs = [(g[1][0], g[0][0]) for g in groups if len(g[0]) == 1 and len(g[1]) == 1]
    return dict(pairs)


def euclid_counterparts(cfg, cdir, lsst):
    groups = pickle.load(open(cdir / "groups" / "E_no_overlap.pkl", "rb"))["data"]
    pair = one_to_one(groups)
    e_cop = pd.read_parquet(cdir / "catalogs" / "euclid.parquet")
    e_box = euclid_hf.read_box(cfg)
    # copaings does not keep the Euclid identifier: it is recovered from the exact position (same floats).
    key = ["mer_catalogue_right_ascension", "mer_catalogue_declination"]
    e_box = e_box.drop_duplicates(subset=key)
    e_cop = e_cop.reset_index().rename(columns={"index": "euclid_row"})
    e = e_cop.merge(e_box, left_on=["ra", "dec"], right_on=key, how="left", suffixes=("", "_mmu"))
    if e["object_id"].isna().any():
        raise RuntimeError("copaings Euclid objects without an MMU identifier")
    e = e.set_index("euclid_row")

    out = pd.DataFrame(index=lsst.index)
    rows = np.array([pair.get(i, -1) for i in lsst.index])
    has = rows >= 0
    out["has_euclid_counterpart"] = has
    take = {
        "euclid_object_id": "object_id",
        "euclid_ra": "ra", "euclid_dec": "dec",
        "euclid_vis_mag": "vis_mag",
        "euclid_size": "size_true", "euclid_size_minor": "size_minor_true",
        "euclid_position_angle": "position_angle_true", "euclid_ellipticity": "ellipticity",
    }
    take.update({c.replace("mer_catalogue_", "euclid_"): c for c in euclid_hf.CUT1_COLUMNS})
    take.update({c.replace("phz_photo_z_", "euclid_").replace("phz_classification_", "euclid_"): c
                 for c in euclid_hf.PHZ_COLUMNS})
    sub = e.loc[rows[has]]
    where = np.flatnonzero(has)
    for new, old in take.items():
        vals = [None] * len(out)
        for i, v in zip(where, sub[old].values):
            vals[i] = None if v is pd.NA else v
        out[new] = pd.Series(vals, index=out.index, dtype=object)
    out["euclid_object_id"] = pd.array(
        [int(v) if v is not None else None for v in out["euclid_object_id"]], dtype="Int64")
    for c in ("euclid_ra", "euclid_dec", "euclid_vis_mag", "euclid_size", "euclid_size_minor",
              "euclid_position_angle", "euclid_ellipticity", "euclid_point_like_prob", "euclid_spurious_prob",
              "euclid_extended_prob", "euclid_phz_median"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in ("euclid_point_like_flag", "euclid_spurious_flag", "euclid_det_quality_flag", "euclid_vis_det",
              "euclid_phz_flags", "euclid_phz_classification"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def psf_columns(raw):
    # PSF size (sigma, arcsec) and ellipticity (distortion), per band, from the TAP moments.
    out = pd.DataFrame(index=raw.index)
    for b in BANDS:
        ixx, iyy, ixy = raw[f"{b}_ixxPSF"], raw[f"{b}_iyyPSF"], raw[f"{b}_ixyPSF"]
        tr = ixx + iyy
        out[f"psf_sigma_{b}"] = np.sqrt(tr / 2.0) * LSST_PIXEL_SCALE
        out[f"psf_e1_{b}"] = (ixx - iyy) / tr
        out[f"psf_e2_{b}"] = 2.0 * ixy / tr
    return out


def build(cfg):
    cdir = copaings_dir(cfg)
    lsst = pd.read_parquet(cdir / "lsst_sb.parquet")
    raw = pd.read_parquet(cfg.data(RUBIN_DIR, "catalog", "object_box.parquet")).set_index("objectId")

    t = pd.DataFrame(index=lsst.index)
    t["object_id"] = lsst["objectId"].astype("int64")
    t["tract"] = lsst["tract_id"].astype("int64")
    t["patch"] = lsst["patch"].astype("int64")
    t["ra"], t["dec"] = lsst["ra"], lsst["dec"]
    t["healpix5"] = hp.ang2pix(2 ** PARTITION_ORDER, t["ra"].values, t["dec"].values, nest=True, lonlat=True)
    for b in BANDS:
        t[f"{b}_cModel_mag"] = lsst[f"{b}_cModel_mag"]
    t["i_cModelFlux"], t["i_cModelFluxErr"] = lsst["i_cModelFlux"], lsst["i_cModelFluxErr"]
    t["i_blendedness"] = lsst["i_blendedness"]
    t["i_ixx"], t["i_iyy"], t["i_ixy"] = lsst["Ixx_pixel_i"], lsst["Iyy_pixel_i"], lsst["Ixy_pixel_i"]
    t["e1_moments"], t["e2_moments"] = lsst["e1"], lsst["e2"]

    # Single ellipse (half light, tract pixels) and galaxy / PSF resolution factor in the i band.
    a, b_, th = moments_to_ellipse(t["i_ixx"], t["i_iyy"], t["i_ixy"])
    t["ellipse_a"], t["ellipse_b"], t["ellipse_theta"] = a, b_, th
    r = raw.loc[t["object_id"].values]
    psf = psf_columns(r)
    psf.index = t.index
    t = pd.concat([t, psf], axis=1)
    t_psf = r["i_ixxPSF"].values + r["i_iyyPSF"].values
    t["resolution_i"] = 1.0 - t_psf / (t["i_ixx"].values + t["i_iyy"].values)

    # Value of the configured star-galaxy criterion (e.g. griz_model_extendedness); every object in this table
    # already passed it, copaings dropped the others. NaN when no criterion is configured.
    t["star_galaxy_value"] = star_galaxy.values(r, cfg)

    # Rubin's own deblending status, a band-independent isolation criterion: Rubin did not split this object
    # from a neighbour if it was never deblended (detect_isIsolated) or if its blend produced a single child
    # (deblend_blendNChild == 1). Objects in a blend of 2 or more children fail. On the demo box (tract 5063)
    # the number of detect_isIsolated objects equals the number with deblend_blendNChild == 0; the meaning of
    # blendNChild == 1 is inferred from that, not from the DP2 schema documentation: check it when it matters.
    # i_blendedness is not used: its DP2 values are not on the 0-1 scale.
    t["detect_isIsolated"] = r["detect_isIsolated"].fillna(0).to_numpy(float) == 1
    t["deblend_blendNChild"] = r["deblend_blendNChild"].to_numpy(float)
    t["rubin_deblend_value"] = t["deblend_blendNChild"]
    t["rubin_deblend_pass"] = t["detect_isIsolated"] | (t["deblend_blendNChild"] == 1)

    # copaings blending entropy, kept as a value.
    t["sb_euclid"] = lsst["blending_entropy_Euclid"]
    t["purity_euclid"] = lsst["purity_Euclid"]

    t = pd.concat([t, euclid_counterparts(cfg, cdir, lsst)], axis=1)

    # Cut 1: the Euclid counterpart is neither point-like nor spurious (missing value = flag not set).
    # Without a unique counterpart the cut cannot be evaluated and removes nothing: cut 2 rejects the object.
    point = t["euclid_point_like_flag"].fillna(0).astype(int) == 1
    spurious = t["euclid_spurious_flag"].fillna(0).astype(int) == 1
    t["cut1_pass"] = ~(t["has_euclid_counterpart"] & (point | spurious))
    # Cut 2: matched one-to-one after the overlap test. S_b = 0 is not enough: on the demo box, 418 objects have
    # S_b = 0 in groups that are not one-to-one (galaxies split 1 Euclid / 2 Rubin, blends 2 Euclid / 1 Rubin...).
    # Both are required, and both quantities are kept.
    t["cut2_value"] = t["sb_euclid"]
    t["cut2_pass"] = (t["sb_euclid"] == 0) & t["has_euclid_counterpart"]
    # Cut 6: signal-to-noise ratio in the i band.
    with np.errstate(divide="ignore", invalid="ignore"):
        t["cut6_value"] = t["i_cModelFlux"] / t["i_cModelFluxErr"]
    t["cut6_pass"] = t["cut6_value"] > float(cfg["cuts"]["snr_min"])
    return t.reset_index(drop=True)


def run(cfg):
    t = build(cfg)
    out = cfg.data(TABLES_DIR, "catalog.parquet")
    t.to_parquet(out, index=False)
    n = len(t)
    print(f"catalogue: {n:,} objects; one-to-one Euclid counterpart {t['has_euclid_counterpart'].sum():,}; "
          f"cut 1 {t['cut1_pass'].sum():,}; cut 2 {t['cut2_pass'].sum():,}; "
          f"1 and 2 {(t['cut1_pass'] & t['cut2_pass']).sum():,}; cut 6 {t['cut6_pass'].sum():,} -> {out}")
    return t
