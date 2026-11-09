# catalog_table.py
# Anna Preto
# AstroDeep Project
# Table du catalogue, une ligne par objet Rubin de copaings (galaxies au sens de Rubin, magnitude <= mag_cut), sans
# aucune ligne retirée : chaque coupure catalogue écrit sa grandeur continue et sa condition. Joint la contrepartie
# Euclid des groupes un à un, les colonnes de la coupure 1 et du redshift, la tache dans les six bandes, l'ellipse
# unique et le facteur de résolution.

import pickle

import healpy as hp
import numpy as np
import pandas as pd

from ..common.ellipse import LSST_PIXEL_SCALE, moments_to_ellipse
from ..sources import euclid_hf
from ..sources.rubin_tap import BANDS

PARTITION_ORDER = 5  # partitions HEALPix NESTED d'ordre 5 (nside 32), comme les dossiers Norder de HATS


def copaings_dir(cfg):
    return cfg.data("20_copaings", cfg.name, "x").parent


def one_to_one(groups):
    # Groupes un à un après le test de recouvrement : rang Rubin -> rang Euclid dans les tables de copaings.
    pairs = [(g[1][0], g[0][0]) for g in groups if len(g[0]) == 1 and len(g[1]) == 1]
    return dict(pairs)


def euclid_counterparts(cfg, cdir, lsst):
    groups = pickle.load(open(cdir / "groups" / "E_no_overlap.pkl", "rb"))["data"]
    pair = one_to_one(groups)
    e_cop = pd.read_parquet(cdir / "catalogs" / "euclid.parquet")
    e_box = euclid_hf.read_box(cfg)
    # copaings ne garde pas l'identifiant Euclid : on le retrouve par la position exacte (mêmes flottants).
    key = ["mer_catalogue_right_ascension", "mer_catalogue_declination"]
    e_box = e_box.drop_duplicates(subset=key)
    e_cop = e_cop.reset_index().rename(columns={"index": "euclid_row"})
    e = e_cop.merge(e_box, left_on=["ra", "dec"], right_on=key, how="left", suffixes=("", "_mmu"))
    if e["object_id"].isna().any():
        raise RuntimeError("objets Euclid de copaings sans identifiant MMU")
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
    # Taille (sigma, seconde d'arc) et ellipticité (distorsion) de la tache, par bande, depuis les moments TAP.
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
    raw = pd.read_parquet(cfg.data("10_rubin", "catalog", "object_box.parquet")).set_index("objectId")

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

    # Ellipse unique (demi-lumière, pixels du tract) et facteur de résolution galaxie / tache en bande i.
    a, b_, th = moments_to_ellipse(t["i_ixx"], t["i_iyy"], t["i_ixy"])
    t["ellipse_a"], t["ellipse_b"], t["ellipse_theta"] = a, b_, th
    r = raw.loc[t["object_id"].values]
    psf = psf_columns(r)
    psf.index = t.index
    t = pd.concat([t, psf], axis=1)
    t_psf = r["i_ixxPSF"].values + r["i_iyyPSF"].values
    t["resolution_i"] = 1.0 - t_psf / (t["i_ixx"].values + t["i_iyy"].values)

    # Blending entropy de copaings, gardée en valeur.
    t["sb_euclid"] = lsst["blending_entropy_Euclid"]
    t["purity_euclid"] = lsst["purity_Euclid"]

    t = pd.concat([t, euclid_counterparts(cfg, cdir, lsst)], axis=1)

    # Coupure 1 : la contrepartie Euclid n'est ni ponctuelle ni un artefact (valeur manquante = drapeau non levé).
    # Sans contrepartie unique, la coupure n'est pas évaluable et ne retire rien : c'est la coupure 2 qui l'écarte.
    point = t["euclid_point_like_flag"].fillna(0).astype(int) == 1
    spurious = t["euclid_spurious_flag"].fillna(0).astype(int) == 1
    t["cut1_pass"] = ~(t["has_euclid_counterpart"] & (point | spurious))
    # Coupure 2 : appariés un à un après le test de recouvrement. S_b = 0 ne suffit pas : sur la boîte de démo,
    # 418 objets ont S_b = 0 dans des groupes qui ne sont pas un à un (galaxies découpées 1 Euclid / 2 Rubin,
    # mélanges 2 Euclid / 1 Rubin...). On exige les deux, et on garde les deux grandeurs.
    t["cut2_value"] = t["sb_euclid"]
    t["cut2_pass"] = (t["sb_euclid"] == 0) & t["has_euclid_counterpart"]
    # Coupure 6 : rapport signal sur bruit en bande i.
    with np.errstate(divide="ignore", invalid="ignore"):
        t["cut6_value"] = t["i_cModelFlux"] / t["i_cModelFluxErr"]
    t["cut6_pass"] = t["cut6_value"] > float(cfg["cuts"]["snr_min"])
    return t.reset_index(drop=True)


def run(cfg):
    t = build(cfg)
    out = cfg.data("30_processed", "catalog.parquet")
    t.to_parquet(out, index=False)
    n = len(t)
    print(f"catalogue : {n:,} objets ; contrepartie Euclid un à un {t['has_euclid_counterpart'].sum():,} ; "
          f"coupure 1 {t['cut1_pass'].sum():,} ; coupure 2 {t['cut2_pass'].sum():,} ; "
          f"1 et 2 {(t['cut1_pass'] & t['cut2_pass']).sum():,} ; coupure 6 {t['cut6_pass'].sum():,} -> {out}")
    return t
