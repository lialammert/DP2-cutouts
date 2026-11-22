# patches.py
# Anna Preto
# AstroDeep Project
# List of the patches to process, from SIA (no pixel downloaded): for each deep_coadd patch of the search area, its
# outer polygon on the sky, its size in pixels, the bands present and the identifiers of its six images.
# Each patch then becomes a case of the existing stages, with a sky box = outer polygon + catalogue margin.

import os

import numpy as np
import pandas as pd
from pyvo.dal.sia2 import SIA2Service

from ..sources.rubin_soda import BANDS, SIA_URL, session


def patch_name(tract, patch):
    return f"{int(tract)}_{int(patch)}"


def patches_path(cfg):
    return cfg.data("plan", "patches.parquet")


def list_patches(cfg):
    pc = cfg["patches"]
    s = session(cfg["rubin"]["token_file"])
    rows = []
    for area in pc["search"]:
        t = SIA2Service(SIA_URL, session=s).search(pos=(float(area["ra"]), float(area["dec"]),
                                                        float(area["radius_deg"])), calib_level=3).to_table()
        for r in t:
            if str(r["dataproduct_subtype"]) != "lsst.deep_coadd":
                continue
            poly = np.array([float(v) for v in str(r["s_region"]).split()[2:]]).reshape(-1, 2)
            rows.append(dict(tract=int(r["lsst_tract"]), patch=int(r["lsst_patch"]), band=str(r["lsst_band"]),
                             dataset_id=str(r["obs_publisher_did"]), s_ra=float(r["s_ra"]),
                             s_dec=float(r["s_dec"]), s_xel1=int(r["s_xel1"]), s_xel2=int(r["s_xel2"]),
                             ra_min=poly[:, 0].min(), ra_max=poly[:, 0].max(),
                             dec_min=poly[:, 1].min(), dec_max=poly[:, 1].max()))
    bands = pd.DataFrame(rows).drop_duplicates(subset=["tract", "patch", "band"])
    if pc.get("tracts"):
        bands = bands[bands["tract"].isin(pc["tracts"])]
    if pc.get("include"):
        bands = bands[bands["patch"].isin(pc["include"])]
    if pc.get("names_file"):
        # Patch list of scripts/inspect/area_census.py (column name = tract_patch), kept where Euclid covers at least
        # euclid_cover_min of the patch (centre and four corners); the census needs IRSA, the plan does not.
        sel = pd.read_csv(os.path.expanduser(pc["names_file"]))
        if "euclid_cover_frac" in sel:
            sel = sel[sel["euclid_cover_frac"] >= float(pc.get("euclid_cover_min", 1.0))]
        names = set(sel["name"].astype(str))
        bands = bands[[patch_name(t, p) in names for t, p in zip(bands["tract"], bands["patch"])]]
    geo = ["s_ra", "s_dec", "s_xel1", "s_xel2", "ra_min", "ra_max", "dec_min", "dec_max"]
    patches = (bands.groupby(["tract", "patch"])
               .agg(**{c: (c, "first") for c in geo}, bands=("band", lambda b: "".join(x for x in BANDS if x in set(b))))
               .reset_index())
    patches["complete"] = patches["bands"] == BANDS
    patches["name"] = [patch_name(t, p) for t, p in zip(patches["tract"], patches["patch"])]
    bands.to_parquet(cfg.data("plan", "patch_bands.parquet"), index=False)
    patches.to_parquet(patches_path(cfg), index=False)
    print(f"{len(patches)} patches ({int(patches['complete'].sum())} with all six bands) -> {patches_path(cfg)}")
    return patches


def load_patches(cfg, names=None, index=None):
    # names: list of "tract_patch"; index: position in the list (Slurm array task id). Default: every patch.
    patches = pd.read_parquet(patches_path(cfg))
    if names:
        missing = set(names) - set(patches["name"])
        if missing:
            raise KeyError(f"patches not in the plan: {sorted(missing)}")
        patches = patches[patches["name"].isin(names)]
    if index is not None:
        patches = patches.iloc[[int(index)]]
    return [r._asdict() for r in patches.itertuples(index=False)]


def case_dir(cfg, patch):
    return cfg.data_root / "patches" / patch["name"]


def patch_case(cfg, patch):
    # Sub-case of one patch: outer polygon of the patch + catalogue margin, in its own folder. Objects of the
    # neighbouring patches inside the margin are context only: the patch owns the objects whose (tract, patch) in
    # the Rubin Object table are its own.
    m = float(cfg["patches"]["catalog_margin_arcsec"]) / 3600.0
    dec_c = 0.5 * (patch["dec_min"] + patch["dec_max"])
    mra = m / np.cos(np.radians(dec_c))
    box = [patch["ra_min"] - mra, patch["ra_max"] + mra, patch["dec_min"] - m, patch["dec_max"] + m]
    return cfg.derive(patch["name"], case_dir(cfg, patch), box,
                      rubin={"tracts": [int(patch["tract"])], "tap_tract_filter": False},
                      pixels={"tract": int(patch["tract"]), "patch": int(patch["patch"])})


def owned(table, patch):
    return (table["tract"] == int(patch["tract"])) & (table["patch"] == int(patch["patch"]))
