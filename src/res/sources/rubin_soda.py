# rubin_soda.py
# Anna Preto
# AstroDeep Project
# Téléchargement des pixels Rubin DP2 hors de la plateforme : recherche des deep_coadd d'un patch par SIA, puis
# découpe image + masque + variance par SODA (service cutout-sync-maskedimage), une découpe par bande sur la même
# région. Les vignettes sont ensuite découpées en local dans ces six images, comme le butler le fera sur la
# plateforme. Écrit aussi l'emprise exacte du patch dans le repère du tract, nécessaire à la coupure 3.

import io
import json
import os

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.wcs import WCS
from pyvo.dal.adhoc import DatalinkResults, SodaQuery
from pyvo.dal.sia2 import SIA2Service

BANDS = "ugrizy"
SIA_URL = "https://data.lsst.cloud/api/sia/dp2/query"


def session(token_file):
    # Le jeton est lu depuis le fichier et passé en en-tête ; il n'est jamais affiché ni journalisé.
    with open(os.path.expanduser(token_file)) as f:
        token = f.read().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    del token
    return s


def find_patch_images(s, ra, dec, tract, patch):
    # Les six deep_coadd (une par bande) du patch demandé qui contiennent la position.
    t = SIA2Service(SIA_URL, session=s).search(pos=(ra, dec, 0.0005), calib_level=3).to_table()
    keep = [(str(r["dataproduct_subtype"]) == "lsst.deep_coadd" and int(r["lsst_tract"]) == tract
             and int(r["lsst_patch"]) == patch) for r in t]
    t = t[keep]
    rows = {str(r["lsst_band"]): r for r in t}
    missing = [b for b in BANDS if b not in rows]
    if missing:
        raise RuntimeError(f"bandes absentes pour tract {tract} patch {patch} : {missing}")
    return rows


def soda_cutout(s, access_url, ra, dec, radius_deg):
    dl = DatalinkResults.from_result_url(access_url, session=s)
    sq = SodaQuery.from_resource(dl, dl.get_adhocservice_by_id("cutout-sync-maskedimage"), session=s)
    sq.circle = (ra * u.deg, dec * u.deg, radius_deg * u.deg)
    data = sq.execute_stream().read()
    sq.raise_if_error()
    return data


def tract_origin(header):
    # Origine de la découpe dans le repère du tract : le WCS « A » donne la position en pixels du tract du premier
    # pixel FITS (CRPIX = 1), c'est-à-dire de l'indice 0 du tableau.
    assert header["CRPIX1A"] == 1.0 and header["CRPIX2A"] == 1.0
    return int(header["CRVAL1A"]), int(header["CRVAL2A"])


def patch_bbox(s, row):
    # Emprise exacte du patch en pixels du tract, [x_min, x_max] x [y_min, y_max] inclus. On demande une petite
    # découpe sur le coin (x_min, y_min) du patch : SODA la rogne à l'image, et son origine est celle du patch.
    poly = [float(v) for v in str(row["s_region"]).split()[2:]]
    corners = np.array(poly).reshape(-1, 2)
    probe = fits.open(io.BytesIO(soda_cutout(s, row["access_url"], float(row["s_ra"]), float(row["s_dec"]), 0.001)))
    w = WCS(probe["IMAGE"].header)
    x0p, y0p = tract_origin(probe["IMAGE"].header)
    px, py = w.world_to_pixel_values(corners[:, 0], corners[:, 1])
    k = np.argmin((px + x0p) + (py + y0p))
    ra_c, dec_c = corners[k]
    corner = fits.open(io.BytesIO(soda_cutout(s, row["access_url"], ra_c, dec_c, 0.002)))
    x_min, y_min = tract_origin(corner["IMAGE"].header)
    nx, ny = int(row["s_xel1"]), int(row["s_xel2"])
    return dict(x_min=x_min, x_max=x_min + nx - 1, y_min=y_min, y_max=y_min + ny - 1)


def run(cfg):
    pc = cfg["pixels"]
    s = session(cfg["rubin"]["token_file"])
    tract, patch = int(pc["tract"]), int(pc["patch"])
    # Centre de la région : donné, ou à défaut le centre du patch (trouvé à partir des objets que le catalogue
    # attribue à ce patch).
    if pc.get("center"):
        ra, dec = (float(v) for v in pc["center"])
    else:
        cat = pd.read_parquet(cfg.data("10_rubin", "catalog", "object_box.parquet"),
                              columns=["coord_ra", "coord_dec", "tract", "patch"])
        sub = cat[(cat["tract"] == tract) & (cat["patch"] == patch)]
        row = find_patch_images(s, float(sub["coord_ra"].median()), float(sub["coord_dec"].median()), tract, patch)["i"]
        ra, dec = float(row["s_ra"]), float(row["s_dec"])
    rows = find_patch_images(s, ra, dec, tract, patch)

    radius = 0.5 * int(pc["size_px"]) * 0.2 / 3600.0
    out_dir = cfg.data("10_rubin", "images", "x").parent
    files = {}
    for b in BANDS:
        data = soda_cutout(s, rows[b]["access_url"], ra, dec, radius)
        path = out_dir / f"deep_coadd_{tract}_{patch}_{b}.fits"
        with open(path, "wb") as f:
            f.write(data)
        files[b] = path.name
        print(f"bande {b} : {len(data) / 1e6:.1f} Mo -> {path}")

    meta = dict(tract=tract, patch=patch, center=[ra, dec], size_px=int(pc["size_px"]), files=files,
                patch_bbox=patch_bbox(s, rows["i"]))
    with open(out_dir / "images.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("emprise du patch (pixels du tract) :", meta["patch_bbox"])
    return meta
