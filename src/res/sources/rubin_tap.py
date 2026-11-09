# rubin_tap.py
# Anna Preto
# AstroDeep Project
# Lecture du catalogue Object de Rubin DP2 par le service TAP de la plateforme Rubin, dans une boîte du ciel.
# Remplace la lecture par le butler de l'étape prep de copaings : la table TAP est déjà réduite aux objets
# primaires (pas de doublon entre patches, pas de parent), vérifié sur la boîte de démo.

import os

import pandas as pd
import pyvo
import requests

BANDS = "ugrizy"

# Colonnes lues par copaings (DEFAULT_LSST_COLUMNS), moins i_centroid_x/y et i_fwhm qui n'existent pas en DP2 :
# copaings recalcule x, y depuis ra/dec (common_frame_xy).
COPAINGS_COLUMNS = [
    "coord_ra", "coord_dec",
    "i_ixx", "i_iyy", "i_ixy", "i_ixxPSF", "i_iyyPSF",
    "i_cModelFlux", "i_cModelFluxErr", "i_blendedness", "i_extendedness",
    "r_cModelFlux", "r_cModelFluxErr", "r_blendedness",
    "g_cModelFlux", "g_cModelFluxErr", "g_blendedness",
    "u_cModelFlux", "z_cModelFlux", "y_cModelFlux",
]

# Colonnes supplémentaires à garder pour la suite : identifiants, position dans le découpage du ciel,
# moments de la PSF dans les six bandes (taille et ellipticité de la tache par couleur).
EXTRA_COLUMNS = (
    ["objectId", "parentObjectId", "tract", "patch", "detect_isIsolated"]
    + [f"{b}_{m}PSF" for b in BANDS for m in ("ixx", "iyy", "ixy")]
)


def tap_service(url, token_file):
    # Le jeton est lu depuis le fichier et passé en en-tête ; il n'est jamais affiché ni journalisé.
    with open(os.path.expanduser(token_file)) as f:
        token = f.read().strip()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    del token
    return pyvo.dal.TAPService(url, session=session)


def box_query(table, columns, sky_box, tracts=None):
    ra0, ra1, dec0, dec1 = sky_box
    where = (f"CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
             f"POLYGON('ICRS', {ra0}, {dec0}, {ra1}, {dec0}, {ra1}, {dec1}, {ra0}, {dec1})) = 1")
    if tracts:
        where += f" AND tract IN ({', '.join(str(int(t)) for t in tracts)})"
    return f"SELECT {', '.join(columns)} FROM {table} WHERE {where}"


def fetch_objects(cfg):
    # Toutes les lignes de la boîte, sans aucune coupure : le catalogue d'avant coupure sert à la repondération.
    rc = cfg["rubin"]
    tap = tap_service(rc["tap_url"], rc["token_file"])
    columns = list(dict.fromkeys(COPAINGS_COLUMNS + EXTRA_COLUMNS))
    # Les côtés d'un POLYGON ADQL sont des arcs de grand cercle, pas des lignes de déclinaison constante :
    # on lit avec une marge, et la boîte exacte en ra/dec est appliquée ensuite (in_box de copaings).
    m = float(rc.get("margin_deg", 0.0))
    ra0, ra1, dec0, dec1 = cfg.sky_box
    box = [ra0 - m, ra1 + m, dec0 - m, dec1 + m]
    job = tap.submit_job(box_query(rc["table"], columns, box, rc.get("tracts")))
    try:
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=1800)
        job.raise_if_error()
        df = job.fetch_result().to_table().to_pandas()
    finally:
        job.delete()
    if df["objectId"].duplicated().any():
        raise RuntimeError("objectId en double dans la réponse TAP")
    return df


def run(cfg):
    df = fetch_objects(cfg)
    out = cfg.data("10_rubin", "catalog", "object_box.parquet")
    df.to_parquet(out, index=False)
    print(f"Rubin : {len(df):,} objets, {df.shape[1]} colonnes -> {out}")
    return df
