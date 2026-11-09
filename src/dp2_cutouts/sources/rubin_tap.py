# rubin_tap.py
# Anna Preto
# AstroDeep Project
# Reads the Rubin DP2 Object catalogue through the TAP service of the Rubin Science Platform, within a sky box.
# Replaces the butler read of the copaings prep step: the TAP table is already restricted to primary objects
# (no duplicates between patches, no parents), checked on the demo box.

import os

import pandas as pd
import pyvo
import requests

BANDS = "ugrizy"

# Columns read by copaings (DEFAULT_LSST_COLUMNS), minus i_centroid_x/y and i_fwhm which do not exist in DP2:
# copaings recomputes x, y from ra/dec (common_frame_xy).
COPAINGS_COLUMNS = [
    "coord_ra", "coord_dec",
    "i_ixx", "i_iyy", "i_ixy", "i_ixxPSF", "i_iyyPSF",
    "i_cModelFlux", "i_cModelFluxErr", "i_blendedness", "i_extendedness",
    "r_cModelFlux", "r_cModelFluxErr", "r_blendedness",
    "g_cModelFlux", "g_cModelFluxErr", "g_blendedness",
    "u_cModelFlux", "z_cModelFlux", "y_cModelFlux",
]

# Extra columns kept for later steps: identifiers, position in the sky tiling, PSF moments in all six bands
# (PSF size and ellipticity per band).
EXTRA_COLUMNS = (
    ["objectId", "parentObjectId", "tract", "patch", "detect_isIsolated"]
    + [f"{b}_{m}PSF" for b in BANDS for m in ("ixx", "iyy", "ixy")]
)


def tap_service(url, token_file):
    # The token is read from the file and passed as a header; it is never printed or logged.
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
    # Every row in the box, without any cut: the pre-cut catalogue is used for reweighting.
    rc = cfg["rubin"]
    tap = tap_service(rc["tap_url"], rc["token_file"])
    columns = list(dict.fromkeys(COPAINGS_COLUMNS + EXTRA_COLUMNS))
    # The edges of an ADQL POLYGON are great-circle arcs, not lines of constant declination: we read with a
    # margin, and the exact ra/dec box is applied afterwards (copaings in_box).
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
        raise RuntimeError("duplicate objectId in the TAP response")
    return df


def run(cfg):
    df = fetch_objects(cfg)
    out = cfg.data("10_rubin", "catalog", "object_box.parquet")
    df.to_parquet(out, index=False)
    print(f"Rubin: {len(df):,} objects, {df.shape[1]} columns -> {out}")
    return df
