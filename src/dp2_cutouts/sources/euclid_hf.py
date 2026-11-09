# euclid_hf.py
# Anna Preto
# AstroDeep Project
# Reads the Euclid Q1 catalogue (MER) from the MultimodalUniverse HuggingFace repository, HATS format, within a
# sky box and without the image column. Writes FITS tiles in the format expected by copaings
# (euclid_q1_mer_final_<healpix>.fits, nside 64, NESTED, MER catalogue column names).

import os

import healpy as hp
import lsdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

# MMU column -> MER name read by copaings (catalogs.build_euclid).
COPAINGS_COLUMNS = {
    "mer_catalogue_right_ascension": "RIGHT_ASCENSION",
    "mer_catalogue_declination": "DECLINATION",
    "mer_catalogue_position_angle": "POSITION_ANGLE",
    "mer_catalogue_semimajor_axis": "SEMIMAJOR_AXIS",
    "mer_catalogue_ellipticity": "ELLIPTICITY",
    "mer_catalogue_flux_vis_1fwhm_aper": "FLUX_VIS_1FWHM_APER",
    "mer_catalogue_fluxerr_vis_1fwhm_aper": "FLUXERR_VIS_1FWHM_APER",
    "mer_catalogue_extended_flag": "EXTENDED_FLAG",
    "object_id": "OBJECT_ID",
}

# Cut 1 columns (neither point-like nor spurious), kept as continuous quantities next to the flags.
CUT1_COLUMNS = [
    "mer_catalogue_point_like_flag", "mer_catalogue_point_like_prob",
    "mer_catalogue_spurious_flag", "mer_catalogue_spurious_prob",
    "mer_catalogue_det_quality_flag", "mer_catalogue_extended_prob", "mer_catalogue_vis_det",
]

# Photometric redshift: probability density, median, intervals and quality flag (recorded, never cut on).
PHZ_COLUMNS = [
    "phz_photo_z_phz_pdf", "phz_photo_z_phz_median",
    "phz_photo_z_phz_70_int", "phz_photo_z_phz_90_int", "phz_photo_z_phz_95_int",
    "phz_photo_z_phz_flags", "phz_classification_phz_classification",
]

COPAINGS_NSIDE = 64

# Angle convention of the Euclid ellipses. copaings computes theta = 270 - (PA + 90) = -PA in the Rubin pixel
# frame, whereas matched pairs give theta = PA + 90 (median offset 1.5 degrees over 385 elongated pairs, against
# 46 degrees for -PA). "measured" writes PA' = 90 - PA in the tiles, so that the copaings formula returns
# theta = PA + 90 without modifying copaings.
PA_CONVENTIONS = ("copaings", "measured")


def tiles_dir(cfg):
    conv = cfg["euclid"].get("pa_convention", "copaings")
    if conv not in PA_CONVENTIONS:
        raise ValueError(f"pa_convention must be one of {PA_CONVENTIONS}")
    return cfg.data("00_euclid", f"copaings_tiles_{conv}", "x").parent


def fetch_box(cfg):
    ec = cfg["euclid"]
    ra0, ra1, dec0, dec1 = cfg.sky_box
    m = float(ec.get("margin_deg", 0.0))
    columns = list(dict.fromkeys(["ra", "dec"] + list(COPAINGS_COLUMNS) + CUT1_COLUMNS + PHZ_COLUMNS))
    cat = lsdb.open_catalog(ec["hats_url"], columns=columns)
    df = cat.box_search(ra=(ra0 - m, ra1 + m), dec=(dec0 - m, dec1 + m)).compute()
    # lsdb returns a NestedFrame; convert back to a plain pandas DataFrame.
    return numpy_backed(pd.DataFrame(df).reset_index(drop=True))


def numpy_backed(df):
    # pyarrow columns -> numpy types: floats with NaN, nullable integers, lists as numpy arrays.
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.ArrowDtype):
            pt = s.dtype.pyarrow_dtype
            if pa.types.is_list(pt) or pa.types.is_large_list(pt):
                vals = pa.chunked_array(s.array._pa_array).to_pylist()
                out[c] = pd.Series([None if v is None else np.asarray(v, dtype=float) for v in vals],
                                   index=df.index, dtype=object)
            elif pa.types.is_floating(pt):
                out[c] = s.to_numpy(dtype=float, na_value=np.nan)
            elif pa.types.is_integer(pt):
                out[c] = s.astype("Int64")
            else:
                out[c] = s.astype(object)
        else:
            out[c] = s
    return out


def read_box(cfg):
    path = cfg.data("00_euclid", "euclid_box.parquet")
    return numpy_backed(pq.read_table(path).to_pandas(ignore_metadata=True, types_mapper=pd.ArrowDtype))


def write_copaings_tiles(df, out_dir, pa_convention="copaings"):
    # One FITS tile per HEALPix pixel at nside 64 NESTED, like the MER files that copaings reads.
    os.makedirs(out_dir, exist_ok=True)
    pix = hp.ang2pix(COPAINGS_NSIDE, df["mer_catalogue_right_ascension"].values,
                     df["mer_catalogue_declination"].values, nest=True, lonlat=True)
    written = []
    for p in np.unique(pix):
        sub = df[pix == p]
        # pyarrow columns with missing values: floats with NaN for FITS, identifier as a 64-bit integer.
        cols = {new: sub[old].to_numpy(dtype=float, na_value=np.nan)
                for old, new in COPAINGS_COLUMNS.items() if old != "object_id"}
        cols["OBJECT_ID"] = sub["object_id"].astype("int64").to_numpy()
        if pa_convention == "measured":
            cols["POSITION_ANGLE"] = 90.0 - cols["POSITION_ANGLE"]
        t = Table(cols)
        path = os.path.join(out_dir, f"euclid_q1_mer_final_{int(p)}.fits")
        t.write(path, overwrite=True)
        written.append(int(p))
    return written


def run(cfg, refetch=True):
    out = cfg.data("00_euclid", "euclid_box.parquet")
    if refetch or not out.exists():
        df = fetch_box(cfg)
        df.to_parquet(out, index=False)
    else:
        df = read_box(cfg)
    conv = cfg["euclid"].get("pa_convention", "copaings")
    tiles = write_copaings_tiles(df, tiles_dir(cfg), conv)
    print(f"Euclid: {len(df):,} objects -> {out}; copaings tiles ({conv}) {tiles}")
    return df
