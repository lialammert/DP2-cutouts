# catalog_near.py
# Anna Preto
# AstroDeep Project
# Every catalogued object around one target of a processed patch, straight from the raw catalogues: the Rubin
# Object table of the patch box (object_box.parquet, no cut) and the Euclid MER catalogue of the box
# (euclid_box.parquet, no cut), with the MER flags. Says for each Euclid source whether it enters the companion list
# used for masking (stamps.companions drops the sources without a finite size). Offsets are on the sky: west
# (right on the stamps, x grows westwards) and north (up). Light: catalogues only, fine on a login node.
# Usage: python scripts/inspect/catalog_near.py <pipeline config> <tract_patch> <object_id> [--radius 6]

import argparse

import numpy as np
import pandas as pd

from dp2_cutouts.config import Config
from dp2_cutouts.pipeline.patches import load_patches, patch_case


def offsets(ra, dec, ra0, dec0):
    # Small-angle offsets in arcsec: west (= -east) and north.
    east = (np.asarray(ra) - ra0) * np.cos(np.radians(dec0)) * 3600.0
    north = (np.asarray(dec) - dec0) * 3600.0
    return -east, north


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("patch")
    ap.add_argument("object_id", type=int)
    ap.add_argument("--radius", type=float, default=6.0, help="arcsec")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    case = patch_case(cfg, load_patches(cfg, [args.patch])[0])
    cat = pd.read_parquet(case.data("30_processed", "catalog.parquet"))
    t = cat[cat["object_id"] == args.object_id]
    if t.empty:
        raise SystemExit(f"object {args.object_id} is not in {args.patch}")
    t = t.iloc[0]
    ra0, dec0 = float(t["ra"]), float(t["dec"])
    print(f"target {args.object_id}: ra {ra0:.6f} dec {dec0:.6f}, i = {t['i_cModel_mag']:.2f}; Euclid counterpart "
          f"{t.get('euclid_object_id')} (VIS {t.get('euclid_vis_mag')})")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    rub = pd.read_parquet(case.data("10_rubin", "catalog", "object_box.parquet"))
    w_, n_ = offsets(rub["coord_ra"], rub["coord_dec"], ra0, dec0)
    rub = rub.assign(west=w_, north=n_, sep=np.hypot(w_, n_))
    rub = rub[rub["sep"] <= args.radius].sort_values("sep")
    with np.errstate(divide="ignore", invalid="ignore"):
        rub["i_mag"] = 31.4 - 2.5 * np.log10(rub["i_cModelFlux"])
    cols = [c for c in ("objectId", "sep", "west", "north", "i_mag", "i_extendedness", "parentObjectId",
                        "detect_isIsolated") if c in rub]
    print(f"\nRubin Object table within {args.radius}\" ({len(rub)}):")
    print(rub[cols].round(3).to_string(index=False))

    eu = pd.read_parquet(case.data("00_euclid", "euclid_box.parquet"))
    w_, n_ = offsets(eu["mer_catalogue_right_ascension"], eu["mer_catalogue_declination"], ra0, dec0)
    eu = eu.assign(west=w_, north=n_, sep=np.hypot(w_, n_))
    eu = eu[eu["sep"] <= args.radius].sort_values("sep")
    with np.errstate(divide="ignore", invalid="ignore"):
        eu["vis_mag"] = 31.4 - 2.5 * np.log10(eu["mer_catalogue_flux_vis_1fwhm_aper"] * 1e3)
    size = eu["mer_catalogue_semimajor_axis"].astype(float)
    eu["in_companion_list"] = np.isfinite(size) & np.isfinite(eu["mer_catalogue_ellipticity"].astype(float))
    eu["is_counterpart"] = eu["object_id"].astype("Int64") == t.get("euclid_object_id")
    cols = ["object_id", "sep", "west", "north", "vis_mag", "mer_catalogue_semimajor_axis",
            "mer_catalogue_point_like_flag", "mer_catalogue_spurious_flag", "mer_catalogue_det_quality_flag",
            "mer_catalogue_extended_flag", "in_companion_list", "is_counterpart"]
    print(f"\nEuclid MER catalogue within {args.radius}\" ({len(eu)}):")
    print(eu[[c for c in cols if c in eu]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
