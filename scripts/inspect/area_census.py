# area_census.py
# Anna Preto
# AstroDeep Project
# How many DP2 patches a wider run would take, and how many of them Euclid Q1 covers, before writing its pipeline
# configuration. For each search area: every deep_coadd patch that the Rubin SIA service finds (as the plan step of
# pipeline/patches.py), its six bands, and the fraction of its centre and four corners inside a Euclid MER tile (tile
# outlines from one IRSA SIA query per area, tested locally). No pixel is downloaded; network needed (login node).
# Writes a CSV (one row per patch) and prints the counts per area and tract.
# Usage: python scripts/inspect/area_census.py <pipeline config> <out csv>
#            [--area NAME RA DEC RADIUS_DEG ...]   (default: EDF-F and EDF-S)

import argparse
import re

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from pyvo.dal.sia2 import SIA2Service

from dp2_cutouts.config import Config
from dp2_cutouts.sources.euclid_seg import IRSA_SIA, MOSAIC_COLLECTION
from dp2_cutouts.sources.rubin_soda import BANDS, SIA_URL, session

# Euclid Q1 deep fields (centres of the Q1 release; radii generous: the census keeps only what both cover).
AREAS = [("EDF-F", 52.93, -28.09, 2.0), ("EDF-S", 61.24, -48.42, 3.0)]


def polygon(s_region):
    # "POLYGON [ICRS] ra1 dec1 ra2 dec2 ..." -> (n, 2) array of vertices.
    v = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", str(s_region))]
    return np.array(v[: len(v) // 2 * 2]).reshape(-1, 2)


def rubin_patches(s, ra, dec, r):
    t = SIA2Service(SIA_URL, session=s).search(pos=(ra, dec, r), calib_level=3).to_table()
    rows = []
    for x in t:
        if str(x["dataproduct_subtype"]) != "lsst.deep_coadd":
            continue
        p = polygon(x["s_region"])
        rows.append(dict(tract=int(x["lsst_tract"]), patch=int(x["lsst_patch"]), band=str(x["lsst_band"]),
                         s_ra=float(x["s_ra"]), s_dec=float(x["s_dec"]), ra_min=p[:, 0].min(), ra_max=p[:, 0].max(),
                         dec_min=p[:, 1].min(), dec_max=p[:, 1].max()))
    if not rows:
        return pd.DataFrame()
    b = pd.DataFrame(rows).drop_duplicates(["tract", "patch", "band"])
    geo = ["s_ra", "s_dec", "ra_min", "ra_max", "dec_min", "dec_max"]
    g = b.groupby(["tract", "patch"]).agg(**{c: (c, "first") for c in geo},
                                          bands=("band", lambda x: "".join(k for k in BANDS if k in set(x))))
    g = g.reset_index()
    g["complete"] = g["bands"] == BANDS
    return g


def euclid_tiles(ra, dec, r):
    t = SIA2Service(IRSA_SIA).search(pos=(ra, dec, r), collection=MOSAIC_COLLECTION).to_table()
    polys, seen = [], set()
    for x in t:
        key = str(x["s_region"])
        if key in seen:
            continue
        seen.add(key)
        p = polygon(key)
        if len(p) >= 3:
            polys.append(MplPath(p))
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--area", nargs=4, action="append", metavar=("NAME", "RA", "DEC", "RADIUS_DEG"))
    args = ap.parse_args()
    areas = [(a[0], float(a[1]), float(a[2]), float(a[3])) for a in args.area] if args.area else AREAS
    cfg = Config.load(args.config)
    s = session(cfg["rubin"]["token_file"])
    out = []
    for name, ra, dec, r in areas:
        p = rubin_patches(s, ra, dec, r)
        tiles = euclid_tiles(ra, dec, r)
        print(f"{name}: {len(p)} DP2 patches, {len(tiles)} Euclid MER tile outlines", flush=True)
        if not len(p):
            continue
        pts = np.stack([np.c_[p["s_ra"], p["s_dec"]], np.c_[p["ra_min"], p["dec_min"]],
                        np.c_[p["ra_min"], p["dec_max"]], np.c_[p["ra_max"], p["dec_min"]],
                        np.c_[p["ra_max"], p["dec_max"]]], axis=1)                   # (n patches, 5 points, 2)
        inside = np.zeros(pts.shape[:2], bool)
        for t in tiles:
            inside |= t.contains_points(pts.reshape(-1, 2)).reshape(pts.shape[:2])
        p["euclid_cover_frac"] = inside.mean(axis=1)
        p["area"] = name
        out.append(p)
    t = pd.concat(out, ignore_index=True).drop_duplicates(["tract", "patch"])
    t["name"] = [f"{a}_{b}" for a, b in zip(t["tract"], t["patch"])]
    t.to_csv(args.out, index=False)
    full = t["complete"] & (t["euclid_cover_frac"] == 1)
    part = t["complete"] & (t["euclid_cover_frac"] > 0) & (t["euclid_cover_frac"] < 1)
    print(f"\n{'area':6s} {'patches':>8s} {'6 bands':>8s} {'Euclid full':>12s} {'Euclid part':>12s}")
    for name, g in t.groupby("area"):
        f, q = full[g.index], part[g.index]
        print(f"{name:6s} {len(g):8d} {int(g['complete'].sum()):8d} {int(f.sum()):12d} {int(q.sum()):12d}")
    print(f"{'all':6s} {len(t):8d} {int(t['complete'].sum()):8d} {int(full.sum()):12d} {int(part.sum()):12d}")
    print("\npatches with six bands and full Euclid coverage, per tract:")
    print(t[full].groupby(["area", "tract"]).size().to_string())
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
