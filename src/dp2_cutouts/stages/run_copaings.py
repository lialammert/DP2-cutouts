# run_copaings.py
# Anna Preto
# AstroDeep Project
# Runs copaings against Euclid only, from the Rubin catalogue read through TAP and the Euclid tiles written from
# HuggingFace. Only the prep step is rewritten (no butler, no HST catalogue); the fof, eot, eot_merge, magfit,
# sb and sb_merge steps are those of copaings, called unmodified.

import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from ..sources import euclid_hf


def _import_copaings(cfg):
    # friendly is imported through FRIENDLY_PATH: its pyproject pins numpy < 2, so it is not installed with pip.
    os.environ.setdefault("FRIENDLY_PATH", os.path.expanduser(cfg["copaings"]["friendly_path"]))
    from copaings import catalogs, stages
    from copaings.config import Case
    return catalogs, stages, Case


def make_case(cfg):
    _, _, Case = _import_copaings(cfg)
    cc = cfg["copaings"]
    return Case(
        name=cfg.name,
        out_root=str(cfg.data("20_copaings", "x").parent),
        tracts=list(cfg["rubin"]["tracts"]),
        sky_box=cfg.sky_box,
        common_frame_xy=True,
        euclid_dir=str(euclid_hf.tiles_dir(cfg)),
        euclid_region=cc["euclid_region"],
        linking_length=float(cc["linking_length"]),
        mag_cut=float(cc["mag_cut"]),
        eot_dense_max=int(cc["eot_dense_max"]),
        n_eot_tasks=int(cc["n_eot_tasks"]),
        n_sb_tasks=int(cc["n_sb_tasks"]),
        telescopes=["Euclid"],
    )


def prep(cfg, case):
    # Same sequence as copaings.catalogs.assemble, without HST.
    catalogs, _, _ = _import_copaings(cfg)
    raw = pd.read_parquet(cfg.data("10_rubin", "catalog", "object_box.parquet"))
    raw["tract_id"] = raw["tract"]

    def in_box(df):
        r0, r1, d0, d1 = case.sky_box
        keep = (df["ra"] >= r0) & (df["ra"] <= r1) & (df["dec"] >= d0) & (df["dec"] <= d1)
        return df[keep].reset_index(drop=True)

    lsst = in_box(catalogs.build_lsst(case, raw))
    # objectId is not in the copaings column table: it is re-attached by row position.
    kept = raw[raw["i_extendedness"] == 1].reset_index(drop=True)
    ra0, ra1, d0, d1 = case.sky_box
    kept = kept[(kept["coord_ra"] >= ra0) & (kept["coord_ra"] <= ra1)
                & (kept["coord_dec"] >= d0) & (kept["coord_dec"] <= d1)].reset_index(drop=True)
    if len(kept) != len(lsst) or not (kept["coord_ra"].values == lsst["ra"].values).all():
        raise RuntimeError("cannot re-attach objectId to the copaings catalogue")
    lsst["objectId"] = kept["objectId"].values
    lsst["patch"] = kept["patch"].values

    euclid = in_box(catalogs.build_euclid(case, lsst))

    meta = {"n_raw": {"LSST": len(lsst), "Euclid": len(euclid)}}
    ll, ee = catalogs._radec(lsst), catalogs._radec(euclid)
    meta["area_deg2"] = {"LSST": catalogs.occupied_area(lsst["ra"], lsst["dec"]),
                         "Euclid": catalogs.occupied_area(euclid["ra"], euclid["dec"])}
    euclid["in_lsst_footprint"] = catalogs.in_hull(ee, ll)
    lsst["in_euclid_footprint"] = catalogs.in_hull(ll, ee)
    lsst["in_hst_footprint"] = False

    lsst["vis_blendedness"] = lsst["i_blendedness"]
    lsst["blendedness"] = lsst["i_blendedness"]
    euclid["mag_i"] = euclid["vis_mag"]

    m = case.mag_cut
    lsst = lsst[lsst["i_cModel_mag"] <= m].reset_index(drop=True)
    euclid = euclid[euclid["vis_mag"] <= m].reset_index(drop=True)
    meta["n_after_cut"] = {"LSST": len(lsst), "Euclid": len(euclid)}

    lsst.to_parquet(case.path("catalogs", "lsst.parquet"))
    euclid.to_parquet(case.path("catalogs", "euclid.parquet"))
    # copaings always reads hst.parquet; an empty table is enough since only Euclid is processed.
    pd.DataFrame({"ra": [], "dec": [], "F814_MAG": [], "mag_i": []}).to_parquet(case.path("catalogs", "hst.parquet"))
    catalogs.save_meta(case.path("catalogs", "meta.json"), meta)
    return meta


def _eot_task(args):
    cfg, task_id = args
    _, stages, _ = _import_copaings(cfg)
    case = make_case(cfg)
    stages.eot(case, "Euclid", task_id, case.n_eot_tasks)


def _sb_task(args):
    cfg, task_id = args
    _, stages, _ = _import_copaings(cfg)
    case = make_case(cfg)
    stages.sb(case, "Euclid", task_id, case.n_sb_tasks)


def run(cfg):
    _, stages, _ = _import_copaings(cfg)
    case = make_case(cfg)
    meta = prep(cfg, case)
    print("prep:", json.dumps(meta))

    stages.fof(case, "Euclid")
    with ProcessPoolExecutor(max_workers=case.n_eot_tasks) as ex:
        list(ex.map(_eot_task, [(cfg, i) for i in range(case.n_eot_tasks)]))
    stages.eot_merge(case, "Euclid", case.n_eot_tasks)

    # magfit loops over the TELESCOPES constant (Euclid and HST), not over case.telescopes.
    stages.TELESCOPES = ("Euclid",)
    stages.magfit(case)

    with ProcessPoolExecutor(max_workers=case.n_sb_tasks) as ex:
        list(ex.map(_sb_task, [(cfg, i) for i in range(case.n_sb_tasks)]))
    stages.sb_merge(case, case.n_sb_tasks)
    compare_reference(cfg, case)


def compare_reference(cfg, case):
    ref = cfg.raw.get("reference")
    if not ref:
        return
    meta = json.load(open(case.path("catalogs", "meta.json")))
    fof = json.load(open(case.path("groups", "E_fof_stats.json")))
    sbs = json.load(open(case.path("sb", "sb_merge_stats.json")))["Euclid"]
    eot = len(stages_load_groups(cfg, case))
    rows = []
    for k in ("LSST", "Euclid"):
        rows.append((f"n_raw {k}", ref["n_raw"][k], meta["n_raw"][k]))
        rows.append((f"n_after_cut {k}", ref["n_after_cut"][k], meta["n_after_cut"][k]))
    rows.append(("fof groups Euclid", ref["fof_groups_Euclid"], fof["n_groups"]))
    rows.append(("eot groups Euclid", ref["eot_groups_Euclid"], eot))
    for k, v in ref["sb_merge_Euclid"].items():
        rows.append((k, v, sbs[k]))
    print(f"\n{'quantity':<26}{'reference':>12}{'obtained':>12}")
    for name, r, o in rows:
        print(f"{name:<26}{r:>12}{o:>12}{'' if r == o else '   <-- mismatch'}")


def stages_load_groups(cfg, case):
    _, stages, _ = _import_copaings(cfg)
    return stages.load_groups(case, "Euclid")
