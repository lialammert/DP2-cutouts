# probe.py
# Anna Preto
# AstroDeep Project
# Which Euclid image source works from here, and what it returns. To be run first, on a node with internet
# (prepost on Jean Zay), on a handful of objects of one planned patch:
#   - MMU (Hugging Face, HATS): the raw structure of the image column of the first object (band names, array sizes,
#     scale), then the parsed cutouts and the time taken;
#   - IRSA: the SIA query and one cutout.
# Nothing is written outside eval_dir("probe_euclid"); the report says which source to put in euclid.image_source.

import json
import time
import traceback

import numpy as np
import pandas as pd

from ..sources import euclid_images


def _describe(v, depth=0):
    if depth > 3:
        return type(v).__name__
    if isinstance(v, dict):
        return {k: _describe(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [type(v).__name__, len(v), _describe(v[0], depth + 1) if len(v) else None]
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape}{v.dtype}"
    return f"{type(v).__name__}:{str(v)[:40]}"


def objects_of_patch(cfg, patch_name, n):
    from ..pipeline.patches import load_patches, patch_case
    pcfg = cfg.pipeline()
    patch = load_patches(pcfg, [patch_name])[0]
    case = patch_case(pcfg, patch)
    cat = pd.read_parquet(case.data("30_processed", "catalog.parquet"))
    cat = cat[cat["has_euclid_counterpart"].astype(bool)].sort_values("euclid_vis_mag").head(n)
    return pd.DataFrame(dict(object_id=cat["euclid_object_id"].astype("int64"), ra=cat["euclid_ra"].astype(float),
                             dec=cat["euclid_dec"].astype(float))), case.sky_box


def run(cfg, patch_name, n=5):
    ec = cfg["euclid"]
    out_dir = cfg.eval_dir("probe_euclid")
    out_dir.mkdir(parents=True, exist_ok=True)
    objs, box = objects_of_patch(cfg, patch_name, n)
    rep = dict(patch=patch_name, objects=objs.to_dict("records"))

    try:
        t0 = time.time()
        ids = [int(v) for v in objs["object_id"]]
        files = euclid_images.mmu_partition_files(ec["hats_url"], box)
        rep["mmu_layout"] = euclid_images.mmu_layout(files)
        print(json.dumps(rep["mmu_layout"], indent=2), flush=True)
        df = euclid_images.read_mmu_rows(ec["hats_url"], ids, ["object_id", "image"], box,
                                         float(ec.get("mmu_max_group_mb", 4000)))
        rep["mmu_rows"] = int(len(df))
        if len(df):
            first = df["image"].iloc[0]
            recs = euclid_images._as_records(first)
            rep["mmu_raw_structure"] = _describe(recs[0]) if recs else _describe(first)
            rep["mmu_bands_raw"] = [str(r.get("band")) for r in recs]
        cuts = euclid_images.fetch_mmu(ec["hats_url"], objs, list(ec["bands"]), int(ec["stamp_px"]),
                                       float(ec["pixel_scale"]), box, float(ec.get("mmu_max_group_mb", 4000)))
        rep["mmu"] = dict(ok=len(cuts) > 0, n=len(cuts), seconds=round(time.time() - t0, 1),
                          cutouts={str(k): dict(bands=v["bands"], shape=list(v["flux"].shape), source_px=v["source_px"],
                                                source_scale=v["source_scale"], scale_ok=v["scale_ok"],
                                                vis_peak=float(v["flux"][..., 0].max()),
                                                valid_fraction=float((~v["mask"]).mean()))
                                   for k, v in cuts.items()})
    except Exception as e:  # noqa: BLE001 - the report must say what failed
        rep["mmu"] = dict(ok=False, error=repr(e), traceback=traceback.format_exc()[-2000:])

    try:
        t0 = time.time()
        irsa = euclid_images.IrsaCutouts(ec["irsa_sia_url"], ec["irsa_collection"], list(ec["bands"]),
                                         int(ec["stamp_px"]), float(ec["pixel_scale"]))
        r0 = objs.iloc[0]
        c = irsa.cutout(float(r0["ra"]), float(r0["dec"]))
        rep["irsa"] = dict(ok=c is not None, seconds=round(time.time() - t0, 1),
                           bands=c["bands"] if c else None, shape=list(c["flux"].shape) if c else None)
    except Exception as e:  # noqa: BLE001
        rep["irsa"] = dict(ok=False, error=repr(e), traceback=traceback.format_exc()[-2000:])

    works = [s for s in ("mmu", "irsa") if rep.get(s, {}).get("ok")]
    rep["recommendation"] = works[0] if works else "none: neither source answered from this node"
    (out_dir / f"probe_{patch_name}.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("mmu", "irsa", "recommendation") if k in rep}, indent=2, default=str))
    return rep
