# phases.py
# Anna Preto
# AstroDeep Project
# The three phases of a patch and the final assembly.
#   plan    (network: Rubin catalogue, Euclid catalogue) : copaings, catalogue table, cuts 1, 2, 6
#   fetch   (network: Rubin pixels)                      : the six images of the full patch, only if it has targets
#   process (no network)                                 : pixel cuts 3, 4, 5, 7, 8, masking
#   assemble                                             : final dataset from the objects owned by each patch
# A patch owns the objects whose (tract, patch) in the Rubin Object table are its own; objects of the catalogue
# margin are context for copaings and masking only, and never produce a row.

import hashlib
import json
import shutil

import numpy as np
import pandas as pd

from .. import adapters
from ..stages import catalog_table, package, run_copaings, stamps
from . import manifest
from .patches import load_patches, owned, patch_case


def plan_patch(cfg, patch):
    case = patch_case(cfg, patch)
    adapters.fetch_rubin_catalog(case)
    adapters.fetch_euclid_catalog(case)
    run_copaings.run(case)
    t = catalog_table.run(case)
    own = owned(t, patch)
    targets = own & t["cut1_pass"] & t["cut2_pass"]
    return "done", dict(n_objects_in_box=int(len(t)), n_owned=int(own.sum()), n_targets=int(targets.sum()),
                        sky_box=case.sky_box)


def _images_intact(case, patch):
    # Pixels already on disk for this patch, complete and unchanged (relaunch after an interruption).
    p = case.data("10_rubin", "images", "images.json", makedirs=False)
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    for b, name in meta["files"].items():
        f = p.parent / name
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != meta["sha256"][b]:
            return None
    return meta


def fetch_patch(cfg, patch):
    case = patch_case(cfg, patch)
    man = manifest.load(cfg, patch)
    if man["phases"]["plan"]["n_targets"] == 0:
        return "skipped", dict(reason="no target after cuts 1 and 2")
    if not patch["complete"]:
        return "skipped", dict(reason=f"bands present: {patch['bands']}")
    meta = _images_intact(case, patch)
    reused = meta is not None
    if not reused:
        meta = adapters.fetch_rubin_pixels(case, patch)
    info = dict(reused=reused, patch_bbox=meta["patch_bbox"], source_ids=meta["source_ids"], sha256=meta["sha256"])
    # Euclid MER segmentation tiles for the segmentation maskings (cached, shared by the patches).
    from ..sources import euclid_seg
    mc = case["masking"]
    if mc["method"] in euclid_seg.SEG_METHODS:
        paths = euclid_seg.fetch_for_sky_box(case.sky_box, euclid_seg.cache_dir(mc))
        info["seg_tiles"] = [str(p) for p in paths]
    return "done", info


def process_patch(cfg, patch):
    case = patch_case(cfg, patch)
    man = manifest.load(cfg, patch)
    if man["phases"]["fetch"]["status"] == "skipped":
        return "skipped", dict(reason="no pixels")
    table = stamps.run(case)
    t = table[owned(table, patch) & table["pixel_target"]]
    sel = package.selection(t, cfg["package"]["apply"])
    return "done", dict(n_pixel_targets=int(len(t)), n_final=int(sel.sum()))


PHASE_FUNCTIONS = {"plan": plan_patch, "fetch": fetch_patch, "process": process_patch}


def run(cfg, phase, names=None, index=None, force=False):
    patches = load_patches(cfg, names, index)
    return [manifest.run_phase(cfg, p, phase, PHASE_FUNCTIONS[phase], force) for p in patches]


def assemble(cfg):
    # Final dataset of every processed patch, written one patch at a time (memory independent of the number of
    # patches), and full catalogue (before cuts) of every planned patch.
    root = cfg.data("final", "x").parent
    for sub in ("data", "catalog"):
        if (root / sub).exists():
            shutil.rmtree(root / sub)
    max_rows = int(cfg["package"]["max_rows_per_file"])
    planned = []
    for patch in load_patches(cfg):
        man = manifest.load(cfg, patch)
        if manifest.is_done(cfg, man, "plan"):
            processed = (man["phases"].get("process", {}).get("status") == "done"
                         and manifest.is_done(cfg, man, "process"))
            planned.append((patch, processed))
    if not planned:
        raise RuntimeError("no planned patch")

    def owned_table(patch, processed):
        case = patch_case(cfg, patch)
        t = pd.read_parquet(case.data("30_processed", "table.parquet" if processed else "catalog.parquet"))
        t = t[owned(t, patch)].reset_index(drop=True)
        for c in ("in_region", "pixel_target"):
            if c in t.columns:
                t[c] = t[c].astype("boolean").fillna(False).astype(bool)
        return t.drop(columns=[c for c in package.DROP if c in t.columns])

    # Common catalogue schema: that of a processed patch (patches without pixels get empty pixel columns).
    ref = next(((p, pr) for p, pr in planned if pr), planned[0])
    schema = package.to_arrow(owned_table(*ref)).schema

    ids, moments, meta0, n_patches = [], [], None, 0
    for patch, processed in planned:
        t = owned_table(patch, processed)
        ids.append(t["object_id"].to_numpy(np.int64))
        cat = package.to_arrow(t.reindex(columns=schema.names)).cast(schema)
        package.write_partitioned(cat, root / "catalog", 100_000, basename=f"part-{patch['name']}-{{i}}.parquet",
                                  append=True)
        if not processed:
            continue
        case = patch_case(cfg, patch)
        meta = json.loads(case.data("30_processed", "stamps.json").read_text())
        if meta0 is not None and meta["plane_bits"] != meta0["plane_bits"]:
            raise RuntimeError(f"patch {patch['name']}: mask planes differ from the other patches")
        meta0 = meta0 or meta
        keep = package.selection(t, cfg["package"]["apply"])
        final_ids = t.loc[keep, "object_id"].tolist()
        st = pd.read_parquet(case.data("30_processed", "stamps.parquet"),
                             columns=["object_id", "flux", "ivar", "maskbits"], filters=[("object_id", "in", final_ids)])
        final, arrow = package.final_rows(cfg, t, st, meta)
        package.write_partitioned(arrow, root / "data", max_rows, basename=f"part-{patch['name']}-{{i}}.parquet",
                                  append=True)
        moments.append(final["has_valid_i_moments"])
        n_patches += 1
        print(f"[{patch['name']}] {len(final):,} stamps")
    all_ids = np.concatenate(ids)
    if len(np.unique(all_ids)) != len(all_ids):
        raise RuntimeError("an object is owned by two patches")
    if meta0 is None:
        raise RuntimeError("no processed patch")
    finals = pd.concat(moments, ignore_index=True)
    package.write_info(cfg, meta0, root, len(all_ids), finals)
    print(f"final dataset: {len(finals):,} stamps from {n_patches} patches, catalogue {len(all_ids):,} objects "
          f"from {len(planned)} patches -> {root}")
    return finals
