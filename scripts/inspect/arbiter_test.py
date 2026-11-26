# arbiter_test.py
# Anna Preto
# AstroDeep Project
# Consequences of masking.euclid_arbiter ("nothing is masked where Euclid sees no object", stages/stamps.py) on the
# D1 targets of a few patches: each target's stamp is processed twice by the stamp pipeline (stamps.process_target),
# with the current masking and with the arbiter, in memory only (no product of the full run is touched). Compared:
#   masked_fraction        share of the Rubin stamp replaced by sky
#   halo_ring_z            light measured just outside the masks (a mask on real light: the target's halo cut)
#   companion_light_ratio  light the unmasked catalogued neighbours are predicted to leave on the target: what the
#                          arbiter leaves unmasked (Rubin objects Euclid does not see) shows up here
#   delta_e_mask, mask_pass  shape change by the masking, and the pipeline's masking cut
# Prints medians and rates per i magnitude, and the D1 cut on companion light (dataset1.max_companion_light_ratio)
# each masking would apply; writes the per-object table. Memory: one patch's six coadds (~1.5 GB): a CPU job.
# Every D1 target brighter than i = 22 of each patch and --max-faint of the fainter ones (column weight: how many
# faint targets each one stands for). About 10-30 min per patch.
# Usage: python scripts/inspect/arbiter_test.py <ML config> <out dir> [--patches 2561_67 ...] [--n-patches 6]
#            [--max-faint 300] [--seed 0]

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from dp2_cutouts.mlconfig import MLConfig

COLS = ("masked_fraction", "masked_fraction_in_ellipse", "halo_ring_z", "companion_light_ratio", "delta_e_mask",
        "mask_pass", "n_companions")


def with_arbiter(case, on):
    c = copy.copy(case)
    c.raw = copy.deepcopy(case.raw)
    c.raw["masking"]["euclid_arbiter"] = bool(on)
    return c


def run_patch(cfg, pcfg, patch, max_faint, rng):
    from dp2_cutouts.common import masks
    from dp2_cutouts.pipeline.patches import patch_case
    from dp2_cutouts.sources import euclid_seg
    from dp2_cutouts.sources.rubin_soda import BANDS
    from dp2_cutouts.stages import stamps
    case = patch_case(pcfg, patch)
    d1_meta = cfg.dataset_dir("d1") / "shards" / patch["name"] / "meta.parquet"
    if not d1_meta.exists():
        return None
    ids = set(pd.read_parquet(d1_meta, columns=["object_id"])["object_id"].astype("int64"))
    if not ids:
        return None
    img = stamps.load_images(case)
    cat = pd.read_parquet(case.data("30_processed", "catalog.parquet"))
    rubin, euclid = stamps.companions(case, img, cat)
    mc = case["masking"]
    tiles = euclid_seg.cached_tiles(euclid_seg.cache_dir(mc))
    img["seg"] = euclid_seg.seg_on_grid(tiles, img["wcs"][BANDS.index("i")], img["flux"].shape[1:])
    cc = case["cuts"]
    unusable = masks.bitmask(img["bits"], ["DETECTED", "NO_DATA"] + list(cc["bad_planes"]))
    banks = {n: stamps.SkyBank(img, unusable, n) for n in (cc["stamp_px"], cc["diag_px"])}
    targets = cat[cat["object_id"].astype("int64").isin(ids)]
    # Every bright target (i < 22, where the halos get cut) and a sample of the faint ones (enough to see what the
    # arbiter leaves unmasked around them).
    bright = targets["i_cModel_mag"].astype(float).to_numpy() < 22
    faint = np.flatnonzero(~bright)
    keep = bright.copy()
    keep[rng.permutation(faint)[:max_faint]] = True
    targets = targets[keep]
    variants = {"pipeline": with_arbiter(case, False), "arbiter": with_arbiter(case, True)}
    rows = []
    for row in targets.itertuples(index=False):
        rec = dict(patch=patch["name"], object_id=int(row.object_id), i_mag=float(row.i_cModel_mag),
                   weight=1.0 if row.i_cModel_mag < 22 else len(faint) / max(min(len(faint), max_faint), 1))
        for name, c in variants.items():
            rng = np.random.default_rng([int(cc["seed"]), int(row.object_id) & 0x7FFFFFFF])
            out, _ = stamps.process_target(row, img, c, rubin, euclid, banks, rng)
            for k in COLS:
                rec[f"{name}_{k}"] = out.get(k, np.nan)
        rows.append(rec)
    print(f"{patch['name']}: {len(rows)} D1 targets", flush=True)
    return pd.DataFrame(rows)


def summary(t, cfg):
    lim = cfg["dataset1"].get("max_companion_light_ratio")
    bins = pd.cut(t["i_mag"], [0, 19, 20, 21, 22, 23, 24, 30])
    out = {}
    for name in ("pipeline", "arbiter"):
        g = t.groupby(bins, observed=True)
        out[name] = pd.DataFrame(dict(
            n=g.size(),
            masked_median=g[f"{name}_masked_fraction"].median(),
            halo_z_median=g[f"{name}_halo_ring_z"].median(),
            halo_z_above_10=g[f"{name}_halo_ring_z"].apply(lambda s: (s > 10).mean()),
            light_p90=g[f"{name}_companion_light_ratio"].quantile(0.9),
            light_above_cut=g[f"{name}_companion_light_ratio"].apply(
                lambda s: (s > float(lim)).mean() if lim is not None else np.nan),
            mask_pass=g[f"{name}_mask_pass"].mean()))
    for name, s in out.items():
        print(f"\n== {name} masking (per i magnitude; light_above_cut: share the D1 companion-light cut "
              f"{lim} would reject)\n{s.round(3).to_string()}")
    d = t["arbiter_masked_fraction"] - t["pipeline_masked_fraction"]
    print(f"\nmasked fraction, arbiter - pipeline: median {d.median():.3f}, "
          f"{(d < -0.01).mean():.1%} of the targets lose more than 1 % of masked pixels, "
          f"{(d > 0.01).mean():.1%} gain more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--patches", nargs="*")
    ap.add_argument("--n-patches", type=int, default=6)
    ap.add_argument("--max-faint", type=int, default=300, help="faint D1 targets (i >= 22) per patch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from dp2_cutouts.pipeline.patches import load_patches
    cfg = MLConfig.load(args.config)
    pcfg = cfg.pipeline()
    patches = load_patches(pcfg)
    by_name = {p["name"]: p for p in patches}
    chosen = [by_name[n] for n in (args.patches or [])]
    rng = np.random.default_rng(args.seed)
    for k in rng.permutation(len(patches)):
        if len(chosen) >= len(args.patches or []) + args.n_patches:
            break
        if patches[k] not in chosen:
            chosen.append(patches[k])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = [r for r in (run_patch(cfg, pcfg, p, args.max_faint, rng) for p in chosen) if r is not None]
    t = pd.concat(parts, ignore_index=True)
    t.to_parquet(out / "arbiter_test.parquet", index=False)
    summary(t, cfg)
    print(f"\n-> {out / 'arbiter_test.parquet'} ({len(t)} D1 targets in {len(parts)} patches)")


if __name__ == "__main__":
    main()
