# d1_views.py
# Anna Preto
# AstroDeep Project
# Pixels of D1 objects (and of objects the cuts rejected) for the private viewer (export_preview.py), read from the
# products the stamp pipeline left in each patch folder (no network):
#   10_rubin/images/deep_coadd_<patch>_<band>.fits   raw Rubin stamp (before neighbours are replaced by sky) and the
#                                                   DETECTED plane, cut at the stamp's box (stages/stamps.py)
#   30_processed/stamps.parquet                      companion_mask (the neighbour mask the pipeline applied) and the
#                                                   final stamp (neighbours replaced by sky)
#   30_processed/table.parquet                       every catalogue row with each cut's value and pass flag
#   Euclid MMU mirror + MER segmentation             VIS cutout and its neighbour mask (rejected objects only: D1
#                                                   objects take them from D1)
# Neighbour mask split by origin ("pipeline" masking, stamps.companion_mask_segmentation_catalog):
#   rubin        pixels of the Rubin detection footprints of other objects (stamps.rubin_footprint_mask), plus the
#                rim the Euclid segments of those same neighbours add around them;
#   euclid_only  pieces of the mask touching no Rubin footprint of another object: neighbours inside the target's own
#                Rubin footprint (Euclid sees them apart; deblender children of the same parent, e.g. a bright galaxy
#                shredded in its wings) or not detected by Rubin, masked by their Euclid segment or catalogue ellipse.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, label

from dp2_cutouts.common import masks
from dp2_cutouts.sources.rubin_soda import BANDS
from dp2_cutouts.stages.stamps import rubin_footprint_mask

# Order in which the final dataset applies the cuts (stages/package.py, package.apply); viz.STEPS labels them.
STEPS = ("cut1", "cut2", "in_region", "cut3", "cut4", "cut5", "cut6", "cut7", "cut8", "mask")
REASONS = {"cut1": "cut 1: Euclid point-like or spurious", "cut2": "cut 2: not one-to-one (S_b > 0)",
           "in_region": "stamp outside the region", "cut3": "cut 3: too close to the patch edge",
           "cut4": "cut 4: unexposed pixel", "cut5": "cut 5: damaged pixel near the target",
           "cut6": "cut 6: S/N_i <= 5", "cut7": "cut 7: not the same pixel in the six bands",
           "cut8": "cut 8: |de| between apertures > 0.02", "mask": "masking: |de| before / after > 0.02",
           "no_euclid_counterpart": "D1: no Euclid counterpart", "companion_light": "D1: companion light on the target",
           "halo_ring": "D1: neighbour halo past the masks", "no_euclid_image": "D1: no Euclid image",
           "no_vis_band": "D1: no VIS band", "euclid_scale_mismatch": "D1: Euclid pixel scale mismatch",
           "euclid_invalid_pixels": "D1: too many invalid VIS pixels"}


def first_failure(t):
    # Per catalogue row of table.parquet: the first cut of STEPS it fails, None when it passes them all.
    reason = np.full(len(t), None, dtype=object)
    for step in STEPS:
        col = "in_region" if step == "in_region" else f"{step}_pass"
        if col not in t:
            continue
        ok = t[col].astype("boolean").fillna(False).to_numpy(dtype=bool)
        reason[(reason == None) & ~ok] = step  # noqa: E711 (element-wise)
    return reason


def split_mask(need, rubin_fp):
    # need: the pipeline's neighbour mask; rubin_fp: its Rubin-footprint part. Returns (rubin, euclid_only): the
    # connected pieces of need outside the footprints that touch none of them are the Euclid-only neighbours.
    rubin_fp = rubin_fp & need
    rest = need & ~rubin_fp
    euclid_only = np.zeros_like(need)
    if rest.any():
        lab, n = label(rest, structure=np.ones((3, 3), int))
        near = binary_dilation(rubin_fp, iterations=1) if rubin_fp.any() else np.zeros_like(need)
        touching = set(np.unique(lab[near & rest]))
        for k in range(1, n + 1):
            if k not in touching:
                euclid_only |= lab == k
    return need & ~euclid_only, euclid_only


class PatchPixels:
    # Raw Rubin stamps and Rubin-footprint masks of one patch, read through FITS sections (the coadds stay on disk).
    def __init__(self, pcfg, patch):
        from dp2_cutouts.pipeline.patches import patch_case
        self.case = patch_case(pcfg, patch)
        img_dir = self.case.data("10_rubin", "images", "x").parent
        files = json.load(open(img_dir / "images.json"))["files"]
        self.hdus = {b: fits.open(img_dir / files[b], memmap=True) for b in BANDS}
        hi = self.hdus["i"]
        self.wcs = WCS(hi["IMAGE"].header)
        self.shape = (int(hi["IMAGE"].header["NAXIS2"]), int(hi["IMAGE"].header["NAXIS1"]))
        self.det_bit = masks.bitmask(masks.plane_bits(hi["MASK"].header), ["DETECTED"])
        self.n = int(self.case["cuts"]["stamp_px"])
        self.grow = int(self.case["masking"]["grow_px"])
        self.method = self.case["masking"]["method"]

    def close(self):
        for h in self.hdus.values():
            h.close()

    def _cut(self, b, ext, oy, ox):
        hdu = self.hdus[b][ext]
        try:
            return np.asarray(hdu.section[oy:oy + self.n, ox:ox + self.n])
        except (AttributeError, TypeError):
            return np.asarray(hdu.data[oy:oy + self.n, ox:ox + self.n])

    def raw(self, ra, dec):
        # Raw six-band stamp (n, n, 6) and variance, and the Rubin-footprint mask, on the pipeline's stamp box
        # (stamps.process_target: integer centre floor(x + 0.5) of the i-band position). None off the image.
        x, y = self.wcs.world_to_pixel_values(float(ra), float(dec))
        half = self.n // 2
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        ox, oy = ix - half, iy - half
        if ox < 0 or oy < 0 or ox + self.n > self.shape[1] or oy + self.n > self.shape[0]:
            return None
        flux = np.stack([self._cut(b, "IMAGE", oy, ox) for b in BANDS], -1).astype(np.float32)
        var = np.stack([self._cut(b, "VARIANCE", oy, ox) for b in BANDS], -1).astype(np.float32)
        det = np.any(np.stack([(self._cut(b, "MASK", oy, ox).astype(np.int64) & self.det_bit) != 0 for b in BANDS]),
                     axis=0)
        fp, _ = rubin_footprint_mask(det, x - ox, y - oy, self.grow)
        return dict(flux=flux, var=var, rubin_fp=fp)

    def stamps(self, ids, final=False):
        # companion_mask (and the final stamp, (n, n, 6), with final=True) of the given objects from stamps.parquet.
        path = self.case.data("30_processed", "stamps.parquet")
        cols = ["object_id", "companion_mask"] + (["flux", "ivar"] if final else [])
        t = pq.read_table(path, columns=cols, filters=[("object_id", "in", [int(i) for i in ids])]).to_pandas()
        n, out = self.n, {}
        for r in t.itertuples(index=False):
            d = dict(need=np.asarray(r.companion_mask, bool).reshape(n, n))
            if final:
                d["flux"] = np.asarray(r.flux, np.float32).reshape(len(BANDS), n, n).transpose(1, 2, 0)
                d["ivar"] = np.asarray(r.ivar, np.float32).reshape(len(BANDS), n, n).transpose(1, 2, 0)
            out[int(r.object_id)] = d
        return out


def vis_cutouts(cfg, pcfg, case, objs):
    # VIS cutouts of rejected objects and their Euclid neighbour mask (D1's rules, dataset1.euclid_neighbour_need).
    # objs: euclid_object_id, euclid_ra, euclid_dec. Returns {euclid_object_id: (flux, ivar, mask, fill)}.
    from dp2_cutouts.pairs import common
    from dp2_cutouts.pairs.dataset1 import euclid_neighbour_need
    from dp2_cutouts.pairs.dataset2 import euclid_ellipses, euclid_table, rubin_to_euclid_px
    from dp2_cutouts.sources import euclid_images, euclid_seg
    objs = objs.dropna(subset=["euclid_object_id", "euclid_ra", "euclid_dec"])
    if not len(objs):
        return {}
    ec = cfg["euclid"]
    eb = list(ec["bands"])
    vis, n_e = eb.index("VIS"), int(ec["stamp_px"])
    objects = pd.DataFrame(dict(object_id=objs["euclid_object_id"].astype("int64").to_numpy(),
                                ra=objs["euclid_ra"].astype(float).to_numpy(),
                                dec=objs["euclid_dec"].astype(float).to_numpy()))
    cuts = euclid_images.fetch(ec, objects, case.sky_box)
    e_tab = euclid_table(case)
    a_r, b_r, th = euclid_ellipses(e_tab, cfg, common.RUBIN_SCALE)
    k_px = rubin_to_euclid_px(cfg)
    e_tab["a_e"], e_tab["b_e"], e_tab["theta"] = a_r * k_px, b_r * k_px, th
    paths = euclid_seg.cached_tiles(euclid_seg.cache_dir(pcfg["masking"]))
    tiles = euclid_seg.TileSet(paths).__enter__() if paths else None
    out = {}
    try:
        for eid, cut in cuts.items():
            f, iv, m, present = euclid_images.to_band_array(cut, eb)
            if not present[vis]:
                continue
            fill, _ = euclid_neighbour_need(cut, eid, e_tab, cfg, tiles, n_e)
            out[int(eid)] = (f[..., vis], iv[..., vis], m[..., vis], fill)
    finally:
        if tiles is not None:
            tiles.__exit__(None, None, None)
    return out


def rejected_candidates(cfg, pcfg, patches, d1_shards):
    # Every rejected object of the given patches with its reason: the first pipeline cut it fails (owned rows of
    # table.parquet), or D1's own reason (rejected.parquet of the patch's D1 shard). Columns of the table kept.
    from dp2_cutouts.pipeline.patches import owned, patch_case
    keep_cols = ["object_id", "ra", "dec", "tract", "patch", "euclid_object_id", "euclid_ra", "euclid_dec",
                 "euclid_vis_mag", "i_cModel_mag", "masked_fraction", "delta_e_mask", "halo_ring_z",
                 "companion_light_ratio", "in_region"] + [f"cut{k}_{s}" for k in range(1, 9) for s in ("pass", "value")] \
        + ["mask_pass"]
    rows = []
    for p in patches:
        case = patch_case(pcfg, p)
        path = case.data("30_processed", "table.parquet")
        if not Path(path).exists():
            continue
        names = set(pq.read_schema(path).names)
        t = pd.read_parquet(path, columns=[c for c in keep_cols if c in names])
        t = t[owned(t, p).to_numpy()].reset_index(drop=True)
        if "euclid_object_id" in t:                     # nullable 64-bit: concatenating with an all-missing column
            t["euclid_object_id"] = t["euclid_object_id"].astype("Int64")   # of another patch made it float64
        t["reason"] = first_failure(t)
        rej = d1_shards / p["name"] / "rejected.parquet"
        if rej.exists():
            r = pd.read_parquet(rej)
            if len(r) and "object_id" in r:
                why = dict(zip(r["object_id"].astype("int64"), r["reason"].astype(str)))
                passed = t["reason"].isna().to_numpy()
                ids = t["object_id"].astype("int64").to_numpy()
                t.loc[passed, "reason"] = [why.get(int(i)) for i in ids[passed]]
        t = t[t["reason"].notna()]
        t["patch_name"] = p["name"]
        rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
