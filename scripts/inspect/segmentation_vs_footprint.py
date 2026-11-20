# segmentation_vs_footprint.py
# Anna Preto
# AstroDeep Project
# Companion masking with the Euclid segmentation maps against the current footprint masking, on the targets of one
# patch of the pipeline run (by default only those of the local demo region: the 1000 px square at the patch
# centre, the 422 targets of notebooks 02 and 03). Each target goes through the stamp stage once per method (same
# random sky windows); prints how many are cut by the masking decision (mask_pass), by cut 8 and in total, and
# draws raw | footprint masking | segmentation masking for the targets whose fate changes.
# Downloads the MER_SEG tiles that cover the region (about 2.7 GB each, cached): run it on prepost.
# Usage: python scripts/inspect/segmentation_vs_footprint.py <pipeline config> <tract_patch> <out dir>
#            [--seg-grow 0 2 4] [--demo-px 1000 | --demo-px 0 for the whole patch] [--cache DIR]

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dp2_cutouts.common import masks  # noqa: E402
from dp2_cutouts.config import Config  # noqa: E402
from dp2_cutouts.pipeline.patches import load_patches, patch_case  # noqa: E402
from dp2_cutouts.sources import euclid_seg  # noqa: E402
from dp2_cutouts.stages import package, stamps  # noqa: E402


def stretch(img, s):
    return np.arcsinh(img / (0.1 * s)) / np.arcsinh(10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("patch")
    ap.add_argument("out")
    ap.add_argument("--seg-grow", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--demo-px", type=int, default=1000)
    ap.add_argument("--cache", default=os.path.join(os.environ.get("SCRATCH", "."), "euclid_mer_seg"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = Config.load(args.config)
    patch = load_patches(cfg, [args.patch])[0]
    case = patch_case(cfg, patch)
    img = stamps.load_images(case)
    meta = img["meta"]
    cc = case["cuts"]
    H, W = img["flux"].shape[1:]
    cat = pd.read_parquet(case.data("30_processed", "catalog.parquet"))
    targets = cat[cat["cut1_pass"] & cat["cut2_pass"] & (cat["tract"] == meta["tract"])
                  & (cat["patch"] == meta["patch"])].copy()
    w = img["wcs"][stamps.BANDS.index("i")]
    tx, ty = w.world_to_pixel_values(targets["ra"].values, targets["dec"].values)
    targets["ix"], targets["iy"] = np.floor(tx + 0.5).astype(int), np.floor(ty + 0.5).astype(int)

    # Region: the demo square, centred like the local demo cutout on the centre of the patch image (s_ra, s_dec,
    # recorded as "center" in images.json).
    xc, yc = (float(v) for v in w.world_to_pixel_values(*meta["center"]))
    half, dhalf = cc["stamp_px"] // 2, cc["diag_px"] // 2
    if args.demo_px > 0:
        r = args.demo_px // 2
        rx0, rx1, ry0, ry1 = int(xc - r), int(xc + r), int(yc - r), int(yc + r)
        sel = ((targets["ix"] >= rx0 + half) & (targets["ix"] < rx1 - half)
               & (targets["iy"] >= ry0 + half) & (targets["iy"] < ry1 - half))
        targets = targets[sel]
    else:
        rx0, rx1, ry0, ry1 = 0, W, 0, H
    print(f"{len(targets)} targets in the region ({'demo ' + str(args.demo_px) + ' px' if args.demo_px else 'patch'})")

    # Euclid segmentation on the Rubin grid, over the region plus the diagnostic stamp margin.
    gx0, gx1 = max(rx0 - dhalf - 2, 0), min(rx1 + dhalf + 2, W)
    gy0, gy1 = max(ry0 - dhalf - 2, 0), min(ry1 + dhalf + 2, H)
    ra_c, dec_c = w.pixel_to_world_values(0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1))
    radius = 0.2 / 3600 * 0.5 * np.hypot(gx1 - gx0, gy1 - gy0) * 1.1
    tiles = euclid_seg.find_tiles(float(ra_c), float(dec_c), radius)
    print(f"MER tiles: {tiles}  ({time.time() - t0:.0f} s)")
    paths = [euclid_seg.download(euclid_seg.seg_url(t), Path(args.cache) / t) for t in tiles]
    for p in paths:
        print(f"  {p}  {p.stat().st_size / 1e9:.2f} GB")
    from astropy.io import fits
    from astropy.wcs.utils import proj_plane_pixel_scales
    with fits.open(paths[0], memmap=True) as h:
        hdu = euclid_seg._image_hdu(h)
        print("  first tile:", hdu.header.get("BITPIX"), "bits,", hdu.header["NAXIS1"], "x", hdu.header["NAXIS2"],
              "px, scale", np.round(proj_plane_pixel_scales(euclid_seg.WCS(hdu.header)) * 3600, 4), "arcsec")
    seg = np.zeros((4, H, W), np.int64)
    seg[:, gy0:gy1, gx0:gx1] = euclid_seg.seg_on_grid(paths, w.slice((slice(gy0, gy1), slice(gx0, gx1))),
                                                      (gy1 - gy0, gx1 - gx0))
    img["seg"] = seg
    cover = (seg[:, ry0:ry1, rx0:rx1] != 0).any(axis=0)
    print(f"segmentation on the Rubin grid: {cover.mean():.1%} of the region in a segment  ({time.time() - t0:.0f} s)")

    # Same stamp stage, one configuration per method.
    rubin, euclid = stamps.companions(case, img, cat)
    unusable = masks.bitmask(img["bits"], ["DETECTED", "NO_DATA"] + list(cc["bad_planes"]))
    banks = {n: stamps.SkyBank(img, unusable, n) for n in (cc["stamp_px"], cc["diag_px"])}
    methods = {"footprint": case}
    for g in args.seg_grow:
        methods[f"segmentation_grow{g}"] = case.derive(case.name, case.data_root, case.sky_box,
                                                       masking={"method": "segmentation", "seg_grow_px": g})
    ib = stamps.BANDS.index("i")
    n = cc["stamp_px"]
    res, pix = {}, {}
    for name, c in methods.items():
        rows = []
        for row in targets.itertuples(index=False):
            rng = np.random.default_rng([int(cc["seed"]), int(row.object_id) & 0x7FFFFFFF])
            o, arr = stamps.process_target(row, img, c, rubin, euclid, banks, rng)
            rows.append(o)
            if arr is not None:
                pix[(name, int(row.object_id))] = (
                    np.asarray(arr["flux_raw"]).reshape(-1, n, n)[ib], np.asarray(arr["flux"]).reshape(-1, n, n)[ib],
                    np.asarray(arr["companion_mask"]).reshape(n, n))
        df = targets.drop(columns=["ix", "iy"]).merge(pd.DataFrame(rows), on="object_id", how="left")
        df["final"] = package.selection(df, case["package"]["apply"])
        res[name] = df
        print(f"  {name}: done ({time.time() - t0:.0f} s)")

    # Numbers.
    summary = []
    for name, df in res.items():
        R = df[df["in_region"].astype(bool)]
        summary.append(dict(method=name, targets=len(R), cut_by_mask=int((~R["mask_pass"].astype(bool)).sum()),
                            cut_by_cut8=int((~R["cut8_pass"].astype(bool)).sum()),
                            cut_by_mask_or_cut8=int((~(R["mask_pass"].astype(bool) & R["cut8_pass"].astype(bool))).sum()),
                            final_kept=int(R["final"].sum()),
                            median_masked_fraction=round(float(R["masked_fraction"].median()), 3),
                            median_companion_light=round(float(R["companion_light_ratio"].median()), 4),
                            companion_light_above_0p1=int((R["companion_light_ratio"] > 0.1).sum())))
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False))
    summary.to_csv(out / "summary.csv", index=False)
    for name, df in res.items():
        df[["object_id", "in_region", "mask_pass", "delta_e_mask", "cut8_pass", "cut8_value", "masked_fraction",
            "companion_light_ratio", "final"]].to_csv(out / f"targets_{name}.csv", index=False)

    # Changes of fate, footprint -> main segmentation setting (the middle grow value).
    main_name = f"segmentation_grow{args.seg_grow[len(args.seg_grow) // 2]}"
    a = res["footprint"].set_index("object_id")
    b = res[main_name].set_index("object_id")
    lost, gained = a.index[a["final"] & ~b["final"]], a.index[~a["final"] & b["final"]]
    print(f"{main_name} vs footprint: {len(lost)} kept before and cut now, {len(gained)} cut before and kept now")

    # Figures: raw | footprint | segmentation, red = filled pixels.
    def draw(pick, path, title):
        pick = [i for i in pick if ("footprint", int(i)) in pix and (main_name, int(i)) in pix]
        if not pick:
            return
        fig, axes = plt.subplots(len(pick), 3, figsize=(6.3, 2.2 * len(pick)), squeeze=False)
        for r_, oid in enumerate(pick):
            raw, f_old, m_old = pix[("footprint", int(oid))]
            _, f_new, m_new = pix[(main_name, int(oid))]
            s = float(np.nanpercentile(np.abs(raw), 99.5)) or 1.0
            for ax, im, m, t in ((axes[r_, 0], raw, None, "raw"),
                                 (axes[r_, 1], f_old, m_old, "footprint (now)"),
                                 (axes[r_, 2], f_new, m_new, main_name.replace("_", " "))):
                ax.imshow(stretch(im, s), origin="lower", cmap="gray_r")
                if m is not None and m.any():
                    ax.contour(m, levels=[0.5], colors="tab:red", linewidths=0.6)
                ax.set_xticks([])
                ax.set_yticks([])
                if r_ == 0:
                    ax.set_title(t, fontsize=8)
            fa, fb = ("kept" if a.loc[oid, "final"] else "cut"), ("kept" if b.loc[oid, "final"] else "cut")
            axes[r_, 0].set_ylabel(f"S/N_i {a.loc[oid, 'cut6_value']:.0f}: {fa} -> {fb}\ndelta_e mask "
                                   f"{a.loc[oid, 'delta_e_mask']:.3f} -> {b.loc[oid, 'delta_e_mask']:.3f}", fontsize=6)
        fig.suptitle(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / path, dpi=130)

    diff = (b["masked_fraction"] - a["masked_fraction"]).abs()
    pick = list(lost[:5]) + list(gained[:3])
    pick += [i for i in diff.sort_values(ascending=False).index if i not in pick][:max(0, 10 - len(pick))]
    draw(pick, "segmentation_vs_footprint.png", "i band, red: pixels replaced by sky (changes of fate first)")
    # High S/N targets (catalogue i-band S/N), where the two maskings differ: the galaxy itself is visible.
    differ = [i for i in a.index if ("footprint", int(i)) in pix and (main_name, int(i)) in pix
              and (pix[("footprint", int(i))][2] != pix[(main_name, int(i))][2]).any()]
    bright = a.loc[differ, "cut6_value"].astype(float).sort_values(ascending=False).index[:12]
    draw(list(bright), "segmentation_vs_footprint_bright.png",
         "highest catalogue S/N_i targets where the two maskings differ; red: pixels replaced by sky")
    (out / "run.json").write_text(json.dumps(dict(patch=args.patch, demo_px=args.demo_px, tiles=tiles,
                                                  seg_grow=args.seg_grow, main=main_name), indent=2))
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
