# mask_origin.py
# Anna Preto
# AstroDeep Project
# Where the neighbour mask of one Rubin stamp comes from ("pipeline" masking, stages/stamps.py): recomputes on the
# stamp box the detection map, the target's footprint and the Rubin-footprint part of the mask, reads the mask the
# pipeline applied (stamps.parquet companion_mask), and lists the catalogued objects of the box (Rubin Object table,
# Euclid MER catalogue) with the MER segment found at their position. Prints the numbers and writes a figure:
# raw i, detection map (target footprint outlined), the applied mask split by origin, catalogue positions.
# Reads the patch folder of the stamp pipeline only (coadds, catalogues, segmentation cache): no network.
# Lists the Rubin siblings too (same parentObjectId): deblender children of a bright galaxy, e.g. pieces of its halo,
# are masked like neighbours when Euclid sees nothing there (case found on 2561_67, 2026-11-26).
# Usage: python scripts/inspect/mask_origin.py <ML config> <patch, e.g. 2561_67> <Rubin object_id> [out.png]

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hf"))
from d1_views import PatchPixels, split_mask  # noqa: E402

from dp2_cutouts.mlconfig import MLConfig  # noqa: E402
from dp2_cutouts.pipeline.patches import load_patches  # noqa: E402
from dp2_cutouts.sources.rubin_soda import BANDS  # noqa: E402
from dp2_cutouts.stages.stamps import rubin_footprint_mask  # noqa: E402


def main():
    cfg = MLConfig.load(sys.argv[1])
    name, oid = sys.argv[2], int(sys.argv[3])
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(f"mask_origin_{oid}.png")
    pcfg = cfg.pipeline()
    patch = next(p for p in load_patches(pcfg) if p["name"] == name)
    pix = PatchPixels(pcfg, patch)
    case = pix.case
    t = pd.read_parquet(case.data("30_processed", "table.parquet"))
    row = t[t["object_id"].astype("int64") == oid]
    if not len(row):
        raise SystemExit(f"object {oid} not in {name}/30_processed/table.parquet")
    row = row.iloc[0]
    n, half = pix.n, pix.n // 2
    x, y = pix.wcs.world_to_pixel_values(float(row["ra"]), float(row["dec"]))
    ox, oy = int(np.floor(x + 0.5)) - half, int(np.floor(y + 0.5)) - half
    xc, yc = x - ox, y - oy
    raw = pix.raw(row["ra"], row["dec"])
    det = np.any(np.stack([(pix._cut(b, "MASK", oy, ox).astype(np.int64) & pix.det_bit) != 0 for b in BANDS]), 0)
    fp, target_fp = rubin_footprint_mask(det, xc, yc, pix.grow)
    need = pix.stamps([oid]).get(oid, {}).get("need")
    print(f"{name} object {oid}: i = {row.get('i_cModel_mag', np.nan):.2f}, masking method {pix.method}")
    print(f"  target pixel detected: {bool(det[int(np.floor(yc + 0.5)), int(np.floor(xc + 0.5))])}; "
          f"target footprint {int(target_fp.sum())} px ({target_fp.mean():.1%} of the stamp); detected {det.mean():.1%}")
    for c in ("masked_fraction", "masked_fraction_in_ellipse", "delta_e_mask", "n_companions", "n_shared_companions",
              "n_euclid_below_depth", "companion_light_ratio", "halo_ring_z", "psf_sigma_i"):
        if c in row:
            print(f"  {c}: {row[c]}")
    if need is None:
        raise SystemExit("no companion_mask in stamps.parquet for this object")
    rub, outside = split_mask(need, fp)
    print(f"  applied mask {need.mean():.1%} of the stamp: Rubin footprints {rub.mean():.1%}, "
          f"outside every Rubin footprint {outside.mean():.1%}")

    # Catalogued objects of the box, with the MER segment under them (0 = none).
    from dp2_cutouts.sources import euclid_seg
    w = pix.wcs
    rub_cat = pd.read_parquet(case.data("10_rubin", "catalog", "object_box.parquet"))
    rx, ry = w.world_to_pixel_values(rub_cat["coord_ra"].to_numpy(float), rub_cat["coord_dec"].to_numpy(float))
    inb = (rx - ox > -0.5) & (rx - ox < n - 0.5) & (ry - oy > -0.5) & (ry - oy < n - 0.5)
    rc = rub_cat[inb].assign(sx=rx[inb] - ox, sy=ry[inb] - oy)
    from dp2_cutouts.sources import euclid_hf
    eb = euclid_hf.read_box(case)
    ex, ey = w.world_to_pixel_values(eb["mer_catalogue_right_ascension"].to_numpy(float),
                                     eb["mer_catalogue_declination"].to_numpy(float))
    ine = (ex - ox > -0.5) & (ex - ox < n - 0.5) & (ey - oy > -0.5) & (ey - oy < n - 0.5)
    ec = eb[ine].assign(sx=ex[ine] - ox, sy=ey[ine] - oy)
    tiles = euclid_seg.cached_tiles(euclid_seg.cache_dir(pcfg["masking"]))
    seg = None
    if tiles:
        seg = euclid_seg.seg_on_grid(tiles, w[oy:oy + n, ox:ox + n], (n, n))
        seg = seg.max(axis=0) if seg.ndim == 3 else seg

    def seg_at(sx, sy):
        if seg is None:
            return -1
        i, j = int(np.floor(sy + 0.5)), int(np.floor(sx + 0.5))
        return int(seg[i, j]) if (0 <= i < n and 0 <= j < n) else -1

    print(f"\n  Rubin objects in the stamp: {len(rc)}")
    cols = [c for c in ("objectId", "parentObjectId", "i_cModelFlux", "i_extendedness", "detect_isIsolated") if c in rc]
    rc = rc.assign(sep_arcsec=np.hypot(rc["sx"] - xc, rc["sy"] - yc) * 0.2,
                   in_mask=[bool(need[int(np.clip(np.floor(b + 0.5), 0, n - 1)), int(np.clip(np.floor(a + 0.5), 0, n - 1))])
                            for a, b in zip(rc["sx"], rc["sy"])],
                   segment=[seg_at(a, b) for a, b in zip(rc["sx"], rc["sy"])])
    print(rc[cols + ["sep_arcsec", "in_mask", "segment"]].sort_values("sep_arcsec").to_string())
    print(f"\n  Euclid objects in the stamp: {len(ec)}")
    ecols = [c for c in ("object_id", "mer_catalogue_flux_vis_1fwhm_aper", "mer_catalogue_point_like_flag",
                         "mer_catalogue_spurious_flag") if c in ec]
    ec = ec.assign(sep_arcsec=np.hypot(ec["sx"] - xc, ec["sy"] - yc) * 0.2,
                   segment=[seg_at(a, b) for a, b in zip(ec["sx"], ec["sy"])])
    print(ec[ecols + ["sep_arcsec", "segment"]].sort_values("sep_arcsec").to_string())

    # Figure.
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
    i_raw = raw["flux"][..., BANDS.index("i")]
    s = np.sqrt(np.median(raw["var"][..., BANDS.index("i")]))
    top = max(np.nanmax(i_raw[n // 3:-n // 3, n // 3:-n // 3]) / s, 5)
    show = np.arcsinh(i_raw / s) / np.arcsinh(top)
    ax[0].imshow(show, origin="lower", cmap="gray_r", vmin=0, vmax=0.6)
    ax[0].set_title("raw i (asinh, noise units)")
    ax[1].imshow(det, origin="lower", cmap="Greys", vmin=0, vmax=2)
    ax[1].contour(target_fp, levels=[0.5], colors="orange")
    ax[1].set_title("DETECTED (any band), target footprint")
    rgb = np.ones((n, n, 3))
    rgb[rub] = (0.23, 0.51, 0.96)
    rgb[outside] = (1, 0.17, 0.17)
    ax[2].imshow(rgb, origin="lower")
    ax[2].set_title("applied mask: blue Rubin fp, red outside")
    ax[3].imshow(seg if seg is not None else np.zeros((n, n)), origin="lower", cmap="tab20", interpolation="nearest")
    ax[3].set_title("MER segments on the Rubin grid")
    for a in ax:
        a.plot(rc["sx"], rc["sy"], "o", mfc="none", mec="cyan", ms=7, label="Rubin objects")
        a.plot(ec["sx"], ec["sy"], "x", color="magenta", ms=6, label="Euclid objects")
        a.plot([xc], [yc], "+", color="orange", ms=12)
        a.set_xticks([]), a.set_yticks([])
    ax[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"{name} {oid}")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\n-> {out}")
    pix.close()


if __name__ == "__main__":
    main()
