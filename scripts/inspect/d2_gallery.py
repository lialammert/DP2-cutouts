# d2_gallery.py
# Anna Preto
# AstroDeep Project
# Looks at Dataset 2.
#   kept:     every D2 target, from the stamps stored in the dataset: Rubin blend (i) | Euclid VIS before masking |
#             VIS after masking, the masked pixels (euclid_fill) outlined in red. Two targets per row, one PDF.
#   rejected: the targets that failed the Euclid separation test (rejected.parquet, reason euclid_separation). D2
#             keeps only their measurements, so the test is run again on a sample per failure category (same code,
#             same seed: same pixels), and drawn the same way. Needs the MMU Euclid cache and the MER tiles of the
#             D2 build (no network).
# Usage: python scripts/inspect/d2_gallery.py <ML config> <out dir> [--mode kept rejected] [--per-category 24]
#            [--patches 8] [--d2 DIR]

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from dp2_cutouts.mlconfig import MLConfig  # noqa: E402
from dp2_cutouts.pairs import dataset2  # noqa: E402
from dp2_cutouts.pairs.store import Store  # noqa: E402

ROWS = 6          # rows per page, two targets per row
I_BAND = 3        # ugrizy: i
ALL_FOUR = "delta_e+centroid_shift+size_change+flux_change"


def noise_sigma(flux, ivar):
    # Typical pixel noise of a stamp, for the stretch (median of 1/sqrt(ivar) over usable pixels).
    ok = (ivar > 0) & np.isfinite(flux)
    if not ok.any():
        s = np.nanstd(flux)
        return float(s) if s > 0 else 1.0
    return float(np.median(1.0 / np.sqrt(ivar[ok])))


def show(ax, img, sigma, title=None, contour=None):
    # asinh stretch: linear up to ~1 sigma, logarithmic above, same scale for before / after.
    ax.imshow(np.arcsinh(np.nan_to_num(img) / sigma), origin="lower", cmap="gray_r", vmin=-1, vmax=6,
              interpolation="nearest")
    if contour is not None and contour.any():
        ax.contour(contour.astype(float), levels=[0.5], colors="red", linewidths=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=6)


def label(r):
    # One-line description of a target under its row.
    def g(k, fmt):
        v = r.get(k, np.nan)
        try:
            return fmt.format(v) if np.isfinite(v) else "nan"
        except (TypeError, ValueError):
            return str(v)
    why = f"  fails: {r['recomputed_reasons']}" if r.get("recomputed_reasons") else ""
    return (f"{int(r['euclid_object_id'])}  VIS={g('euclid_vis_mag', '{:.1f}')}  Sb={g('sb_rubin', '{:.2f}')}{why}\n"
            f"de={g('sep_delta_e', '{:.3f}')} dc={g('sep_centroid_shift_arcsec', '{:.3f}')}\" "
            f"ds={g('sep_rel_size_change', '{:.3f}')} df={g('sep_rel_flux_change', '{:.3f}')} "
            f"mask_t={g('euclid_masked_fraction_in_target', '{:.2f}')}")


def draw_pages(pdf, items, title):
    # items: list of (row dict, rubin_i or None, rubin_ivar_i or None, vis_raw, vis_masked, vis_ivar, need).
    per_page = 2 * ROWS
    for p0 in range(0, len(items), per_page):
        fig, axes = plt.subplots(ROWS, 6, figsize=(8.27, 11.0))
        for ax in axes.ravel():
            ax.axis("off")
        for k, (r, ri, riv, raw, masked, eiv, need) in enumerate(items[p0:p0 + per_page]):
            row, col = divmod(k, 2)
            a = axes[row, 3 * col:3 * col + 3]
            for x in a:
                x.axis("on")
            if ri is not None:
                show(a[0], ri, noise_sigma(ri, riv), "Rubin i (blend)")
            s = noise_sigma(raw, eiv)
            show(a[1], raw, s, "VIS before")
            show(a[2], masked, s, "VIS after masking", contour=need)
            a[0].text(0.0, -0.08, label(r), transform=a[0].transAxes, fontsize=5, va="top", ha="left")
        fig.suptitle(f"{title}  ({p0 + 1}-{min(p0 + per_page, len(items))} of {len(items)})", fontsize=9)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.04, wspace=0.04, hspace=0.42)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def summary_page(pdf, lines):
    fig = plt.figure(figsize=(8.27, 11.0))
    fig.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def kept(cfg, d2, out):
    st = Store(d2)
    meta = st.meta.reset_index(drop=True)
    order = np.argsort(meta["euclid_vis_mag"].to_numpy(float), kind="stable")
    vis = list(cfg["euclid"]["bands"]).index("VIS")
    path = out / "d2_kept.pdf"
    with PdfPages(path) as pdf:
        lines = [f"D2 kept targets: {len(meta)}  (sorted by VIS magnitude, brightest first)",
                 f"splits: {meta['split'].value_counts().to_dict()}",
                 f"masking: {meta['euclid_masking'].value_counts().to_dict()}",
                 "", "columns per target: Rubin i blend (centred on the Euclid target, never masked) |",
                 "  VIS before masking | VIS after masking (red: pixels replaced by sky)",
                 "label: object id, VIS mag, Rubin blending entropy Sb;",
                 "  de / dc / ds / df = change of ellipticity / centroid (\") / size / flux when masking,",
                 "  mask_t = masked fraction inside the target ellipse", "",
                 "quantiles (10 / 50 / 90 %):"]
        for k in ("euclid_vis_mag", "sb_rubin", "sep_delta_e", "sep_centroid_shift_arcsec", "sep_rel_size_change",
                  "sep_rel_flux_change", "euclid_masked_fraction", "euclid_masked_fraction_in_target"):
            if k in meta:
                q = np.nanquantile(meta[k].to_numpy(float), [0.1, 0.5, 0.9])
                lines.append(f"  {k:36s} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f}")
        summary_page(pdf, lines)
        for c0 in range(0, len(order), 240):          # read the memory maps by chunks
            idx = order[c0:c0 + 240]
            rf, riv = st.take("rubin_flux", idx), st.take("rubin_ivar", idx)
            ef, eiv = st.take("euclid_flux", idx), st.take("euclid_ivar", idx)
            raw, need = st.take("euclid_flux_unmasked", idx), st.take("euclid_fill", idx)
            items = [(meta.iloc[i].to_dict(), rf[j, ..., I_BAND], riv[j, ..., I_BAND], raw[j, ..., vis],
                      ef[j, ..., vis], eiv[j, ..., vis], need[j]) for j, i in enumerate(idx)]
            draw_pages(pdf, items, "D2 kept")
            print(f"kept: {c0 + len(idx)} / {len(order)}", flush=True)
    print(f"-> {path}")


def category(detail):
    if detail == ALL_FOUR:
        return "all_four"
    if detail == "hsm_failed":
        return "hsm_failed"
    if detail == "delta_e":
        return "delta_e_only"
    return "other"


def rejected(cfg, d2, out, per_cat, n_patches, seed):
    from dp2_cutouts.pipeline.patches import load_patches, patch_case
    from dp2_cutouts.sources import euclid_images, euclid_seg
    from dp2_cutouts.sources.rubin_soda import BANDS
    from dp2_cutouts.stages import stamps as stamp_stage

    rej = Store(d2).rejected()
    rej = rej[rej["reason"] == "euclid_separation"].copy()
    rej["category"] = rej["detail"].map(category)
    rej["patch"] = rej["group_id"].str.split(":").str[0]
    # Sample drawn from the patches with the most rejections only: each patch costs a load of its Rubin images.
    top = rej["patch"].value_counts().index[:n_patches]
    pool = rej[rej["patch"].isin(top)]
    rng = np.random.default_rng(seed)
    pick = []
    for cat_, g in pool.groupby("category"):
        pick.append(g.iloc[rng.permutation(len(g))[:per_cat]])
    pick = pd.concat(pick).drop_duplicates("euclid_object_id")
    print(f"patches {list(top)}: drawing", pick.groupby("category").size().to_dict(), "of",
          rej["category"].value_counts().to_dict(), flush=True)

    pcfg = cfg.pipeline()
    d2c = cfg["dataset2"]
    n_r = int(cfg["rubin"]["stamp_px"])
    vis = list(cfg["euclid"]["bands"]).index("VIS")
    tiles = None
    if d2c.get("euclid_masking", "ellipse") == "segmentation":
        paths = euclid_seg.cached_tiles(euclid_seg.cache_dir(d2c if "seg_cache" in d2c else pcfg["masking"]))
        tiles = euclid_seg.TileSet(paths).__enter__()
    items = {c: [] for c in pick["category"].unique()}
    n_diff = 0                                      # recomputed reasons different from those of the D2 build
    for pname, g in pick.groupby("patch"):
        patch = load_patches(pcfg, [pname])[0]
        case = patch_case(pcfg, patch)
        cand, e = dataset2.candidate_targets(case, cfg, patch)
        cand = cand[cand["euclid_object_id"].isin(g["euclid_object_id"])]
        img = stamp_stage.load_images(case)
        cuts = euclid_images.fetch(cfg["euclid"], pd.DataFrame(dict(
            object_id=cand["euclid_object_id"], ra=cand["euclid_ra"], dec=cand["euclid_dec"])), case.sky_box)
        cat_of = g.set_index("euclid_object_id")["category"]
        detail_of = g.set_index("euclid_object_id")["detail"]
        for r in cand.itertuples(index=False):
            cut = cuts.get(int(r.euclid_object_id))
            if cut is None or "VIS" not in cut["bands"]:
                continue
            row = r._asdict()
            rg = np.random.default_rng([int(cfg["seed"]), int(r.euclid_object_id) & 0x7FFFFFFF])
            tgt = dict(ra=r.euclid_ra, dec=r.euclid_dec, a_e=e.loc[r.euclid_row, "a_e"],
                       b_e=e.loc[r.euclid_row, "b_e"], theta=e.loc[r.euclid_row, "theta"],
                       flux=float(e.loc[r.euclid_row, "vis_flux"]))
            others = e.drop(index=r.euclid_row)
            dra = np.abs(((others["ra"] - r.euclid_ra + 180) % 360) - 180) * np.cos(np.radians(r.euclid_dec)) * 3600
            others = others[(dra < 15) & (np.abs(others["dec"] - r.euclid_dec) * 3600 < 15)]
            sep, reasons, (ef, eiv, em, ep, need, f_raw) = dataset2.euclid_separation(cut, tgt, others, cfg, rg, tiles)
            row.update(sep)
            row["recomputed_reasons"] = "+".join(reasons)
            w = img["wcs"][BANDS.index("i")]
            x, y = w.world_to_pixel_values(r.euclid_ra, r.euclid_dec)
            st = dataset2.rubin_stamp(img, float(x), float(y), n_r)
            ri = riv = None
            if st is not None:
                var = st["var"][I_BAND]
                ri, riv = st["flux"][I_BAND], np.where(var > 0, 1.0 / np.maximum(var, 1e-30), 0.0)
            c = cat_of.loc[int(r.euclid_object_id)]
            if row["recomputed_reasons"] != detail_of.loc[int(r.euclid_object_id)]:
                n_diff += 1
            items[c].append((row, ri, riv, f_raw[..., vis], ef[..., vis], eiv[..., vis], need))
        print(f"rejected: patch {pname} done", flush=True)
    if tiles is not None:
        tiles.__exit__(None, None, None)

    counts = rej["category"].value_counts().to_dict()
    n_drawn = sum(len(v) for v in items.values())
    print(f"recomputed {n_drawn}, reasons different from the D2 build: {n_diff}", flush=True)
    path = out / "d2_rejected_separation.pdf"
    with PdfPages(path) as pdf:
        lines = [f"D2 candidates rejected by the Euclid separation test: {len(rej)}",
                 f"by category: {counts}",
                 f"most frequent details: {rej['detail'].value_counts().head(8).to_dict()}", "",
                 f"limits: {dict(d2c['accept'])}", "",
                 f"Test run again on a random sample per category from the patches {list(top)}",
                 f"(same seed: same sky fill): {n_drawn} drawn, {n_diff} with reasons different from the D2 build.",
                 "Columns as in d2_kept.pdf. Categories:",
                 "  all_four     = delta_e + centroid_shift + size_change + flux_change",
                 "  hsm_failed   = the adaptive moments failed before or after masking",
                 "  delta_e_only = only the ellipticity change is above its limit",
                 "  other        = any other combination (incl. target_masked)"]
        summary_page(pdf, lines)
        for c in ("all_four", "hsm_failed", "delta_e_only", "other"):
            if items.get(c):
                draw_pages(pdf, sorted(items[c], key=lambda t: t[0].get("euclid_vis_mag", 99)),
                           f"rejected: {c} (of {counts.get(c, 0)})")
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--mode", nargs="+", default=["kept", "rejected"], choices=["kept", "rejected"])
    ap.add_argument("--per-category", type=int, default=24, help="rejected targets drawn per failure category")
    ap.add_argument("--patches", type=int, default=8, help="rejected: sample from the N patches with most rejections")
    ap.add_argument("--d2", help="D2 dataset folder (default: datasets/d2 under ml_root)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = MLConfig.load(args.config)
    d2 = Path(args.d2) if args.d2 else cfg.dataset_dir("d2")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "kept" in args.mode:
        kept(cfg, d2, out)
    if "rejected" in args.mode:
        rejected(cfg, d2, out, args.per_category, args.patches, args.seed)


if __name__ == "__main__":
    main()
