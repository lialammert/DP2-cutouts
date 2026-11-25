# d2_shredding.py
# Anna Preto
# AstroDeep Project
# How often is a D2 target a fragment of one galaxy that the Euclid MER segmentation cut into several sources (a
# clumpy or irregular galaxy whose knots were masked as "neighbours"), rather than a galaxy with a real neighbour?
# The D2 acceptance cannot see it: it only limits the masked fraction inside the target's own (fragment) ellipse.
# Measured on the stamps already stored in D2 (no catalogue, no network), on the raw VIS stamp:
#   footprint         connected region of the smoothed VIS image above 2 sigma that contains the target centre
#   foot_masked_frac  fraction of the footprint pixels that the D2 masking replaced (euclid_fill)
#   foot_masked_flux  fraction of the footprint light in those pixels
#   embed_max         for each masked region inside the footprint: median smoothed light on a 2-pixel ring just
#                     outside it (inside the footprint) over its own peak; maximum over the regions. A knot embedded
#                     in a disc keeps disc light all around it (high); a separate neighbour that only touches the
#                     target at a faint isophote has little light around its mask (low).
#   n_masked_regions  masked regions inside the footprint
# Plus the D2 columns halo_ring_z (neighbour light left around the masks) and the masked fraction inside the target
# ellipse. Output: d2_shredding.parquet (one row per target) and d2_shredding.pdf (distributions, then random
# targets per bin of embed_max and of foot_masked_flux, drawn as in d2_gallery.py), to choose a cut by eye.
# Works on the finished D2 shards before the merge, or on the merged D2.
# Usage: python scripts/inspect/d2_shredding.py <ML config> <out dir> [--shards 40] [--d2 DIR] [--per-bin 18]

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import ndimage  # noqa: E402

from d2_gallery import I_BAND, noise_sigma, show  # noqa: E402  (same folder)
from dp2_cutouts.mlconfig import MLConfig  # noqa: E402
from dp2_cutouts.pairs.store import Store, shard_done  # noqa: E402

SMOOTH_PX = 1.0
NOISE_GAIN = 1.0 / (2.0 * np.sqrt(np.pi) * SMOOTH_PX)   # noise of a unit-sum Gaussian-smoothed white-noise image
DETECT_SIGMA = 2.0
EMBED_BINS = [0.0, 0.1, 0.2, 0.35, 1.01]
FLUX_BINS = [0.0, 0.05, 0.15, 0.3, 1.01]


def metrics(raw, ivar, valid, fill):
    n = raw.shape[0]
    c = (n - 1) / 2
    sig = noise_sigma(raw, ivar * valid)
    sm = ndimage.gaussian_filter(np.where(valid, np.nan_to_num(raw), 0.0), SMOOTH_PX)
    lab, _ = ndimage.label(sm > DETECT_SIGMA * NOISE_GAIN * sig, structure=np.ones((3, 3)))
    yy, xx = np.indices(raw.shape)
    near = (lab > 0) & (np.hypot(xx - c, yy - c) <= 3)
    if not near.any():
        return dict(foot_npix=0, foot_masked_frac=np.nan, foot_masked_flux=np.nan, embed_max=np.nan,
                    n_masked_regions=0)
    d = np.where(near, np.hypot(xx - c, yy - c), np.inf)
    foot = lab == lab.flat[np.argmin(d)]
    inside = fill & foot
    pos = np.clip(sm, 0, None)
    out = dict(foot_npix=int(foot.sum()), foot_masked_frac=float(inside.sum() / foot.sum()),
               foot_masked_flux=float(pos[inside].sum() / max(pos[foot].sum(), 1e-30)), embed_max=0.0,
               n_masked_regions=0)
    blobs, nb = ndimage.label(fill, structure=np.ones((3, 3)))
    for k in range(1, nb + 1):
        b = blobs == k
        if not (b & foot).any():
            continue
        out["n_masked_regions"] += 1
        ring = ndimage.binary_dilation(b, iterations=2) & ~fill & foot & valid
        peak = sm[b].max()
        if ring.sum() >= 3 and peak > 0:
            out["embed_max"] = max(out["embed_max"], float(np.median(sm[ring]) / peak))
    return out


def stores(cfg, d2, n_shards, seed):
    if (d2 / "info.json").exists():
        return [d2]
    shards = [s for s in sorted((d2 / "shards").glob("*")) if s.is_dir() and shard_done(s)]
    rng = np.random.default_rng(seed)
    return [shards[i] for i in sorted(rng.permutation(len(shards))[:n_shards])]


def measure(cfg, paths):
    vis = list(cfg["euclid"]["bands"]).index("VIS")
    rows, keep = [], []
    for p in paths:
        st = Store(p)
        if len(st) == 0:
            continue
        idx = np.arange(len(st))
        raw, iv = st.take("euclid_flux_unmasked", idx)[..., vis], st.take("euclid_ivar", idx)[..., vis]
        mk, fill = st.take("euclid_mask", idx)[..., vis], st.take("euclid_fill", idx)
        for i in idx:
            m = metrics(raw[i], iv[i], (~mk[i]) & (iv[i] > 0), fill[i])
            r = st.meta.iloc[i]
            m.update(store=str(p), row=int(i), euclid_object_id=int(r["euclid_object_id"]),
                     group_id=r["group_id"], split=r["split"], euclid_vis_mag=float(r["euclid_vis_mag"]),
                     sb_rubin=float(r["sb_rubin"]), halo_ring_z=float(r.get("halo_ring_z", np.nan)),
                     masked_in_target=float(r.get("euclid_masked_fraction_in_target", np.nan)),
                     partner_sep_arcsec=float(r.get("partner_sep_arcsec", np.nan)))
            rows.append(m)
        print(f"{p.name}: {len(st)} targets", flush=True)
    return pd.DataFrame(rows)


def gallery(pdf, cfg, tab, col, bins, per_bin, seed):
    vis = list(cfg["euclid"]["bands"]).index("VIS")
    rng = np.random.default_rng(seed)
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = tab[(tab[col] >= lo) & (tab[col] < hi)]
        if len(g) == 0:
            continue
        g = g.iloc[rng.permutation(len(g))[:per_bin]]
        rows = 6
        fig, axes = plt.subplots(rows, 6, figsize=(8.27, 11.0))
        for ax in axes.ravel():
            ax.axis("off")
        for k, r in enumerate(g.itertuples(index=False)):
            st = Store(r.store)
            i = [r.row]
            ri, riv = st.take("rubin_flux", i)[0, ..., I_BAND], st.take("rubin_ivar", i)[0, ..., I_BAND]
            raw, iv = st.take("euclid_flux_unmasked", i)[0, ..., vis], st.take("euclid_ivar", i)[0, ..., vis]
            ef, fill = st.take("euclid_flux", i)[0, ..., vis], st.take("euclid_fill", i)[0]
            row, c0 = divmod(k, 2)
            a = axes[row, 3 * c0:3 * c0 + 3]
            for x in a:
                x.axis("on")
            show(a[0], ri, noise_sigma(ri, riv), "Rubin i")
            s = noise_sigma(raw, iv)
            show(a[1], raw, s, "VIS before")
            show(a[2], ef, s, "VIS after masking", contour=fill)
            a[0].text(0.0, -0.08, f"{r.euclid_object_id}  VIS={r.euclid_vis_mag:.1f}\n"
                                  f"embed={r.embed_max:.2f} fflux={r.foot_masked_flux:.2f} "
                                  f"ffrac={r.foot_masked_frac:.2f} halo_z={r.halo_ring_z:.1f}",
                      transform=a[0].transAxes, fontsize=5, va="top", ha="left")
        n_bin = int(((tab[col] >= lo) & (tab[col] < hi)).sum())
        fig.suptitle(f"{col} in [{lo}, {hi}): {n_bin} targets ({100 * n_bin / len(tab):.1f} %), "
                     f"{len(g)} drawn at random", fontsize=9)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.04, wspace=0.04, hspace=0.42)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--d2", help="D2 folder (default: datasets/d2 under ml_root; its shards before the merge)")
    ap.add_argument("--shards", type=int, default=40, help="random finished shards measured before the merge")
    ap.add_argument("--per-bin", type=int, default=12, help="targets drawn per bin (12 = one page)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = MLConfig.load(args.config)
    d2 = Path(args.d2) if args.d2 else cfg.dataset_dir("d2")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tab = measure(cfg, stores(cfg, d2, args.shards, args.seed))
    tab.to_parquet(out / "d2_shredding.parquet", index=False)
    ok = tab[np.isfinite(tab["embed_max"])]
    lines = [f"D2 targets measured: {len(tab)} from {tab['store'].nunique()} stores; footprint found: {len(ok)}",
             "", "quantiles (10 / 25 / 50 / 75 / 90 %):"]
    for k in ("foot_masked_frac", "foot_masked_flux", "embed_max", "n_masked_regions", "halo_ring_z",
              "masked_in_target", "foot_npix"):
        q = np.nanquantile(ok[k].to_numpy(float), [0.1, 0.25, 0.5, 0.75, 0.9])
        lines.append(f"  {k:18s} " + " ".join(f"{v:8.3f}" for v in q))
    lines += ["", "fraction of targets with:"]
    for k, cuts in (("embed_max", (0.1, 0.2, 0.35)), ("foot_masked_flux", (0.05, 0.15, 0.3))):
        for v in cuts:
            lines.append(f"  {k} > {v:<5}  {100 * float((ok[k] > v).mean()):5.1f} %")
    print("\n".join(lines))
    with PdfPages(out / "d2_shredding.pdf") as pdf:
        fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.0))
        axes[0, 0].axis("off")
        axes[0, 0].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=5.5)
        for a, k in zip(axes.ravel()[1:], ("embed_max", "foot_masked_flux", "foot_masked_frac", "halo_ring_z")):
            a.hist(ok[k].to_numpy(float), bins=60)
            a.set_xlabel(k)
            a.set_yscale("log")
        a = axes[2, 1]
        a.scatter(ok["foot_masked_flux"], ok["embed_max"], s=2, alpha=0.3)
        a.set_xlabel("foot_masked_flux")
        a.set_ylabel("embed_max")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        gallery(pdf, cfg, ok, "embed_max", EMBED_BINS, args.per_bin, args.seed)
        gallery(pdf, cfg, ok, "foot_masked_flux", FLUX_BINS, args.per_bin, args.seed + 1)
    print(f"-> {out / 'd2_shredding.pdf'}, {out / 'd2_shredding.parquet'}")


if __name__ == "__main__":
    main()
