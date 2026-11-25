# mask_damage.py
# Anna Preto
# AstroDeep Project
# Of the D2 targets whose VIS measurement changes when the neighbours are masked, which ones had only the
# neighbour removed (fine) and which ones were damaged by the mask (target light lost under it)? D2 has no truth for
# its targets, so the damage is estimated by an indicator and the indicator is calibrated on D1, where it does:
#   indicator   lost_sym: target light hidden by the masks, estimated by point symmetry about the target centre (a
#               masked pixel whose mirror pixel shows the target, unmasked, is given that pixel's light), over the
#               light of the target that stays visible. 0: no target light under the masks.
#   calibration real D2 mask patterns (the masked regions touching their D2 target, rotated / flipped at random) laid
#               on isolated D1 galaxies of similar size to that target (adaptive-moment sigma within 20 %: D1 galaxies
#               are brighter and larger than D2 targets), filled with sky as D2 does; the true change of the VIS adaptive moments
#               (shape |de|, size, flux) is measured against the unmasked galaxy, together with lost_sym.
#   result      the largest lost_sym below which the injected masks keep |de| under --tol (per bin, 90th
#               percentile), and the share of D2 above it, overall, by VIS magnitude, and for the targets that
#               pass / fail the former before/after test (de < 0.02, centroid < 0.05", size and flux < 5 %).
# The other kind of damage, neighbour light left outside the masks, is not calibrated here: D2 records it as
# halo_ring_z (its distribution is printed).
# Works on the finished shards before the merges, or on merged D1 / D2. VIS only. No network.
# Usage: python scripts/inspect/mask_damage.py <ML config> <out dir> [--shards 40] [--tol 0.01] [--n-inject 20000]

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
from dp2_cutouts.pairs.common import moments  # noqa: E402
from dp2_cutouts.pairs.dataset2 import fill_with_sky  # noqa: E402
from dp2_cutouts.pairs.store import Store, shard_done  # noqa: E402

SMOOTH_PX = 1.0
NOISE_GAIN = 1.0 / (2.0 * np.sqrt(np.pi) * SMOOTH_PX)
DETECT_SIGMA = 2.0
LOST_BINS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 1.01]
OLD_TEST = dict(de=0.02, dc=0.05, ds=0.05, df=0.05)


# ----------------------------------------------------------------------------- measurements
def support(raw, valid, fill, sig):
    # Visible light of the target: connected region of the smoothed image (masked pixels set to 0) above
    # DETECT_SIGMA that contains the stamp centre, masked pixels excluded. (None, smoothed) if nothing there.
    n = raw.shape[0]
    c = (n - 1) / 2
    sm = ndimage.gaussian_filter(np.where(valid & ~fill, np.nan_to_num(raw), 0.0), SMOOTH_PX)
    lab, _ = ndimage.label(sm > DETECT_SIGMA * NOISE_GAIN * sig, structure=np.ones((3, 3)))
    yy, xx = np.indices(raw.shape)
    r = np.hypot(xx - c, yy - c)
    near = (lab > 0) & (r <= 3)
    if not near.any():
        return None, sm
    t = (lab == lab.flat[np.argmin(np.where(near, r, np.inf))]) & ~fill
    return t, sm


def lost_sym(raw, valid, fill, sig):
    # Target light under the masks estimated by point symmetry, over the visible target light (stamps of odd size
    # centred on the target: the mirror of pixel (y, x) is (n-1-y, n-1-x)).
    t, sm = support(raw, valid, fill, sig)
    if t is None:
        return np.nan, np.nan
    near = fill & ndimage.binary_dilation(t, iterations=4)
    mirror_t = t[::-1, ::-1]
    known = near & mirror_t
    pos = np.clip(sm, 0, None)
    lost = float(pos[::-1, ::-1][known].sum())
    vis = float(pos[t].sum())
    # Masked pixels whose mirror is masked too: their loss is not estimated (lost_sym is a lower bound there).
    unknown = float((near & fill[::-1, ::-1]).sum()) / max(float(near.sum()), 1.0) if near.any() else 0.0
    return lost / max(vis + lost, 1e-30), unknown


def masked_in_ellipse(fill, e1, e2, sigma, k=2.0):
    # Fraction of the pixels within k "sigma" of the adaptive-moment ellipse (centred on the stamp) that are masked.
    if not (np.isfinite(e1) and np.isfinite(e2) and np.isfinite(sigma)) or sigma <= 0:
        return np.nan
    e2_ = min(e1 * e1 + e2 * e2, 0.95 ** 2)
    m = sigma ** 2 / np.sqrt(1 - e2_) * np.array([[1 - e1, e2], [e2, 1 + e1]])   # [[yy, xy], [xy, xx]]
    n = fill.shape[0]
    c = (n - 1) / 2
    yy, xx = np.indices(fill.shape)
    v = np.stack([yy - c, xx - c], -1)
    d2 = np.einsum("...i,ij,...j->...", v, np.linalg.inv(m), v)
    inside = d2 <= k * k
    return float(fill[inside].mean()) if inside.any() else np.nan


def stores(cfg, name, n_shards, seed):
    d = cfg.dataset_dir(name)
    if (d / "info.json").exists():
        return [d]
    shards = [s for s in sorted((d / "shards").glob("*")) if s.is_dir() and shard_done(s)]
    rng = np.random.default_rng(seed)
    return [shards[i] for i in sorted(rng.permutation(len(shards))[:n_shards])]


def vis_index(cfg):
    return list(cfg["euclid"]["bands"]).index("VIS")


# ----------------------------------------------------------------------------- D2: indicator and mask patterns
def measure_d2(cfg, paths, max_patterns, seed):
    vis = vis_index(cfg)
    rows, patterns = [], []
    for p in paths:
        st = Store(p)
        if len(st) == 0:
            continue
        idx = np.arange(len(st))
        raw, iv = st.take("euclid_flux_unmasked", idx)[..., vis], st.take("euclid_ivar", idx)[..., vis]
        mk, fill = st.take("euclid_mask", idx)[..., vis], st.take("euclid_fill", idx)
        meta = st.meta
        for i in idx:
            valid = (~mk[i]) & (iv[i] > 0)
            sig = noise_sigma(raw[i], iv[i] * valid)
            ls, unk = lost_sym(raw[i], valid, fill[i], sig)
            r = meta.iloc[i]
            g = {k: float(r.get(k, np.nan)) for k in ("sep_delta_e", "sep_centroid_shift_arcsec",
                                                      "sep_rel_size_change", "sep_rel_flux_change", "halo_ring_z",
                                                      "euclid_vis_mag", "e1_vis", "e2_vis", "sigma_vis_px")}
            old_ok = (g["sep_delta_e"] < OLD_TEST["de"]) and (g["sep_centroid_shift_arcsec"] < OLD_TEST["dc"]) and \
                (g["sep_rel_size_change"] < OLD_TEST["ds"]) and (g["sep_rel_flux_change"] < OLD_TEST["df"])
            rows.append(dict(store=str(p), row=int(i), euclid_object_id=int(r["euclid_object_id"]), split=r["split"],
                             lost_sym=ls, lost_unknown=unk, old_test_ok=bool(old_ok),
                             masked_in_moment_ellipse=masked_in_ellipse(fill[i], g["e1_vis"], g["e2_vis"],
                                                                        g["sigma_vis_px"]), **g))
            # Mask pattern for the injection: the masked regions that touch the target's visible light.
            if len(patterns) < max_patterns:
                t, _ = support(raw[i], valid, fill[i], sig)
                if t is not None:
                    lab, nb = ndimage.label(fill[i], structure=np.ones((3, 3)))
                    touch = np.unique(lab[ndimage.binary_dilation(t, iterations=4) & (lab > 0)])
                    if len(touch) and np.isfinite(g["sigma_vis_px"]) and g["sigma_vis_px"] > 0:
                        # Packed bits (a 119 x 119 mask is 14 kB unpacked), with the size of the D2 target.
                        patterns.append((np.packbits(np.isin(lab, touch)), float(g["sigma_vis_px"])))
        print(f"D2 {p.name}: {len(st)} targets", flush=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(patterns)
    return pd.DataFrame(rows), patterns


# ----------------------------------------------------------------------------- D1: injection
def inject_d1(cfg, paths, patterns, n_inject, seed):
    vis = vis_index(cfg)
    rng = np.random.default_rng(seed)
    out = []
    per_store = int(np.ceil(n_inject / max(len(paths), 1)))
    # Size matching: a pattern is laid on a D1 galaxy of similar size (adaptive-moment sigma within 20 %) to the D2
    # target it was cut around, so that it bites into the galaxy as much as it did into its target.
    lsig = np.log(np.array([q[1] for q in patterns])) if patterns else np.zeros(0)
    order = np.argsort(lsig)
    lsig_sorted = lsig[order]
    n_unmatched = 0
    for p in paths:
        st = Store(p)
        if len(st) == 0 or not patterns:
            continue
        has_fill = "euclid_fill" in st.names()
        rows = np.flatnonzero(st.meta["split"].to_numpy() != "holdout")
        rows = np.sort(rng.permutation(rows)[:3 * per_store])     # spare galaxies for the size matching
        if len(rows) == 0:
            continue
        f, iv = st.take("euclid_flux", rows)[..., vis], st.take("euclid_ivar", rows)[..., vis]
        mk = st.take("euclid_mask", rows)[..., vis]
        own = st.take("euclid_fill", rows) if has_fill else np.zeros(f.shape, bool)
        done = 0
        for j, row in enumerate(rows):
            if done >= per_store:
                break
            valid = (~mk[j]) & (iv[j] > 0)
            w = (valid & ~own[j]).astype(np.int32)
            truth = moments(f[j], w)
            if not truth["ok"]:
                continue
            ls_g = np.log(truth["sigma"]) if truth["sigma"] > 0 else np.nan
            lo, hi = np.searchsorted(lsig_sorted, [ls_g - 0.18, ls_g + 0.18])
            if not np.isfinite(ls_g) or hi <= lo:
                n_unmatched += 1
                continue
            bits, sig_t = patterns[order[rng.integers(lo, hi)]]
            pat = np.unpackbits(bits, count=f[j].size).reshape(f[j].shape).astype(bool)
            need = np.rot90(pat, int(rng.integers(4)))
            if rng.random() < 0.5:
                need = need[:, ::-1]
            need = need & valid & ~own[j]
            sig = noise_sigma(f[j], iv[j] * valid)
            sm = ndimage.gaussian_filter(np.where(valid, f[j], 0.0), SMOOTH_PX)
            src = ndimage.binary_dilation(sm > DETECT_SIGMA * NOISE_GAIN * sig, iterations=2)
            sky = valid & ~src & ~own[j]
            ff, _ = fill_with_sky(f[j][..., None], iv[j][..., None], need, sky[..., None], rng)
            after = moments(ff[..., 0], w)
            ls, unk = lost_sym(f[j], valid, need | own[j], sig)
            rec = dict(store=str(p), row=int(row), lost_sym=ls, lost_unknown=unk, ok=bool(after["ok"]),
                       sigma_truth_px=float(truth["sigma"]), sigma_pattern_target_px=sig_t,
                       masked_frac=float(need.mean()),
                       masked_in_moment_ellipse=masked_in_ellipse(need, after["e1"], after["e2"], after["sigma"]))
            if after["ok"]:
                rec.update(de=float(np.hypot(after["e1"] - truth["e1"], after["e2"] - truth["e2"])),
                           dsize=float(after["sigma"] / truth["sigma"] - 1.0),
                           dflux=float(after["flux"] / truth["flux"] - 1.0) if truth["flux"] else np.nan,
                           dc_px=float(np.hypot(after["x"] - truth["x"], after["y"] - truth["y"])))
            out.append(rec)
            done += 1
        print(f"D1 {p.name}: {done} injections, {n_unmatched} without a size-matched pattern so far", flush=True)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- calibration and report
def calibration(inj, tol):
    ok = inj[inj["ok"] & np.isfinite(inj["lost_sym"])]
    tab = []
    for lo, hi in zip(LOST_BINS[:-1], LOST_BINS[1:]):
        g = ok[(ok["lost_sym"] >= lo) & (ok["lost_sym"] < hi)]
        if len(g) == 0:
            tab.append(dict(lo=lo, hi=hi, n=0))
            continue
        tab.append(dict(lo=lo, hi=hi, n=len(g), de_med=g["de"].median(), de_p90=g["de"].quantile(0.9),
                        de_p95=g["de"].quantile(0.95), dsize_med=g["dsize"].median(),
                        dflux_med=g["dflux"].median(), dflux_p10=g["dflux"].quantile(0.1)))
    return pd.DataFrame(tab)


def threshold(tab, tol, min_n=20):
    # Lower edge of the first lost_sym bin (with at least min_n injections) whose |de| p90 exceeds tol; 1 if none.
    for r in tab.itertuples(index=False):
        if r.n >= min_n and r.de_p90 > tol:
            return float(r.lo)
    return 1.0


def report(d2, inj, tab, thr, tol, tols):
    lines = [f"D2 targets: {len(d2)}  |  D1 injections: {len(inj)} ({int(inj['ok'].sum())} measured after masking)",
             "", f"Calibration on D1 (real D2 mask patterns on isolated galaxies), tolerance |de| p90 <= {tol}:",
             "  lost_sym bin        n    |de| med   p90    p95    dsize med  dflux med  dflux p10"]
    for r in tab.itertuples(index=False):
        if r.n:
            lines.append(f"  [{r.lo:5.3f},{r.hi:5.3f})  {r.n:6d}  {r.de_med:7.4f} {r.de_p90:6.4f} {r.de_p95:6.4f}"
                         f"   {r.dsize_med:+8.4f}  {r.dflux_med:+8.4f}  {r.dflux_p10:+8.4f}")
        else:
            lines.append(f"  [{r.lo:5.3f},{r.hi:5.3f})       0")
    ok = d2[np.isfinite(d2["lost_sym"])]
    lines += ["", "threshold = start of the first bin (>= 20 injections) whose true |de| p90 exceeds the tolerance:"]
    for t in tols:
        th = threshold(tab, t)
        share = 100 * (ok["lost_sym"] >= th).mean()
        lines.append(f"  tolerance {t:5.3f}: lost_sym < {th:.3f}, D2 judged damaged {share:5.1f} %")
    if thr <= 0:
        lines += ["", f"-> no lost_sym value keeps the true |de| p90 within {tol}: even with no visible loss, the",
                  "   masks change the shape beyond it in more than 10 % of the injections (the lines below then",
                  "   count every target as damaged and are not a result).", ""]
    else:
        lines += ["", f"-> with tolerance {tol}: lost_sym < {thr:.3f}", ""]
    okj = inj[inj["ok"]]
    q = np.nanquantile(okj["de"], [0.5, 0.75, 0.9, 0.95])
    lines.append(f"All injections (size-matched D2 masks touching their target): true |de| median {q[0]:.4f}, "
                 f"75 % {q[1]:.4f}, 90 % {q[2]:.4f}, 95 % {q[3]:.4f}")
    for t in (0.01, 0.02, 0.05):
        lines.append(f"  share with true |de| > {t}: {100 * float((okj['de'] > t).mean()):5.1f} %")
    # D2 targets through the calibration: each target gets the injection statistics of its lost_sym bin.
    b = pd.cut(ok["lost_sym"], LOST_BINS, right=False, labels=False).to_numpy().astype(int)
    med = tab["de_med"].to_numpy() if "de_med" in tab else np.full(len(tab), np.nan)
    p90 = tab["de_p90"].to_numpy() if "de_p90" in tab else np.full(len(tab), np.nan)
    exp_med, exp_p90 = med[b], p90[b]
    lines += ["", "D2 through the calibration (expected true |de| of its lost_sym bin):",
              "  group                                         n      lost_sym<0.005  bin median>0.02  bin p90>0.05"]
    for name, sel in (("all D2", np.ones(len(ok), bool)), ("pass the former before/after test", ok["old_test_ok"]),
                      ("fail it", ~ok["old_test_ok"])):
        sel = np.asarray(sel, bool)
        lines.append(f"  {name:42s} {int(sel.sum()):6d}   {100 * float((ok['lost_sym'][sel] < 0.005).mean()):10.1f} %"
                     f"   {100 * float(np.nanmean(exp_med[sel] > 0.02)):10.1f} %   "
                     f"{100 * float(np.nanmean(exp_p90[sel] > 0.05)):9.1f} %")
    lines.append("")
    if thr > 0:
        bad = ok["lost_sym"] >= thr
        lines.append(f"D2 damaged by the masks (lost_sym >= {thr:.3f}): {100 * bad.mean():.1f} % of {len(ok)}")
        for name, sel in (("pass the former before/after test", ok["old_test_ok"]),
                          ("fail it (neighbour entered the measurement)", ~ok["old_test_ok"])):
            lines.append(f"  {name:45s} {int(sel.sum()):6d} targets, damaged {100 * bad[sel].mean():5.1f} %")
        mag = pd.cut(ok["euclid_vis_mag"], [0, 24, 25, 26, 30])
        for k, g in bad.groupby(mag, observed=True):
            lines.append(f"  VIS {str(k):12s} {len(g):6d} targets, damaged {100 * g.mean():5.1f} %")
    q = np.nanquantile(ok["halo_ring_z"].to_numpy(float), [0.5, 0.9, 0.99])
    lines += ["", f"neighbour light left around the masks, halo_ring_z (not calibrated): median {q[0]:.2f}, "
                  f"90 % {q[1]:.2f}, 99 % {q[2]:.2f}",
              f"lost_unknown (masked pixels near the target whose mirror is masked too), D2 median "
              f"{ok['lost_unknown'].median():.2f}: lost_sym is a lower bound there"]
    return lines


def gallery_page(pdf, cfg, rows, title):
    vis = vis_index(cfg)
    fig, axes = plt.subplots(6, 6, figsize=(8.27, 11.0))
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in enumerate(rows.itertuples(index=False)):
        if k >= 12:
            break
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
        a[0].text(0.0, -0.08, f"{r.euclid_object_id} VIS={r.euclid_vis_mag:.1f}\nlost_sym={r.lost_sym:.3f} "
                              f"de={r.sep_delta_e:.2f} halo_z={r.halo_ring_z:.1f}",
                  transform=a[0].transAxes, fontsize=5, va="top", ha="left")
    fig.suptitle(title, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.04, wspace=0.04, hspace=0.42)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--shards", type=int, default=40, help="random finished shards of D2 and of D1 (before merging)")
    ap.add_argument("--tol", type=float, default=0.01, help="tolerated true |de| (90th percentile per bin)")
    ap.add_argument("--n-inject", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = MLConfig.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d2, patterns = measure_d2(cfg, stores(cfg, "d2", args.shards, args.seed), 20000, args.seed)
    print(f"{len(patterns)} D2 mask patterns touching their target", flush=True)
    inj = inject_d1(cfg, stores(cfg, "d1", args.shards, args.seed + 1), patterns, args.n_inject, args.seed)
    d2.to_parquet(out / "mask_damage_d2.parquet", index=False)
    inj.to_parquet(out / "mask_damage_injection.parquet", index=False)
    tab = calibration(inj, args.tol)
    thr = threshold(tab, args.tol)
    lines = report(d2, inj, tab, thr, args.tol, sorted({0.005, 0.01, 0.02, 0.05, args.tol}))
    print("\n".join(lines))
    with PdfPages(out / "mask_damage.pdf") as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.0), gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
        axes[0].axis("off")
        axes[0].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=5.5)
        ok = inj[inj["ok"] & np.isfinite(inj["lost_sym"])]
        axes[1].scatter(np.clip(ok["lost_sym"], 1e-4, None), np.clip(ok["de"], 1e-4, None), s=2, alpha=0.2)
        axes[1].axvline(max(thr, 1e-4), color="red")
        axes[1].axhline(args.tol, color="grey", ls="--")
        axes[1].set(xscale="log", yscale="log", xlabel="lost_sym (indicator)", ylabel="true |de| (D1 injection)")
        v = d2["lost_sym"].to_numpy(float)
        axes[2].hist(np.clip(v[np.isfinite(v)], 1e-4, 1), bins=np.logspace(-4, 0, 50), label="D2")
        axes[2].axvline(max(thr, 1e-4), color="red", label="threshold")
        axes[2].set(xscale="log", xlabel="lost_sym", ylabel="D2 targets")
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        okd = d2[np.isfinite(d2["lost_sym"])]
        rng = np.random.default_rng(args.seed)
        above = okd[okd["lost_sym"] >= thr]
        below = okd[(okd["lost_sym"] < thr) & ~okd["old_test_ok"]]
        if len(above):
            gallery_page(pdf, cfg, above.iloc[rng.permutation(len(above))[:12]],
                         f"D2 judged damaged (lost_sym >= {thr:.3f}), random")
        if len(below):
            gallery_page(pdf, cfg, below.iloc[rng.permutation(len(below))[:12]],
                         f"D2 judged intact (lost_sym < {thr:.3f}) although the former test fails them, random")
    print(f"-> {out / 'mask_damage.pdf'}")


if __name__ == "__main__":
    main()
