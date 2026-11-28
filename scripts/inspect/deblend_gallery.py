# deblend_gallery.py
# Anna Preto
# AstroDeep Project
# What a deblend looks like, step by step: one PDF page per D2 target (the campaign's D2 selection), every trained
# stage applied to it. Rubin panels in the i band, one asinh stretch per page for all of them (fluxes compare); VIS
# panels with their own stretch. Row by row:
#   input / Euclid   Rubin blend (gri, i), the input X; Euclid VIS target E_c with its neighbours masked (red)
#   Run A / B steps  K1: the VIS target redrawn from its K1 latent (what K1 understood); K2 o K1: the same latent drawn
#                    as Rubin would see the isolated target (the teacher's view, not available at inference)
#   Run C steps      C2 teacher r(z | X, E_c) mean, drawn by C1 as Rubin and as VIS; C2 student s(z | X) mean drawn
#                    as Rubin (Rubin only, before the CFM student C3)
#   deblends         mean of K Rubin-only samples of Runs A, B, C (C = the C3 student), and blend - mean (the
#                    residual: the target should vanish, the neighbours remain; colour scale +-5 sigma)
# Targets: random among the D2 targets passing --min-snr-vis (VIS S/N of the target) and either --max-sep (partner
# within this many arcsec) or --min-sb (Rubin blendedness); without them, random among all.
# Usage: python scripts/inspect/deblend_gallery.py <ML config> <out.pdf> [--suffix _pilot] [--n 12] [--k 10]
#            [--split val] [--seed 0] [--min-snr-vis 50] [--max-sep 1.5] [--min-sb 0.3]

import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "jax")            # keras is imported below, before dp2_cutouts.deblend
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from dp2_cutouts.mlconfig import MLConfig  # noqa: E402


def stretch(img, top, soft):
    return np.clip(np.arcsinh(img / soft) / np.arcsinh(top / soft), -0.05, 1)


def pick_rows(ds, args):
    m = ds.st.meta.iloc[ds.rows_all]
    ok = np.ones(len(m), bool)
    if args.min_snr_vis is not None and "snr_vis" in m:
        ok &= m["snr_vis"].to_numpy(float) >= args.min_snr_vis
    if args.max_sep is not None and "partner_sep_arcsec" in m:
        ok &= m["partner_sep_arcsec"].to_numpy(float) <= args.max_sep
    if args.min_sb is not None and "sb_rubin" in m:
        ok &= m["sb_rubin"].to_numpy(float) >= args.min_sb
    rows = ds.rows_all[ok]
    print(f"{len(rows)} D2 {args.split} targets pass the filters (of {len(ds.rows_all)})", flush=True)
    if not len(rows):
        raise SystemExit("no target passes the filters: loosen --min-snr-vis / --max-sep / --min-sb")
    return np.sort(np.random.default_rng(args.seed).choice(rows, min(args.n, len(rows)), replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-snr-vis", type=float)
    ap.add_argument("--max-sep", type=float)
    ap.add_argument("--min-sb", type=float)
    args = ap.parse_args()
    sets = [f"deblend.run_suffix={args.suffix}"] if args.suffix else []
    cfg = MLConfig.load(args.config, sets)
    if not (cfg.dataset_dir("d2") / "mask_loss.parquet").exists() and \
            (cfg["deblend"].get("d2_selection") or {}).get("max_lost_flux_frac") is not None:
        sets.append("deblend.d2_selection.max_lost_flux_frac=null")      # not computed yet: examples all the same
        cfg = MLConfig.load(args.config, sets)
    import json

    from keras import ops

    from dp2_cutouts.deblend import data, train
    from dp2_cutouts.deblend.evaluate import Campaign
    camp = Campaign(cfg)
    if camp.missing:
        print(f"stages not trained (left out): {camp.missing}")
    rows = pick_rows(data.D2Batches(cfg, args.split, 1, False), args)
    ds = data.D2Batches(cfg, args.split, len(rows), False, indices=rows)
    x, _ = ds[0]
    meta = ds.st.meta.iloc[rows].reset_index(drop=True)
    ib = 3                                                     # i in ugrizy

    def np_(t):
        return np.asarray(ops.convert_to_numpy(t))

    rubin, vis = {}, {}                                        # title -> (B, n, n, bands) / (B, ne, ne)
    cond = x.get("x_psf")                                      # Rubin PSF of the stamps (deblend.psf_condition)
    tc = None if cond is None else ops.convert_to_tensor(cond)
    if "k1" in camp.parts:
        mu_k1, _ = camp.parts["k1"][0](ops.convert_to_tensor(x["e_in"]))
        vis["K1: VIS target redrawn\nfrom its latent"] = camp.decode(camp.parts["k1"][1], np_(mu_k1), camp.se)[..., 0]
        if "k2" in camp.parts:
            rubin["K2 o K1: Euclid -> Rubin\n(teacher's view, Runs A/B)"] = camp.decode(camp.parts["k2"], np_(mu_k1),
                                                                                        camp.sr, cond)
    rd = train.run_dir(cfg, "c2")
    if "c1" in camp.parts and (rd / "DONE").exists():
        s_c2 = json.loads((rd / "run.json").read_text())["settings"]
        br = train.load_c2_branches(cfg, rd, camp.d, s_c2)
        mu_t, _ = br.teacher(ops.convert_to_tensor(x["x_in"]), ops.convert_to_tensor(x["e_in"]), cond=tc)
        mu_s, _ = br.student(ops.convert_to_tensor(x["x_in"]), cond=tc)
        _, dec_r, dec_e = camp.parts["c1"]
        rubin["C2 teacher (blend + VIS)\ndrawn by C1 as Rubin"] = camp.decode(dec_r, np_(mu_t), camp.sr, cond)
        vis["C2 teacher (blend + VIS)\ndrawn by C1 as VIS"] = camp.decode(dec_e, np_(mu_t), camp.se)[..., 0]
        rubin["C2 student (blend only)\ndrawn by C1 as Rubin"] = camp.decode(dec_r, np_(mu_s), camp.sr, cond)
    means = {run: smp(x["x_in"], args.k, cond=cond)[1].mean(1) for run, smp in camp.samplers.items()}
    noise_r = np.median(1 / np.sqrt(np.where(x["x_w"] > 0, x["x_w"], np.nan)), axis=(1, 2))   # (B, bands)
    e_noise = np.median(1 / np.sqrt(np.where(x["e_w"][..., 0] > 0, x["e_w"][..., 0], np.nan)), axis=(1, 2))

    with PdfPages(args.out) as pdf:
        for j in range(len(rows)):
            m = meta.iloc[j]
            blend = x["x_flux"][j]
            n = blend.shape[0]
            c = n // 2
            s = float(noise_r[j, ib])
            top = max(float(np.nanmax(blend[c - n // 6:c + n // 6, c - n // 6:c + n // 6, ib])), 5 * s)
            soft = max(top / 10, s)
            e = x["e_flux"][j, ..., 0]
            ne = e.shape[0]
            es = float(e_noise[j])
            etop = max(float(np.nanmax(e[ne // 3:-ne // 3, ne // 3:-ne // 3])), 5 * es)
            tx, ty = -float(m.get("rc_offset_x_px", 0)), -float(m.get("rc_offset_y_px", 0))
            rows_p = [[("Rubin blend (gri)\nthe input X", "rgb", blend),
                       ("Rubin blend, i", "i", blend[..., ib]),
                       ("Euclid VIS target E_c\n(neighbours masked, red)", "vis_in", e)]]
            rows_p.append([(t, "vis", v[j]) for t, v in vis.items() if t.startswith("K1")]
                          + [(t, "i", r[j, ..., ib]) for t, r in rubin.items() if t.startswith("K2")])
            rows_p.append([(t, "i", r[j, ..., ib]) for t, r in rubin.items() if t.startswith("C2")]
                          + [(t, "vis", v[j]) for t, v in vis.items() if t.startswith("C2")])
            rows_p.append([(f"Run {r}: mean of {args.k}\n(Rubin only)", "i", mm[j, ..., ib]) for r, mm in means.items()])
            rows_p.append([(f"Run {r}: blend - mean", "res", blend[..., ib] - mm[j, ..., ib]) for r, mm in means.items()])
            rows_p = [rw for rw in rows_p if rw]
            cols = max(len(rw) for rw in rows_p)
            fig, axs = plt.subplots(len(rows_p), cols, figsize=(3.0 * cols, 3.25 * len(rows_p) + 0.5), squeeze=False)
            for i, rw in enumerate(rows_p):
                for k in range(cols):
                    ax = axs[i, k]
                    if k >= len(rw):
                        ax.axis("off")
                        continue
                    title, kind, img = rw[k]
                    if kind == "rgb":
                        rgb = np.dstack([stretch(blend[..., b], top, soft) for b in (3, 2, 1)])
                        ax.imshow(np.clip(rgb, 0, 1), origin="lower")
                    elif kind == "i":
                        ax.imshow(stretch(img, top, soft), origin="lower", cmap="gray_r", vmin=0, vmax=1)
                    elif kind == "res":
                        ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-5 * s, vmax=5 * s)
                    else:
                        show = np.where(x["e_w"][j, ..., 0] > 0, img, 0) if kind == "vis_in" else img
                        ax.imshow(stretch(show, etop, max(etop / 10, es)), origin="lower", cmap="gray_r", vmin=0,
                                  vmax=1)
                        bad = (x["e_w"][j, ..., 0] <= 0).astype(float)
                        if kind == "vis_in" and bad.any():
                            ax.contour(bad, levels=[0.5], colors="red", linewidths=0.7)
                    if kind in ("rgb", "i", "res"):
                        ax.plot([c + tx], [c + ty], "+", color="orange", ms=10, mew=1.2)
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks([]), ax.set_yticks([])
            txt = (f"D2 {args.split} | euclid_object_id {m.get('euclid_object_id')} | VIS {m.get('euclid_vis_mag', np.nan):.2f}"
                   f" (S/N {m.get('snr_vis', np.nan):.0f}) | partner VIS {m.get('partner_vis_mag', np.nan):.2f} at "
                   f"{m.get('partner_sep_arcsec', np.nan):.2f}\" | Rubin blendedness {m.get('sb_rubin', np.nan):.2f}"
                   f" | runs{args.suffix or ''}")
            fig.suptitle(txt, fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig, dpi=110)
            plt.close(fig)
    print(f"-> {args.out} ({len(rows)} targets, runs {sorted(means)}, steps {sorted(rubin) + sorted(vis)})")


if __name__ == "__main__":
    main()
