# deblend_gallery.py
# Anna Preto
# AstroDeep Project
# What a deblend looks like: one PDF page per D2 target, the trained runs of the campaign applied to the Rubin blend
# (the only input at inference) next to what Euclid says. Panels (i band unless stated, one asinh stretch per page for
# every Rubin panel, so that fluxes compare):
#   Rubin blend (gri, i)        the input X, centred on the Rubin object; orange + = the Euclid target
#   Euclid VIS target           the teacher's target E_c: VIS with the neighbours masked (red)
#   Euclid -> Rubin             K2 decoder applied to the K1 latent of E_c: the isolated target as Rubin would see it,
#                               inferred from Euclid (the teacher's view; not available at inference)
#   Run A / B / C mean          posterior mean of K decoded samples (Rubin only)
#   Run A / B / C residual      blend - mean: the neighbours should remain, the target should vanish
# Models are those of the campaign (Campaign of deblend/evaluate.py); --suffix _pilot reads the pilot runs.
# Usage: python scripts/inspect/deblend_gallery.py <ML config> <out.pdf> [--suffix _pilot] [--n 12] [--k 10]
#            [--split val] [--seed 0]

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from dp2_cutouts.mlconfig import MLConfig  # noqa: E402


def stretch(img, top, soft):
    return np.clip(np.arcsinh(img / soft) / np.arcsinh(top / soft), -0.05, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sets = [f"deblend.run_suffix={args.suffix}"] if args.suffix else []
    cfg = MLConfig.load(args.config, sets)
    if not (cfg.dataset_dir("d2") / "mask_loss.parquet").exists() and             (cfg["deblend"].get("d2_selection") or {}).get("max_lost_flux_frac") is not None:
        sets.append("deblend.d2_selection.max_lost_flux_frac=null")      # not computed yet: examples all the same
        cfg = MLConfig.load(args.config, sets)
    from keras import ops

    from dp2_cutouts.deblend import data
    from dp2_cutouts.deblend.evaluate import Campaign
    camp = Campaign(cfg)
    if camp.missing:
        print(f"stages not trained (left out): {camp.missing}")
    probe = data.D2Batches(cfg, args.split, 1, False)
    rows = np.sort(np.random.default_rng(args.seed).choice(probe.rows_all, min(args.n, len(probe.rows_all)),
                                                           replace=False))
    ds = data.D2Batches(cfg, args.split, len(rows), False, indices=rows)
    x, _ = ds[0]
    meta = ds.st.meta.iloc[rows].reset_index(drop=True)
    ib = 3                                                     # i in ugrizy
    means = {run: smp(x["x_in"], args.k)[1].mean(1) for run, smp in camp.samplers.items()}
    ref = None
    if "k1" in camp.parts and "k2" in camp.parts:
        mu, _ = camp.parts["k1"][0](ops.convert_to_tensor(x["e_in"]))
        ref = camp.decode(camp.parts["k2"], np.asarray(ops.convert_to_numpy(mu)), camp.sr)
    noise_r = np.median(1 / np.sqrt(np.where(x["x_w"] > 0, x["x_w"], np.nan)), axis=(1, 2))   # (B, bands)
    with PdfPages(args.out) as pdf:
        for j in range(len(rows)):
            m = meta.iloc[j]
            blend = x["x_flux"][j]
            n = blend.shape[0]
            c = n // 2
            s = float(noise_r[j, ib])
            top = max(float(np.nanmax(blend[c - n // 6:c + n // 6, c - n // 6:c + n // 6, ib])), 5 * s)
            soft = max(top / 10, s)
            tx, ty = -float(m.get("rc_offset_x_px", 0)), -float(m.get("rc_offset_y_px", 0))
            panels = [("Rubin blend (gri)", "rgb", blend), ("Rubin blend, i", "i", blend[..., ib]),
                      ("Euclid VIS target\n(neighbours masked, red)", "vis", None)]
            if ref is not None:
                panels.append(("Euclid -> Rubin (K2 o K1)\nteacher's view", "i", ref[j, ..., ib]))
            for run, mean in means.items():
                panels.append((f"Run {run}: mean of {args.k}", "i", mean[j, ..., ib]))
            for run, mean in means.items():
                panels.append((f"Run {run}: blend - mean", "res", blend[..., ib] - mean[j, ..., ib]))
            cols = 5
            rws = -(-len(panels) // cols)
            fig, axs = plt.subplots(rws, cols, figsize=(3.0 * cols, 3.2 * rws + 0.6))
            for ax in axs.ravel():
                ax.axis("off")
            for ax, (title, kind, img) in zip(axs.ravel(), panels):
                if kind == "rgb":
                    rgb = np.dstack([stretch(blend[..., b], top, soft) for b in (3, 2, 1)])
                    ax.imshow(np.clip(rgb, 0, 1), origin="lower")
                elif kind == "i":
                    ax.imshow(stretch(img, top, soft), origin="lower", cmap="gray_r", vmin=0, vmax=1)
                elif kind == "res":
                    lim = 5 * s
                    ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
                else:
                    e = x["e_flux"][j, ..., 0]
                    ne = e.shape[0]
                    es = float(np.median(1 / np.sqrt(np.where(x["e_w"][j, ..., 0] > 0, x["e_w"][j, ..., 0], np.nan))))
                    etop = max(float(np.nanmax(e[ne // 3:-ne // 3, ne // 3:-ne // 3])), 5 * es)
                    ax.imshow(stretch(np.where(x["e_w"][j, ..., 0] > 0, e, 0), etop, max(etop / 10, es)), origin="lower",
                              cmap="gray_r", vmin=0, vmax=1)
                    bad = (x["e_w"][j, ..., 0] <= 0).astype(float)
                    if bad.any():
                        ax.contour(bad, levels=[0.5], colors="red", linewidths=0.7)
                if kind in ("rgb", "i", "res"):
                    ax.plot([c + tx], [c + ty], "+", color="orange", ms=10, mew=1.2)
                ax.set_title(title, fontsize=8)
                ax.axis("on")
                ax.set_xticks([]), ax.set_yticks([])
            txt = (f"D2 {args.split} | euclid_object_id {m.get('euclid_object_id')} | VIS {m.get('euclid_vis_mag', np.nan):.2f}"
                   f" | partner VIS {m.get('partner_vis_mag', np.nan):.2f} at {m.get('partner_sep_arcsec', np.nan):.2f}\""
                   f" | residual colour scale +-5 sigma | runs{args.suffix or ''}")
            fig.suptitle(txt, fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig, dpi=110)
            plt.close(fig)
    print(f"-> {args.out} ({len(rows)} targets, runs {sorted(means)})")


if __name__ == "__main__":
    main()
