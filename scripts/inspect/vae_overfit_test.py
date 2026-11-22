# vae_overfit_test.py
# Anna Preto
# AstroDeep Project
# Can the DebVAder architecture reconstruct the y band at all? A small over-fitting test that isolates the output
# activation of the decoder mean: the same model (the run's config.json), from the same initialisation (same seed),
# trained on the same few stamps that are bright in y by catalogue photometry, once per variant of the output
# activations (default: mean ReLU vs mean linear, the scale activation identical). Every eval_every epochs, on those
# stamps: y aperture flux reconstructed / observed, share of y mean pre-activations <= 0, NLL per band. No latent
# head (pure VAE). Writes curves.csv, curves.png and examples.png into --out.
# Usage: python scripts/inspect/vae_overfit_test.py --run <debvader run dir (config.json)> --out <dir>
#            [--data <rubin_vae folder>] [--d1 <D1 dir>] [--n 64] [--epochs 300] [--eval-every 20]
#            [--variants relu,relu linear,relu] [--seed 0]

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dp2_cutouts.ml import vae_audit as A  # noqa: E402  (sets KERAS_BACKEND before keras)
from dp2_cutouts.pairs.store import Store  # noqa: E402

BANDS = list("ugrizy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data")
    ap.add_argument("--d1")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--variants", nargs="+", default=["relu,relu", "linear,relu"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import keras
    from keras import ops
    from debvader.core.vae import DebvaderVAE
    from debvader.data.normalization import DEFAULT_NONLINEAR_BETA, normalize_nonlinear_numpy
    from debvader.utils.config import Config

    run = Path(args.run)
    conf_path = run / "config.json"
    data = Path(args.data or json.loads(conf_path.read_text())["data"]["dataset_path"])
    d1 = Store(Path(args.d1) if args.d1 else data.parent / "d1")
    images = np.load(data / "images.npy", mmap_mode="r")
    meta = dict(np.load(data / "meta.npz"))
    scale = np.asarray(json.loads((data / "stats.json").read_text())["nonlinear"]["scale_per_band"], np.float32)
    cat = d1.meta.iloc[meta["d1_row"]].reset_index(drop=True)

    # The stamps brightest in y by catalogue photometry (every split: this is a capacity test, not a generalisation one).
    snr_y = A.catalogue_snr(cat["y_cModel_mag"], cat["depth5_y"])
    rows = np.sort(np.argsort(-np.nan_to_num(snr_y, nan=-1))[:args.n])
    raw = np.asarray(images[rows], np.float32)
    x = normalize_nonlinear_numpy(raw, scale, DEFAULT_NONLINEAR_BETA).astype(np.float32)
    n = x.shape[1]
    c = (n - 1) / 2.0
    m = cat.iloc[rows]
    gal, _ = A.regions((n, n), np.column_stack([np.full(len(m), c) + m["rubin_subpixel_dx"].fillna(0).to_numpy(),
                                                np.full(len(m), c) + m["rubin_subpixel_dy"].fillna(0).to_numpy()]),
                       m[["ellipse_a", "ellipse_b", "ellipse_theta"]].to_numpy(float))
    obs_flux = A.aperture_flux(x, gal)                       # normalised units: a relative comparison is enough
    iy = BANDS.index("y")
    print(f"{len(rows)} stamps, catalogue S/N_y {np.nanmin(snr_y[rows]):.0f} - {np.nanmax(snr_y[rows]):.0f}  "
          f"({time.time() - t0:.0f} s)")

    records, finals = [], {}
    for variant in args.variants:
        mean_act, scale_act = variant.split(",")
        cfg = Config.from_json(conf_path)
        cfg.model.use_latent_head = False
        cfg.model.use_z_head = False
        cfg.model.use_cvae = False
        cfg.model.output_mean_activation, cfg.model.output_scale_activation = mean_act, scale_act
        keras.utils.set_random_seed(args.seed)
        vae = DebvaderVAE(cfg.model, cfg.training, tuple(cfg.data.input_shape), normalization="nonlinear")
        vae(ops.zeros((1, *x.shape[1:])), training=False)
        vae.compile(optimizer=keras.optimizers.Adam(float(getattr(cfg.training, "learning_rate", 1e-3))))
        C = x.shape[-1]
        dec = vae.decoder

        def evaluate(epoch):
            params = vae.encoder(ops.convert_to_tensor(x), training=False)
            mu_lat = params[:, :cfg.model.latent_dim]
            o = np.asarray(ops.convert_to_numpy(dec(mu_lat, training=False)))
            fc, act = dec.final_conv, dec.final_conv.activation
            saved = (getattr(dec, "_mean_act", None), getattr(dec, "_scale_act", None))
            try:
                fc.activation = keras.activations.linear
                if hasattr(dec, "_mean_act"):
                    dec._mean_act = dec._scale_act = None
                pre = np.asarray(ops.convert_to_numpy(dec(mu_lat, training=False)))
            finally:
                fc.activation = act
                if hasattr(dec, "_mean_act"):
                    dec._mean_act, dec._scale_act = saved
            mu, sig = o[..., :C], 1e-4 + o[..., C:]
            rec = A.aperture_flux(mu, gal)
            nll = A.gaussian_nll(x, mu, sig)
            rec_y = rec[:, iy] / np.where(obs_flux[:, iy] != 0, obs_flux[:, iy], np.nan)
            r = dict(variant=variant, epoch=epoch, y_flux_ratio_median=float(np.nanmedian(rec_y)),
                     y_pre_mu_nonpositive=float((pre[..., iy][gal] <= 0).mean()),
                     **{f"nll_{b}": float(nll[..., k][gal].mean()) for k, b in enumerate(BANDS)},
                     **{f"flux_ratio_{b}": float(np.nanmedian(rec[:, k] / np.where(obs_flux[:, k] != 0,
                                                                                    obs_flux[:, k], np.nan)))
                        for k, b in enumerate(BANDS)})
            records.append(r)
            print(f"  {variant:>14s} epoch {epoch:4d}: y flux rec/obs {r['y_flux_ratio_median']:.3f}, y pre-act <= 0 "
                  f"{r['y_pre_mu_nonpositive']:.0%}, NLL y {r['nll_y']:.3f}  ({time.time() - t0:.0f} s)")
            return mu

        last = {"mu": evaluate(0)}

        class Every(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
                    last["mu"] = evaluate(epoch + 1)

        vae.fit(x, x, batch_size=min(32, len(x)), epochs=args.epochs, verbose=0, shuffle=True, callbacks=[Every()])
        finals[variant] = last["mu"]

    df = pd.DataFrame(records)
    df.to_csv(out / "curves.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for v, g in df.groupby("variant"):
        ax[0].plot(g["epoch"], g["y_flux_ratio_median"], marker="o", ms=3, label=v)
        ax[1].plot(g["epoch"], g["y_pre_mu_nonpositive"], marker="o", ms=3, label=v)
        ax[2].plot(g["epoch"], g["nll_y"], marker="o", ms=3, label=v)
    ax[0].axhline(1, c="0.6", lw=0.8)
    ax[0].set_ylabel("y flux reconstructed / observed (median)")
    ax[1].set_ylabel("y mean pre-activations <= 0 (galaxy)")
    ax[2].set_ylabel("NLL per pixel, y (galaxy)")
    for a_ in ax:
        a_.set_xlabel("epoch")
        a_.legend(fontsize=7)
    fig.suptitle(f"over-fitting test on the {len(rows)} stamps brightest in y (catalogue); same seed, same data",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)

    k_show = np.argsort(-np.nan_to_num(snr_y[rows], nan=-1))[:6]
    fig, axes = plt.subplots(len(k_show), 1 + len(finals), figsize=(2.2 * (1 + len(finals)), 2.2 * len(k_show)),
                             squeeze=False)
    for r_, j in enumerate(k_show):
        vmax = float(np.nanpercentile(np.abs(x[j][..., iy]), 99.5)) or 1.0
        panels = [("obs y", x[j][..., iy])] + [(f"mean y ({v})", mu_[j][..., iy]) for v, mu_ in finals.items()]
        for a_, (t, im) in zip(axes[r_], panels):
            a_.imshow(im, origin="lower", cmap="gray_r", vmin=-0.2 * vmax, vmax=vmax)
            a_.set_xticks([])
            a_.set_yticks([])
            if r_ == 0:
                a_.set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=120)
    print(f"-> {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
