# encoder_finetune_test.py
# Anna Preto
# AstroDeep Project
# Can the current Rubin VAE encoder, as it is, close its amortisation gap? (2026-11-28, after latent_refine_test.py:
# per-galaxy optimised posteriors improved the ELBO by ~180 nats at S/N_i 80-200 and ~1 700 nats at S/N_i >= 200,
# and the shapes, with the decoder unchanged.) From the beta = 1 checkpoint, the decoder frozen and the encoder's
# architecture and weights kept, only the encoder is trained further: a fresh Adam at a constant lr 1e-4, the same
# ELBO (one reparametrised draw per galaxy, beta = 1), at most `steps` (300) updates of `batch` (128) train
# galaxies at S/N_i >= 80 (with the training augmentation). Evaluated on the same 64 validation galaxies as
# latent_refine_test.py (same selection, same fixed draws): NLL, KL, ELBO, m and |delta e| before / after, and the
# fraction of the amortisation gap recovered, with the per-galaxy optimised latents (recomputed here, as in
# latent_refine_test.py) as the reference: (ELBO_finetuned - ELBO_encoder) / (ELBO_optimised - ELBO_encoder).
# The validation ELBO is also followed every 50 updates (is it still improving?).
# A large gain: the architecture has unused capacity (optimisation, not design). No gain: no proof of a lack of
# capacity. Nothing is written but the printed table (and --out json); no checkpoint is saved.
# Hard limit: `max_seconds` (300) for everything, JAX compilation included.
# Usage (one V100): python scripts/inspect/encoder_finetune_test.py config/ml_jeanzay.yaml

import time

T0 = time.time()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latent_refine_test import select_rows  # noqa: E402


class OutOfTime(RuntimeError):
    pass


def note(msg):
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - T0:6.1f} s, peak {rss:5.1f} GB] {msg}", flush=True)


def run(cfg, groups=((80, 200), (200, 1e9)), n=32, train_snr=80.0, steps=300, batch=128, lr=1e-4,
        opt_lr=0.03, opt_steps=500, eval_draws=16, seed=0, max_seconds=300):
    import jax
    import jax.numpy as jnp
    from keras import ops

    from dp2_cutouts.deblend import data, evaluate
    from latent_refine_test import converged

    from latent_refine_test import load_rubin_vae

    # The budget counts the GPU work only (compilation included), from the end of the data reading: the reading of
    # the weights and the metadata on Lustre took 262 s once (job 941720) and is not GPU time. Slurm's --time bounds
    # the whole job.
    t_gpu = [None]

    def left():
        return max_seconds - (time.time() - (t_gpu[0] or time.time()))

    enc, dec, sr = load_rubin_vae(cfg)     # the Rubin VAE only
    enc.trainable = True            # loaded frozen (train.load_rubin_encoder); its weights are the ones trained here
    note("models loaded")
    cols = list(dict.fromkeys(list(data.train_columns(cfg, "d1")) + ["snr_i", "psf_sigma_i", "psf_e1_i",
                                                                     "psf_e2_i"]))
    st = data.Store(cfg.dataset_dir("d1"), cols, shared=True)
    keep, _ = data.d1_keep(cfg, st.meta)
    rows, labels = select_rows(st.meta, keep, groups, n, seed)          # the same 64 as latent_refine_test.py
    xv, _ = data.D1Batches(cfg, "val", len(rows), False, indices=rows, fields=("rubin",), meta_columns=cols)[0]
    mrows = st.meta.iloc[rows].reset_index(drop=True)
    snr = st.meta["snr_i"].to_numpy(float)
    tr_rows = np.flatnonzero((st.meta["split"] == "train").to_numpy() & np.asarray(keep, bool) & (snr >= train_snr))
    ds = data.D1Batches(cfg, "train", batch, True, seed, indices=tr_rows, fields=("rubin",), meta_columns=cols,
                        block="auto")
    note(f"{len(rows)} validation galaxies; {len(tr_rows)} train galaxies at S/N_i >= {train_snr:g}")

    t_gpu[0] = time.time()
    note(f"GPU budget of {max_seconds:g} s starts")
    scale = jnp.asarray(sr, jnp.float32)
    tv0 = [jnp.asarray(v.value) for v in enc.trainable_variables]
    ntv = [jnp.asarray(v.value) for v in enc.non_trainable_variables]

    def posterior(tv, x_in, cond):
        (mu, lv), _ = enc.stateless_call(tv, ntv, x_in, cond=cond)
        return mu, lv

    def decode(z, cond):        # z (B, K, d) -> (B, K, n, n, C)
        B, K, d = z.shape
        cc = None if cond is None else jnp.repeat(cond[:, None], K, 1).reshape(B * K, -1)
        img = dec(z.reshape(B * K, d), cond=cc) * scale
        return img.reshape(B, K, *img.shape[1:])

    def terms(mu, lv, eps, flux, w, cond):       # per galaxy: NLL (mean over the draws), KL
        z = mu[:, None] + jnp.exp(0.5 * lv)[:, None] * eps
        img = decode(z, cond)
        nll = 0.5 * jnp.sum(w[:, None] * (flux[:, None] - img) ** 2, axis=(2, 3, 4)).mean(1)
        kl = 0.5 * jnp.sum(jnp.exp(lv) + mu ** 2 - 1.0 - lv, axis=1)
        return nll, kl

    def enc_loss(tv, x_in, cond, eps, flux, w):  # the training ELBO: one draw per galaxy, beta = 1
        mu, lv = posterior(tv, x_in, cond)
        nll, kl = terms(mu, lv, eps, flux, w, cond)
        return jnp.mean(nll + kl)

    enc_step = jax.jit(jax.value_and_grad(enc_loss))
    post_j = jax.jit(posterior)
    ev = jax.jit(terms)
    key = jax.random.PRNGKey(seed)
    vin = jnp.asarray(xv["r_in"])
    vc = None if xv.get("r_psf") is None else jnp.asarray(xv["r_psf"])
    vflux, vw = jnp.asarray(xv["r_flux"]), jnp.asarray(xv["r_w"])
    B = len(rows)
    mu0, lv0 = post_j(tv0, vin, vc)
    d = mu0.shape[1]
    eps_eval = jax.random.normal(jax.random.fold_in(key, 1), (B, eval_draws, d))   # as latent_refine_test.py

    def val_summary(mu, lv):
        nll, kl = (np.asarray(a) for a in ev(mu, lv, eps_eval, vflux, vw, vc))
        img = np.asarray(decode(mu[:, None], vc))[:, 0]
        return nll, kl, img

    before = val_summary(mu0, lv0)
    note("evaluation of the encoder (compiled)")

    # Reference: per-galaxy optimised posteriors (as latent_refine_test.py, until converged).
    def lat_loss(p, eps):
        nll, kl = terms(p[0], p[1], eps, vflux, vw, vc)
        return jnp.mean(nll + kl)

    lat_step = jax.jit(jax.value_and_grad(lat_loss))
    p, m_, v_ = [mu0, lv0], [jnp.zeros_like(mu0), jnp.zeros_like(lv0)], [jnp.zeros_like(mu0), jnp.zeros_like(lv0)]
    curve_lat = []
    for t in range(1, opt_steps + 1):
        val, g = lat_step(p, jax.random.normal(jax.random.fold_in(key, 100 + t), (B, 2, d)))
        curve_lat.append(float(val))
        for i in range(2):
            m_[i] = 0.9 * m_[i] + 0.1 * g[i]
            v_[i] = 0.999 * v_[i] + 0.001 * g[i] ** 2
            p[i] = p[i] - opt_lr * (m_[i] / (1 - 0.9 ** t)) / (jnp.sqrt(v_[i] / (1 - 0.999 ** t)) + 1e-8)
        if converged(curve_lat) or left() < 200:
            break
    reference = val_summary(p[0], p[1])
    note(f"reference: per-galaxy latents optimised in {len(curve_lat)} steps, converged {converged(curve_lat)}")

    # The encoder, trained further: fresh Adam, constant lr.
    tv = list(tv0)
    m_ = [jnp.zeros_like(a) for a in tv]
    v_ = [jnp.zeros_like(a) for a in tv]
    curve, val_curve = [], {}
    for t in range(1, steps + 1):
        x, _ = ds[(t - 1) % len(ds)]
        eps = jax.random.normal(jax.random.fold_in(key, 10_000 + t), (len(x["r_in"]), 1, d))
        c = None if x.get("r_psf") is None else jnp.asarray(x["r_psf"])
        val, g = enc_step(tv, jnp.asarray(x["r_in"]), c, eps, jnp.asarray(x["r_flux"]), jnp.asarray(x["r_w"]))
        curve.append(float(val))
        for i in range(len(tv)):
            m_[i] = 0.9 * m_[i] + 0.1 * g[i]
            v_[i] = 0.999 * v_[i] + 0.001 * g[i] ** 2
            tv[i] = tv[i] - lr * (m_[i] / (1 - 0.9 ** t)) / (jnp.sqrt(v_[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 50 == 0 or t == steps:
            mu, lv = post_j(tv, vin, vc)
            nll, kl = (np.asarray(a) for a in ev(mu, lv, eps_eval, vflux, vw, vc))
            val_curve[t] = round(float(np.median(-(nll + kl))), 1)
        if left() < 45:
            break
    mu1, lv1 = post_j(tv, vin, vc)
    after = val_summary(mu1, lv1)
    note(f"encoder trained {len(curve)} updates")

    valid = np.asarray(xv["r_w"]) > 0
    obs = evaluate.measure_rubin(np.asarray(xv["r_flux"]), valid, mrows, cfg, "o", 6.0)
    shapes = {k: evaluate.measure_rubin(s[2], valid, mrows, cfg, "s", 6.0)
              for k, s in (("encoder", before), ("finetuned", after), ("optimised", reference))}
    res = dict(n=int(B), updates=len(curve), lr=lr, batch=batch, train_galaxies=int(len(tr_rows)),
               elapsed_s=round(time.time() - T0, 1), reference_steps=len(curve_lat),
               reference_converged=bool(converged(curve_lat)),
               train_loss={int(s): round(float(np.mean(curve[max(0, s - 10):s])), 1)
                           for s in sorted(set([10, 50, 100, 150, 200, 250, 300, len(curve)])) if s <= len(curve)},
               val_elbo_median=val_curve, groups={})
    for gname in sorted(set(labels)):
        sel = labels == gname
        r = dict(n=int(sel.sum()))
        for tag, (nll, kl, _) in (("encoder", before), ("finetuned", after), ("optimised", reference)):
            r[f"nll_{tag}"] = float(np.median(nll[sel]))
            r[f"kl_{tag}"] = float(np.median(kl[sel]))
            r[f"elbo_{tag}"] = float(np.median(-(nll[sel] + kl[sel])))
            for q in ("e1", "e2"):
                y = evaluate.valid_e(shapes[tag][f"s_{q}_rg"].to_numpy(float)[sel])
                xo = evaluate.valid_e(obs[f"o_{q}_rg"].to_numpy(float)[sel])
                r[f"m_{q}_{tag}"] = evaluate.shape_metrics(y, xo).get("m")
                r[f"abs_de_{q}_{tag}"] = float(np.nanmedian(np.abs(y - xo)))
        gap = r["elbo_optimised"] - r["elbo_encoder"]
        r["gap_recovered"] = float((r["elbo_finetuned"] - r["elbo_encoder"]) / gap) if gap > 0 else None
        res["groups"][gname] = r
    res["gpu_s"] = round(time.time() - t_gpu[0], 1)
    if time.time() - t_gpu[0] > max_seconds:
        raise OutOfTime(f"GPU part {time.time() - t_gpu[0]:.0f} s > {max_seconds} s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-seconds", type=float, default=300)
    ap.add_argument("--out")
    a = ap.parse_args()
    from dp2_cutouts.mlconfig import MLConfig
    try:
        res = run(MLConfig.load(a.config, []), max_seconds=a.max_seconds)
    except OutOfTime as e:
        print(f"ABANDON: time limit reached ({e}); no conclusion")
        return
    print(f"{res['n']} validation galaxies; encoder trained {res['updates']} updates of {res['batch']} "
          f"({res['train_galaxies']} train galaxies at S/N_i >= 80), lr {res['lr']}; {res['elapsed_s']} s in all, "
          f"{res['gpu_s']} s of GPU work; "
          f"reference latents: {res['reference_steps']} steps, converged {res['reference_converged']}")
    print("train loss (mean of the 10 updates up to):", ", ".join(f"{k}: {v}" for k, v in res["train_loss"].items()))
    print("validation ELBO, median over the 64 galaxies:", ", ".join(f"{k}: {v}" for k, v in res["val_elbo_median"].items()))
    print(f"{'S/N_i':>9s} {'':10s} {'NLL':>10s} {'KL':>7s} {'ELBO':>10s} {'m e1':>7s} {'m e2':>7s} {'|de1|':>6s} {'|de2|':>6s}")
    for g, r in res["groups"].items():
        for tag in ("encoder", "finetuned", "optimised"):
            def f(k):
                v = r.get(k)
                return float("nan") if v is None else v
            print(f"{g:>9s} {tag:10s} {f('nll_' + tag):10.1f} {f('kl_' + tag):7.1f} {f('elbo_' + tag):10.1f} "
                  f"{f('m_e1_' + tag):+7.3f} {f('m_e2_' + tag):+7.3f} {f('abs_de_e1_' + tag):6.3f} {f('abs_de_e2_' + tag):6.3f}")
        gr = r.get("gap_recovered")
        print(f"{g:>9s} amortisation gap recovered by the fine-tuned encoder: "
              + ("n/a" if gr is None else f"{100 * gr:.0f} %"))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
