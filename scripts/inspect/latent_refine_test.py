# latent_refine_test.py
# Anna Preto
# AstroDeep Project
# Is the Rubin VAE limited by its encoder (amortised inference) or by its decoder? (2026-11-28, after the beta = 1
# checkpoint: shapes shrunk by 35-50 % at S/N_i >= 100, chi2 excess of hundreds to thousands of nats on bright
# galaxies.) With the trained networks frozen, the posterior of each galaxy (mu, log sigma^2, initialised from the
# encoder) is optimised directly on the same ELBO (beta = 1): Adam, `steps` steps, lr 0.03, `draws` latent draws per
# galaxy and step, at least 50 steps and until the loss stops improving (relative gain < 1e-3 between the last two
# 10-step windows; at most 500 steps). Before / after, on the same fixed draws: NLL, KL, ELBO per galaxy; and the REGAUSS shape of the
# image decoded at mu against the observed one (i band), per S/N group.
#   ELBO and shapes much better    -> the encoder's inference is the limit (amortisation gap)
#   chi2 better, shapes not        -> look at what the decoder renders
#   nothing better (converged)     -> the decoder is the limit: change the decoder
# Hard limit: the whole run (imports and JAX compilation included) stops after `max_seconds` (300) and then draws no
# conclusion. Read-only: nothing is written but the printed table (and --out json).
# Usage (one V100, a few minutes; or CPU): python scripts/inspect/latent_refine_test.py config/ml_jeanzay.yaml

import time

T0 = time.time()

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402


class OutOfTime(RuntimeError):
    pass


def note(msg):
    # Progress with elapsed time and peak memory of the process (to locate a memory kill).
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # kB -> GB (Linux)
    print(f"[{time.time() - T0:6.1f} s, peak {rss:5.1f} GB] {msg}", flush=True)


def check_time(max_seconds, where):
    if time.time() - T0 > max_seconds:
        raise OutOfTime(f"{where}: {time.time() - T0:.0f} s > {max_seconds} s")


def load_rubin_vae(cfg):
    # The Rubin VAE only (best weights): encoder, decoder, per-band output scale. (The first versions loaded every
    # finished VAE through evaluate.Campaign, K1 and C1 with their VIS decoders included: 262 s of reading on a slow
    # Lustre, job 941720.)
    from dp2_cutouts.deblend import data, train
    rd = train.run_dir(cfg, "rubin_vae")
    if not (rd / "DONE").exists():
        raise RuntimeError(f"no finished Rubin VAE in {rd}")
    d = int(cfg["deblend"]["latent_dim"])
    n_r, b_r, _ = train.sizes(cfg)
    enc = train.load_rubin_encoder(cfg, rd, d)
    dec = train.load_decoder(rd / "decoder_best.weights.h5", n_r, b_r, d, cfg)
    return enc, dec, np.asarray(data.load_norm_stats(cfg)["rubin"]["scale"], np.float32)


def select_rows(meta, keep, groups, n, seed):
    # n validation rows per S/N_i group [lo, hi), fixed seed.
    rng = np.random.default_rng(seed)
    snr = meta["snr_i"].to_numpy(float)
    val = (meta["split"] == "val").to_numpy() & np.asarray(keep, bool)
    out, labels = [], []
    for lo, hi in groups:
        rows = np.flatnonzero(val & (snr >= lo) & (snr < hi))
        pick = np.sort(rng.choice(rows, min(n, len(rows)), replace=False)) if len(rows) else rows
        out.append(pick)
        labels += [f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}"] * len(pick)
    return np.concatenate(out), np.array(labels)


def converged(curve, window=10, tol=1e-3, min_steps=50):
    # The loss of the last `window` steps (mean) improves on the `window` before by less than tol (relative), after
    # at least min_steps steps. The loss is noisy (2 draws per step): window means.
    if len(curve) < max(min_steps, 2 * window):
        return False
    a, b = np.mean(curve[-2 * window:-window]), np.mean(curve[-window:])
    return (a - b) < tol * abs(a)


def run(cfg, groups=((80, 200), (200, 1e9)), n=32, steps=500, lr=0.03, draws=2, eval_draws=16, seed=0,
        max_seconds=300, min_steps=50, tol=1e-3):
    # steps: at most; the optimisation stops once converged (see converged) or 40 s before max_seconds, leaving time
    # for the evaluation; res["converged"] says which.
    import jax
    import jax.numpy as jnp
    from keras import ops

    from dp2_cutouts.deblend import data, evaluate
    enc, dec, sr = load_rubin_vae(cfg)
    note("models loaded")
    # Only the metadata columns used here (the first run, with every column of the 1.5 M rows, was killed by the
    # 40 GB memory limit of the job).
    cols = list(dict.fromkeys(list(data.train_columns(cfg, "d1")) + ["snr_i", "psf_sigma_i", "psf_e1_i",
                                                                     "psf_e2_i"]))
    st = data.Store(cfg.dataset_dir("d1"), cols, shared=True)
    note("metadata read")
    keep, _ = data.d1_keep(cfg, st.meta)
    rows, labels = select_rows(st.meta, keep, groups, n, seed)
    if len(rows) == 0:
        raise RuntimeError("no validation galaxy in the S/N groups")
    x, _ = data.D1Batches(cfg, "val", len(rows), False, indices=rows, fields=("rubin",), meta_columns=cols)[0]
    note(f"{len(rows)} galaxies read")
    mrows = st.meta.iloc[rows].reset_index(drop=True)
    check_time(max_seconds, "data")
    cond = x.get("r_psf")
    c = None if cond is None else jnp.asarray(cond)
    flux, w = jnp.asarray(x["r_flux"]), jnp.asarray(x["r_w"])
    scale = jnp.asarray(sr, jnp.float32)
    mu0, lv0 = (jnp.asarray(ops.convert_to_numpy(a)) for a in enc(jnp.asarray(x["r_in"]), cond=c))
    B, d = mu0.shape

    def decode(z):            # z (B, K, d) -> (B, K, n, n, C) physical units
        K = z.shape[1]
        cc = None if c is None else jnp.repeat(c[:, None], K, 1).reshape(B * K, -1)
        img = dec(z.reshape(B * K, d), cond=cc) * scale
        return img.reshape(B, K, *img.shape[1:])

    def terms(mu, lv, eps):   # per galaxy: NLL (mean over draws), KL
        z = mu[:, None] + jnp.exp(0.5 * lv)[:, None] * eps
        img = decode(z)
        nll = 0.5 * jnp.sum(w[:, None] * (flux[:, None] - img) ** 2, axis=(2, 3, 4)).mean(1)
        kl = 0.5 * jnp.sum(jnp.exp(lv) + mu ** 2 - 1.0 - lv, axis=1)
        return nll, kl

    def loss(p, eps):
        nll, kl = terms(p[0], p[1], eps)
        return jnp.mean(nll + kl)

    grad = jax.jit(jax.value_and_grad(loss))
    ev = jax.jit(terms)
    key = jax.random.PRNGKey(seed)
    eps_eval = jax.random.normal(jax.random.fold_in(key, 1), (B, eval_draws, d))

    def summary(mu, lv):
        nll, kl = (np.asarray(a) for a in ev(mu, lv, eps_eval))
        img = np.asarray(decode(mu[:, None]))[:, 0]
        return nll, kl, img

    nll0, kl0, img0 = summary(mu0, lv0)
    note("evaluation before (compiled)")
    check_time(max_seconds, "evaluation before (compilation included)")
    p = [mu0, lv0]
    m = [jnp.zeros_like(a) for a in p]
    v = [jnp.zeros_like(a) for a in p]
    b1, b2, eps_ = 0.9, 0.999, 1e-8
    curve, done = [], False
    for t in range(1, steps + 1):
        e = jax.random.normal(jax.random.fold_in(key, 100 + t), (B, draws, d))
        val, g = grad(p, e)
        curve.append(float(val))
        for i in range(2):
            m[i] = b1 * m[i] + (1 - b1) * g[i]
            v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
            p[i] = p[i] - lr * (m[i] / (1 - b1 ** t)) / (jnp.sqrt(v[i] / (1 - b2 ** t)) + eps_)
        if converged(curve, tol=tol, min_steps=min_steps):
            done = True
            break
        if time.time() - T0 > max_seconds - 40:
            break
    note(f"optimisation: {len(curve)} steps, converged {done}")
    nll1, kl1, img1 = summary(p[0], p[1])
    obs = evaluate.measure_rubin(np.asarray(x["r_flux"]), np.asarray(x["r_w"]) > 0, mrows, cfg, "o", 6.0)
    s0 = evaluate.measure_rubin(img0, np.asarray(x["r_w"]) > 0, mrows, cfg, "s", 6.0)
    s1 = evaluate.measure_rubin(img1, np.asarray(x["r_w"]) > 0, mrows, cfg, "s", 6.0)
    check_time(max_seconds, "shapes")
    marks = [s for s in (1, 10, 25, 50, 100, 200, 300, 400, 500) if s <= len(curve)] + [len(curve)]
    res = dict(n=int(B), steps=len(curve), lr=lr, draws=draws, elapsed_s=round(time.time() - T0, 1),
               converged=done, loss_first_last=[curve[0], curve[-1]],
               curve={int(s): round(float(np.mean(curve[max(0, s - 5):s])), 2) for s in sorted(set(marks))},
               groups={})
    for g in sorted(set(labels)):
        sel = labels == g
        r = dict(n=int(sel.sum()))
        for tag, nll, kl in (("before", nll0, kl0), ("after", nll1, kl1)):
            r[f"nll_{tag}"] = float(np.median(nll[sel]))
            r[f"kl_{tag}"] = float(np.median(kl[sel]))
            r[f"elbo_{tag}"] = float(np.median(-(nll[sel] + kl[sel])))
        for tag, s in (("before", s0), ("after", s1)):
            for q in ("e1", "e2"):
                y = evaluate.valid_e(s[f"s_{q}_rg"].to_numpy(float)[sel])
                xo = evaluate.valid_e(obs[f"o_{q}_rg"].to_numpy(float)[sel])
                mm = evaluate.shape_metrics(y, xo)
                r[f"m_{q}_{tag}"] = mm.get("m")
                r[f"abs_de_{q}_{tag}"] = float(np.nanmedian(np.abs(y - xo)))
        res["groups"][g] = r
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
    print(f"{res['n']} galaxies, {res['steps']} steps, {res['elapsed_s']} s in all; "
          f"ELBO loss {res['loss_first_last'][0]:.1f} -> {res['loss_first_last'][1]:.1f}; "
          + ("CONVERGED" if res["converged"] else "NOT CONVERGED (stopped by the time limit): no conclusion"))
    print("loss curve (mean of the 5 steps up to):", ", ".join(f"{k}: {v}" for k, v in res["curve"].items()))
    print(f"{'S/N_i':>9s} {'':6s} {'NLL':>10s} {'KL':>7s} {'ELBO':>10s} {'m e1':>7s} {'m e2':>7s} {'|de1|':>6s} {'|de2|':>6s}")
    for g, r in res["groups"].items():
        for tag in ("before", "after"):
            def f(k):
                v = r.get(k)
                return float("nan") if v is None else v
            print(f"{g:>9s} {tag:6s} {f('nll_' + tag):10.1f} {f('kl_' + tag):7.1f} {f('elbo_' + tag):10.1f} "
                  f"{f('m_e1_' + tag):+7.3f} {f('m_e2_' + tag):+7.3f} {f('abs_de_e1_' + tag):6.3f} {f('abs_de_e2_' + tag):6.3f}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
