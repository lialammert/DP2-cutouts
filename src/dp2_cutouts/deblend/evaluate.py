# evaluate.py
# Anna Preto
# AstroDeep Project
# Evaluations of the deblending campaign: dp2-ml <cfg> deblend-eval <what> [--split test] [--seed S].
#   d1       isolated galaxies (D1): what each latent restores in Rubin (Rubin VAE R->R, K1+K2 E->R, C1 (R,E)->R)
#            and the three final models applied to isolated Rubin stamps, against the observed Rubin stamp;
#            observational predictive calibration (samples + a noise draw at the observed ivar, then measured).
#   masking  robustness of the K1 latent to real D2 neighbour masks applied to D1 Euclid stamps (latent shift in
#            posterior sigma units, decoded VIS flux / size / shape change), by mask severity.
#   d2       real blends (D2): the three runs on the same objects; shapes (adaptive moments and PSF-corrected
#            REGAUSS), sizes, fluxes, colours, failures, alignment with the neighbour, over-subtraction of the
#            blend; Euclid VIS REGAUSS shapes as an independent reference, read through the Rubin-vs-VIS relation
#            fitted on D1 (an empirical map, not assumed transferable without bias); performance by difficulty
#            (blendedness, S/N, magnitude, separation, contrast, size / PSF, mask severity, field); paired
#            differences between runs with a patch block bootstrap.
#   cost     inference cost: parameters, time per galaxy, throughput, CFM latent convergence vs NFE (Euler,
#            Heun) against a fine Heun reference, MC dropout passes.
#   latent   latent spaces of the Rubin VAE, K1 and C1 (variance of the posterior mean, posterior variance and KL
#            per dimension, active dimensions, PCA spectrum, effective dimension) and the C1 decoder Jacobians:
#            directions that move the Rubin image but barely the Euclid one, and their effect on Rubin fluxes,
#            colours, size and shape (JAX).
#   check_vaes / check_k2  the checkpoint of the VAEs before their dependents train (campaign gates; val split):
#            the D1 reconstructions of the Rubin VAE, C1 and K1 (VIS; with K2: E -> R) and their latent spaces.
#            Then dp2-ml <cfg> deblend-approve <stage>.
# Nothing here uses the holdout split unless --split holdout is given explicitly (final evaluation only).
# Outputs: <ml_root>/deblend/evaluations/<what>[_<split>][_s<seed>]/ (parquet tables, summary.json, figures).
# There is no isolated Rubin truth for D2: every D2 number is a comparison with references that are themselves
# measurements (Euclid through D1, the blend itself), never a calibration of the Rubin posterior.

import json

import numpy as np
import pandas as pd

from ..ml.evaluate import bootstrap, linear_mc, robust_scatter
from ..ml.training import write_json
from ..pairs import common
from . import data, infer, train

RUNS = ("A", "B", "C")
STAGE_OF = dict(A="A", B="B", C="c3")


# ----------------------------------------------------------------------------- measurements
def gaussian_psf_image(sigma_px, e1, e2, n=25):
    import galsim
    psf = galsim.Gaussian(sigma=max(float(sigma_px), 0.3)).shear(e1=float(np.clip(e1, -0.5, 0.5)),
                                                                 e2=float(np.clip(e2, -0.5, 0.5)))
    return psf.drawImage(nx=n, ny=n, scale=1.0).array


def shapes(img, valid, psf_sigma_px, psf_e1=0.0, psf_e2=0.0):
    # Adaptive moments (observed, PSF-convolved) and REGAUSS PSF-corrected distortion, with a Gaussian PSF model
    # built from the catalogue PSF moments (the PSF images are not in the datasets). NaN on failure.
    import galsim
    out = dict(e1=np.nan, e2=np.nan, sigma=np.nan, ok=False, e1_rg=np.nan, e2_rg=np.nan, ok_rg=False)
    n = img.shape[0]
    c = (n - 1) / 2.0
    m = common.moments(img, valid.astype(np.int32), (c, c))
    out.update(e1=m["e1"], e2=m["e2"], sigma=m["sigma"], ok=bool(m["ok"]))
    if not np.isfinite(psf_sigma_px):
        return out
    try:
        gi = galsim.Image(np.ascontiguousarray(img, np.float64), scale=1.0)
        bad = galsim.Image((~valid).astype(np.int32), scale=1.0)
        pi = galsim.Image(gaussian_psf_image(psf_sigma_px, psf_e1, psf_e2), scale=1.0)
        r = galsim.hsm.EstimateShear(gi, pi, badpix=bad, shear_est="REGAUSS", strict=False)
        if not r.error_message:
            out.update(e1_rg=float(r.corrected_e1), e2_rg=float(r.corrected_e2), ok_rg=True)
    except Exception:  # noqa: BLE001 - a failed measurement is recorded as such
        pass
    return out


def measure_rubin(images, valid, meta_rows, cfg, prefix, aperture_px):
    # images (N, n, n, C) nJy -> one row per image: i-band shapes, aperture fluxes per band.
    bands = list(cfg["rubin"]["bands"])
    ib = bands.index("i")
    n = images.shape[1]
    c = (n - 1) / 2.0
    rr, cc = np.indices((n, n))
    ap = (rr - c) ** 2 + (cc - c) ** 2 <= aperture_px ** 2
    rows = []
    for k in range(len(images)):
        mr = meta_rows.iloc[k]
        s = shapes(images[k][..., ib], valid[k][..., ib] if valid is not None else np.ones((n, n), bool),
                   float(mr.get("psf_sigma_i", np.nan)) / common.RUBIN_SCALE if "psf_sigma_i" in mr else np.nan,
                   float(mr.get("psf_e1_i", 0.0)), float(mr.get("psf_e2_i", 0.0)))
        r = {f"{prefix}_{q}": v for q, v in s.items()}
        for j, b in enumerate(bands):
            r[f"{prefix}_flux_{b}"] = float((images[k][..., j] * ap).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def psf_sigma_arcsec_from_fwhm(fwhm):
    return float(fwhm) / 2.3548


def measure_vis(images, valid, cfg, prefix):
    s_px = psf_sigma_arcsec_from_fwhm(cfg["euclid"].get("vis_psf_fwhm", 0.16)) / float(cfg["euclid"]["pixel_scale"])
    rows = []
    for k in range(len(images)):
        s = shapes(images[k][..., 0], valid[k][..., 0], s_px)
        rows.append({f"{prefix}_{q}": v for q, v in s.items()})
    return pd.DataFrame(rows)


def add_noise(images, w, rng):
    # Observational predictive: a noise draw at the observed inverse variance (0-weight pixels get none).
    sig = np.where(w > 0, 1.0 / np.sqrt(np.maximum(w, 1e-30)), 0.0)
    return images + rng.standard_normal(images.shape).astype(np.float32) * sig


# ----------------------------------------------------------------------------- probabilistic scores
def coverage(samples, ref, levels=(0.5, 0.68, 0.9, 0.95)):
    # samples (N, K), ref (N,): fraction of refs inside the central interval of each level.
    out = {}
    ok = np.isfinite(ref) & (np.isfinite(samples).sum(1) >= 3)
    for lv in levels:
        lo = np.nanpercentile(samples[ok], 50 * (1 - lv), axis=1)
        hi = np.nanpercentile(samples[ok], 50 * (1 + lv), axis=1)
        out[f"cov{int(lv * 100)}"] = float(np.mean((ref[ok] >= lo) & (ref[ok] <= hi))) if ok.any() else np.nan
    return out


def pit(samples, ref):
    ok = np.isfinite(ref) & (np.isfinite(samples).sum(1) >= 3)
    return np.array([np.nanmean(samples[i] <= ref[i]) for i in np.flatnonzero(ok)])


def crps(samples, ref):
    # Sample CRPS: E|X - y| - 1/2 E|X - X'|, mean over objects.
    vals = []
    for i in range(len(ref)):
        x = samples[i][np.isfinite(samples[i])]
        if x.size < 3 or not np.isfinite(ref[i]):
            continue
        vals.append(np.mean(np.abs(x - ref[i])) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :])))
    return float(np.mean(vals)) if vals else np.nan


def energy_score(samples, ref):
    # samples (N, K, p), ref (N, p): E||X - y|| - 1/2 E||X - X'||.
    vals = []
    for i in range(len(ref)):
        x = samples[i][np.all(np.isfinite(samples[i]), axis=1)]
        if len(x) < 3 or not np.all(np.isfinite(ref[i])):
            continue
        d1 = np.linalg.norm(x - ref[i], axis=1).mean()
        d2 = np.linalg.norm(x[:, None] - x[None, :], axis=-1).mean()
        vals.append(d1 - 0.5 * d2)
    return float(np.mean(vals)) if vals else np.nan


def summary_stats(d):
    d = np.asarray(d, float)
    return dict(bias=float(np.nanmedian(d)), nmad=robust_scatter(d), n=int(np.isfinite(d).sum()))


# ----------------------------------------------------------------------------- loading the trained runs
class Campaign:
    # Everything trained, loaded once; missing stages are skipped (listed in .missing).
    def __init__(self, cfg, seed=None, students=True):
        self.cfg, self.seed = cfg, seed
        self.d = int(cfg["deblend"]["latent_dim"])
        self.n_r, self.b_r, self.n_e = train.sizes(cfg)
        st = data.load_norm_stats(cfg)
        self.sr, self.se = np.asarray(st["rubin"]["scale"], np.float32), np.asarray(st["euclid"]["scale"], np.float32)
        self.missing = []
        self.parts = {}

        def done(stage):
            rd = train.run_dir(cfg, stage, seed if stage in ("A", "B", "c2", "c3") else None)
            if not (rd / "DONE").exists():
                rd = train.run_dir(cfg, stage)
            ok = (rd / "DONE").exists()
            if not ok:
                self.missing.append(stage)
            return rd if ok else None

        d = self.d
        # Rubin encoders and decoders take the Rubin PSF (deblend.psf_condition): train.load_* with cfg.
        if (rd := done("rubin_vae")):
            self.parts["rvae"] = (train.load_rubin_encoder(cfg, rd, d),
                                  train.load_decoder(rd / "decoder_best.weights.h5", self.n_r, self.b_r, d, cfg))
        if (rd := done("k1")):
            self.parts["k1"] = (train.load_k1_encoder(cfg, rd, d),
                                train.load_decoder(rd / "decoder_best.weights.h5", self.n_e, 1, d))
        if (rd := done("k2")):
            self.parts["k2"] = train.load_decoder(rd / "decoder_best.weights.h5", self.n_r, self.b_r, d, cfg)
        if (rd := done("c1")):
            s_c1 = json.loads((rd / "run.json").read_text())["settings"]
            self.parts["c1"] = (train.load_joint_encoder(cfg, rd, d, s_c1),
                                train.load_decoder(rd / "dec_r_best.weights.h5", self.n_r, self.b_r, d, cfg),
                                train.load_decoder(rd / "dec_e_best.weights.h5", self.n_e, 1, d))
        ev = cfg["deblend"].get("eval", {}) or {}
        self.samplers = {}
        if not students:
            return
        for run in RUNS:
            stage = STAGE_OF[run]
            rd = done(stage)
            dec = self.parts.get("k2") if run in ("A", "B") else (self.parts["c1"][1] if "c1" in self.parts else None)
            if rd is None or dec is None:
                continue
            s = json.loads((rd / "run.json").read_text())["settings"]
            stu = train.load_student(cfg, stage, rd, d, s)
            if run == "A":
                self.samplers[run] = infer.RunASampler(stu, dec, self.sr, int(cfg["deblend"]["seed"]))
            else:
                self.samplers[run] = infer.CFMSampler(stu, dec, self.sr, int(ev.get("nfe", 40)),
                                                      ev.get("method", "heun"), int(cfg["deblend"]["seed"]))

    def decode(self, dec, z, scale, cond=None):
        from keras import ops
        c = None if cond is None else ops.convert_to_tensor(cond)
        return np.asarray(ops.convert_to_numpy(dec(ops.convert_to_tensor(z), cond=c))) * scale

    def post(self, enc, *inputs, cond=None):
        # Posterior (mean, log variance) of a diagonal encoder.
        from keras import ops
        kw = {} if cond is None else dict(cond=ops.convert_to_tensor(cond))
        mu, lv = enc(*[ops.convert_to_tensor(a) for a in inputs], **kw)
        return np.asarray(ops.convert_to_numpy(mu)), np.asarray(ops.convert_to_numpy(lv))

    def mu(self, enc, *inputs, cond=None):
        return self.post(enc, *inputs, cond=cond)[0]


def _iterate(ds, max_n):
    n = 0
    for i in range(len(ds)):
        x, _ = ds[i]
        rows = ds.rows(i)
        yield x, rows
        n += len(rows)
        if n >= max_n:
            return


def out_dir(cfg, what, split, seed):
    from .campaign import eval_dir
    d = eval_dir(cfg, what, split, seed)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- D1
def vis_measure(images, valid, cfg, prefix, aperture_px):
    # VIS shapes (measure_vis) and the flux in a circular aperture of the valid pixels.
    m = measure_vis(images, valid, cfg, prefix)
    n = images.shape[1]
    rr, cc = np.indices((n, n))
    ap = ((rr - (n - 1) / 2) ** 2 + (cc - (n - 1) / 2) ** 2 <= aperture_px ** 2)[None, :, :, None]
    m[f"{prefix}_flux"] = (images * valid * ap).sum((1, 2, 3))
    return m


def eval_d1(cfg, split, seed, students=True, name="d1", only=None):
    # students False: the VAEs only (Rubin VAE, K1 with K2, C1), the checkpoint before their dependents are trained
    # (eval_check). only: the paths to evaluate among rvae, k1k2, c1 (None: all those trained).
    ev = cfg["deblend"].get("eval", {}) or {}
    K = int(ev.get("k_samples", 10))
    ap = float(ev.get("flux_aperture_px", 6))
    ap_vis = 2.0 * ap                                      # the same sky radius on the 0.1" VIS pixels
    camp = Campaign(cfg, seed, students=students)
    want = (lambda p: only is None or p in only)
    ds = data.D1Batches(cfg, split, int(ev.get("batch_size", 64)), False)
    meta = ds.st.meta
    rng = np.random.default_rng(int(cfg["deblend"]["seed"]))
    tables, prob = [], {r: {"e1": [], "e2": [], "e1_ref": [], "e2_ref": [], "flux_i": [], "flux_i_ref": []}
                        for r in camp.samplers}
    for x, rows in _iterate(ds, int(ev.get("max_examples", 5000))):
        mrows = meta.iloc[rows].reset_index(drop=True)
        valid = x["r_w"] > 0
        parts = [mrows[["object_id", "tract", "patch", "split"]].copy() if "tract" in mrows else
                 mrows[["object_id"]].copy() if "object_id" in mrows else pd.DataFrame(index=mrows.index)]
        for c in ("snr_i", "i_cModel_mag", "psf_sigma_i", "resolution_i", "euclid_vis_mag", "dec"):
            if c in mrows:
                parts[0][c] = mrows[c].to_numpy()
        obs = measure_rubin(x["r_flux"], valid, mrows, cfg, "obs", ap)
        parts.append(obs)
        cond = x.get("r_psf")
        paths = {}
        if "rvae" in camp.parts and want("rvae"):
            enc, dec = camp.parts["rvae"]
            paths["rvae"] = camp.decode(dec, camp.mu(enc, x["r_in"], cond=cond), camp.sr, cond)
        if "k1" in camp.parts and "k2" in camp.parts and want("k1k2"):
            paths["k1k2"] = camp.decode(camp.parts["k2"], camp.mu(camp.parts["k1"][0], x["e_in"]), camp.sr, cond)
        if "c1" in camp.parts and want("c1"):
            enc, dec_r, _ = camp.parts["c1"]
            paths["c1"] = camp.decode(dec_r, camp.mu(enc, x["r_in"], x["e_in"], cond=cond), camp.sr, cond)
        for pname, img in paths.items():
            parts.append(measure_rubin(img, valid, mrows, cfg, pname, ap))
            r2 = x["r_w"] * (x["r_flux"] - img) ** 2
            parts.append(pd.DataFrame({f"{pname}_chi2": (r2.sum((1, 2, 3)) / np.maximum(valid.sum((1, 2, 3)), 1))}))
        # VIS reconstructions of the Euclid decoders (K1; C1's Euclid side) against the observed VIS.
        vpaths = {}
        if "k1" in camp.parts and want("k1k2"):
            enc, dec = camp.parts["k1"]
            vpaths["k1_vis"] = camp.decode(dec, camp.mu(enc, x["e_in"]), camp.se)
        if "c1" in camp.parts and want("c1"):
            enc, _, dec_e = camp.parts["c1"]
            vpaths["c1_vis"] = camp.decode(dec_e, camp.mu(enc, x["r_in"], x["e_in"], cond=cond), camp.se)
        if vpaths:
            ve = x["e_w"] > 0
            parts.append(vis_measure(x["e_flux"], ve, cfg, "vobs", ap_vis))
            for pname, img in vpaths.items():
                parts.append(vis_measure(img, ve, cfg, pname, ap_vis))
                r2 = x["e_w"] * (x["e_flux"] - img) ** 2
                parts.append(pd.DataFrame({f"{pname}_chi2": r2.sum((1, 2, 3)) / np.maximum(ve.sum((1, 2, 3)), 1)}))
        # The final models applied to isolated Rubin stamps: K samples each, measured with a noise draw.
        for run, smp in camp.samplers.items():
            _, imgs = smp(x["r_in"], K, cond=cond)
            mean_img = imgs.mean(1)
            parts.append(measure_rubin(mean_img, valid, mrows, cfg, f"{run}_meanimg", ap))
            e1s, e2s, fis = (np.full((len(rows), K), np.nan) for _ in range(3))
            for k in range(K):
                m = measure_rubin(add_noise(imgs[:, k], x["r_w"], rng), valid, mrows, cfg, "s", ap)
                e1s[:, k], e2s[:, k], fis[:, k] = m["s_e1"], m["s_e2"], m["s_flux_i"]
            prob[run]["e1"].append(e1s)
            prob[run]["e2"].append(e2s)
            prob[run]["flux_i"].append(fis)
            prob[run]["e1_ref"].append(obs["obs_e1"].to_numpy())
            prob[run]["e2_ref"].append(obs["obs_e2"].to_numpy())
            prob[run]["flux_i_ref"].append(obs["obs_flux_i"].to_numpy())
            parts.append(pd.DataFrame({f"{run}_e1_meanofsamples": np.nanmean(e1s, 1),
                                       f"{run}_e2_meanofsamples": np.nanmean(e2s, 1),
                                       f"{run}_e_std": np.nanstd(np.hypot(e1s, e2s), 1)}))
        tables.append(pd.concat(parts, axis=1))
    df = pd.concat(tables, ignore_index=True)
    out = out_dir(cfg, name, split, seed)
    df.to_parquet(out / "objects.parquet")
    summ = dict(split=split, n=int(len(df)), missing_stages=camp.missing, paths={}, probabilistic={}, vis_paths={})
    for pname in ("k1_vis", "c1_vis"):
        if f"{pname}_e1" not in df:
            continue
        ok = df[f"{pname}_ok"].astype(bool) & df["vobs_ok"].astype(bool)
        s = dict(hsm_failure=float(1 - df[f"{pname}_ok"].astype(bool).mean()),
                 size=summary_stats((df[f"{pname}_sigma"] / df["vobs_sigma"] - 1)[ok]),
                 flux=summary_stats(df[f"{pname}_flux"] / df["vobs_flux"] - 1),
                 chi2_per_pixel_median=float(np.nanmedian(df[f"{pname}_chi2"])))
        for q in ("e1", "e2"):
            s[q] = summary_stats((df[f"{pname}_{q}"] - df[f"vobs_{q}"])[ok])
            s[f"{q}_mc"] = linear_mc(df[f"{pname}_{q}"][ok].to_numpy(), df[f"vobs_{q}"][ok].to_numpy())
        summ["vis_paths"][pname] = s
    for pname in ("rvae", "k1k2", "c1") + tuple(f"{r}_meanimg" for r in camp.samplers):
        if f"{pname}_e1" not in df:
            continue
        ok = df[f"{pname}_ok"].astype(bool) & df["obs_ok"].astype(bool)
        s = dict(hsm_failure=float(1 - df[f"{pname}_ok"].astype(bool).mean()))
        for q in ("e1", "e2"):
            s[q] = summary_stats((df[f"{pname}_{q}"] - df[f"obs_{q}"])[ok])
            s[f"{q}_mc"] = linear_mc(df[f"{pname}_{q}"][ok].to_numpy(), df[f"obs_{q}"][ok].to_numpy())
            okr = df[f"{pname}_ok_rg"].astype(bool) & df["obs_ok_rg"].astype(bool)
            s[f"{q}_regauss"] = summary_stats((df[f"{pname}_{q}_rg"] - df[f"obs_{q}_rg"])[okr])
            if "snr_i" in df:
                # PSF-corrected shape of the reconstruction against the observed one, by S/N_i: m < 0 at high S/N
                # = shapes pulled towards the prior (the beta = 1 checkpoint of 2026-11-27).
                sub = df[okr]
                s[f"{q}_regauss_by_snr"] = by_snr(sub, f"{pname}_{q}_rg", f"obs_{q}_rg",
                                                  cfg["deblend"].get("eval", {}).get("snr_bins"))
        s["size"] = summary_stats((df[f"{pname}_sigma"] / df["obs_sigma"] - 1)[ok])
        for b in cfg["rubin"]["bands"]:
            s[f"flux_{b}"] = summary_stats(df[f"{pname}_flux_{b}"] / df[f"obs_flux_{b}"] - 1)
        for b1, b2 in (("g", "r"), ("r", "i"), ("i", "z")):
            col = lambda p: -2.5 * np.log10(np.clip(df[f"{p}_flux_{b1}"], 1e-3, None) /  # noqa: E731
                                            np.clip(df[f"{p}_flux_{b2}"], 1e-3, None))
            s[f"colour_{b1}{b2}"] = summary_stats(col(pname) - col("obs"))
        if f"{pname}_chi2" in df:
            s["chi2_per_pixel_median"] = float(np.nanmedian(df[f"{pname}_chi2"]))
        summ["paths"][pname] = s
    for run, p in prob.items():
        cat = {k: np.concatenate(v) for k, v in p.items()}
        r = {}
        for q in ("e1", "e2", "flux_i"):
            r[q] = dict(coverage(cat[q], cat[f"{q}_ref"]), crps=crps(cat[q], cat[f"{q}_ref"]),
                        pit_hist=np.histogram(pit(cat[q], cat[f"{q}_ref"]), bins=10, range=(0, 1))[0].tolist())
        r["energy_score_e"] = energy_score(np.stack([cat["e1"], cat["e2"]], -1),
                                           np.stack([cat["e1_ref"], cat["e2_ref"]], -1))
        r["note"] = ("observational predictive: samples + noise draw at the observed ivar, measured like the "
                     "observation; not a calibration of the noise-free Rubin signal")
        summ["probabilistic"][run] = r
    write_json(out / "summary.json", summ)
    figures_d1(df, out, list(summ["paths"]))
    return summ


# ----------------------------------------------------------------------------- masking robustness
def eval_masking(cfg, split, seed):
    from keras import ops
    ev = cfg["deblend"].get("eval", {}) or {}
    camp = Campaign(cfg, seed)
    if "k1" not in camp.parts:
        raise RuntimeError("masking evaluation needs a finished K1")
    enc, dec = camp.parts["k1"]
    pool = data.d2_mask_pool(cfg, int(ev.get("mask_pool", 2000)), int(cfg["deblend"]["seed"]))
    ds = data.D1Batches(cfg, split, int(ev.get("batch_size", 64)), False, fields=("euclid",))
    rng = np.random.default_rng(1)
    rows_out = []
    for x, rows in _iterate(ds, int(ev.get("max_examples", 5000))):
        st = ds.st
        f, iv, m = (st.take(k, rows) for k in ("euclid_flux", "euclid_ivar", "euclid_mask"))
        fill0 = st.take("euclid_fill", rows) if ds.has_fill else np.zeros(f.shape[:3], bool)
        extra = pool[rng.integers(0, len(pool), len(rows))]
        prep = ds.prep
        x0, f0, w0 = prep.euclid(f, iv, m, fill0, np.random.default_rng(2))
        x1, _, w1 = prep.euclid(f, iv, m, fill0 | extra, np.random.default_rng(2))
        mu0, lv0 = (np.asarray(ops.convert_to_numpy(a)) for a in enc(x0))
        mu1, _ = (np.asarray(ops.convert_to_numpy(a)) for a in enc(x1))
        shift = np.sqrt(np.mean(((mu1 - mu0) / np.exp(0.5 * lv0)) ** 2, axis=1))
        im0, im1 = camp.decode(dec, mu0, camp.se), camp.decode(dec, mu1, camp.se)
        full = np.ones_like(im0, bool)
        a0, a1 = measure_vis(im0, full, cfg, "a"), measure_vis(im1, full, cfg, "b")
        n = im0.shape[1]
        rr, cc = np.indices((n, n))
        ap = (rr - n // 2) ** 2 + (cc - n // 2) ** 2 <= 12 ** 2
        for k in range(len(rows)):
            rows_out.append(dict(masked_fraction=float(extra[k].mean()),
                                 masked_fraction_centre=float(extra[k][ap].mean()),
                                 latent_shift_sigma=float(shift[k]),
                                 dflux=float((im1[k][..., 0] * ap).sum() / max((im0[k][..., 0] * ap).sum(), 1e-9) - 1),
                                 dsize=float(a1["b_sigma"][k] / a0["a_sigma"][k] - 1),
                                 de=float(np.hypot(a1["b_e1"][k] - a0["a_e1"][k], a1["b_e2"][k] - a0["a_e2"][k]))))
    df = pd.DataFrame(rows_out)
    out = out_dir(cfg, "masking", split, seed)
    df.to_parquet(out / "objects.parquet")
    edges = [0, 0.02, 0.05, 0.1, 0.2, 1.0]
    by = df.groupby(pd.cut(df["masked_fraction"], edges), observed=False)
    summ = dict(n=int(len(df)), overall={c: summary_stats(df[c]) for c in ("latent_shift_sigma", "dflux", "dsize",
                                                                           "de")},
                by_mask_fraction=[dict(bin=str(k), n=int(len(g)),
                                       latent_shift_sigma_median=float(g["latent_shift_sigma"].median())
                                       if len(g) else np.nan,
                                       de_median=float(g["de"].median()) if len(g) else np.nan,
                                       dflux_median=float(g["dflux"].median()) if len(g) else np.nan)
                                  for k, g in by],
                note="K1 encoder is mask-aware (validity channel); masks are real D2 neighbour masks")
    write_json(out / "summary.json", summ)
    return summ


# ----------------------------------------------------------------------------- D2
E_MAX = 1.0          # |e| >= 1 is a failed REGAUSS correction (physically impossible): left out of every comparison


def valid_e(e):
    e = np.asarray(e, float)
    return np.where(np.abs(e) < E_MAX, e, np.nan)


SNR_BINS = [0, 10, 20, 50, 80, 100, 1e9]   # S/N_i bins of the shape summaries (shapes reliable from ~50-80)


def shape_metrics(y, x):
    # Shape y against reference x (one component), failed measurements (|e| >= 1, NaN) left out: n, R^2 (about the
    # reference's mean), multiplicative / additive bias (y = (1 + m) x + c), median offset.
    y, x = valid_e(y), valid_e(x)
    ok = np.isfinite(y) & np.isfinite(x)
    out = dict(n=int(ok.sum()))
    if ok.sum() < 10:
        return out
    ss = float(((x[ok] - x[ok].mean()) ** 2).sum())
    out["R2"] = float(1 - ((y[ok] - x[ok]) ** 2).sum() / ss) if ss > 0 else np.nan
    out["m"], out["c"] = linear_mc(y[ok], x[ok])
    out["median_offset"] = float(np.median(y[ok] - x[ok]))
    return out


def by_snr(df, ycol, xcol, edges=None):
    # shape_metrics of ycol against xcol in bins of snr_i, and on S/N_i >= each edge.
    edges = list(edges or SNR_BINS)
    snr = df["snr_i"].to_numpy(float)
    res = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (snr >= lo) & (snr < hi)
        res[f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}"] = shape_metrics(df[ycol].to_numpy(float)[sel],
                                                                         df[xcol].to_numpy(float)[sel])
    return res


def d1_relation(cfg, ev):
    # Rubin i REGAUSS shape vs Euclid VIS REGAUSS shape on D1 (val split: never the evaluation split), per
    # component: e_R = a e_VIS + b (Theil-Sen), residual scatter. Only where both measurements are reliable: S/N_i >=
    # eval.relation_min_snr (default 50) and |e| < 1. Measured on 3000 D1 val pairs (scripts/inspect/
    # shape_relation.py, 2026-11-27): slope 1.00 / 0.96-1.00 and r = 0.82-0.88 at S/N_i >= 50, while below S/N_i ~50
    # REGAUSS on Rubin fails for 25-31 % of the pairs; the least-squares fit on all pairs that the first version used
    # gave slopes of -0.6 / -3.6. Raises when the slope departs from 1 by more than eval.relation_max_slope_dev or the
    # offset exceeds eval.relation_max_offset: the reference would not be the same shape.
    from scipy.stats import theilslopes
    bs = int(ev.get("batch_size", 64))
    min_snr = float(ev.get("relation_min_snr", 50))
    ds0 = data.D1Batches(cfg, "val", bs, False)
    meta = ds0.st.meta
    pool = ds0.rows_all[meta["snr_i"].to_numpy(float)[ds0.rows_all] >= min_snr]
    nmax = int(ev.get("relation_max_examples", 3000))
    if len(pool) > nmax:
        pool = np.sort(np.random.default_rng(0).choice(pool, nmax, replace=False))
    ds = data.D1Batches(cfg, "val", bs, False, indices=pool)
    rows = []
    for x, idx in _iterate(ds, nmax):
        mrows = meta.iloc[idx].reset_index(drop=True)
        r = measure_rubin(x["r_flux"], x["r_w"] > 0, mrows, cfg, "r", 6.0)
        v = measure_vis(x["e_flux"], x["e_w"] > 0, cfg, "v")
        rows.append(pd.concat([r, v], axis=1))
    df = pd.concat(rows, ignore_index=True)
    rel = dict(min_snr_i=min_snr, method="Theil-Sen, |e| < 1, both REGAUSS measurements valid")
    ok = df["r_ok_rg"].astype(bool).to_numpy() & df["v_ok_rg"].astype(bool).to_numpy()
    dev, off = float(ev.get("relation_max_slope_dev", 0.2)), float(ev.get("relation_max_offset", 0.05))
    for q in ("e1", "e2"):
        xv, yr = valid_e(df[f"v_{q}_rg"]), valid_e(df[f"r_{q}_rg"])
        good = ok & np.isfinite(xv) & np.isfinite(yr)
        min_pairs = int(ev.get("relation_min_pairs", 30))
        if good.sum() < max(min_pairs, 3):
            if min_pairs == 0:                          # check disabled (synthetic tests): no relation, no guard
                rel[q] = dict(a=np.nan, b=np.nan, scatter=np.nan, n=int(good.sum()))
                continue
            raise RuntimeError(f"D1 relation: only {int(good.sum())} reliable pairs at S/N_i >= {min_snr:g}")
        a, b, _, _ = theilslopes(yr[good], xv[good])
        res = yr[good] - (a * xv[good] + b)
        rel[q] = dict(a=float(a), b=float(b), scatter=robust_scatter(res), n=int(good.sum()),
                      r=float(np.corrcoef(xv[good], yr[good])[0, 1]),
                      failed_fraction=float(1 - good.sum() / max(int(ok.sum()), 1)))
        if abs(a - 1) > dev or abs(b) > off:
            raise RuntimeError(f"D1 relation {q}: slope {a:.3f}, offset {b:.3f} (allowed 1 +- {dev}, |b| <= {off}): "
                               "the Rubin and VIS shapes do not agree; check shape_relation.py before evaluating")
    rel["size_ratio_median"] = float(np.nanmedian(df["r_sigma"] / (df["v_sigma"] * float(cfg["euclid"]["pixel_scale"])
                                                                   / common.RUBIN_SCALE)))
    return rel


def eval_d2(cfg, split, seed):
    ev = cfg["deblend"].get("eval", {}) or {}
    K = int(ev.get("k_samples", 10))
    ap = float(ev.get("flux_aperture_px", 6))
    camp = Campaign(cfg, seed)
    rel = d1_relation(cfg, ev)
    ds = data.D2Batches(cfg, split, int(ev.get("batch_size", 64)), False)
    meta = ds.st.meta
    tables = []
    for x, rows in _iterate(ds, int(ev.get("max_examples", 5000))):
        mrows = meta.iloc[rows].reset_index(drop=True)
        keep = [c for c in ("euclid_object_id", "group_id", "tract", "patch", "sb_rubin", "snr_i", "rubin_i_cModel_mag",
                            "euclid_vis_mag", "partner_vis_mag", "partner_sep_arcsec", "resolution_i",
                            "euclid_masked_fraction_in_target", "euclid_masked_fraction", "n_rubin_in_group",
                            "euclid_dec", "rc_offset_x_px", "rc_offset_y_px", "partner_dx_px", "partner_dy_px",
                            "partner_kept", "scene_complete") if c in mrows]
        base = mrows[keep].copy()
        vis = measure_vis(x["e_flux"], x["e_w"] > 0, cfg, "vis")
        parts = [base, vis]
        valid = x["x_w"] > 0
        n = x["x_flux"].shape[1]
        rr, cc = np.indices((n, n))
        apm = (rr - (n - 1) / 2) ** 2 + (cc - (n - 1) / 2) ** 2 <= ap ** 2
        for run, smp in camp.samplers.items():
            _, imgs = smp(x["x_in"], K, cond=x.get("x_psf"))
            mean_img = imgs.mean(1)
            mm = measure_rubin(mean_img, valid, mrows, cfg, f"{run}_meanimg", ap)
            e1s, e2s, sz = (np.full((len(rows), K), np.nan) for _ in range(3))
            e1r, e2r = (np.full((len(rows), K), np.nan) for _ in range(2))
            for k in range(K):
                m = measure_rubin(imgs[:, k], valid, mrows, cfg, "s", ap)
                e1s[:, k], e2s[:, k], sz[:, k] = m["s_e1"], m["s_e2"], m["s_sigma"]
                g1, g2 = m["s_e1_rg"].to_numpy(float), m["s_e2_rg"].to_numpy(float)
                bad = ~(np.abs(g1) < E_MAX) | ~(np.abs(g2) < E_MAX)       # failed corrections left out
                e1r[:, k] = np.where(bad, np.nan, g1)
                e2r[:, k] = np.where(bad, np.nan, g2)
            # Over-subtraction: central pixels where the blend lies below the reconstructed central by > 3 sigma.
            res = (x["x_flux"] - mean_img) * np.sqrt(x["x_w"])
            over = ((res < -3) & valid & apm[None, :, :, None]).sum((1, 2, 3)) / np.maximum(
                (valid & apm[None, :, :, None]).sum((1, 2, 3)), 1)
            parts += [mm, pd.DataFrame({
                f"{run}_e1_mos": np.nanmean(e1s, 1), f"{run}_e2_mos": np.nanmean(e2s, 1),
                f"{run}_e1_rg_mos": np.nanmean(e1r, 1), f"{run}_e2_rg_mos": np.nanmean(e2r, 1),
                f"{run}_e1_rg_sd": np.nanstd(e1r, 1), f"{run}_e2_rg_sd": np.nanstd(e2r, 1),
                f"{run}_sigma_mos": np.nanmean(sz, 1), f"{run}_hsm_fail": np.mean(~np.isfinite(e1s), 1),
                f"{run}_oversub": over})]
            for lv in (0.68, 0.9):
                for q, arr in (("e1", e1r), ("e2", e2r)):
                    lo = np.nanpercentile(arr, 50 * (1 - lv), axis=1)
                    hi = np.nanpercentile(arr, 50 * (1 + lv), axis=1)
                    parts.append(pd.DataFrame({f"{run}_{q}_rg_lo{int(lv*100)}": lo,
                                               f"{run}_{q}_rg_hi{int(lv*100)}": hi}))
        tables.append(pd.concat(parts, axis=1))
    df = pd.concat(tables, ignore_index=True)
    # Euclid reference: the target's PSF-corrected VIS shape (eval.reference: vis, the default, as the D1 relation is
    # the identity where both measurements are reliable), or read through the fitted relation (fit). Failed VIS
    # corrections (|e| >= 1) give no reference.
    mode = ev.get("reference", "vis")
    bad_vis = ~(np.abs(df["vis_e1_rg"]) < E_MAX) | ~(np.abs(df["vis_e2_rg"]) < E_MAX)
    for q in ("e1", "e2"):
        a, b = (rel[q]["a"], rel[q]["b"]) if mode == "fit" else (1.0, 0.0)
        df[f"ref_{q}"] = np.where(bad_vis, np.nan, a * df[f"vis_{q}_rg"] + b)
    rel["reference"] = mode
    if "euclid_dec" in df:
        df["field"] = np.where(df["euclid_dec"] > -38, "EDF-F", "EDF-S")
    if "partner_vis_mag" in df:
        df["contrast_mag"] = df["partner_vis_mag"] - df["euclid_vis_mag"]
    out = out_dir(cfg, "d2", split, seed)
    df.to_parquet(out / "objects.parquet")
    write_json(out / "d1_relation.json", rel)
    summ = dict(split=split, n=int(len(df)), missing_stages=camp.missing, d1_relation=rel, runs={}, paired={},
                binned={}, note=("no isolated Rubin truth on D2: comparisons with Euclid read through the D1 "
                                 "relation; coverage uses the D1 relation scatter as reference noise"))
    groups = (df["tract"].astype(int) * 1000 + df["patch"].astype(int)).to_numpy() if "tract" in df else \
        np.arange(len(df))
    B = int(ev.get("bootstrap", 200))
    for run in camp.samplers:
        s = {}
        for q in ("e1", "e2"):
            d = df[f"{run}_{q}_rg_mos"] - df[f"ref_{q}"]
            s[q] = summary_stats(d)
            s[f"{q}_mc_vs_ref"] = linear_mc(df[f"{run}_{q}_rg_mos"].to_numpy(), df[f"ref_{q}"].to_numpy())
            s[f"{q}_meanimg_minus_mos"] = summary_stats(df[f"{run}_meanimg_{q}_rg"] - df[f"{run}_{q}_rg_mos"])
            for lv in (68, 90):
                inside = (df[f"ref_{q}"] >= df[f"{run}_{q}_rg_lo{lv}"]) & (df[f"ref_{q}"] <= df[f"{run}_{q}_rg_hi{lv}"])
                s[f"{q}_coverage{lv}_vs_ref"] = float(inside[np.isfinite(df[f"ref_{q}"])].mean())
        e_run = np.hypot(df[f"{run}_e1_rg_mos"], df[f"{run}_e2_rg_mos"])
        e_ref = np.hypot(df["ref_e1"], df["ref_e2"])
        from scipy.stats import ks_2samp
        ok = np.isfinite(e_run) & np.isfinite(e_ref)
        s["abs_e_ks_vs_ref"] = float(ks_2samp(e_run[ok], e_ref[ok]).statistic) if ok.sum() > 10 else np.nan
        ang = 0.5 * np.arctan2(df[f"{run}_e2_rg_mos"], df[f"{run}_e1_rg_mos"])
        ang_ref = 0.5 * np.arctan2(df["ref_e2"], df["ref_e1"])
        s["position_angle_diff_deg"] = summary_stats(np.degrees(np.angle(np.exp(2j * (ang - ang_ref))) / 2))
        if "partner_dx_px" in df:
            has = df["partner_kept"].astype(bool) if "partner_kept" in df else np.ones(len(df), bool)
            phi = np.arctan2(df["partner_dy_px"], df["partner_dx_px"])
            s["alignment_with_partner"] = dict(
                run=float(np.nanmean((np.cos(2 * (ang - phi)) * e_run)[has])),
                euclid_ref=float(np.nanmean((np.cos(2 * (ang_ref - phi)) * e_ref)[has])), n=int(has.sum()))
        s["hsm_failure"] = float(df[f"{run}_hsm_fail"].mean())
        s["oversubtraction_fraction_median"] = float(df[f"{run}_oversub"].median())
        s["size_rubin_over_vis"] = summary_stats(df[f"{run}_sigma_mos"] / (df["vis_sigma"] * float(
            cfg["euclid"]["pixel_scale"]) / common.RUBIN_SCALE) / rel["size_ratio_median"] - 1)
        summ["runs"][run] = s
        edges = (ev.get("bins") or {})
        bins = dict(sb_rubin=edges.get("sb_rubin", [0, 0.1, 0.3, 0.6, 10]),
                    snr_i=edges.get("snr_i", ev.get("snr_bins") or SNR_BINS),
                    rubin_i_cModel_mag=edges.get("mag", [15, 22, 23, 24, 25, 30]),
                    partner_sep_arcsec=edges.get("sep", [0, 0.5, 1.0, 1.5, 3, 10]),
                    contrast_mag=edges.get("contrast", [-10, -1, 0, 1, 10]),
                    resolution_i=edges.get("resolution", [0, 0.2, 0.4, 0.6, 1.01]),
                    euclid_masked_fraction_in_target=edges.get("mask", [-0.01, 0.0, 0.02, 0.05, 0.1]))
        for by, ed in bins.items():
            if by not in df:
                continue
            cut = pd.cut(df[by].astype(float), ed)
            summ["binned"].setdefault(by, {})[run] = [
                dict(bin=str(k), n=int(len(g)),
                     e1_bias=float(np.nanmedian(g[f"{run}_e1_rg_mos"] - g["ref_e1"])) if len(g) else np.nan,
                     e1_nmad=robust_scatter(g[f"{run}_e1_rg_mos"] - g["ref_e1"]),
                     e2_bias=float(np.nanmedian(g[f"{run}_e2_rg_mos"] - g["ref_e2"])) if len(g) else np.nan,
                     e2_nmad=robust_scatter(g[f"{run}_e2_rg_mos"] - g["ref_e2"]),
                     hsm_failure=float(g[f"{run}_hsm_fail"].mean()) if len(g) else np.nan)
                for k, g in df.groupby(cut, observed=False)]
        if "field" in df:
            summ["binned"].setdefault("field", {})[run] = {
                f: dict(n=int(len(g)), e1=summary_stats(g[f"{run}_e1_rg_mos"] - g["ref_e1"]),
                        e2=summary_stats(g[f"{run}_e2_rg_mos"] - g["ref_e2"])) for f, g in df.groupby("field")}
    # Paired differences of |e - ref| between runs, patch block bootstrap.
    runs = list(camp.samplers)
    ug = np.unique(groups)
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            da = np.hypot(df[f"{a}_e1_rg_mos"] - df["ref_e1"], df[f"{a}_e2_rg_mos"] - df["ref_e2"])
            db = np.hypot(df[f"{b}_e1_rg_mos"] - df["ref_e1"], df[f"{b}_e2_rg_mos"] - df["ref_e2"])
            diff = (da - db).to_numpy()

            def stat(gsel, diff=diff):
                sel = np.isin(groups, ug[gsel])
                return float(np.nanmedian(diff[sel])) if sel.any() else np.nan

            summ["paired"][f"{a}_minus_{b}"] = dict(median_abs_error_difference=float(np.nanmedian(diff)),
                                                    ci68_patch_bootstrap=bootstrap(stat, len(ug), B, 0),
                                                    n_objects=int(np.isfinite(diff).sum()), n_patches=int(len(ug)))
    # R^2, m, c of every run against the reference in S/N_i bins (where each run starts to fail).
    if "snr_i" in df:
        summ["shape_by_snr"] = {run: {q: by_snr(df, f"{run}_{q}_rg_mos", f"ref_{q}", ev.get("snr_bins"))
                                      for q in ("e1", "e2")} for run in runs}
    # Core shape metrics again on the weak-lensing-like sample (S/N_i >= 10) and on the targets whose shapes are
    # reliable (S/N_i >= eval.reliable_snr, default 80): R^2, m, c, bias, coverages against the reference.
    hi = float(ev.get("reliable_snr", 80))
    for label, lo in (("snr_i_ge_10", 10.0), ("snr_i_ge_%d" % int(hi), hi)):
        if "snr_i" not in df:
            break
        g = df[df["snr_i"].astype(float) >= lo]
        summ[label] = {"n": int(len(g))}
        for run in runs:
            s = {}
            for q in ("e1", "e2"):
                y, x = g[f"{run}_{q}_rg_mos"].to_numpy(float), g[f"ref_{q}"].to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(x)
                ss = float(((x[ok] - x[ok].mean()) ** 2).sum()) if ok.sum() > 2 else np.nan
                r2 = float(1 - ((y[ok] - x[ok]) ** 2).sum() / ss) if ss and np.isfinite(ss) else np.nan
                s[q] = dict(n=int(ok.sum()), R2=r2, mc_vs_ref=linear_mc(y, x), bias=summary_stats(y - x))
                for lv in (68, 90):
                    inside = (g[f"ref_{q}"] >= g[f"{run}_{q}_rg_lo{lv}"]) & (g[f"ref_{q}"] <= g[f"{run}_{q}_rg_hi{lv}"])
                    fin = np.isfinite(g[f"ref_{q}"])
                    s[q][f"coverage{lv}"] = float(inside[fin].mean()) if fin.any() else np.nan
            summ[label][run] = s
    write_json(out / "summary.json", summ)
    figures_d2(df, out, runs)
    return summ


# ----------------------------------------------------------------------------- cost
def eval_cost(cfg, split, seed):
    import time

    from keras import ops

    from . import cfm
    ev = cfg["deblend"].get("eval", {}) or {}
    camp = Campaign(cfg, seed)
    ds = data.D2Batches(cfg, split, int(ev.get("cost_batch", 64)), False, fields=("blend",))
    x, _ = ds[0]
    xin = x["x_in"]
    xc = x.get("x_psf")
    cc = None if xc is None else ops.convert_to_tensor(xc)
    out = dict(n_galaxies=int(len(xin)), runs={})
    for run, smp in camp.samplers.items():
        r = dict(student_params=int(smp.student.count_params()), decoder_params=int(smp.decoder.count_params()))
        if run == "A":
            r["timing"] = {str(k): infer.timing(smp, xin, k, cond=xc) for k in (10, 50, 100)}
        else:
            r["field_params"] = int(smp.student.field.count_params())
            h = smp.student.context(ops.convert_to_tensor(xin), cond=cc)
            t0 = time.perf_counter()
            h = smp.student.context(ops.convert_to_tensor(xin), cond=cc)
            r["context_s_per_galaxy"] = (time.perf_counter() - t0) / len(xin)
            Kc = int(ev.get("cost_k", 10))
            z0 = np.random.default_rng(0).standard_normal((len(xin) * Kc, smp.student.field.d)).astype(np.float32)
            hh = np.repeat(np.asarray(ops.convert_to_numpy(h)), Kc, axis=0)
            ref = np.asarray(ops.convert_to_numpy(cfm.integrate(smp.student.field, z0, hh, 200, "heun")))
            conv = {}
            for method in ("euler", "heun"):
                for nfe in (5, 10, 20, 50, 100):
                    if method == "heun" and nfe % 2:
                        continue
                    t0 = time.perf_counter()
                    z = np.asarray(ops.convert_to_numpy(cfm.integrate(smp.student.field, z0, hh,
                                                                      cfm.steps_for_nfe(nfe, method), method)))
                    dt = time.perf_counter() - t0
                    conv[f"{method}_{nfe}"] = dict(nfe=nfe, steps=cfm.steps_for_nfe(nfe, method),
                                                   rms_to_reference=float(np.sqrt(np.mean((z - ref) ** 2))),
                                                   field_s_per_sample=dt / len(z))
            r["convergence"] = conv
            r["timing"] = {str(k): infer.timing(smp, xin, k, cond=xc) for k in (10, 50, 100)}
        out["runs"][run] = r
    try:
        import jax
        out["devices"] = [str(dv) for dv in jax.devices()]
        st = jax.devices()[0].memory_stats() or {}
        out["peak_device_memory_bytes"] = int(st.get("peak_bytes_in_use", -1))
    except Exception:  # noqa: BLE001
        pass
    write_json(out_dir(cfg, "cost", split, seed) / "summary.json", out)
    return out


# ----------------------------------------------------------------------------- latent spaces
def eval_latent(cfg, split, seed):
    ev = cfg["deblend"].get("eval", {}) or {}
    camp = Campaign(cfg, seed, students=False)
    out = latent_summary(cfg, camp, split, int(ev.get("latent_max_examples", 5000)))
    if "c1" in camp.parts:
        out["jacobian"] = jacobian_analysis(cfg, camp, out.pop("_first_batch"), int(ev.get("jacobian_n", 32)))
    out.pop("_first_batch", None)
    write_json(out_dir(cfg, "latent", split, seed) / "summary.json", out)
    return out


def latent_summary(cfg, camp, split, max_n, threshold=0.01):
    # Latent spaces of the VAEs on D1 (rvae: Rubin VAE, k1, c1): per dimension the variance of the posterior mean
    # over the objects, the mean posterior variance and the mean KL to the prior; active dimensions (variance of
    # the mean above threshold, Burda et al. 2016); PCA spectrum of the means.
    ds = data.D1Batches(cfg, split, int((cfg["deblend"].get("eval") or {}).get("batch_size", 64)), False)
    post = {k: ([], []) for k in ("rvae", "k1", "c1") if k in camp.parts}
    first = None
    for x, rows in _iterate(ds, max_n):
        cond = x.get("r_psf")
        for k, (mus, lvs) in post.items():
            enc = camp.parts[k][0]
            mu, lv = (camp.post(enc, x["e_in"]) if k == "k1" else
                      camp.post(enc, x["r_in"], cond=cond) if k == "rvae" else
                      camp.post(enc, x["r_in"], x["e_in"], cond=cond))
            mus.append(mu)
            lvs.append(lv)
        if first is None:
            first = x
    out = dict(spaces={}, _first_batch=first)
    for k, (mus, lvs) in post.items():
        if not mus:
            continue
        z, lv = np.concatenate(mus), np.concatenate(lvs)
        ev_ = np.sort(np.linalg.eigvalsh(np.cov(z.T)))[::-1]
        kl = 0.5 * (np.exp(lv) + z ** 2 - 1.0 - lv).mean(0)
        var_mean = z.var(0)
        out["spaces"][k] = dict(var_per_dim=var_mean.tolist(), posterior_var_per_dim=np.exp(lv).mean(0).tolist(),
                                kl_per_dim=kl.tolist(), kl_total=float(kl.sum()),
                                active_dims=int((var_mean > threshold).sum()), active_threshold=threshold,
                                pca_eigenvalues=ev_.tolist(),
                                participation_ratio=float(ev_.sum() ** 2 / (ev_ ** 2).sum()),
                                n_dims_90pct=int(np.searchsorted(np.cumsum(ev_) / ev_.sum(), 0.9) + 1))
    return out


def eval_check(cfg, split, seed, which):
    # Checkpoint of the VAEs before the stages that depend on them are trained (campaign gates, 2026-11-28), on the
    # val split: check_vaes = Rubin VAE, K1 (its VIS reconstructions) and C1 (Rubin and VIS); check_k2 = K1 + K2
    # (E -> R). Reconstructions (shapes by S/N_i, sizes, fluxes, colours, chi2) and the latent spaces (active
    # dimensions, KL per dimension). Approve each with dp2-ml <cfg> deblend-approve <stage>.
    only = dict(check_vaes=("rvae", "c1", "k1k2"), check_k2=("k1k2",))[which]
    summ = eval_d1(cfg, split, seed, students=False, name=which, only=only)
    camp = Campaign(cfg, seed, students=False)
    lat = latent_summary(cfg, camp, split, int((cfg["deblend"].get("eval") or {}).get("latent_max_examples", 5000)))
    lat.pop("_first_batch", None)
    summ["latent"] = lat["spaces"]
    summ["note"] = ("VAE checkpoint: look at m, c by S/N_i (>= 80) of every path, sizes, fluxes, chi2, the K1 / C1 "
                    "VIS reconstructions and the active dimensions before dp2-ml deblend-approve")
    write_json(out_dir(cfg, which, split, seed) / "summary.json", summ)
    return summ


def jacobian_analysis(cfg, camp, x, n):
    # Noise-weighted Jacobians of the C1 decoders at the posterior means of n D1 examples; per example, the
    # generalised eigenvectors of (J_R^T J_R, J_E^T J_E + eps I): directions that change the Rubin image most per
    # unit of Euclid change. Effect of the top direction (unit latent step) on Rubin fluxes, g-r / r-i colours, size
    # and shape, measured on the decoded images.
    import keras
    if keras.backend.backend() != "jax":
        return dict(status="skipped: JAX backend needed")
    import jax
    enc, dec_r, dec_e = camp.parts["c1"]
    st = data.load_norm_stats(cfg)
    w_r = 1.0 / np.asarray(st["rubin"]["sky_sigma"], np.float32)
    w_e = 1.0 / np.asarray(st["euclid"]["sky_sigma"], np.float32)
    cond = x.get("r_psf")
    mu = camp.mu(enc, x["r_in"][:n], x["e_in"][:n], cond=None if cond is None else cond[:n])
    fe = lambda z: (dec_e(z[None])[0] * camp.se * w_e).reshape(-1)  # noqa: E731
    ratios, effects = [], []
    bands = list(cfg["rubin"]["bands"])
    for i in range(len(mu)):
        ci = None if cond is None else jax.numpy.asarray(cond[i:i + 1])
        fr = lambda z, ci=ci: (dec_r(z[None], cond=ci)[0] * camp.sr * w_r).reshape(-1)  # noqa: E731
        z = jax.numpy.asarray(mu[i])
        Jr = np.asarray(jax.jacfwd(fr)(z))
        Je = np.asarray(jax.jacfwd(fe)(z))
        Mr, Me = Jr.T @ Jr, Je.T @ Je
        eps = 1e-6 * np.trace(Me) / len(Me)
        from scipy.linalg import eigh
        vals, vecs = eigh(Mr, Me + eps * np.eye(len(Me)))
        ratios.append(vals[::-1].tolist())
        dz = vecs[:, -1] / np.linalg.norm(vecs[:, -1])
        imgs = camp.decode(dec_r, np.stack([mu[i] - 0.5 * dz, mu[i] + 0.5 * dz]), camp.sr,
                           None if cond is None else np.repeat(cond[i:i + 1], 2, axis=0))
        m = measure_rubin(imgs, np.ones_like(imgs, bool), pd.DataFrame([{}, {}]), cfg, "j", 6.0)
        eff = {f"dflux_{b}": float(m[f"j_flux_{b}"][1] / max(m[f"j_flux_{b}"][0], 1e-9) - 1) for b in bands}
        eff.update(de=float(np.hypot(m["j_e1"][1] - m["j_e1"][0], m["j_e2"][1] - m["j_e2"][0])),
                   dsize=float(m["j_sigma"][1] / m["j_sigma"][0] - 1))
        effects.append(eff)
    eff = pd.DataFrame(effects)
    return dict(n=int(len(mu)), generalized_eigenvalues_median=np.median(np.asarray(ratios), 0).tolist(),
                top_direction_effect_median=eff.median().to_dict(),
                note="noise-weighted (1 / sky sigma per band); unit step in the latent along the top direction")


# ----------------------------------------------------------------------------- figures
def figures_d1(df, out, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, max(len(paths), 1), figsize=(3.2 * max(len(paths), 1), 3.2), squeeze=False)
    for ax, p in zip(axes[0], paths):
        ok = df[f"{p}_ok"].astype(bool) & df["obs_ok"].astype(bool)
        ax.scatter(df["obs_e1"][ok], df[f"{p}_e1"][ok], s=2, alpha=0.4)
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
        ax.set(xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), xlabel="observed Rubin e1 (i)", ylabel=f"{p} e1", title=p)
    fig.tight_layout()
    fig.savefig(out / "d1_e1_scatter.png", dpi=120)
    plt.close(fig)


def figures_d2(df, out, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, max(len(runs), 1), figsize=(3.2 * max(len(runs), 1), 3.2), squeeze=False)
    for ax, r in zip(axes[0], runs):
        ax.scatter(df["ref_e1"], df[f"{r}_e1_rg_mos"], s=2, alpha=0.4)
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
        ax.set(xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), xlabel="Euclid VIS through D1 relation, e1",
               ylabel=f"Run {r} e1 (REGAUSS, mean of samples)", title=f"Run {r}")
    fig.tight_layout()
    fig.savefig(out / "d2_e1_vs_euclid.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- dispatch
def run(cfg, what, seed=None, split="test"):
    fns = dict(d1=eval_d1, masking=eval_masking, d2=eval_d2, cost=eval_cost, latent=eval_latent,
               check_vaes=lambda c, sp, s: eval_check(c, sp, s, "check_vaes"),
               check_k2=lambda c, sp, s: eval_check(c, sp, s, "check_k2"))
    if what == "all":
        return {k: f(cfg, split, seed) for k, f in fns.items() if not k.startswith("check_")}
    if what not in fns:
        raise SystemExit(f"deblend-eval needs one of {sorted(fns)} or all")
    return fns[what](cfg, split, seed)
