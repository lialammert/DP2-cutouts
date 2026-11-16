# evaluate.py
# Anna Preto
# AstroDeep Project
# Evaluation of Stage 1 (D1 test split) and Stage 2 (D2 test split and controlled blends). Every measurement is made
# the same way on every image it compares: HSM adaptive moments in the shape band (ellipticity, size, centroid) and
# aperture fluxes, on images in VAE flux units. Results: one parquet row per object and path, a summary JSON with
# bootstrap intervals, binned metrics, the acceptance gate of Stage 1, and a sample of images for the figures.
#
# Paths compared
#   Stage 1   ref    D_R(mu(E_R(x_R)))          the reference Rubin path
#             cross  D_R(mu(E_E(x_E)))          the Euclid path being tested
#             mean   D_R(mean Rubin latent)     floor: what a path that knows nothing gives
#   Stage 2   student   D_R(mu(E_S(blend)))     the trained blend encoder
#             teacher   D_R(mu(E_E(x_E)))       the Euclid teacher of the target (no Rubin truth on real blends)
#             deblender D_R(mu(E_DebVAder-stage2(blend)))   the untouched DebVAder deblender (baseline)
#             naive     D_R(mu(E_R(blend)))     the isolated-galaxy encoder fed the blend (baseline)
#   Controlled blends add the truth: the isolated Rubin stamp of the target and its reference reconstruction.

import json

import numpy as np
import pandas as pd
from keras import ops

from ..pairs import common
from ..pairs.store import Store
from . import losses
from .data import Normaliser
from .debvader_bridge import FrozenRubinVAE


# ----------------------------------------------------------------------------- measurements
def measure(images, weight, cfg, prefix):
    # images, weight (N, H, W, C) in VAE flux units -> DataFrame of shape, centroid and aperture fluxes.
    ib = list(cfg["rubin"]["bands"]).index(cfg["rubin"]["shape_band"])
    acc = cfg["stage1_acceptance"]
    rad = float(acc["flux_aperture_px"])
    n = images.shape[1]
    c = (n - 1) / 2.0
    rows = []
    for k in range(len(images)):
        w = (weight[k][..., ib] > 0).astype(np.int32)
        m = common.moments(images[k][..., ib], w, (c, c))
        r = {f"{prefix}_e1": m["e1"], f"{prefix}_e2": m["e2"], f"{prefix}_sigma": m["sigma"],
             f"{prefix}_x": m["x"] - c, f"{prefix}_y": m["y"] - c, f"{prefix}_ok": m["ok"]}
        rr, cc = np.indices(images[k].shape[:2])
        ap = (rr - c) ** 2 + (cc - c) ** 2 <= rad ** 2
        for j, b in enumerate(cfg["rubin"]["bands"]):
            r[f"{prefix}_flux_{b}"] = float((images[k][..., j] * ap).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def compare(df, a, b, cfg):
    # Differences between path a and path b (columns a_*, b_*), per object.
    out = pd.DataFrame(index=df.index)
    ok = df[f"{a}_ok"].astype(bool) & df[f"{b}_ok"].astype(bool)
    for q in ("e1", "e2"):
        out[f"d{q}_{a}_{b}"] = np.where(ok, df[f"{a}_{q}"] - df[f"{b}_{q}"], np.nan)
    out[f"dcentroid_px_{a}_{b}"] = np.where(ok, np.hypot(df[f"{a}_x"] - df[f"{b}_x"], df[f"{a}_y"] - df[f"{b}_y"]),
                                            np.nan)
    out[f"rsize_{a}_{b}"] = np.where(ok, df[f"{a}_sigma"] / df[f"{b}_sigma"] - 1.0, np.nan)
    for band in cfg["rubin"]["bands"]:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"rflux_{band}_{a}_{b}"] = df[f"{a}_flux_{band}"] / df[f"{b}_flux_{band}"] - 1.0
    return out


def robust_scatter(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def linear_mc(e_test, e_ref):
    # e_test = (1 + m) e_ref + c, least squares.
    ok = np.isfinite(e_test) & np.isfinite(e_ref)
    if ok.sum() < 10:
        return np.nan, np.nan
    slope, intercept = np.polyfit(e_ref[ok], e_test[ok], 1)
    return float(slope - 1.0), float(intercept)


def bootstrap(fn, n, B, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return (float(np.percentile(vals, 16)), float(np.percentile(vals, 84))) if vals.size else (np.nan, np.nan)


def binned(df, by, edges, metrics):
    # metrics: name -> function(sub DataFrame) -> float
    cut = pd.cut(df[by].astype(float), edges)
    rows = []
    for interval, sub in df.groupby(cut, observed=False):
        r = dict(bin=str(interval), lo=float(interval.left), hi=float(interval.right), n=int(len(sub)))
        for name, fn in metrics.items():
            r[name] = fn(sub) if len(sub) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).assign(by=by)


# ----------------------------------------------------------------------------- running the networks
def batched(fn, arrays, batch=256):
    outs = None
    n = len(next(iter(arrays.values())))
    for i in range(0, n, batch):
        r = fn({k: v[i:i + batch] for k, v in arrays.items()})
        r = {k: np.asarray(ops.convert_to_numpy(v)) for k, v in r.items()}
        outs = {k: [v] for k, v in r.items()} if outs is None else {k: outs[k] + [v] for k, v in r.items()}
    return {k: np.concatenate(v) for k, v in outs.items()}


def decode_phys(vae, mu):
    mean = vae.decoder(ops.convert_to_tensor(mu), training=False)[..., :vae.n_bands]
    return vae.denormalise(mean)


def save_examples(path, arrays, k, seed=0):
    n = len(next(iter(arrays.values())))
    pick = np.sort(np.random.default_rng(seed).choice(n, size=min(k, n), replace=False))
    np.savez_compressed(path, index=pick, **{name: a[pick] for name, a in arrays.items()})


# ----------------------------------------------------------------------------- Stage 1
def stage1_gate(summary, acc):
    checks = {
        "mean_delta_e1": abs(summary["mean_de1_cross_ref"]) <= acc["max_mean_delta_e"],
        "mean_delta_e2": abs(summary["mean_de2_cross_ref"]) <= acc["max_mean_delta_e"],
        "m_e1": abs(summary["m_e1"]) <= acc["max_abs_m"], "m_e2": abs(summary["m_e2"]) <= acc["max_abs_m"],
        "c_e1": abs(summary["c_e1"]) <= acc["max_abs_c"], "c_e2": abs(summary["c_e2"]) <= acc["max_abs_c"],
        "scatter_ratio": summary["scatter_ratio_vs_obs"] <= acc["max_scatter_ratio"],
        "centroid": summary["median_centroid_offset_px"] <= acc["max_median_centroid_offset_px"],
    }
    for b in acc["flux_bands"]:
        checks[f"flux_bias_{b}"] = abs(summary[f"median_rflux_{b}"]) <= acc["max_median_flux_bias"]
        checks[f"flux_scatter_{b}"] = summary[f"scatter_rflux_{b}"] <= acc["max_flux_scatter"]
    checks = {k: bool(v) for k, v in checks.items()}
    return dict(passed=all(checks.values()), checks=checks,
                note="thresholds are proposals to be validated (config stage1_acceptance)")


def evaluate_stage1(cfg, debvader_run, run_name, stats_path=None, split="test", tag="best"):
    from .stage1 import load_euclid_encoder

    out_dir = cfg.eval_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    vae = FrozenRubinVAE(debvader_run, stage="stage1", stats_path=stats_path)
    run_dir = cfg.run_dir(run_name)
    enc = load_euclid_encoder(cfg, run_dir, vae.latent_dim, tag)
    ref = dict(np.load(run_dir / "latent_reference.npz"))
    st = Store(cfg.dataset_dir("d1"))
    norm = Normaliser(cfg, vae, st.path)
    rows = np.flatnonzero((st.meta["split"] == split).to_numpy())
    cond = norm.cond(st.meta[norm.cond_columns].to_numpy(np.float32))

    res, ex = [], {k: [] for k in ("obs", "ref", "cross", "euclid", "weight")}
    mu_all = {"mu_r": [], "mu_e": []}
    for s in range(0, len(rows), 1024):
        r = rows[s:s + 1024]
        xr, wr, fr, ivr = norm.rubin(*(st.take(k, r) for k in ("rubin_flux", "rubin_ivar", "rubin_mask")))
        xe = norm.euclid(st.take("euclid_flux", r), st.take("euclid_mask", r))

        def run(b):
            mu_r = vae.mu(vae.encode_params(ops.convert_to_tensor(b["xr"])))
            mu_e = enc([ops.convert_to_tensor(b["xe"]), ops.convert_to_tensor(b["cond"])], training=False)[:, :vae.latent_dim]
            return dict(mu_r=mu_r, mu_e=mu_e, ref=decode_phys(vae, mu_r), cross=decode_phys(vae, mu_e))
        o = batched(run, dict(xr=xr, xe=xe, cond=cond[r]))
        mean_img = np.asarray(ops.convert_to_numpy(decode_phys(vae, ref["mean"][None, :])))
        df = pd.concat([measure(fr, wr, cfg, "obs"), measure(o["ref"], np.ones_like(wr), cfg, "ref"),
                        measure(o["cross"], np.ones_like(wr), cfg, "cross"),
                        measure(np.repeat(mean_img, len(r), 0), np.ones_like(wr), cfg, "mean")], axis=1)
        d = o["mu_e"] - o["mu_r"]
        df["latent_whitened_dist"] = np.sqrt(((d @ ref["W"].T) ** 2).sum(-1) / vae.latent_dim)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = lambda img: (((img - fr) ** 2 * ivr * wr).sum((1, 2, 3)) / np.maximum(wr.sum((1, 2, 3)), 1))  # noqa: E731
        df["chi2_ref_obs"], df["chi2_cross_obs"] = chi(o["ref"]), chi(o["cross"])
        df["row"] = r
        res.append(df)
        for k in mu_all:
            mu_all[k].append(o[k])
        if s == 0:
            ex["obs"].append(fr)
            ex["ref"].append(o["ref"])
            ex["cross"].append(o["cross"])
            ex["euclid"].append(xe)
            ex["weight"].append(wr)
    df = pd.concat(res, ignore_index=True)
    meta = st.meta.iloc[df["row"].to_numpy()].reset_index(drop=True)
    df = pd.concat([meta, df.drop(columns=["row"])], axis=1)
    df = pd.concat([df, compare(df, "cross", "ref", cfg), compare(df, "ref", "obs", cfg),
                    compare(df, "cross", "obs", cfg), compare(df, "mean", "ref", cfg)], axis=1)
    df.to_parquet(out_dir / f"stage1_{split}.parquet", index=False)

    mu_r, mu_e = np.concatenate(mu_all["mu_r"]), np.concatenate(mu_all["mu_e"])
    acc = cfg["stage1_acceptance"]
    s = {}
    for q in ("e1", "e2"):
        s[f"mean_d{q}_cross_ref"] = float(np.nanmean(df[f"d{q}_cross_ref"]))
        s[f"m_{q}"], s[f"c_{q}"] = linear_mc(df[f"cross_{q}"].to_numpy(float), df[f"ref_{q}"].to_numpy(float))
        s[f"scatter_d{q}_cross_ref"] = robust_scatter(df[f"d{q}_cross_ref"])
        s[f"scatter_d{q}_mean_ref"] = robust_scatter(df[f"d{q}_mean_ref"])
    num = np.hypot(robust_scatter(df["de1_cross_obs"]), robust_scatter(df["de2_cross_obs"]))
    den = np.hypot(robust_scatter(df["de1_ref_obs"]), robust_scatter(df["de2_ref_obs"]))
    s["scatter_ratio_vs_obs"] = float(num / den) if den > 0 else np.nan
    s["median_centroid_offset_px"] = float(np.nanmedian(df["dcentroid_px_cross_ref"]))
    for b in cfg["rubin"]["bands"]:
        s[f"median_rflux_{b}"] = float(np.nanmedian(df[f"rflux_{b}_cross_ref"]))
        s[f"scatter_rflux_{b}"] = robust_scatter(df[f"rflux_{b}_cross_ref"])
    s["median_latent_whitened_dist"] = float(df["latent_whitened_dist"].median())
    s["median_chi2_ref_obs"] = float(df["chi2_ref_obs"].median())
    s["median_chi2_cross_obs"] = float(df["chi2_cross_obs"].median())
    s["hsm_failure_cross"] = float(1 - df["cross_ok"].mean())
    s["hsm_failure_ref"] = float(1 - df["ref_ok"].mean())
    Wd = lambda a: a @ ref["W"].T  # noqa: E731
    sub = np.random.default_rng(0).choice(len(mu_r), size=min(5000, len(mu_r)), replace=False)
    a, b = Wd(mu_e[sub]), Wd(mu_r[sub])
    dist = losses.pairwise_sq_dist(a, b)
    rank = (dist < np.diag(dist)[:, None]).sum(1)
    s["retrieval"] = {f"top{k}": float((rank < k).mean()) for k in cfg["evaluation"]["retrieval_k"]}
    s["retrieval_chance"] = {f"top{k}": k / len(sub) for k in cfg["evaluation"]["retrieval_k"]}
    s["retrieval_pool"] = int(len(sub))
    B = int(cfg["evaluation"]["bootstrap"])
    e1c, e1r = df["cross_e1"].to_numpy(float), df["ref_e1"].to_numpy(float)
    s["ci_m_e1"] = bootstrap(lambda i: linear_mc(e1c[i], e1r[i])[0], len(df), B)
    de1 = df["de1_cross_ref"].to_numpy(float)
    s["ci_mean_de1_cross_ref"] = bootstrap(lambda i: float(np.nanmean(de1[i])), len(df), B)
    s["n"] = int(len(df))
    gate = stage1_gate(s, acc)

    metrics = {"mean_de1": lambda d: float(np.nanmean(d["de1_cross_ref"])),
               "mean_de2": lambda d: float(np.nanmean(d["de2_cross_ref"])),
               "scatter_de1": lambda d: robust_scatter(d["de1_cross_ref"]),
               "median_rflux_i": lambda d: float(np.nanmedian(d["rflux_i_cross_ref"])),
               "median_latent_dist": lambda d: float(d["latent_whitened_dist"].median())}
    bins = pd.concat([binned(df, by, edges, metrics) for by, edges in cfg["evaluation"]["bins"].items()
                      if by in df.columns], ignore_index=True)
    bins.to_parquet(out_dir / f"stage1_{split}_bins.parquet", index=False)
    (out_dir / f"stage1_{split}_summary.json").write_text(json.dumps(dict(summary=s, gate=gate), indent=2,
                                                                     default=float))
    save_examples(out_dir / f"stage1_{split}_examples.npz", {k: np.concatenate(v) for k, v in ex.items()},
                  int(cfg["evaluation"]["n_examples_figures"]))
    print(f"Stage 1 {run_name} on {split}: gate {'PASSED' if gate['passed'] else 'FAILED'} -> {out_dir}")
    for k, v in gate["checks"].items():
        print(f"  {k:20s} {'ok' if v else 'FAIL'}")
    return s, gate


# ----------------------------------------------------------------------------- Stage 2
def _encode_paths(vae, encoders, xr):
    # encoders: name -> callable(tensor) -> params; returns name -> reconstruction (VAE units) and mu.
    out = {}
    for name, enc in encoders.items():
        mu = enc(ops.convert_to_tensor(xr))[:, :vae.latent_dim]
        out[f"mu_{name}"] = mu
        out[name] = decode_phys(vae, mu)
    return out


def evaluate_stage2(cfg, debvader_run, stage1_run, run_name, stats_path=None, split="test", tag="best"):
    from .stage1 import load_euclid_encoder
    from .stage2 import load_student, student_encoder

    out_dir = cfg.eval_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    vae = FrozenRubinVAE(debvader_run, stage="stage1", stats_path=stats_path)
    teacher = load_euclid_encoder(cfg, cfg.run_dir(stage1_run), vae.latent_dim)
    student = load_student(debvader_run, cfg.run_dir(run_name), stats_path, tag)
    deblender = student_encoder(debvader_run, "stage2", stats_path)
    deblender.trainable = False
    ref = dict(np.load(cfg.run_dir(stage1_run) / "latent_reference.npz"))
    run2 = json.loads((cfg.run_dir(run_name) / "run.json").read_text())
    centering = run2["settings"]["centering"]
    encoders = dict(student=lambda t: student(t, training=False), deblender=lambda t: deblender(t, training=False),
                    naive=lambda t: vae.encode_params(t))
    norm = Normaliser(cfg, vae, cfg.dataset_dir("d1"))
    W = ref["W"]
    wdist = lambda a, b: np.sqrt((((a - b) @ W.T) ** 2).sum(-1) / vae.latent_dim)  # noqa: E731
    summary = {}

    # --- D2: real blends, scored against the Euclid teacher.
    d2 = Store(cfg.dataset_dir("d2"))
    rows = np.flatnonzero((d2.meta["split"] == split).to_numpy())
    sfx = "_rc" if centering == "rubin" else ""
    cond = norm.cond(d2.meta[norm.cond_columns].to_numpy(np.float32))
    res, ex = [], None
    for s in range(0, len(rows), 1024):
        r = rows[s:s + 1024]
        xr, wr, fr, ivr = norm.rubin(*(d2.take(k + sfx, r) for k in ("rubin_flux", "rubin_ivar", "rubin_mask")))
        xe = norm.euclid(d2.take("euclid_flux", r), d2.take("euclid_mask", r))

        def run(b):
            o = _encode_paths(vae, encoders, b["xr"])
            mu_t = teacher([ops.convert_to_tensor(b["xe"]), ops.convert_to_tensor(b["cond"])],
                           training=False)[:, :vae.latent_dim]
            o.update(mu_teacher=mu_t, teacher=decode_phys(vae, mu_t))
            return o
        o = batched(run, dict(xr=xr, xe=xe, cond=cond[r]))
        ones = np.ones_like(wr)
        df = pd.concat([measure(o[p], ones, cfg, p) for p in ("teacher", "student", "deblender", "naive")], axis=1)
        for p in ("student", "deblender", "naive"):
            df[f"latent_dist_{p}_teacher"] = wdist(o[f"mu_{p}"], o["mu_teacher"])
        df["row"] = r
        res.append(df)
        if ex is None:
            ex = dict(blend=fr, teacher=o["teacher"], student=o["student"], deblender=o["deblender"],
                      naive=o["naive"], euclid=xe)
    df = pd.concat(res, ignore_index=True)
    df = pd.concat([d2.meta.iloc[df["row"].to_numpy()].reset_index(drop=True), df.drop(columns=["row"])], axis=1)
    df = pd.concat([df] + [compare(df, p, "teacher", cfg) for p in ("student", "deblender", "naive")], axis=1)
    df["separation_arcsec"] = df["partner_sep_arcsec"]
    df["i_cModel_mag"] = df["rubin_i_cModel_mag"]  # magnitude of the Rubin object of the group, for the bins
    df.to_parquet(out_dir / f"stage2_d2_{split}.parquet", index=False)
    s = {}
    for p in ("student", "deblender", "naive"):
        s[p] = dict(median_latent_dist=float(df[f"latent_dist_{p}_teacher"].median()),
                    scatter_de1=robust_scatter(df[f"de1_{p}_teacher"]),
                    scatter_de2=robust_scatter(df[f"de2_{p}_teacher"]),
                    mean_de1=float(np.nanmean(df[f"de1_{p}_teacher"])),
                    mean_de2=float(np.nanmean(df[f"de2_{p}_teacher"])),
                    median_rflux_i=float(np.nanmedian(df[f"rflux_i_{p}_teacher"])),
                    scatter_rflux_i=robust_scatter(df[f"rflux_i_{p}_teacher"]),
                    median_centroid_px=float(np.nanmedian(df[f"dcentroid_px_{p}_teacher"])))
    summary["d2_vs_teacher"] = s
    summary["d2_n"] = int(len(df))
    metrics = {f"latent_dist_{p}": (lambda d, p=p: float(d[f"latent_dist_{p}_teacher"].median()))
               for p in ("student", "deblender", "naive")}
    metrics.update({f"scatter_de1_{p}": (lambda d, p=p: robust_scatter(d[f"de1_{p}_teacher"]))
                    for p in ("student", "deblender", "naive")})
    bins = pd.concat([binned(df, by, edges, metrics) for by, edges in cfg["evaluation"]["bins"].items()
                      if by in df.columns], ignore_index=True)
    bins.to_parquet(out_dir / f"stage2_d2_{split}_bins.parquet", index=False)
    save_examples(out_dir / f"stage2_d2_{split}_examples.npz", ex, int(cfg["evaluation"]["n_examples_figures"]))

    # --- Controlled blends: scored against the isolated Rubin stamp of the target and its reference reconstruction.
    bpath = cfg.dataset_dir("blends")
    if (bpath / "info.json").exists():
        bl = Store(bpath)
        d1 = Store(cfg.dataset_dir("d1"))
        rows = np.flatnonzero((bl.meta["split"] == split).to_numpy())
        res, exb = [], None
        for s0 in range(0, len(rows), 1024):
            r = rows[s0:s0 + 1024]
            t = bl.meta["d1_row"].to_numpy()[r]
            xb, wb, fb, ivb = norm.rubin(*(bl.take(k, r) for k in ("rubin_flux", "rubin_ivar", "rubin_mask")))
            xt, wt, ft, ivt = norm.rubin(*(d1.take(k, t) for k in ("rubin_flux", "rubin_ivar", "rubin_mask")))

            def run(b):
                o = _encode_paths(vae, encoders, b["xb"])
                mu_ref = vae.mu(vae.encode_params(ops.convert_to_tensor(b["xt"])))
                o.update(mu_ref=mu_ref, ref=decode_phys(vae, mu_ref))
                return o
            o = batched(run, dict(xb=xb, xt=xt))
            ones = np.ones_like(wb)
            df_b = pd.concat([measure(ft, wt, cfg, "truth"), measure(o["ref"], ones, cfg, "ref")]
                             + [measure(o[p], ones, cfg, p) for p in ("student", "deblender", "naive")], axis=1)
            for p in ("student", "deblender", "naive"):
                df_b[f"latent_dist_{p}_ref"] = wdist(o[f"mu_{p}"], o["mu_ref"])
                df_b[f"chi2_{p}_truth"] = ((o[p] - ft) ** 2 * ivt * wt).sum((1, 2, 3)) / np.maximum(wt.sum((1, 2, 3)), 1)
            df_b["row"] = r
            res.append(df_b)
            if exb is None:
                exb = dict(blend=fb, truth=ft, ref=o["ref"], student=o["student"], deblender=o["deblender"],
                           naive=o["naive"])
        db = pd.concat(res, ignore_index=True)
        db = pd.concat([bl.meta.iloc[db["row"].to_numpy()].reset_index(drop=True), db.drop(columns=["row"])], axis=1)
        db = pd.concat([db] + [compare(db, p, "ref", cfg) for p in ("student", "deblender", "naive")]
                       + [compare(db, p, "truth", cfg) for p in ("student", "deblender", "naive")], axis=1)
        db.to_parquet(out_dir / f"stage2_blends_{split}.parquet", index=False)
        sb = {}
        for p in ("student", "deblender", "naive"):
            m_e1, c_e1 = linear_mc(db[f"{p}_e1"].to_numpy(float), db["ref_e1"].to_numpy(float))
            sb[p] = dict(median_latent_dist=float(db[f"latent_dist_{p}_ref"].median()),
                         median_chi2_truth=float(db[f"chi2_{p}_truth"].median()),
                         scatter_de1_ref=robust_scatter(db[f"de1_{p}_ref"]),
                         mean_de1_ref=float(np.nanmean(db[f"de1_{p}_ref"])), m_e1=m_e1, c_e1=c_e1,
                         median_rflux_i_ref=float(np.nanmedian(db[f"rflux_i_{p}_ref"])),
                         scatter_rflux_i_ref=robust_scatter(db[f"rflux_i_{p}_ref"]))
        summary["blends_vs_reference"] = sb
        summary["blends_n"] = int(len(db))
        db["separation_arcsec"] = db["separation_px"] * common.RUBIN_SCALE
        bm = {f"chi2_{p}": (lambda d, p=p: float(d[f"chi2_{p}_truth"].median())) for p in ("student", "deblender",
                                                                                          "naive")}
        bm.update({f"scatter_de1_{p}": (lambda d, p=p: robust_scatter(d[f"de1_{p}_ref"]))
                   for p in ("student", "deblender", "naive")})
        edges = dict(separation_arcsec=cfg["evaluation"]["bins"]["separation_arcsec"],
                     flux_ratio=[0.0, 0.25, 0.5, 0.75, 1.0])
        pd.concat([binned(db, k, e, bm) for k, e in edges.items()], ignore_index=True).to_parquet(
            out_dir / f"stage2_blends_{split}_bins.parquet", index=False)
        save_examples(out_dir / f"stage2_blends_{split}_examples.npz", exb, int(cfg["evaluation"]["n_examples_figures"]))

    (out_dir / f"stage2_{split}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Stage 2 {run_name} on {split} -> {out_dir}")
    print(json.dumps(summary, indent=2, default=float))
    return summary
