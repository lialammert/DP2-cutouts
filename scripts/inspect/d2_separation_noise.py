# d2_separation_noise.py
# Anna Preto
# AstroDeep Project
# Is the D2 Euclid separation threshold above the noise? The test masks the other sources on the VIS cutout, fills
# the holes with sky drawn from the same cutout, and compares the target's moments before / after (absolute limits:
# dataset2.accept). Where the masked pixels reach the target's moment window, a new noise draw alone moves the
# measurement. Here the test is run K times per target with different sky draws:
#   noise  = spread of the after-masking measurement over the draws (what the fill alone does),
#   effect = |mean over the draws - before| (what removing the neighbours does),
#   z      = effect / noise (about 1 when the neighbours do not matter).
# Targets: a sample per D2 outcome (kept, and the euclid_separation rejects by category) from the patches with the
# most rejections. Writes d2_separation_noise.csv and .png, prints per category the noise of each metric against
# its limit and how many targets would pass with a noise-relative limit.
# Usage: python scripts/inspect/d2_separation_noise.py <ML config> <out dir> [--draws 10] [--per-category 60]
#            [--patches 4] [--z-max 3] [--d2 DIR]

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dp2_cutouts.mlconfig import MLConfig  # noqa: E402
from dp2_cutouts.pairs import common, dataset2  # noqa: E402
from dp2_cutouts.pairs.store import Store  # noqa: E402

ALL_FOUR = "delta_e+centroid_shift+size_change+flux_change"
# metric: (limit key in dataset2.accept, unit)
METRICS = {"delta_e": ("max_delta_e", ""), "centroid": ("max_centroid_shift_arcsec", "arcsec"),
           "size": ("max_rel_size_change", "rel"), "flux": ("max_rel_flux_change", "rel")}


def category(detail):
    if detail == "":
        return "kept"
    if detail == ALL_FOUR:
        return "all_four"
    if detail == "hsm_failed":
        return "hsm_failed"
    if detail == "delta_e":
        return "delta_e_only"
    return "other"


def sample(st, n_patches, per_cat, seed):
    rej = st.rejected()
    rej = rej[rej["reason"] == "euclid_separation"][["euclid_object_id", "group_id", "detail"]]
    kept = st.meta[["euclid_object_id", "group_id"]].assign(detail="")
    t = pd.concat([rej, kept], ignore_index=True)
    t["patch"] = t["group_id"].str.split(":").str[0]
    t["category"] = t["detail"].map(category)
    top = rej["group_id"].str.split(":").str[0].value_counts().index[:n_patches]
    t = t[t["patch"].isin(top)]
    rng = np.random.default_rng(seed)
    pick = [g.iloc[rng.permutation(len(g))[:per_cat]] for _, g in t.groupby("category")]
    return pd.concat(pick).drop_duplicates("euclid_object_id"), list(top)


def measure(cut, tgt, others, cfg, tiles, seeds, vis):
    # Moments before (once) and after masking for each sky draw. Returns (before dict, list of after dicts).
    w = cut["wcs"]
    tx, ty = w.world_to_pixel_values(tgt["ra"], tgt["dec"])
    before, after = None, []
    for s in seeds:
        _, _, (ff, _, m, _, need, f) = dataset2.euclid_separation(cut, tgt, others, cfg, np.random.default_rng(s),
                                                                   tiles)
        weight = (~m[..., vis]).astype(np.int32)
        if before is None:
            before = common.moments(f[..., vis], weight, (float(tx), float(ty)))
            before["masked_px"] = int(need.sum())
        after.append(common.moments(ff[..., vis], weight, (float(tx), float(ty))))
    return before, after


def summarise(before, after, scale):
    # Per metric: effect (mean after - before), noise (spread of after), z, and the value of the build-like single
    # draw (the first seed).
    ok = [a for a in after if a["ok"]]
    r = dict(n_draws=len(after), n_draws_ok=len(ok), hsm_before_ok=bool(before["ok"]))
    if not before["ok"] or len(ok) < 3:
        return r
    e1 = np.array([a["e1"] for a in ok])
    e2 = np.array([a["e2"] for a in ok])
    x = np.array([a["x"] for a in ok])
    y = np.array([a["y"] for a in ok])
    sg = np.array([a["sigma"] for a in ok])
    fl = np.array([a["flux"] for a in ok])
    de = np.hypot(e1.mean() - before["e1"], e2.mean() - before["e2"])
    ne = np.sqrt((e1.var(ddof=1) + e2.var(ddof=1)) / 2)
    dc = np.hypot(x.mean() - before["x"], y.mean() - before["y"]) * scale
    nc = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) * scale
    ds = abs(sg.mean() / before["sigma"] - 1)
    ns = sg.std(ddof=1) / before["sigma"]
    df = abs(fl.mean() / before["flux"] - 1) if before["flux"] else np.nan
    nf = fl.std(ddof=1) / abs(before["flux"]) if before["flux"] else np.nan
    a0 = after[0]
    single = dict(
        delta_e=np.hypot(a0["e1"] - before["e1"], a0["e2"] - before["e2"]) if a0["ok"] else np.nan,
        centroid=np.hypot(a0["x"] - before["x"], a0["y"] - before["y"]) * scale if a0["ok"] else np.nan,
        size=abs(a0["sigma"] / before["sigma"] - 1) if a0["ok"] else np.nan,
        flux=abs(a0["flux"] / before["flux"] - 1) if a0["ok"] and before["flux"] else np.nan)
    for k, (eff, noi) in dict(delta_e=(de, ne), centroid=(dc, nc), size=(ds, ns), flux=(df, nf)).items():
        r[f"{k}_effect"] = float(eff)
        r[f"{k}_noise"] = float(noi)
        # no noise and no effect: the mask does not reach the moment window (pixels unchanged)
        r[f"{k}_z"] = float(eff / noi) if noi > 0 else (0.0 if eff == 0 else np.inf)
        r[f"{k}_single"] = float(single[k])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--per-category", type=int, default=60)
    ap.add_argument("--patches", type=int, default=4)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d2")
    args = ap.parse_args()
    from dp2_cutouts.pipeline.patches import load_patches, patch_case
    from dp2_cutouts.sources import euclid_images, euclid_seg

    cfg = MLConfig.load(args.config)
    d2 = Path(args.d2) if args.d2 else cfg.dataset_dir("d2")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st = Store(d2)
    pick, top = sample(st, args.patches, args.per_category, args.seed)
    print(f"patches {top}: {pick['category'].value_counts().to_dict()}", flush=True)

    pcfg = cfg.pipeline()
    d2c = cfg["dataset2"]
    acc = d2c["accept"]
    vis = list(cfg["euclid"]["bands"]).index("VIS")
    scale = float(cfg["euclid"]["pixel_scale"])
    tiles = None
    if d2c.get("euclid_masking", "ellipse") == "segmentation":
        paths = euclid_seg.cached_tiles(euclid_seg.cache_dir(d2c if "seg_cache" in d2c else pcfg["masking"]))
        tiles = euclid_seg.TileSet(paths).__enter__()
    rows = []
    for pname, g in pick.groupby("patch"):
        patch = load_patches(pcfg, [pname])[0]
        case = patch_case(pcfg, patch)
        cand, e = dataset2.candidate_targets(case, cfg, patch)
        cand = cand[cand["euclid_object_id"].isin(g["euclid_object_id"])]
        cuts = euclid_images.fetch(cfg["euclid"], pd.DataFrame(dict(
            object_id=cand["euclid_object_id"], ra=cand["euclid_ra"], dec=cand["euclid_dec"])), case.sky_box)
        info = g.set_index("euclid_object_id")
        for r in cand.itertuples(index=False):
            cut = cuts.get(int(r.euclid_object_id))
            if cut is None or "VIS" not in cut["bands"]:
                continue
            tgt = dict(ra=r.euclid_ra, dec=r.euclid_dec, a_e=e.loc[r.euclid_row, "a_e"],
                       b_e=e.loc[r.euclid_row, "b_e"], theta=e.loc[r.euclid_row, "theta"],
                       flux=float(e.loc[r.euclid_row, "vis_flux"]))
            others = e.drop(index=r.euclid_row)
            dra = np.abs(((others["ra"] - r.euclid_ra + 180) % 360) - 180) * np.cos(np.radians(r.euclid_dec)) * 3600
            others = others[(dra < 15) & (np.abs(others["dec"] - r.euclid_dec) * 3600 < 15)]
            # first seed = the one of the D2 build: draw 0 reproduces the recorded decision
            seeds = [[int(cfg["seed"]), int(r.euclid_object_id) & 0x7FFFFFFF]] + \
                    [[int(cfg["seed"]), int(r.euclid_object_id) & 0x7FFFFFFF, k] for k in range(1, args.draws)]
            before, after = measure(cut, tgt, others, cfg, tiles, seeds, vis)
            row = dict(euclid_object_id=int(r.euclid_object_id), patch=pname,
                       category=info.loc[int(r.euclid_object_id), "category"],
                       detail=info.loc[int(r.euclid_object_id), "detail"], euclid_vis_mag=float(r.euclid_vis_mag),
                       sb_rubin=float(r.sb_rubin), partner_sep_arcsec=float(r.partner_sep_arcsec),
                       masked_px=before["masked_px"])
            row.update(summarise(before, after, scale))
            rows.append(row)
        print(f"{pname}: {len(cand)} targets done", flush=True)
    if tiles is not None:
        tiles.__exit__(None, None, None)

    t = pd.DataFrame(rows)
    t.to_csv(out / "d2_separation_noise.csv", index=False)
    print(f"\n{len(t)} targets, {args.draws} sky draws each; limits {dict(acc)}")
    print("\nHSM: draws that fail after masking (fraction), per category")
    print((1 - t["n_draws_ok"] / t["n_draws"]).groupby(t["category"]).describe()[["mean", "50%", "max"]].round(3)
          .to_string())
    m = t[t["delta_e_noise"].notna()] if "delta_e_noise" in t else t.iloc[0:0]
    for k, (key, unit) in METRICS.items():
        if f"{k}_noise" not in m:
            continue
        lim = float(acc[key])
        g = m.groupby("category")
        s = pd.DataFrame(dict(
            n=g.size(), noise_median=g[f"{k}_noise"].median(), noise_over_limit=(m[f"{k}_noise"] > lim).groupby(
                m["category"]).mean(), effect_median=g[f"{k}_effect"].median(), z_median=g[f"{k}_z"].median(),
            frac_z_below=(m[f"{k}_z"] < args.z_max).groupby(m["category"]).mean(),
            frac_single_below_limit=(m[f"{k}_single"] <= lim).groupby(m["category"]).mean()))
        print(f"\n{k} (limit {lim} {unit}): noise of one sky fill vs limit, effect = |mean after - before|, "
              f"z = effect / noise\n{s.round(4).to_string()}")
    if len(m):
        zcols = [f"{k}_z" for k in METRICS]
        m = m.assign(pass_z=(m[zcols] < args.z_max).all(axis=1))
        print(f"\nwould pass with all four z < {args.z_max} (noise-relative limit), per category:\n"
              f"{m.groupby('category')['pass_z'].agg(['sum', 'size', 'mean']).round(3).to_string()}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for c, gg in m.groupby("category"):
            axes[0].scatter(gg["delta_e_noise"], gg["delta_e_single"], s=8, label=f"{c} ({len(gg)})")
            axes[1].scatter(gg["euclid_vis_mag"], gg["delta_e_z"], s=8)
        lim = float(acc["max_delta_e"])
        lo = max(1e-5, float(np.nanmin(m["delta_e_noise"].replace(0, np.nan))) / 2)
        xx = np.logspace(np.log10(lo), 0, 50)
        axes[0].plot(xx, xx * np.sqrt(np.pi / 2) * np.sqrt(2), "k:", lw=0.8, label="pure noise (mean)")
        axes[0].axhline(lim, color="r", lw=0.8, label=f"limit {lim}")
        axes[0].set(xscale="log", yscale="log", xlabel="noise of e after one sky fill",
                    ylabel="delta_e of the build draw")
        axes[0].legend(fontsize=7)
        axes[1].axhline(args.z_max, color="r", lw=0.8)
        axes[1].set(yscale="log", xlabel="VIS mag", ylabel="z (delta_e effect / noise)")
        fig.tight_layout()
        fig.savefig(out / "d2_separation_noise.png", dpi=130)
    print(f"\n-> {out / 'd2_separation_noise.csv'}")


if __name__ == "__main__":
    main()
