# dataset_bias.py
# Anna Preto
# AstroDeep Project
# Selection bias of the paired datasets: how D1 (isolated pairs) and D2 (Rubin blends resolved by Euclid) differ
# from the Rubin-Euclid matches they come from, and from each other. Every population is described by the same
# Euclid catalogue quantities (the Rubin photometry of a D2 target is that of the blend): VIS magnitude, size,
# ellipticity, photometric redshift, extended probability, PHZ class, and the local density of Euclid sources
# (neighbours within 5" and 10"). Populations: matches (Rubin objects of the patches with a one-to-one Euclid
# counterpart), D1, D2 candidates (Euclid targets of the blended copaings groups), D2.
# Prints quantiles, KS distances and median shifts (overall, within each field EDF-F / EDF-S, and between the two
# fields), PHZ class fractions, and how much of D2 lies where D1 has
# (almost) no example (what a reweighting of D1 towards D2 could not reach); writes the table and histograms.
# Catalogues only (no image): a few minutes on cpu_p1.
# Usage: python scripts/inspect/dataset_bias.py <ML config> <out dir> [--d1 DIR] [--d2 DIR]

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from dp2_cutouts.mlconfig import MLConfig  # noqa: E402
from dp2_cutouts.pairs import common, dataset2  # noqa: E402
from dp2_cutouts.pairs.store import Store  # noqa: E402

PROPS = ["vis_mag", "size", "ellipticity", "photo_z", "extended_prob", "n_neigh_5", "n_neigh_10"]
SETS = ["matches", "d1", "d2_candidates", "d2"]
BOX_COLS = {"phz_photo_z_phz_median": "photo_z", "mer_catalogue_extended_prob": "extended_prob",
            "phz_classification_phz_classification": "phz_class"}
PAIRS = [("matches", "d1"), ("matches", "d2"), ("d1", "d2"), ("d2_candidates", "d2")]
FIELD_DEC_SPLIT = -40.0          # EDF-F around -28 deg, EDF-S around -48 deg (as ml/phase0.py)


def ks_table(pops, pairs):
    # KS distance and median shift (second - first) of every property for each pair of populations.
    rows = []
    for c in PROPS:
        for a, b in pairs:
            x, y = pops[a][c].dropna().to_numpy(), pops[b][c].dropna().to_numpy()
            if len(x) and len(y):
                rows.append(dict(prop=c, pair=f"{a} -> {b}", ks=ks_2samp(x, y).statistic,
                                 median_shift=np.median(y) - np.median(x)))
    return pd.DataFrame(rows)


def euclid_properties(case):
    # Every Euclid object of the patch box: copaings table (vis_mag, size, ellipticity) + MMU columns, and the
    # number of other Euclid sources within 5" and 10".
    from dp2_cutouts.sources import euclid_hf
    e = dataset2.euclid_table(case)
    box = euclid_hf.read_box(case)
    keep = [c for c in BOX_COLS if c in box]
    box = box.drop_duplicates("object_id")[["object_id"] + keep].rename(columns=BOX_COLS)
    box["object_id"] = box["object_id"].astype("int64")
    out = pd.DataFrame(dict(object_id=e["object_id"].astype("int64").to_numpy(),
                            vis_mag=e["vis_mag"].to_numpy(float), size=e["size_true"].to_numpy(float),
                            ellipticity=e["ellipticity"].to_numpy(float),
                            field=np.where(e["dec"].to_numpy(float) < FIELD_DEC_SPLIT, "EDF-S", "EDF-F")))
    wx, ny = common.sky_offset_arcsec(float(e["ra"].mean()), float(e["dec"].mean()), e["ra"].to_numpy(float),
                                      e["dec"].to_numpy(float))
    tree = cKDTree(np.c_[wx, ny])
    for r in (5, 10):
        out[f"n_neigh_{r}"] = np.array([len(v) - 1 for v in tree.query_ball_point(np.c_[wx, ny], r)])
    out = out.merge(box, on="object_id", how="left")
    for c in ("photo_z", "extended_prob"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--d1")
    ap.add_argument("--d2")
    args = ap.parse_args()
    from dp2_cutouts.pipeline.patches import load_patches, owned, patch_case

    cfg = MLConfig.load(args.config)
    pcfg = cfg.pipeline()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d1 = Store(Path(args.d1) if args.d1 else cfg.dataset_dir("d1")).meta
    d2 = Store(Path(args.d2) if args.d2 else cfg.dataset_dir("d2")).meta

    props, members = [], {s: set() for s in SETS}
    for p in load_patches(pcfg):
        case = patch_case(pcfg, p)
        try:
            props.append(euclid_properties(case))
        except FileNotFoundError as err:
            print(f"{p['name']}: skipped ({err})", flush=True)
            continue
        cat = pd.read_parquet(case.data("30_processed", "catalog.parquet"))
        cat = cat[owned(cat, p).to_numpy() & cat["has_euclid_counterpart"].astype(bool).to_numpy()]
        members["matches"] |= set(cat["euclid_object_id"].astype("int64"))
        cand, _ = dataset2.candidate_targets(case, cfg, p)
        if len(cand):
            members["d2_candidates"] |= set(cand["euclid_object_id"].astype("int64"))
        print(f"{p['name']}: {len(cat)} matches, {len(cand)} D2 candidates", flush=True)
    members["d1"] = set(d1["euclid_object_id"].dropna().astype("int64"))
    members["d2"] = set(d2["euclid_object_id"].astype("int64"))
    allp = pd.concat(props, ignore_index=True).drop_duplicates("object_id").set_index("object_id")

    pops = {}
    for s in SETS:
        ids = [i for i in members[s] if i in allp.index]
        pops[s] = allp.loc[ids]
        print(f"{s}: {len(members[s])} objects, {len(ids)} with Euclid catalogue properties")
    rows = []
    for s, t in pops.items():
        rows.append(t.assign(population=s))
    pd.concat(rows).reset_index().to_parquet(out / "dataset_bias.parquet", index=False)

    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    for c in PROPS:
        tab = pd.DataFrame({s: t[c].quantile(q) for s, t in pops.items() if c in t}).T.round(3)
        tab.insert(0, "n", [int(pops[s][c].notna().sum()) for s in tab.index])
        print(f"\n{c} (quantiles 10/25/50/75/90 %)\n{tab.to_string()}")

    print("\nKS distance (0 = same distribution, 1 = disjoint) and median shift (second - first)")
    ks = ks_table(pops, PAIRS)
    print(ks.pivot(index="prop", columns="pair", values="ks").round(3).to_string())
    print(ks.pivot(index="prop", columns="pair", values="median_shift").round(3).to_string())

    # Per field: each dataset against the population of its own field, and the two fields against each other.
    fr = pd.DataFrame({s: t["field"].value_counts() for s, t in pops.items()}).fillna(0).astype(int)
    print(f"\nobjects per field\n{fr.to_string()}\nfraction per field\n{(fr / fr.sum()).round(3).to_string()}")
    fields = sorted(allp["field"].dropna().unique())
    for f in fields:
        sub = {s: t[t["field"] == f] for s, t in pops.items()}
        k = ks_table(sub, PAIRS)
        if len(k):
            print(f"\n[{f}] KS distance\n{k.pivot(index='prop', columns='pair', values='ks').round(3).to_string()}")
    if len(fields) == 2:
        a, b = fields
        cross = {f"{s}_{f}": t[t["field"] == f] for s, t in pops.items() for f in fields}
        k = ks_table(cross, [(f"{s}_{a}", f"{s}_{b}") for s in SETS])
        if len(k):
            print(f"\n{a} -> {b}, same population: KS distance and median shift")
            print(k.pivot(index="prop", columns="pair", values="ks").round(3).to_string())
            print(k.pivot(index="prop", columns="pair", values="median_shift").round(3).to_string())

    if "phz_class" in allp:
        cls = pd.DataFrame({s: t["phz_class"].astype(str).value_counts(normalize=True) for s, t in pops.items()})
        print(f"\nPHZ class fractions\n{cls.fillna(0).round(3).to_string()}")

    # D2 where D1 has (almost) no example: bins of VIS mag (0.5), log size (0.15 dex), photo-z (0.25).
    def bins(t):
        return pd.DataFrame(dict(m=np.floor(t["vis_mag"] / 0.5),
                                 s=np.floor(np.log10(t["size"].clip(lower=1e-3)) / 0.15),
                                 z=np.floor(t["photo_z"] / 0.25))).dropna().astype("int64")
    b1 = bins(pops["d1"]).value_counts()
    b2 = bins(pops["d2"])
    n1 = np.array([b1.get(tuple(r), 0) for r in b2.itertuples(index=False)])
    for k in (1, 5, 20):
        print(f"D2 targets in a (mag, size, z) bin with fewer than {k} D1 examples: {np.mean(n1 < k):.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), PROPS):
        v = pd.concat([pops[s][c] for s in SETS]).dropna()
        if v.empty:                                     # column empty in the catalogue (e.g. extended_prob in Q1)
            ax.set_title(f"{c}: no value", fontsize=8)
            continue
        lo, hi = np.nanquantile(v, [0.005, 0.995])
        edges = np.linspace(lo, hi, 40)
        for s in SETS:
            x = pops[s][c].dropna().clip(lo, hi)
            if len(x):
                ax.hist(x, edges, histtype="step", density=True, lw=1.3, label=f"{s} ({len(x)})")
        ax.set_xlabel(c)
        ax.set_yticks([])
    axes.ravel()[0].legend(fontsize=7)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Euclid catalogue properties: matches, D1, D2 candidates, D2 (normalised histograms)")
    fig.tight_layout()
    fig.savefig(out / "dataset_bias.png", dpi=130)
    print(f"\n-> {out / 'dataset_bias.parquet'}, {out / 'dataset_bias.png'}")


if __name__ == "__main__":
    main()
