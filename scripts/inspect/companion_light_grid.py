# companion_light_grid.py
# Anna Preto
# AstroDeep Project
# Visual check of a contamination measure on one processed patch (companion_light_ratio by default, or any table
# column such as i_blendedness): i-band stamps drawn in bins of that column, each
# shown before masking (flux_raw) and after (flux, the masked pixels outlined). Also prints sanity numbers (is the
# target itself ever masked?). Reads the stamp table in small batches: fine on a login node.
# Usage: python scripts/inspect/companion_light_grid.py <patch dir .../patches/<tract>_<patch>> <out.png> [column]

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

BINS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, np.inf)]
PER_BIN = 4


def stretch(img, s):
    return np.arcsinh(img / (0.1 * s)) / np.arcsinh(10.0)


def main(patch_dir, out, column="companion_light_ratio"):
    proc = f"{patch_dir}/30_processed"
    info = json.load(open(f"{proc}/stamps.json"))
    n, ib = int(info["stamp_px"]), list(info["bands"]).index("i")
    wanted = ["object_id", "in_region", column, "companion_light_ratio", "i_blendedness", "shared_flux_ratio_max",
              "n_shared_companions", "masked_fraction_in_ellipse", "i_cModel_mag"]
    names = set(pq.read_schema(f"{proc}/table.parquet").names)
    if column not in names:
        sys.exit(f"no column {column!r} in {proc}/table.parquet (rerun process for the new measures)")
    t = pd.read_parquet(f"{proc}/table.parquet", columns=[c for c in dict.fromkeys(wanted) if c in names])
    for c in wanted:
        if c not in t:
            t[c] = np.nan
    t = t[t["in_region"].astype(bool)]
    s = t[column].astype(float)
    print(f"{len(t)} targets in {patch_dir}")
    print("masked_fraction_in_ellipse > 0.5 (target itself masked):",
          round(float((t["masked_fraction_in_ellipse"].astype(float) > 0.5).mean()), 3))
    print(f"medians by {column} bin:")
    g = t.groupby(pd.cut(s, [b[0] for b in BINS] + [np.inf], include_lowest=True), observed=True)
    other = [c for c in ("companion_light_ratio", "i_blendedness") if c != column]
    print(pd.concat([g.size().rename("n"), g[other + ["masked_fraction_in_ellipse", "shared_flux_ratio_max",
                                                     "i_cModel_mag"]].median().round(3)], axis=1).to_string())
    print("Spearman correlation companion_light_ratio / i_blendedness:",
          round(float(t["companion_light_ratio"].astype(float).corr(t["i_blendedness"].astype(float),
                                                                     method="spearman")), 3))

    rng = np.random.default_rng(0)
    picks = []
    for lo, hi in BINS:
        ids = t.loc[(s >= lo) & (s < hi), "object_id"].to_numpy()
        picks.append(list(rng.choice(ids, min(PER_BIN, len(ids)), replace=False)) if len(ids) else [])
    want = {int(i) for p in picks for i in p}

    rows = {}
    pf = pq.ParquetFile(f"{proc}/stamps.parquet")
    for batch in pf.iter_batches(batch_size=128, columns=["object_id", "flux_raw", "flux", "companion_mask"]):
        ids = batch.column("object_id").to_numpy()
        for k in np.flatnonzero(np.isin(ids, list(want))):
            r = {c: batch.column(c)[int(k)].as_py() for c in ("flux_raw", "flux", "companion_mask")}
            rows[int(ids[k])] = r
        if len(rows) == len(want):
            break

    meta = t.set_index("object_id")
    fig, axes = plt.subplots(len(BINS), 2 * PER_BIN, figsize=(2 * PER_BIN * 1.7, len(BINS) * 1.9), squeeze=False)
    for i, ((lo, hi), p) in enumerate(zip(BINS, picks)):
        for j in range(PER_BIN):
            a0, a1 = axes[i, 2 * j], axes[i, 2 * j + 1]
            for a in (a0, a1):
                a.set_xticks([])
                a.set_yticks([])
            if j >= len(p):
                a0.axis("off")
                a1.axis("off")
                continue
            oid = int(p[j])
            r = rows[oid]
            raw = np.asarray(r["flux_raw"], np.float32).reshape(-1, n, n)[ib]
            fil = np.asarray(r["flux"], np.float32).reshape(-1, n, n)[ib]
            need = np.asarray(r["companion_mask"], bool).reshape(n, n)
            sc = float(np.nanpercentile(np.abs(raw), 99.5)) or 1.0
            a0.imshow(stretch(raw, sc), origin="lower", cmap="gray_r")
            a1.imshow(stretch(fil, sc), origin="lower", cmap="gray_r")
            if need.any():
                a1.contour(need, levels=[0.5], colors="tab:red", linewidths=0.6)
            m = meta.loc[oid]
            a0.set_title(f"c={m['companion_light_ratio']:.2f} b={m['i_blendedness']:.2f}", fontsize=6)
            a1.set_title(f"i={m['i_cModel_mag']:.1f} r={m['shared_flux_ratio_max']:.1f}", fontsize=6)
        axes[i, 0].set_ylabel(f"{lo:.2f}-{hi:.2f}", fontsize=8)
    fig.suptitle(f"rows: bins of {column}. i band: raw | masked (red: filled pixels); c = companion_light_ratio, "
                 "b = i_blendedness, r = brightest shared companion / target", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
