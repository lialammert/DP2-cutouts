# shape_r2.py
# Anna Preto
# AstroDeep Project
# How well each run's deblended shape follows the target's shape seen by Euclid, from the per-object table of the
# D2 evaluation (deblend/evaluate.py eval_d2: <ml_root>/deblend/evaluations/d2_<split>[...]/objects.parquet).
# For each run and component (e1, e2), and each shape estimate the table holds (mean of the samples' PSF-corrected
# shapes, shape of the mean image), against two references:
#   vis   the target's PSF-corrected (re-Gaussianised) VIS shape, vis_e*_rg
#   ref   the same read through the D1 Rubin / VIS relation (ref_e*), the evaluation's own reference
# Prints n, R^2 = 1 - SS_res / SS_tot (about the reference's mean; negative = worse than the mean), Pearson r, the
# slope m and offset c of a least-squares fit run = (1 + m) ref + c, with all targets and with S/N_i >= 10 (as
# weak-lensing samples) and S/N_i >= 50 (where REGAUSS is reliable on both). |e| >= 1 (failed) left out.
# Usage: python scripts/inspect/shape_r2.py <evaluation dir, e.g. $DP2_ML_ROOT/deblend/evaluations/d2_val_pilot>

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def stats(y, x):
    ok = np.isfinite(y) & np.isfinite(x) & (np.abs(y) < 1.0) & (np.abs(x) < 1.0)   # |e| >= 1: failed REGAUSS
    y, x = y[ok], x[ok]
    if len(y) < 10:
        return dict(n=int(len(y)))
    ss_tot = float(((x - x.mean()) ** 2).sum())
    r2 = 1 - float(((y - x) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    slope, c = np.polyfit(x, y, 1)
    return dict(n=int(len(y)), R2=r2, r=float(np.corrcoef(x, y)[0, 1]), m=float(slope - 1), c=float(c))


def main():
    d = Path(sys.argv[1])
    df = pd.read_parquet(d / "objects.parquet")
    runs = sorted({m.group(1) for c in df.columns if (m := re.match(r"^([A-Z])_e1_", c))})
    subsets = {"all": np.ones(len(df), bool)}
    for col in ("snr_i", "rubin_snr_i"):
        if col in df:
            subsets["S/N_i>=10"] = df[col].to_numpy(float) >= 10
            subsets["S/N_i>=50"] = df[col].to_numpy(float) >= 50       # shapes reliable (shape_relation.py)
            break
    rows = []
    for run in runs:
        for est in [c for c in df.columns if re.match(rf"^{run}_(meanimg_)?e1(_rg)?(_mos)?$", c)]:
            for comp in ("e1", "e2"):
                col = est.replace("e1", comp)
                if col not in df:
                    continue
                for ref in ("vis", "ref"):
                    rc = f"vis_{comp}_rg" if ref == "vis" else f"ref_{comp}"
                    if rc not in df:
                        continue
                    for name, sel in subsets.items():
                        s = stats(df.loc[sel, col].to_numpy(float), df.loc[sel, rc].to_numpy(float))
                        rows.append(dict(run=run, estimate=est.replace(f"{run}_", "").replace("e1", "e*"),
                                         comp=comp, reference=ref, subset=name, **s))
    t = pd.DataFrame(rows)
    if t.empty:
        raise SystemExit(f"no run shape column in {d / 'objects.parquet'}: {list(df.columns)[:40]}")
    pd.set_option("display.width", 160)
    print(t.round(3).to_string(index=False))
    best = t[(t.reference == "vis") & (t.subset == t.subset.iloc[-1])].groupby("run")["R2"].mean()
    print(f"\nmean R^2 per run (e1 and e2, all estimates, against VIS, subset {t.subset.iloc[-1]}):\n"
          f"{best.round(3).to_string()}")


if __name__ == "__main__":
    main()
