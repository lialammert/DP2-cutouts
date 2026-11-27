# d1_truncation.py
# Anna Preto
# AstroDeep Project
# How much of a D1 target's own light the neighbour masks hide on the Rubin stamp (the masked pixels are replaced by
# sky, so that light is lost to the training). Estimated by point symmetry about the target centre, as for D2
# (mask_damage.lost_sym): a masked pixel whose mirror is free is given the light of its mirror. On the preview
# export (export_preview.py: raw Rubin i stamp and the pipeline's masks of each D1 / rejected object), so it runs on
# a login node. halo_ring_z flags any mask lying on light, even a justified one (a real neighbour inside a bright
# halo); lost_flux_frac says how much target light the masks took.
# Prints, per i magnitude, the median loss and the share of targets losing more than 1, 3, 5 % of their light, and
# how lost_flux_frac and halo_ring_z agree; writes lost_flux_frac and lost_unknown into d1.parquet / d1r.parquet of
# the export (the viewer filters on them after upload_preview.py --no-dataset).
# Usage: python scripts/inspect/d1_truncation.py <export dir, e.g. $SCRATCH/hf_preview>

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def mask_damage():
    spec = importlib.util.spec_from_file_location("mask_damage", HERE / "mask_damage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def losses(t, info, lost_sym):
    n = int(info["rubin_px"])
    nb = len(info["rubin_bands"])
    ib = list(info["rubin_bands"]).index("i")
    lost, unknown = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for k, r in enumerate(t.itertuples(index=False)):
        if not (getattr(r, "has_raw", False) and getattr(r, "has_need", False)):
            continue
        raw = np.frombuffer(r.rubin_raw, np.float16).astype(np.float32).reshape(n, n, nb)[..., ib]
        need = (np.unpackbits(np.frombuffer(r.need_rubin, np.uint8), count=n * n)
                | np.unpackbits(np.frombuffer(r.need_euclid, np.uint8), count=n * n)).reshape(n, n).astype(bool)
        # raw is in units of the pixel noise: sigma = 1
        lost[k], unknown[k] = lost_sym(raw, np.isfinite(raw), need, 1.0)
    return lost, unknown


def summary(t, name):
    m = t["i_cModel_mag"].astype(float) if "i_cModel_mag" in t else pd.Series(np.full(len(t), 25.0), index=t.index)
    g = t.groupby(pd.cut(m, [0, 19, 20, 21, 22, 23, 24, 30]), observed=True)
    s = pd.DataFrame(dict(n=g.size(), lost_median=g["lost_flux_frac"].median(),
                          above_1pc=g["lost_flux_frac"].apply(lambda x: (x > 0.01).mean()),
                          above_3pc=g["lost_flux_frac"].apply(lambda x: (x > 0.03).mean()),
                          above_5pc=g["lost_flux_frac"].apply(lambda x: (x > 0.05).mean())))
    print(f"\n== {name}: target light hidden by the masks (lost_flux_frac), per i magnitude\n{s.round(3).to_string()}")
    if "halo_ring_z" in t:
        h = t["halo_ring_z"].astype(float) > 10
        lf = t["lost_flux_frac"]
        print(f"halo_ring_z > 10: {h.mean():.1%} of the objects; among them {(lf[h] > 0.03).mean():.1%} lose > 3 % of "
              f"their light, against {(lf[~h] > 0.03).mean():.1%} among the others")


def main():
    out = Path(sys.argv[1])
    info = json.loads((out / "info.json").read_text())
    md = mask_damage()
    for name in ("d1", "d1r"):
        path = out / f"{name}.parquet"
        if name not in info or not path.exists():
            continue
        t = pd.read_parquet(path)
        if "rubin_raw" not in t:
            print(f"{name}: no raw Rubin stamp in this export (export_preview.py without --no-pixels)")
            continue
        t["lost_flux_frac"], t["lost_unknown"] = losses(t, info[name], md.lost_sym)
        summary(t, name)
        t.to_parquet(path, index=False)
        print(f"-> {path}: lost_flux_frac, lost_unknown added")


if __name__ == "__main__":
    main()
