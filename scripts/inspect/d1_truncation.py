# d1_truncation.py
# Anna Preto
# AstroDeep Project
# How much of a D1 target's own light the neighbour masks hide on the Rubin stamp (the masked pixels are replaced by
# sky, so that light is lost to the training). Estimated by point symmetry about the target centre, as for D2
# (mask_damage.lost_sym): a masked pixel whose mirror is free is given the light of its mirror. On the preview
# export (export_preview.py: raw Rubin i stamp and the pipeline's masks of each D1 / rejected object), so it runs on
# a login node. halo_ring_z flags any mask lying on light, even a justified one (a real neighbour inside a bright
# halo); lost_flux_frac says how much target light the masks took.
# Uncertainty (lost_flux_sigma, same unit): the loss L and the visible light V are sums of smoothed pixels, so each
# has variance sigma^2 * sum_{a,b} C(a - b), C the smoothing kernel convolved with itself (a Gaussian of width
# sqrt(2) * SMOOTH_PX), computed with gaussian_filter on the indicator of those pixels; L / (V + L) by first-order
# propagation. The preview stamps carry one
# noise per stamp (images in noise units: sigma = 1); a full-D1 computation would use the variance of every pixel.
# A target is "really cut" when lost_flux_frac > 3 % and > 3 lost_flux_sigma.
# Prints, per i magnitude, the median loss and its uncertainty, the share of targets losing more than 1, 3, 5 % and
# the share really cut, and how the loss and halo_ring_z agree; writes lost_flux_frac, lost_flux_sigma and
# lost_unknown into d1.parquet / d1r.parquet of the export (the viewer filters on them after
# upload_preview.py --no-dataset).
# Usage: python scripts/inspect/d1_truncation.py <export dir, e.g. $SCRATCH/hf_preview>

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
BINS = [0, 19, 20, 21, 22, 23, 24, 30]


def mask_damage():
    if str(HERE) not in sys.path:                        # mask_damage imports d2_gallery from this folder
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("mask_damage", HERE / "mask_damage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lost_with_sigma(md, raw, valid, fill, sig):
    # mask_damage.lost_sym (same support, same pixels) and its uncertainty; all three as fractions of the target
    # light (visible + lost), except lost_unknown (share of the masked pixels near the target whose mirror is masked).
    t, sm = md.support(raw, valid, fill, sig)
    if t is None:
        return np.nan, np.nan, np.nan
    near = fill & ndimage.binary_dilation(t, iterations=4)
    known = near & t[::-1, ::-1]
    pos = np.clip(sm, 0, None)
    lost = float(pos[::-1, ::-1][known].sum())
    vis = float(pos[t].sum())
    unknown = float((near & fill[::-1, ::-1]).sum()) / max(float(near.sum()), 1.0) if near.any() else 0.0
    def var_sum(sel):                                    # variance of the sum of the smoothed pixels in sel
        w = sel.astype(float)
        return float((ndimage.gaussian_filter(w, md.SMOOTH_PX * np.sqrt(2.0)) * w).sum()) * sig ** 2

    var_l = var_sum(known[::-1, ::-1])                   # the mirror pixels summed into the loss
    var_v = var_sum(t)                                   # the visible target pixels
    total = max(vis + lost, 1e-30)
    frac = lost / total
    # frac = L / (V + L): first-order propagation, L and V from (nearly) disjoint pixels.
    var_f = ((1 - frac) ** 2 * var_l + frac ** 2 * var_v) / total ** 2
    return frac, unknown, np.sqrt(var_f)


def losses(t, info, md):
    n = int(info["rubin_px"])
    nb = len(info["rubin_bands"])
    ib = list(info["rubin_bands"]).index("i")
    out = np.full((len(t), 3), np.nan)
    for k, r in enumerate(t.itertuples(index=False)):
        if not (getattr(r, "has_raw", False) and getattr(r, "has_need", False)):
            continue
        raw = np.frombuffer(r.rubin_raw, np.float16).astype(np.float32).reshape(n, n, nb)[..., ib]
        need = (np.unpackbits(np.frombuffer(r.need_rubin, np.uint8), count=n * n)
                | np.unpackbits(np.frombuffer(r.need_euclid, np.uint8), count=n * n)).reshape(n, n).astype(bool)
        out[k] = lost_with_sigma(md, raw, np.isfinite(raw), need, 1.0)   # raw in units of the pixel noise
    return out


def really_cut(t):
    f, s = t["lost_flux_frac"], t["lost_flux_sigma"]
    return (f > 0.03) & (f > 3 * s)


def summary(t, name):
    m = t["i_cModel_mag"].astype(float) if "i_cModel_mag" in t else pd.Series(np.full(len(t), 25.0), index=t.index)
    bins = pd.cut(m, BINS)
    g = t.groupby(bins, observed=True)
    cut = really_cut(t)
    s = pd.DataFrame(dict(n=g.size(), lost_median=g["lost_flux_frac"].median(),
                          sigma_median=g["lost_flux_sigma"].median(),
                          above_1pc=g["lost_flux_frac"].apply(lambda x: (x > 0.01).mean()),
                          above_3pc=g["lost_flux_frac"].apply(lambda x: (x > 0.03).mean()),
                          above_5pc=g["lost_flux_frac"].apply(lambda x: (x > 0.05).mean()),
                          cut_3pc_3sigma=cut.groupby(bins, observed=True).mean()))
    print(f"\n== {name}: target light hidden by the masks (lost_flux_frac), per i magnitude\n{s.round(3).to_string()}")
    print(f"really cut (> 3 % and > 3 sigma): {cut.mean():.1%} of the {len(t)} objects "
          f"(> 3 % alone: {(t['lost_flux_frac'] > 0.03).mean():.1%})")
    if "halo_ring_z" in t:
        h = t["halo_ring_z"].astype(float) > 10
        print(f"halo_ring_z > 10: {h.mean():.1%} of the objects; among them {cut[h].mean():.1%} really cut, "
              f"against {cut[~h].mean():.1%} among the others")


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
        v = losses(t, info[name], md)
        t["lost_flux_frac"], t["lost_unknown"], t["lost_flux_sigma"] = v[:, 0], v[:, 1], v[:, 2]
        summary(t, name)
        t.to_parquet(path, index=False)
        print(f"-> {path}: lost_flux_frac, lost_flux_sigma, lost_unknown added")


if __name__ == "__main__":
    main()
