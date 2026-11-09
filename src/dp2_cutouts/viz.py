# viz.py
# Anna Preto
# AstroDeep Project
# Notebook tools: cut funnel, reading the intermediate stamps, display (north up, east left: origin="lower",
# x = column towards west) and ellipse drawing.

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.visualization import make_lupton_rgb
from matplotlib.patches import Circle, Ellipse

# Recipe order. "in_region" only exists in the local demo (the stamp must fit inside the downloaded region).
STEPS = [
    ("cut1", "1 · Euclid: neither point-like nor spurious"),
    ("cut2", "2 · S_b = 0 and one-to-one"),
    ("in_region", "demo · inside the downloaded region"),
    ("cut3", "3 · 29 px margin inside the patch"),
    ("cut4", "4 · no unexposed pixel"),
    ("cut5", "5 · no damaged pixel within k × ellipse"),
    ("cut6", "6 · S/N(i) > 5"),
    ("cut7", "7 · same pixel in all six bands"),
    ("cut8", "8 · |Δe| between apertures ≤ 0.02"),
    ("mask", "masking · |Δe| ≤ 0.02"),
]


def passed(table, step):
    col = "in_region" if step == "in_region" else f"{step}_pass"
    return table[col].astype("boolean").fillna(False).to_numpy(dtype=bool)


def funnel(table, steps=STEPS, start_mask=None):
    keep = np.ones(len(table), bool) if start_mask is None else start_mask.copy()
    rows = [("Rubin catalogue (copaings)", int(keep.sum()), np.nan)]
    for key, label in steps:
        before = keep.sum()
        keep &= passed(table, key)
        rows.append((label, int(keep.sum()), 1 - keep.sum() / before if before else np.nan))
    return pd.DataFrame(rows, columns=["step", "remaining", "fraction removed"])


def stage_mask(table, upto):
    # Targets that passed every step up to and including "upto".
    keep = np.ones(len(table), bool)
    for key, _ in STEPS:
        keep &= passed(table, key)
        if key == upto:
            break
    return keep


def load_stamps(cfg):
    st = pd.read_parquet(cfg.data("30_processed", "stamps.parquet")).set_index("object_id")
    meta = json.load(open(cfg.data("30_processed", "stamps.json")))
    nb, n, nd = len(meta["bands"]), meta["stamp_px"], meta["diag_px"]
    out = {}
    for oid, r in st.iterrows():
        d = dict(flux_raw=np.asarray(r.flux_raw).reshape(nb, n, n), flux=np.asarray(r.flux).reshape(nb, n, n),
                 maskbits=np.asarray(r.maskbits).reshape(nb, n, n),
                 companion_mask=np.asarray(r.companion_mask).reshape(n, n).astype(bool))
        if isinstance(r.get("diag_flux"), np.ndarray):
            d["diag_flux"] = np.asarray(r.diag_flux).reshape(nd, nd)
            d["diag_need"] = np.asarray(r.diag_need).reshape(nd, nd).astype(bool)
        out[oid] = d
    return out, meta


def rgb(flux6, stretch=0.5, Q=8):
    # i, r, g composite (indices 3, 2, 1 in ugrizy), flux in nJy.
    s = 1.0 / 100.0
    return make_lupton_rgb(flux6[3] * s, flux6[2] * s, flux6[1] * s, stretch=stretch, Q=Q)


def show(ax, img, title=None, cmap="gray_r", vmin=None, vmax=None):
    if img.ndim == 3:
        ax.imshow(img, origin="lower", interpolation="nearest")
    else:
        if vmin is None:
            vmin, vmax = np.percentile(img, [1, 99.5])
        ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def draw_ellipse(ax, xc, yc, a, b, theta, k=1.0, **kw):
    # An object without moments (NaN ellipse) is not drawn.
    if not all(np.isfinite(v) for v in (xc, yc, a, b, theta)):
        return
    kw.setdefault("fill", False)
    ax.add_patch(Ellipse((xc, yc), 2 * k * a, 2 * k * b, angle=theta, **kw))


def draw_circle(ax, xc, yc, r, **kw):
    kw.setdefault("fill", False)
    ax.add_patch(Circle((xc, yc), r, **kw))


def overlay(ax, mask, color, alpha=0.5):
    rgba = np.zeros(mask.shape + (4,))
    rgba[mask] = list(plt.matplotlib.colors.to_rgba(color, alpha))
    ax.imshow(rgba, origin="lower", interpolation="nearest")


def stamp_center(row, n=59):
    # Fractional position of the target in its stamp (column, row).
    h = n // 2
    return h + (row.x_tract - np.floor(row.x_tract + 0.5)), h + (row.y_tract - np.floor(row.y_tract + 0.5))


def grid(n, ncols=6, size=2.0):
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()
