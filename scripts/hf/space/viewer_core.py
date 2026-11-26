# viewer_core.py
# Anna Preto
# AstroDeep Project
# Decoding and drawing for the private DP2 x Euclid viewer (app.py): no Gradio here, so that it can be tested.
# Panels share the same angular scale (Rubin 0.2"/px, VIS 0.1"/px, both stamps about 12" wide).

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BANDS = "ugrizy"


def decode(row, info):
    # Images of one parquet row, back in noise units (float32) and the fill mask.
    n_r, n_e = int(info["rubin_px"]), int(info["euclid_px"])
    out = dict(rubin=np.frombuffer(row["rubin"], np.float16).reshape(n_r, n_r, -1).astype(np.float32),
               rubin_sigma=np.asarray(json.loads(row["rubin_sigma"]), np.float32),
               vis=np.frombuffer(row["vis"], np.float16).reshape(n_e, n_e).astype(np.float32),
               fill=np.unpackbits(np.frombuffer(row["fill"], np.uint8), count=n_e * n_e).reshape(n_e, n_e)
               .astype(bool))
    if "vis_masked" in row and row["vis_masked"] is not None and not (isinstance(row["vis_masked"], float)):
        out["vis_masked"] = np.frombuffer(row["vis_masked"], np.float16).reshape(n_e, n_e).astype(np.float32)
    return out


def stretch(x, soft=None, top=None):
    # asinh in noise units, 0..1: linear up to ~soft, logarithmic above. top: the brightest pixel of the central
    # third (the target), so that its core is not clipped; soft: top / 10 (at least 1 sigma), so that the inner
    # structure of bright galaxies stays visible while the noise stays faint.
    if top is None:
        n = x.shape[0]
        c = x[n // 3:n - n // 3, n // 3:n - n // 3]
        top = max(float(np.nanmax(c)) if c.size else 5.0, 5.0)
    soft = soft if soft is not None else max(top / 10.0, 1.0)
    return np.clip(np.arcsinh(x / soft) / np.arcsinh(top / soft), 0, 1)


def rgb(rubin):
    # i, r, g composite (noise units), one common stretch so that colours are kept.
    i, r, g = rubin[..., 3], rubin[..., 2], rubin[..., 1]
    n = i.shape[0]
    top = max(float(np.nanmax(np.stack([i, r, g])[:, n // 3:n - n // 3, n // 3:n - n // 3])), 5.0)
    return np.dstack([stretch(c, None, top) for c in (i, r, g)])


def extent(n, scale):
    h = n * scale / 2
    return (-h, h, -h, h)


def figure(row, info, kind):
    im = decode(row, info)
    sr, se = float(info["rubin_scale_arcsec"]), float(info["euclid_scale_arcsec"])
    n_r, n_e = im["rubin"].shape[0], im["vis"].shape[0]
    panels = [("Rubin gri", "rgb"), ("Rubin i", "i"), ("Euclid VIS", "vis")]
    panels.append(("VIS, neighbours masked" if kind == "d2" else "VIS, other sources", "masked"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    tx = float(row.get("target_rubin_x", 0.0) or 0.0) * sr
    ty = float(row.get("target_rubin_y", 0.0) or 0.0) * sr
    for ax, (title, what) in zip(axes, panels):
        if what == "rgb":
            ax.imshow(rgb(im["rubin"]), origin="lower", extent=extent(n_r, sr), interpolation="nearest")
        elif what == "i":
            ax.imshow(stretch(im["rubin"][..., 3]), origin="lower", cmap="gray_r", extent=extent(n_r, sr),
                      interpolation="nearest")
        else:
            img = im.get("vis_masked", im["vis"]) if what == "masked" else im["vis"]
            ax.imshow(stretch(img), origin="lower", cmap="gray_r", extent=extent(n_e, se), interpolation="nearest")
            if what == "masked" and im["fill"].any():
                yy, xx = (np.arange(n_e) - (n_e - 1) / 2) * se, (np.arange(n_e) - (n_e - 1) / 2) * se
                ax.contour(xx, yy, im["fill"].astype(float), levels=[0.5], colors="red", linewidths=0.8)
        if what in ("rgb", "i"):
            ax.plot([tx], [ty], "+", color="orange", ms=10, mew=1.2)      # the Euclid target on the Rubin stamp
        else:
            ax.plot([0], [0], "+", color="orange", ms=10, mew=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('arcsec', fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def describe(row, kind):
    # The metadata shown next to the figure (images and bytes left out).
    skip = {"rubin", "vis", "vis_masked", "fill", "rubin_sigma"}
    out = {}
    for k, v in row.items():
        if k in skip:
            continue
        if isinstance(v, float):
            v = round(v, 4)
        out[k] = v if not isinstance(v, (np.generic,)) else v.item()
    return out


def select(table, kind, splits, fields, mag_range, selection):
    # Row positions matching the filters.
    m = np.ones(len(table), bool)
    if splits:
        m &= table["split"].isin(splits).to_numpy()
    if fields:
        m &= table["field"].isin(fields).to_numpy()
    if mag_range is not None and "euclid_vis_mag" in table:
        v = table["euclid_vis_mag"].to_numpy(float)
        m &= (v >= mag_range[0]) & (v <= mag_range[1])
    if kind == "d2" and selection in ("selected", "rejected") and "selected" in table:
        s = table["selected"].to_numpy(bool)
        m &= s if selection == "selected" else ~s
    return np.flatnonzero(m)
