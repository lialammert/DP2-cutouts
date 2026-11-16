# figures.py
# Anna Preto
# AstroDeep Project
# Figures of the inspection notebooks (notebooks/ml), kept here so that the notebooks stay short and the same
# figure is drawn the same way everywhere. matplotlib only; nothing here imports keras.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def asinh_stretch(img, q=0.1):
    s = np.nanpercentile(np.abs(img), 99.5) or 1.0
    return np.arcsinh(img / (q * s)) / np.arcsinh(1.0 / q)


def default_channel(n_channels):
    # Rubin stamps (6 bands): i; Euclid stamps (VIS first): VIS.
    return 3 if n_channels == 6 else 0


def stamp_grid(columns, titles, n=8, bands=None, size=1.6, row_labels=None):
    # columns: list of arrays (N, H, W[, C]) shown side by side, one example per row; bands: channel per column.
    n = min(n, len(columns[0]))
    fig, axes = plt.subplots(n, len(columns), figsize=(size * len(columns), size * n), squeeze=False)
    for i in range(n):
        for j, col in enumerate(columns):
            img = col[i]
            if img.ndim == 3:
                img = img[..., bands[j] if bands is not None else default_channel(img.shape[-1])]
            ax = axes[i, j]
            ax.imshow(asinh_stretch(img), origin="lower", cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=8)
            if row_labels is not None and j == 0:
                ax.set_ylabel(str(row_labels[i]), fontsize=7)
    fig.tight_layout()
    return fig


def residual_grid(obs, models, names, band, weight=None, n=6, size=1.6):
    # Observation, then for each model: reconstruction and (obs - model), same colour scale per row.
    n = min(n, len(obs))
    ncol = 1 + 2 * len(models)
    fig, axes = plt.subplots(n, ncol, figsize=(size * ncol, size * n), squeeze=False)
    for i in range(n):
        o = obs[i][..., band]
        vmax = np.nanpercentile(np.abs(o), 99.5)
        axes[i, 0].imshow(o, origin="lower", cmap="gray_r", vmin=-0.1 * vmax, vmax=vmax)
        for k, m in enumerate(models):
            r = m[i][..., band]
            axes[i, 1 + 2 * k].imshow(r, origin="lower", cmap="gray_r", vmin=-0.1 * vmax, vmax=vmax)
            res = o - r
            if weight is not None:
                res = res * weight[i][..., band]
            axes[i, 2 + 2 * k].imshow(res, origin="lower", cmap="RdBu_r", vmin=-0.3 * vmax, vmax=0.3 * vmax)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    for k, nm in enumerate(["observed"] + [x for m in names for x in (m, f"obs - {m}")]):
        axes[0, k].set_title(nm, fontsize=8)
    fig.tight_layout()
    return fig


def histories(run_dirs, keys=("total_loss", "latent_loss", "cross_loss", "infonce_loss", "scene_loss")):
    runs = {Path(r).name: pd.read_csv(Path(r) / "history.csv") for r in run_dirs if (Path(r) / "history.csv").exists()}
    keys = [k for k in keys if any(k in h for h in runs.values())]
    fig, axes = plt.subplots(1, max(len(keys), 1), figsize=(4 * max(len(keys), 1), 3), squeeze=False)
    for ax, k in zip(axes[0], keys):
        for name, h in runs.items():
            if k in h:
                ax.plot(h["epoch"], h[k], label=f"{name} train")
            if f"val_{k}" in h:
                ax.plot(h["epoch"], h[f"val_{k}"], "--", label=f"{name} val")
        ax.set_yscale("log")
        ax.set_title(k)
        ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    return fig


def binned_plot(bins, metric_names, by=None):
    bys = [by] if by else list(bins["by"].unique())
    fig, axes = plt.subplots(1, len(bys), figsize=(4 * len(bys), 3), squeeze=False)
    for ax, b in zip(axes[0], bys):
        sub = bins[bins["by"] == b]
        x = 0.5 * (sub["lo"] + sub["hi"])
        for m in metric_names:
            if m in sub:
                ax.plot(x, sub[m], "o-", label=m)
        ax.set_xlabel(b)
        ax.axhline(0, color="k", lw=0.5)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    return fig


def scatter_e(df, a, b, label_a, label_b, lim=0.8):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, q in zip(axes, ("e1", "e2")):
        x, y = df[f"{b}_{q}"].astype(float), df[f"{a}_{q}"].astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.hexbin(x[ok], y[ok], gridsize=60, extent=(-lim, lim, -lim, lim), bins="log", cmap="viridis")
        ax.plot([-lim, lim], [-lim, lim], "w--", lw=0.8)
        ax.set_xlabel(f"{q} {label_b}")
        ax.set_ylabel(f"{q} {label_a}")
    fig.tight_layout()
    return fig


def report_table(report):
    rows = [dict(check=k, status=c["status"]) for k, c in report["checks"].items()]
    return pd.DataFrame(rows)


def load_json(path):
    return json.loads(Path(path).read_text())
