# app.py
# Anna Preto
# AstroDeep Project
# Private Hugging Face Space: browse the DP2 x Euclid preview (scripts/hf/export_preview.py), one object at a time,
# Rubin and Euclid side by side at the same angular scale. Reads the private dataset repository named by the Space
# variable DATASET_REPO with the Space secret HF_TOKEN (a read token). Locally: DATASET_DIR=<export folder>.

import json
import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viewer_core as core


def load():
    local = os.environ.get("DATASET_DIR")
    if local:
        root = Path(local)
        files = {k: root / f"{k}.parquet" for k in ("d1", "d2")}
        info = json.loads((root / "info.json").read_text())
    else:
        from huggingface_hub import hf_hub_download
        repo, token = os.environ["DATASET_REPO"], os.environ.get("HF_TOKEN")
        files = {k: hf_hub_download(repo, f"{k}.parquet", repo_type="dataset", token=token) for k in ("d1", "d2")}
        info = json.loads(Path(hf_hub_download(repo, "info.json", repo_type="dataset", token=token)).read_text())
    return {k: pd.read_parquet(f) for k, f in files.items()}, info


TABLES, INFO = load()


def view(kind, rows, pos):
    if len(rows) == 0:
        return None, {"message": "no object matches the filters"}, "0 / 0"
    pos = int(pos) % len(rows)
    row = TABLES[kind].iloc[int(rows[pos])].to_dict()
    fig = core.figure(row, INFO[kind], kind)
    plt.close(fig)
    return fig, core.describe(row, kind), f"{pos + 1} / {len(rows)}"


def apply_filters(kind, splits, fields, mag_lo, mag_hi, selection):
    rows = core.select(TABLES[kind], kind, splits, fields, (mag_lo, mag_hi), selection)
    fig, meta, where = view(kind, rows, 0)
    return rows, 0, fig, meta, where


def step(kind, rows, pos, delta):
    pos = (int(pos) + delta) % max(len(rows), 1)
    return (pos,) + view(kind, rows, pos)


def jump(kind, rows, pos):
    pos = int(np.random.default_rng().integers(max(len(rows), 1)))
    return (pos,) + view(kind, rows, pos)


def find(kind, rows, text):
    try:
        oid = int(text)
    except (TypeError, ValueError):
        return (0,) + view(kind, rows, 0)
    ids = TABLES[kind]["euclid_object_id"].to_numpy()
    hit = np.flatnonzero(ids[np.asarray(rows, int)] == oid) if len(rows) else []
    pos = int(hit[0]) if len(hit) else 0
    return (pos,) + view(kind, rows, pos)


with gr.Blocks(title="DP2 x Euclid viewer") as demo:
    gr.Markdown("### DP2 x Euclid Q1 - private preview\nD1: isolated galaxies. D2: Rubin blends (stamp centred on "
                "the Rubin object, orange + = the Euclid target), Euclid target before / after masking its "
                "neighbours (red). Same angular scale on every panel.")
    with gr.Row():
        kind = gr.Radio(["d1", "d2"], value="d2", label="dataset")
        splits = gr.CheckboxGroup(["train", "val", "test"], value=["train", "val", "test"], label="split")
        fields = gr.CheckboxGroup(["EDF-F", "EDF-S"], value=["EDF-F", "EDF-S"], label="field")
        selection = gr.Radio(["all", "selected", "rejected"], value="all", label="D2 selection")
    with gr.Row():
        mag_lo = gr.Slider(18, 30, value=18, step=0.1, label="VIS mag min")
        mag_hi = gr.Slider(18, 30, value=30, step=0.1, label="VIS mag max")
        apply = gr.Button("apply filters", variant="primary")
    with gr.Row():
        prev, nxt, rnd = gr.Button("previous"), gr.Button("next"), gr.Button("random")
        where = gr.Textbox(label="object", interactive=False, scale=0)
        oid = gr.Textbox(label="go to euclid_object_id")
    plot = gr.Plot()
    meta = gr.JSON(label="metadata")
    rows, pos = gr.State(np.arange(0)), gr.State(0)

    inputs = [kind, splits, fields, mag_lo, mag_hi, selection]
    apply.click(apply_filters, inputs, [rows, pos, plot, meta, where])
    kind.change(apply_filters, inputs, [rows, pos, plot, meta, where])
    demo.load(apply_filters, inputs, [rows, pos, plot, meta, where])
    nxt.click(lambda k, r, p: step(k, r, p, 1), [kind, rows, pos], [pos, plot, meta, where])
    prev.click(lambda k, r, p: step(k, r, p, -1), [kind, rows, pos], [pos, plot, meta, where])
    rnd.click(jump, [kind, rows, pos], [pos, plot, meta, where])
    oid.submit(find, [kind, rows, oid], [pos, plot, meta, where])

if __name__ == "__main__":
    demo.launch()
