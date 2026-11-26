# static_site.py
# Anna Preto
# AstroDeep Project
# Turns the preview export (export_preview.py: d1.parquet, d2.parquet, info.json) into a static website for a free,
# PRIVATE Hugging Face static Space (Gradio Spaces need a paid subscription): scripts/hf/static/index.html plus
#   data/info.json                 export info (shapes, bands, pixel scales)
#   data/<d1|d2>.json              layout of one object's binary record, chunk size, metadata rows (ids as strings:
#                                  64-bit ids do not fit in JavaScript numbers)
#   data/<d1|d2>_<k>.bin           objects k*CHUNK .. (k+1)*CHUNK-1, one fixed-size record each:
#                                  rubin float16 (n_r, n_r, bands), vis float16 (n_e, n_e), [vis_masked float16
#                                  (n_e, n_e), D2], fill packbits (n_e * n_e bits), zero padding to a multiple of 8
# The page fetches only the chunks it shows. DP2 pixels: only for a PRIVATE Space.
# Usage: python scripts/hf/static_site.py <export dir> <site dir>   (upload_preview.py calls it)

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 100
IMAGES = ("rubin", "vis", "vis_masked", "fill")
HERE = Path(__file__).resolve().parent

SPACE_README = """---
title: DP2 x Euclid viewer
emoji: 🔭
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
---

# DP2 x Euclid viewer - PRIVATE

Static viewer of the private preview of the DP2-cutouts project (Rubin DP2 x Euclid Q1). **Shows Rubin DP2 pixels:
keep this Space private (Rubin data-rights holders only).** Built by scripts/hf/static_site.py of the DP2-cutouts
repository.
"""


def layout(info, name):
    n_r, n_e, nb = int(info["rubin_px"]), int(info["euclid_px"]), len(info["rubin_bands"])
    parts = [("rubin", n_r * n_r * nb * 2), ("vis", n_e * n_e * 2)]
    if name == "d2":
        parts.append(("vis_masked", n_e * n_e * 2))
    parts.append(("fill", math.ceil(n_e * n_e / 8)))
    out, pos = {}, 0
    for k, size in parts:
        out[k] = [pos, size]
        pos += size
    return out, int(math.ceil(pos / 8) * 8)


def record(row, lay, size):
    buf = bytearray(size)
    for k, (pos, n) in lay.items():
        b = bytes(row[k])
        if len(b) != n:
            raise ValueError(f"{k}: {len(b)} bytes, expected {n}")
        buf[pos:pos + n] = b
    return bytes(buf)


def plain(v):
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 5)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def rows_json(t):
    rows = []
    for rec in t.drop(columns=[c for c in IMAGES if c in t]).to_dict("records"):
        out = {}
        for k, v in rec.items():
            v = plain(v)
            if k.endswith("_id") or k == "objectId":
                v = None if v is None else str(v)
            elif k == "rubin_sigma" and isinstance(v, str):
                v = [round(x, 4) for x in json.loads(v)]
            out[k] = v
        rows.append(out)
    return rows


def build_site(export_dir, site_dir):
    export_dir, site_dir = Path(export_dir), Path(site_dir)
    info = json.loads((export_dir / "info.json").read_text())
    if site_dir.exists():
        shutil.rmtree(site_dir)
    (site_dir / "data").mkdir(parents=True)
    for name in ("d1", "d2"):
        t = pd.read_parquet(export_dir / f"{name}.parquet")
        lay, size = layout(info[name], name)
        for c in range(0, len(t), CHUNK):
            with open(site_dir / "data" / f"{name}_{c // CHUNK:03d}.bin", "wb") as f:
                for _, row in t.iloc[c:c + CHUNK].iterrows():
                    f.write(record(row, lay, size))
        (site_dir / "data" / f"{name}.json").write_text(json.dumps(
            dict(layout=lay, record_bytes=size, chunk=CHUNK, rows=rows_json(t)), separators=(",", ":")))
    (site_dir / "data" / "info.json").write_text(json.dumps(info, indent=1, default=str))
    shutil.copy(HERE / "static" / "index.html", site_dir / "index.html")
    (site_dir / "README.md").write_text(SPACE_README, encoding="utf-8")
    return site_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("site_dir")
    args = ap.parse_args()
    site = build_site(args.export_dir, args.site_dir)
    size = sum(f.stat().st_size for f in site.rglob("*") if f.is_file()) / 1e6
    print(f"-> {site} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
