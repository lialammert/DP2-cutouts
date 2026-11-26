# static_site.py
# Anna Preto
# AstroDeep Project
# Turns the preview export (export_preview.py: d1.parquet, d2.parquet, info.json) into a static website for a free,
# PRIVATE Hugging Face static Space (Gradio Spaces need a paid subscription): scripts/hf/static/index.html plus
#   data/info.json                 export info (shapes, bands, pixel scales)
#   data/<name>.json               layout of one object's binary record, chunk size, metadata rows (ids as strings:
#                                  64-bit ids do not fit in JavaScript numbers); name = d1, d2, d1r (rejected)
#   data/<name>_<k>.bin            objects k*CHUNK .. (k+1)*CHUNK-1, one fixed-size record each: every byte column
#                                  of the export (rubin, rubin_raw, vis, vis_masked: float16; fill, need_rubin,
#                                  need_euclid: packbits) at the offset the layout gives, zero padding to a multiple
#                                  of 8 (index.html decodes them by name)
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


def byte_columns(t):
    return [c for c in t.columns if len(t) and isinstance(t[c].iloc[0], (bytes, bytearray))]


def layout(t):
    # Offset and size of every byte column in a record (sizes must not vary between rows); float16 blocks start on
    # even offsets since every size before them is even or they come first (fill-like bit blocks go last).
    cols = byte_columns(t)
    sizes = {c: {len(b) for b in t[c]} for c in cols}
    bad = {c: v for c, v in sizes.items() if len(v) != 1}
    if bad:
        raise ValueError(f"byte columns of varying size: {bad}")
    order = sorted(cols, key=lambda c: (next(iter(sizes[c])) % 2, cols.index(c)))
    out, pos = {}, 0
    for c in order:
        size = next(iter(sizes[c]))
        out[c] = [pos, size]
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
    for rec in t.drop(columns=byte_columns(t)).to_dict("records"):
        out = {}
        for k, v in rec.items():
            v = plain(v)
            if k.endswith("_id") or k == "objectId":
                v = None if v is None else str(v)
            elif k in ("rubin_sigma", "rubin_raw_sigma") and isinstance(v, str):
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
    for name in ("d1", "d2", "d1r"):
        if name not in info or not (export_dir / f"{name}.parquet").exists():
            continue
        t = pd.read_parquet(export_dir / f"{name}.parquet")
        lay, size = layout(t)
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
