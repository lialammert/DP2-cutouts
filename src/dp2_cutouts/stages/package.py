# package.py
# Anna Preto
# AstroDeep Project
# Writes the final dataset: Parquet partitioned by order-5 HEALPix tile (healpix5=<n> folders), files capped in
# number of rows. The image column follows the MultimodalUniverse convention: a list of structs, one per band,
# {band, flux, ivar, mask, scale}, with a boolean mask (True = pixel not to be trusted). The full bits of the
# Rubin mask planes (and the FILLED_SKY bit from masking) are kept separately in image_maskbits.
# The full catalogue, before any cut, is written alongside: it is used for reweighting.

import json
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from ..common import masks

PIXEL_SCALE = 0.2

IMAGE_TYPE = pa.list_(pa.struct([
    ("band", pa.string()),
    ("flux", pa.list_(pa.list_(pa.float32()))),
    ("ivar", pa.list_(pa.list_(pa.float32()))),
    ("mask", pa.list_(pa.list_(pa.bool_()))),
    ("scale", pa.float32()),
]))
MASKBITS_TYPE = pa.list_(pa.list_(pa.list_(pa.int32())))

# Working columns that do not go into the final dataset.
DROP = ["sky_window_x_tract", "sky_window_y_tract"]


def selection(table, apply):
    keep = np.ones(len(table), bool)
    for name in apply:
        col = "in_region" if name == "in_region" else f"{name}_pass"
        keep &= table[col].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return keep


def nested_lists(values, shape):
    # Arrow nested lists built straight from a flat numpy array (no Python objects): each element has the given
    # shape, e.g. (59, 59) gives list<list<value>>. Memory stays that of the pixels.
    arr = pa.array(values)
    for d in reversed(shape):
        count = len(arr) // d
        arr = pa.ListArray.from_arrays(pa.array(np.arange(0, count * d + 1, d, dtype=np.int32)), arr)
    return arr


def image_column(st, bands, n, mmu_mask):
    nb, N = len(bands), len(st)
    f = np.stack([np.asarray(v, np.float32) for v in st["flux"]]) if N else np.zeros((0,), np.float32)
    iv = np.stack([np.asarray(v, np.float32) for v in st["ivar"]]) if N else np.zeros((0,), np.float32)
    mb = np.stack([np.asarray(v, np.int32) for v in st["maskbits"]]) if N else np.zeros((0,), np.int32)
    bad = (mb & mmu_mask) != 0
    fields = list(IMAGE_TYPE.value_type)
    struct = pa.StructArray.from_arrays(
        [pa.array(np.tile(np.asarray(bands, dtype=object), N), pa.string()),
         nested_lists(f.ravel(), (n, n)), nested_lists(iv.ravel(), (n, n)), nested_lists(bad.ravel(), (n, n)),
         pa.array(np.full(N * nb, PIXEL_SCALE, np.float32))], fields=fields)
    images = pa.ListArray.from_arrays(pa.array(np.arange(0, N * nb + 1, nb, dtype=np.int32)), struct)
    maskbits = nested_lists(mb.ravel(), (nb, n, n))
    return images.cast(IMAGE_TYPE), maskbits.cast(MASKBITS_TYPE)


def to_arrow(df):
    # Object columns (redshift density, intervals) as lists of floats; everything else through pyarrow.
    cols, names = [], []
    for c in df.columns:
        s = df[c]
        is_null = [v is None or (np.isscalar(v) and pd.isna(v)) for v in s]
        first = next((v for v, z in zip(s, is_null) if not z), None)
        if s.dtype == object and isinstance(first, (list, tuple, np.ndarray)):
            # Built from one flat numpy array and offsets (no Python float per value).
            parts = [np.asarray(v, float).ravel() for v, z in zip(s, is_null) if not z]
            lens = np.array([0 if z else len(np.ravel(v)) for v, z in zip(s, is_null)], np.int64)
            offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
            values = np.concatenate(parts) if parts else np.zeros(0, float)
            cols.append(pa.ListArray.from_arrays(pa.array(offsets), pa.array(values, pa.float64()),
                                                 type=pa.list_(pa.float64()), mask=pa.array(is_null, pa.bool_())))
        elif s.dtype == object:
            # Pixel columns from the join (booleans or numbers with gaps outside targets).
            cols.append(pa.array([None if z else v for v, z in zip(s, is_null)], from_pandas=True))
        else:
            cols.append(pa.array(s, from_pandas=True))
        names.append(c)
    return pa.Table.from_arrays(cols, names=names)


def write_partitioned(table, base, max_rows, basename="part-{i}.parquet", append=False):
    # append=False replaces the folder; append=True adds files (one batch per patch, distinct basenames).
    if base.exists() and not append:
        shutil.rmtree(base)
    ds.write_dataset(table, base, format="parquet", partitioning=["healpix5"], partitioning_flavor="hive",
                     max_rows_per_file=max_rows, max_rows_per_group=min(max_rows, 1024),
                     basename_template=basename, existing_data_behavior="overwrite_or_ignore")


def final_rows(cfg, table, st, meta):
    # Rows of the final dataset among `table`, with their image columns, as an Arrow table.
    pc = cfg["package"]
    bands, n = meta["bands"], meta["stamp_px"]
    mmu_mask = masks.bitmask(meta["plane_bits"], pc["mmu_mask_planes"])
    keep = selection(table, pc["apply"])
    final = table[keep].drop(columns=[c for c in DROP if c in table.columns]).reset_index(drop=True)
    st = st.set_index("object_id").loc[final["object_id"]].reset_index()
    image, maskbits = image_column(st, bands, n, mmu_mask)
    return final, to_arrow(final).append_column("image", image).append_column("image_maskbits", maskbits)


def write_info(cfg, meta, root, n_catalog, final_columns):
    # final_columns: has_valid_i_moments of the final rows (for the strict selection), or None.
    pc = cfg["package"]
    info = dict(bands=meta["bands"], stamp_px=meta["stamp_px"], pixel_scale=PIXEL_SCALE, plane_bits=meta["plane_bits"],
                filled_bit=masks.FILLED_BIT, mmu_mask_planes=pc["mmu_mask_planes"], cuts_applied=pc["apply"],
                n_catalog=int(n_catalog), n_final=int(len(final_columns)),
                partitioning="healpix5 (HEALPix NESTED, order 5)")
    # Strict selection: the final dataset minus the objects whose isolation rests on copaings' distance fallback
    # (no valid i-band ellipse). Not applied; the column lets analyses filter on it.
    if final_columns.notna().all():
        info["n_final_strict"] = int(final_columns.astype(bool).sum())
        info["strict_selection"] = "final dataset with has_valid_i_moments == True"
    with open(root / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def run(cfg):
    table = pd.read_parquet(cfg.data("30_processed", "table.parquet"))
    st = pd.read_parquet(cfg.data("30_processed", "stamps.parquet"))
    meta = json.load(open(cfg.data("30_processed", "stamps.json")))
    return write(cfg, table, st, meta, cfg.data("40_final", "x").parent)


def write(cfg, table, st, meta, root):
    # table: every catalogue row (with pixel columns for the targets); st: stamp arrays; meta: stamps.json.
    if meta is None:
        raise RuntimeError("no processed stamps to package")
    final, t = final_rows(cfg, table, st, meta)
    root.mkdir(parents=True, exist_ok=True)
    write_partitioned(t, root / "data", int(cfg["package"]["max_rows_per_file"]))
    write_partitioned(to_arrow(table.drop(columns=[c for c in DROP if c in table.columns])), root / "catalog",
                      100_000)
    moments = final["has_valid_i_moments"] if "has_valid_i_moments" in final.columns else pd.Series([None] * len(final))
    write_info(cfg, meta, root, len(table), moments)
    print(f"final dataset: {len(final):,} stamps out of {len(table):,} catalogue objects -> {root}")
    return final
