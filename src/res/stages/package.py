# package.py
# Anna Preto
# AstroDeep Project
# Empaquetage du jeu fini : Parquet partitionné par tuile HEALPix d'ordre 5 (dossiers healpix5=<n>), blocs bornés
# en nombre de lignes. La colonne image suit la convention de MultimodalUniverse : une liste de structures, une par
# bande, {band, flux, ivar, mask, scale}, avec mask booléen (True = pixel à ne pas croire). Les bits complets des
# plans de masque Rubin (et le bit FILLED_SKY du masquage) sont gardés à part dans image_maskbits.
# Le catalogue complet, avant toute coupure, est écrit à côté : il sert à la repondération.

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

# Colonnes de travail qui ne vont pas dans le jeu fini.
DROP = ["sky_window_x_tract", "sky_window_y_tract"]


def selection(table, apply):
    keep = np.ones(len(table), bool)
    for name in apply:
        col = "in_region" if name == "in_region" else f"{name}_pass"
        keep &= table[col].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return keep


def image_column(st, bands, n, mmu_mask):
    images, maskbits = [], []
    for r in st.itertuples():
        f = np.asarray(r.flux, np.float32).reshape(len(bands), n, n)
        iv = np.asarray(r.ivar, np.float32).reshape(len(bands), n, n)
        mb = np.asarray(r.maskbits, np.int32).reshape(len(bands), n, n)
        bad = (mb & mmu_mask) != 0
        images.append([dict(band=b, flux=f[k].tolist(), ivar=iv[k].tolist(), mask=bad[k].tolist(), scale=PIXEL_SCALE)
                       for k, b in enumerate(bands)])
        maskbits.append(mb.tolist())
    return pa.array(images, type=IMAGE_TYPE), pa.array(maskbits, type=MASKBITS_TYPE)


def to_arrow(df):
    # Colonnes objet (densité de redshift, intervalles) en listes de flottants ; le reste par pyarrow.
    cols, names = [], []
    for c in df.columns:
        s = df[c]
        is_null = [v is None or (np.isscalar(v) and pd.isna(v)) for v in s]
        first = next((v for v, z in zip(s, is_null) if not z), None)
        if s.dtype == object and isinstance(first, (list, tuple, np.ndarray)):
            vals = [None if z else np.asarray(v, float).tolist() for v, z in zip(s, is_null)]
            cols.append(pa.array(vals, type=pa.list_(pa.float64())))
        elif s.dtype == object:
            # Colonnes de pixels issues de la jointure (booléens ou nombres avec des trous hors cibles).
            cols.append(pa.array([None if z else v for v, z in zip(s, is_null)], from_pandas=True))
        else:
            cols.append(pa.array(s, from_pandas=True))
        names.append(c)
    return pa.Table.from_arrays(cols, names=names)


def write_partitioned(table, base, max_rows):
    if base.exists():
        shutil.rmtree(base)
    ds.write_dataset(table, base, format="parquet", partitioning=["healpix5"], partitioning_flavor="hive",
                     max_rows_per_file=max_rows, max_rows_per_group=min(max_rows, 1024),
                     basename_template="part-{i}.parquet")


def run(cfg):
    pc = cfg["package"]
    table = pd.read_parquet(cfg.data("30_processed", "table.parquet"))
    st = pd.read_parquet(cfg.data("30_processed", "stamps.parquet"))
    meta = json.load(open(cfg.data("30_processed", "stamps.json")))
    bands, n = meta["bands"], meta["stamp_px"]
    mmu_mask = masks.bitmask(meta["plane_bits"], pc["mmu_mask_planes"])

    keep = selection(table, pc["apply"])
    final = table[keep].drop(columns=[c for c in DROP if c in table.columns]).reset_index(drop=True)
    st = st.set_index("object_id").loc[final["object_id"]].reset_index()
    image, maskbits = image_column(st, bands, n, mmu_mask)
    t = to_arrow(final).append_column("image", image).append_column("image_maskbits", maskbits)

    root = cfg.data("40_final", "x").parent
    write_partitioned(t, root / "data", int(pc["max_rows_per_file"]))
    write_partitioned(to_arrow(table.drop(columns=[c for c in DROP if c in table.columns])), root / "catalog",
                      100_000)
    info = dict(bands=bands, stamp_px=n, pixel_scale=PIXEL_SCALE, plane_bits=meta["plane_bits"],
                filled_bit=masks.FILLED_BIT, mmu_mask_planes=pc["mmu_mask_planes"], cuts_applied=pc["apply"],
                n_catalog=len(table), n_final=len(final), partitioning="healpix5 (HEALPix NESTED, ordre 5)")
    with open(root / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"jeu fini : {len(final):,} vignettes sur {len(table):,} objets du catalogue -> {root}")
    return final
