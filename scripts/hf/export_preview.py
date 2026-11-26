# export_preview.py
# Anna Preto
# AstroDeep Project
# A small random sample of D1 and D2 for the private Hugging Face viewer (scripts/hf/space): one parquet file per
# dataset, one row per object, the stamps as compact bytes plus the metadata worth looking at.
#   rubin        float16 (n_r, n_r, 6), ugrizy, in units of each band's pixel noise; rubin_sigma: that noise (nJy)
#                D1: the stamp of the pair; D2: the blend centred on the Rubin object (the deblenders' input)
#   vis          float16 (n_e, n_e), Euclid VIS in units of its pixel noise; vis_sigma: that noise (uJy)
#                D2: before masking (euclid_flux_unmasked)
#   vis_masked   D2 only: VIS after masking (euclid_flux, the teacher's E_c)
#   fill         packed bits (n_e, n_e): Euclid pixels of other sources (euclid_fill)
#   target_rubin_x/y  D2: position of the Euclid target on the Rubin stamp (pixels from its centre)
#   selected     D2: kept by deblend.d2_selection (strict before/after test, VIS magnitude within D1)
# The holdout split is left out. Reads the per-patch shards (light enough for a login node), the merged dataset only
# when the shards are gone.
# DP2 pixels: the output must only go to a PRIVATE repository (Rubin data rights).
# Usage: python scripts/hf/export_preview.py <ML config> <out dir> [--n-d1 1500] [--n-d2 2500] [--seed 0]

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dp2_cutouts.mlconfig import MLConfig
from dp2_cutouts.pairs.store import Store, shard_done

FIELD_DEC_SPLIT = -40.0          # EDF-F around -28 deg, EDF-S around -48 deg
META = {"d1": ["object_id", "objectId", "euclid_object_id", "split", "tract", "patch", "ra", "dec", "euclid_ra",
               "euclid_dec", "i_cModel_mag", "euclid_vis_mag", "snr_i", "snr_vis_aper", "sigma_vis_px", "e1_vis",
               "e2_vis", "n_euclid_contaminants", "nearest_euclid_arcsec", "euclid_fill_fraction",
               "centroid_offset_arcsec"],
        "d2": ["group_id", "euclid_object_id", "rubin_object_id", "split", "tract", "patch", "euclid_ra",
               "euclid_dec", "rubin_ra", "rubin_dec", "euclid_vis_mag", "rubin_i_cModel_mag", "sb_rubin",
               "n_euclid_in_group", "partner_vis_mag", "partner_sep_arcsec", "sep_delta_e",
               "sep_centroid_shift_arcsec", "sep_rel_size_change", "sep_rel_flux_change", "snr_vis",
               "halo_ring_z", "euclid_masked_fraction", "rc_offset_x_px", "rc_offset_y_px"]}


def sources(cfg, name):
    # The finished shards (one per patch: small enough for a login node, exact 64-bit ids), or the merged dataset
    # when the shards are gone.
    d = cfg.dataset_dir(name)
    shards = [s for s in sorted((d / "shards").glob("*")) if s.is_dir() and shard_done(s)]
    return shards or ([d] if (d / "info.json").exists() else [])


def n_rows(p):
    return int(json.loads((Path(p) / "info.json").read_text())["n"])


def noise(ivar, ok):
    v = ivar[ok & (ivar > 0)]
    return float(np.median(1.0 / np.sqrt(v))) if v.size else 1.0


def f16(x, s):
    return np.clip(np.nan_to_num(x / s), -6e4, 6e4).astype(np.float16).tobytes()


def d2_selected(cfg, meta, d1_meta):
    # Same rule as deblend.data.d2_keep (repeated here so that this export does not import keras).
    sel = cfg["deblend"].get("d2_selection") or {}
    keep = np.ones(len(meta), bool)
    for col, key in (("sep_delta_e", "max_delta_e"), ("sep_centroid_shift_arcsec", "max_centroid_shift_arcsec"),
                     ("sep_rel_size_change", "max_rel_size_change"), ("sep_rel_flux_change", "max_rel_flux_change")):
        lim = (sel.get("before_after") or {}).get(key)
        if lim is not None:
            keep &= meta[col].to_numpy(float) < float(lim)
    lim = sel.get("vis_mag_max")
    if lim == "auto":
        v = d1_meta.loc[d1_meta["split"] == "train", "euclid_vis_mag"].to_numpy(float)
        lim = float(np.quantile(v[np.isfinite(v)], float(sel.get("vis_mag_quantile", 0.95))))
    if lim is not None:
        keep &= meta["euclid_vis_mag"].to_numpy(float) <= float(lim)
    if sel.get("euclid_cut1"):
        from dp2_cutouts.pairs import euclid_flags
        keep &= euclid_flags.cut1_for(cfg, meta["euclid_object_id"])
    return keep, lim


def sample(cfg, name, n, rng):
    # (store, row) pairs drawn uniformly over the non-holdout objects of all sources.
    pairs = []
    for p in sources(cfg, name):
        if not n_rows(p):
            continue
        split = pd.read_parquet(Path(p) / "meta.parquet", columns=["split"])["split"].to_numpy()
        pairs += [(p, int(r)) for r in np.flatnonzero(split != "holdout")]
    if not pairs:
        raise SystemExit(f"no {name} object found under {cfg.dataset_dir(name)}")
    pick = rng.permutation(len(pairs))[:n]
    out = {}
    for k in sorted(pick):
        p, r = pairs[k]
        out.setdefault(p, []).append(r)
    return out, len(pairs)


def export(cfg, name, n, rng, d1_meta=None):
    rb = list(cfg["rubin"]["bands"])
    vis = list(cfg["euclid"]["bands"]).index("VIS")
    picks, total = sample(cfg, name, n, rng)
    recs = []
    for p, rows in picks.items():
        st = Store(p)
        rows = np.asarray(rows)
        rk = "rubin_flux_rc" if name == "d2" else "rubin_flux"
        ik = "rubin_ivar_rc" if name == "d2" else "rubin_ivar"
        mk = "rubin_mask_rc" if name == "d2" else "rubin_mask"
        rf, ri, rm = st.take(rk, rows), st.take(ik, rows), st.take(mk, rows)
        ef, ei, em = st.take("euclid_flux", rows)[..., vis], st.take("euclid_ivar", rows)[..., vis], \
            st.take("euclid_mask", rows)[..., vis]
        raw = st.take("euclid_flux_unmasked", rows)[..., vis] if name == "d2" else ef
        fill = st.take("euclid_fill", rows) if "euclid_fill" in st.names() else np.zeros(ef.shape, bool)
        meta = st.meta.iloc[rows].reset_index(drop=True)
        if name == "d2":
            keep, lim = d2_selected(cfg, meta, d1_meta)
        for j in range(len(rows)):
            rs = [noise(ri[j, ..., b], ~rm[j, ..., b]) for b in range(len(rb))]
            vs = noise(ei[j], ~em[j])
            rec = {c: meta.at[j, c] for c in META[name] if c in meta}
            for c, v in list(rec.items()):
                if isinstance(v, (np.integer,)):
                    rec[c] = int(v)
                elif isinstance(v, (np.floating,)):
                    rec[c] = float(v)
            dec = rec.get("euclid_dec", rec.get("dec", np.nan))
            rec["field"] = "EDF-S" if np.isfinite(dec) and dec < FIELD_DEC_SPLIT else "EDF-F"
            rec["patch_name"] = Path(p).name if Path(p).parent.name == "shards" else ""
            rec["rubin"] = f16(rf[j], np.asarray(rs, np.float32))
            rec["rubin_sigma"] = json.dumps(rs)
            rec["vis"] = f16(raw[j], vs)
            rec["vis_sigma"] = vs
            rec["fill"] = np.packbits(fill[j]).tobytes()
            if name == "d2":
                rec["vis_masked"] = f16(ef[j], vs)
                rec["target_rubin_x"] = -float(meta.at[j, "rc_offset_x_px"])
                rec["target_rubin_y"] = -float(meta.at[j, "rc_offset_y_px"])
                rec["selected"] = bool(keep[j])
            recs.append(rec)
        print(f"{name} {Path(p).name}: {len(rows)} objects", flush=True)
    info = dict(n_total_non_holdout=total, n_exported=len(recs), rubin_px=int(rf.shape[1]),
                euclid_px=int(ef.shape[1]), rubin_bands=rb, rubin_scale_arcsec=0.2,
                euclid_scale_arcsec=float(cfg["euclid"]["pixel_scale"]))
    if name == "d2":
        info["vis_mag_max"] = lim
    return pd.DataFrame(recs), info


CARD = """---
license: other
license_name: rubin-dp2-data-rights
pretty_name: DP2 x Euclid Q1 preview (private)
viewer: false
---
# DP2 x Euclid Q1 preview - PRIVATE

Random sample of the paired datasets of the DP2-cutouts project (Rubin DP2 x Euclid Q1, EDF-F + EDF-S), for
inspection with the companion private Space only. **Contains Rubin DP2 pixels: Rubin data-rights holders only, never
make this repository public.**

- `d1.parquet`: isolated galaxies, Rubin stamp and Euclid VIS cutout of the same object.
- `d2.parquet`: Rubin blends (stamp centred on the Rubin object), Euclid VIS of the target before / after masking its
  neighbours, `selected` = kept by the campaign's D2 selection.
- `info.json`: shapes, bands, pixel scales, counts.

Images are float16 bytes in units of each stamp's pixel noise (`*_sigma` gives that noise: Rubin nJy, VIS uJy);
`fill` is a packed boolean mask (numpy packbits) of the pixels belonging to other Euclid sources.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--n-d1", type=int, default=1500)
    ap.add_argument("--n-d2", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = MLConfig.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    d1, i1 = export(cfg, "d1", args.n_d1, rng)
    d1_meta = pd.concat([m for m in (pd.read_parquet(Path(p) / "meta.parquet", columns=["split", "euclid_vis_mag"])
                                     if n_rows(p) else None for p in sources(cfg, "d1")) if m is not None])
    d2, i2 = export(cfg, "d2", args.n_d2, rng, d1_meta)
    d1.to_parquet(out / "d1.parquet", index=False)
    d2.to_parquet(out / "d2.parquet", index=False)
    (out / "info.json").write_text(json.dumps(dict(d1=i1, d2=i2), indent=2, default=str))
    (out / "README.md").write_text(CARD)
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"-> {out}: d1 {len(d1)}, d2 {len(d2)} objects ({int(d2['selected'].sum())} selected), {size:.0f} MB")


if __name__ == "__main__":
    main()
