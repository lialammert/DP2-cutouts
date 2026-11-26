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
#   rubin_raw    D1 / rejected: the raw Rubin stamp (neighbours not yet replaced by sky), float16 in noise units
#   need_rubin, need_euclid  D1 / rejected: packed bits (n_r, n_r), the pipeline's neighbour mask on the Rubin stamp
#                split by origin (d1_views.split_mask): Rubin detection footprints / Euclid-only neighbours
# d1r.parquet: objects the cuts rejected (--n-rejected per reason, in --rejected-patches random patches), same images
# as D1 and the reason (first pipeline cut failed, or D1's own reason). Pixels are read from the patch folders of the
# stamp pipeline and the Euclid MMU mirror (d1_views.py): run it as a CPU job.
#   selected     D2: kept by deblend.d2_selection (strict before/after test, VIS magnitude within D1, D1's cut 1);
#                with the target's Euclid point-like / spurious flags when d2/euclid_flags.parquet exists
# The holdout split is left out. Reads the per-patch shards (light enough for a login node), the merged dataset only
# when the shards are gone.
# DP2 pixels: the output must only go to a PRIVATE repository (Rubin data rights).
# Usage: python scripts/hf/export_preview.py <ML config> <out dir> [--n-d1 1500] [--n-d2 2500] [--seed 0]
#            [--n-rejected 80] [--rejected-patches 60] [--no-pixels] [--only-rejected]

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dp2_cutouts.mlconfig import MLConfig
from dp2_cutouts.pairs.store import Store, shard_done

HERE = Path(__file__).resolve().parent

FIELD_DEC_SPLIT = -40.0          # EDF-F around -28 deg, EDF-S around -48 deg
META = {"d1": ["object_id", "objectId", "euclid_object_id", "split", "tract", "patch", "ra", "dec", "euclid_ra",
               "euclid_dec", "i_cModel_mag", "euclid_vis_mag", "snr_i", "snr_vis_aper", "sigma_vis_px", "e1_vis",
               "e2_vis", "n_euclid_contaminants", "nearest_euclid_arcsec", "euclid_fill_fraction",
               "centroid_offset_arcsec", "masked_fraction", "masked_fraction_in_ellipse", "delta_e_mask",
               "halo_ring_z", "companion_light_ratio"],
        "d2": ["group_id", "euclid_object_id", "rubin_object_id", "split", "tract", "patch", "euclid_ra",
               "euclid_dec", "rubin_ra", "rubin_dec", "euclid_vis_mag", "rubin_i_cModel_mag", "sb_rubin",
               "n_euclid_in_group", "partner_vis_mag", "partner_sep_arcsec", "sep_delta_e",
               "sep_centroid_shift_arcsec", "sep_rel_size_change", "sep_rel_flux_change", "hsm_vis_ok", "snr_vis",
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


def bits(m):
    return np.packbits(np.asarray(m, bool)).tobytes()


def rubin_views(rec, raw, need, n_r, nb):
    # Raw Rubin stamp and the neighbour mask split by origin (zeros and has_* False when a piece is missing).
    from d1_views import split_mask
    rec["has_raw"] = raw is not None
    if raw is not None:
        rs = [noise(1.0 / np.where(raw["var"][..., b] > 0, raw["var"][..., b], np.inf), np.ones((n_r, n_r), bool))
              for b in range(nb)]
        rec["rubin_raw"] = f16(raw["flux"], np.asarray(rs, np.float32))
        rec["rubin_raw_sigma"] = json.dumps(rs)
    else:
        rec["rubin_raw"] = np.zeros((n_r, n_r, nb), np.float16).tobytes()
    rec["has_need"] = need is not None and raw is not None
    if rec["has_need"]:
        nr_, ne_ = split_mask(need, raw["rubin_fp"])
    else:
        nr_ = ne_ = np.zeros((n_r, n_r), bool)
    rec["need_rubin"], rec["need_euclid"] = bits(nr_), bits(ne_)
    rec["euclid_only_px"] = int(ne_.sum())
    rec["euclid_only_mask"] = bool(ne_.sum() >= 4)
    return rec


def patch_index(cfg):
    from dp2_cutouts.pipeline.patches import load_patches
    pcfg = cfg.pipeline()
    return pcfg, {p["name"]: p for p in load_patches(pcfg)}


def with_views():
    import sys
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))


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


def export(cfg, name, n, rng, d1_meta=None, pixels=True):
    rb = list(cfg["rubin"]["bands"])
    vis = list(cfg["euclid"]["bands"]).index("VIS")
    picks, total = sample(cfg, name, n, rng)
    recs = []
    flags = {}
    fpath = cfg.dataset_dir("d2") / "euclid_flags.parquet"
    if name == "d2" and fpath.exists():           # Euclid flags of the targets (dp2-ml d2-flags), shown with the metadata
        f = pd.read_parquet(fpath, columns=["euclid_object_id", "euclid_point_like_flag", "euclid_spurious_flag"])
        flags = {int(i): (pl, sp) for i, pl, sp in f.itertuples(index=False)}
    pp_pixels = name == "d1" and pixels
    if name == "d1":
        with_views()
    if pp_pixels:
        from d1_views import PatchPixels
        pcfg, by_name = patch_index(cfg)
    for p, rows in picks.items():
        st = Store(p)
        rows = np.asarray(rows)
        pix, stamp = None, {}
        if pp_pixels and Path(p).name in by_name:
            pix = PatchPixels(pcfg, by_name[Path(p).name])
            stamp = pix.stamps(st.meta["object_id"].iloc[rows].astype("int64").tolist())
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
            if name == "d1":
                rraw = need = None
                if pix is not None:
                    rraw = pix.raw(meta.at[j, "ra"], meta.at[j, "dec"])
                    need = stamp.get(int(meta.at[j, "object_id"]), {}).get("need")
                rubin_views(rec, rraw, need, rf.shape[1], len(rb))
                rec["has_final"] = rec["has_vis"] = True
            if name == "d2":
                rec["vis_masked"] = f16(ef[j], vs)
                rec["target_rubin_x"] = -float(meta.at[j, "rc_offset_x_px"])
                rec["target_rubin_y"] = -float(meta.at[j, "rc_offset_y_px"])
                rec["selected"] = bool(keep[j])
                if flags:
                    pl, sp = flags.get(int(meta.at[j, "euclid_object_id"]), (None, None))
                    rec["euclid_point_like_flag"] = None if pd.isna(pl) else int(pl)
                    rec["euclid_spurious_flag"] = None if pd.isna(sp) else int(sp)
            recs.append(rec)
        if pix is not None:
            pix.close()
        print(f"{name} {Path(p).name}: {len(rows)} objects", flush=True)
    info = dict(n_total_non_holdout=total, n_exported=len(recs), rubin_px=int(rf.shape[1]),
                euclid_px=int(ef.shape[1]), rubin_bands=rb, rubin_scale_arcsec=0.2,
                euclid_scale_arcsec=float(cfg["euclid"]["pixel_scale"]))
    if name == "d2":
        info["vis_mag_max"] = lim
    return pd.DataFrame(recs), info


def export_rejected(cfg, n_per_reason, n_patches, rng):
    # Objects the cuts rejected, up to n_per_reason per reason, drawn in n_patches random patches (the final stamps
    # of a patch are read whole: few patches keep the reading short). Same images as D1.
    with_views()
    from d1_views import REASONS, PatchPixels, rejected_candidates, vis_cutouts
    pcfg, by_name = patch_index(cfg)
    rb = list(cfg["rubin"]["bands"])
    n_r, n_e, nb = int(cfg["rubin"]["stamp_px"]), int(cfg["euclid"]["stamp_px"]), len(rb)
    names = sorted(by_name)
    chosen = [by_name[names[k]] for k in rng.permutation(len(names))[:n_patches]]
    cand = rejected_candidates(cfg, pcfg, chosen, cfg.dataset_dir("d1") / "shards")
    if not len(cand):
        return pd.DataFrame(), {}
    cand = pd.concat([g.iloc[rng.permutation(len(g))[:n_per_reason]] for _, g in cand.groupby("reason")],
                     ignore_index=True)
    print("rejected, per reason:", cand["reason"].value_counts().to_dict(), flush=True)
    recs = []
    for name, g in cand.groupby("patch_name"):
        pix = PatchPixels(pcfg, by_name[name])
        stamp = pix.stamps(g["object_id"].astype("int64").tolist(), final=True)
        try:
            vis_cut = vis_cutouts(cfg, pcfg, pix.case, g)
        except Exception as err:  # noqa: BLE001 (a Euclid read failure only leaves the VIS panels empty)
            print(f"{name}: no VIS ({err})", flush=True)
            vis_cut = {}
        for r in g.to_dict("records"):
            rec = {}
            for c, v in r.items():
                if c == "in_region" or c.endswith("_pass"):
                    continue
                if v is None or v is pd.NA:
                    rec[c] = None
                elif isinstance(v, (np.integer, int)) and not isinstance(v, (bool, np.bool_)):
                    rec[c] = int(v)
                elif isinstance(v, (np.floating, float)):
                    rec[c] = float(v) if np.isfinite(v) else None
                else:
                    rec[c] = v
            rec["reason_label"] = REASONS.get(r["reason"], r["reason"])
            rec["split"] = "rejected"
            dec = r.get("euclid_dec")
            dec = float(r["dec"]) if dec is None or pd.isna(dec) else float(dec)
            rec["field"] = "EDF-S" if dec < FIELD_DEC_SPLIT else "EDF-F"
            s_ = stamp.get(int(r["object_id"]))
            rec["has_final"] = s_ is not None
            if s_ is not None:
                rs = [noise(s_["ivar"][..., b], np.ones((n_r, n_r), bool)) for b in range(nb)]
                rec["rubin"] = f16(s_["flux"], np.asarray(rs, np.float32))
                rec["rubin_sigma"] = json.dumps(rs)
            else:
                rec["rubin"] = np.zeros((n_r, n_r, nb), np.float16).tobytes()
            rubin_views(rec, pix.raw(r["ra"], r["dec"]), s_["need"] if s_ is not None else None, n_r, nb)
            eid = r.get("euclid_object_id")
            v = vis_cut.get(int(eid)) if eid is not None and not pd.isna(eid) else None
            rec["has_vis"] = v is not None
            if v is not None:
                vf, vi, vm, fill = v
                vs = noise(vi, ~vm)
                rec["vis"], rec["vis_sigma"], rec["fill"] = f16(vf, vs), vs, bits(fill)
            else:
                rec["vis"] = np.zeros((n_e, n_e), np.float16).tobytes()
                rec["fill"] = bits(np.zeros((n_e, n_e), bool))
            recs.append(rec)
        pix.close()
        print(f"d1r {name}: {len(g)} objects", flush=True)
    info = dict(n_exported=len(recs), rubin_px=n_r, euclid_px=n_e, rubin_bands=rb, rubin_scale_arcsec=0.2,
                euclid_scale_arcsec=float(cfg["euclid"]["pixel_scale"]), reasons=REASONS)
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
- `d1r.parquet`: objects the cuts rejected, with the reason (first cut failed).
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
    ap.add_argument("--n-rejected", type=int, default=80, help="rejected objects per reason (0: none)")
    ap.add_argument("--rejected-patches", type=int, default=60)
    ap.add_argument("--no-pixels", action="store_true", help="D1 without raw stamps and split masks, no rejected")
    ap.add_argument("--only-rejected", action="store_true",
                    help="keep the D1 / D2 already exported in <out dir>, (re)do the rejected objects only")
    args = ap.parse_args()
    cfg = MLConfig.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.only_rejected:
        info = json.loads((out / "info.json").read_text())
    else:
        rng = np.random.default_rng(args.seed)
        d1, i1 = export(cfg, "d1", args.n_d1, rng, pixels=not args.no_pixels)
        d1_meta = pd.concat([m for m in (pd.read_parquet(Path(p) / "meta.parquet", columns=["split", "euclid_vis_mag"])
                                         if n_rows(p) else None for p in sources(cfg, "d1")) if m is not None])
        d2, i2 = export(cfg, "d2", args.n_d2, rng, d1_meta)
        d1.to_parquet(out / "d1.parquet", index=False)
        d2.to_parquet(out / "d2.parquet", index=False)
        info = dict(d1=i1, d2=i2)
        # Written now: the rejected objects are the slow part and a walltime may stop them.
        (out / "info.json").write_text(json.dumps(info, indent=2, default=str))
        (out / "README.md").write_text(CARD)
        print(f"-> {out}: d1 {len(d1)}, d2 {len(d2)} objects ({int(d2['selected'].sum())} selected)", flush=True)
    if args.n_rejected and not args.no_pixels:
        d1r, ir = export_rejected(cfg, args.n_rejected, args.rejected_patches, np.random.default_rng(args.seed + 1))
        if len(d1r):
            d1r.to_parquet(out / "d1r.parquet", index=False)
            info["d1r"] = ir
            (out / "info.json").write_text(json.dumps(info, indent=2, default=str))
            print(f"-> {out}: d1r {len(d1r)} rejected objects", flush=True)
    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
    print(f"-> {out}: {size:.0f} MB")


if __name__ == "__main__":
    main()
