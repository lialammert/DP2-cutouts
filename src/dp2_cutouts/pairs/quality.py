# quality.py
# Anna Preto
# AstroDeep Project
# Quality report of a paired dataset (section 4 of the plan): each check returns a status (pass / warn / fail), its
# numbers and the identifiers of the worst cases, so that the problematic examples can be looked at, not only
# counted. Checks never modify the dataset.
#   1 astrometry        catalogue offsets Euclid - Rubin, and offsets of the measured centroids
#   2 invalid values    NaN / inf in fluxes and ivar, examples without a valid pixel
#   3 ivar              negative, NaN, zero where the mask says the pixel is good
#   4 shapes and bands  array shapes, band presence per example
#   5 duplicates        repeated Rubin or Euclid identifiers, near-duplicate positions
#   6 contamination     Euclid neighbours of the target, masked fraction of the Rubin stamp
#   7 distributions     magnitude, S/N, size, ellipticity (numbers here, figures in the notebook)
#   8 resolution        Euclid VIS size against Rubin size; orientation of the two frames (e1, e2 correlation)
#   9 splits            counts per split and neighbours across splits (border leakage)

import json

import numpy as np
import pandas as pd

from . import splits


def _status(fail, warn=False):
    return "fail" if fail else ("warn" if warn else "pass")


def worst(meta, col, k=10, largest=True):
    s = meta[col].astype(float)
    s = s[np.isfinite(s)]
    s = s.nlargest(k) if largest else s.nsmallest(k)
    return [dict(object_id=int(meta.loc[i, "object_id"]), value=float(v)) for i, v in s.items()]


def check_astrometry(meta):
    off = meta["astrometric_offset_arcsec"].astype(float)
    cen = meta["centroid_offset_arcsec"].astype(float)
    q = dict(median_catalogue_offset=float(off.median()), p99_catalogue_offset=float(off.quantile(0.99)),
             mean_west=float(meta["astrometric_west_arcsec"].mean()),
             mean_north=float(meta["astrometric_north_arcsec"].mean()),
             median_centroid_offset=float(cen.median()), p99_centroid_offset=float(cen.quantile(0.99)),
             n_flagged=int(meta["astrometric_flag"].sum()))
    # A systematic shift (mean offset) is a frame problem, not noise: it must stay well below a Rubin pixel.
    sys_shift = float(np.hypot(q["mean_west"], q["mean_north"]))
    q["systematic_shift_arcsec"] = sys_shift
    return dict(status=_status(sys_shift > 0.1 or q["median_centroid_offset"] > 0.2,
                               q["p99_catalogue_offset"] > 0.3), numbers=q,
                worst=worst(meta, "centroid_offset_arcsec"))


def check_invalid(st, sample):
    out = {}
    bad = []
    for name in ("rubin_flux", "rubin_ivar", "euclid_flux", "euclid_ivar"):
        a = st.take(name, sample)
        nonfinite = ~np.isfinite(a)
        out[f"{name}_nonfinite"] = int(nonfinite.sum())
        bad.extend(sample[nonfinite.reshape(len(sample), -1).any(axis=1)].tolist())
    for name in ("rubin_mask", "euclid_mask"):
        m = st.take(name, sample)
        allbad = m.reshape(len(sample), -1).all(axis=1)
        out[f"{name}_examples_fully_masked"] = int(allbad.sum())
        bad.extend(sample[allbad].tolist())
    fail = any(v > 0 for v in out.values())
    return dict(status=_status(fail), numbers=out,
                worst=[dict(row=int(i), object_id=int(st.meta[_id_column(st.meta)].iloc[i])) for i in sorted(set(bad))[:20]])


def check_ivar(st, sample):
    out = {}
    for s in ("rubin", "euclid"):
        iv, m = st.take(f"{s}_ivar", sample), st.take(f"{s}_mask", sample)
        out[f"{s}_negative"] = int((iv < 0).sum())
        out[f"{s}_zero_on_good_pixel"] = int(((iv == 0) & ~m).sum())
        out[f"{s}_positive_on_masked_pixel"] = int(((iv > 0) & m).sum())
        good = iv[(iv > 0) & ~m]
        out[f"{s}_median_sigma"] = float(np.median(1 / np.sqrt(good))) if good.size else np.nan
    fail = out["rubin_negative"] + out["euclid_negative"] > 0
    warn = out["rubin_zero_on_good_pixel"] + out["euclid_zero_on_good_pixel"] > 0
    return dict(status=_status(fail, warn), numbers=out)


def check_shapes(st, cfg):
    arr = st.info["arrays"]
    nr, ne = int(cfg["rubin"]["stamp_px"]), int(cfg["euclid"]["stamp_px"])
    br, be = len(cfg["rubin"]["bands"]), len(cfg["euclid"]["bands"])
    expect = dict(rubin_flux=[nr, nr, br], euclid_flux=[ne, ne, be], euclid_present=[be])
    wrong = {k: arr[k][0] for k, v in expect.items() if arr[k][0] != v}
    pres = st.array("euclid_present")
    per_band = {b: int(np.asarray(pres[:, k]).sum()) for k, b in enumerate(cfg["euclid"]["bands"])}
    return dict(status=_status(bool(wrong)), numbers=dict(wrong_shapes=wrong, euclid_band_presence=per_band,
                                                          n=len(st)))


def check_duplicates(meta):
    dup_r = int(meta["object_id"].duplicated().sum())
    dup_e = int(meta["euclid_object_id"].duplicated().sum())
    xyz = splits.unit_vectors(meta["ra"], meta["dec"])
    from scipy.spatial import cKDTree
    pairs = cKDTree(xyz).query_pairs(2 * np.sin(np.radians(0.2 / 3600) / 2))
    return dict(status=_status(dup_r + dup_e > 0, len(pairs) > 0),
                numbers=dict(duplicate_rubin_ids=dup_r, duplicate_euclid_ids=dup_e, pairs_closer_than_0p2=len(pairs)))


def check_contamination(meta, cfg):
    n = meta["n_euclid_contaminants"].astype(int)
    q = dict(frac_with_contaminant=float((n > 0).mean()),
             median_nearest_euclid_arcsec=float(meta["nearest_euclid_arcsec"].replace(np.inf, np.nan).median()),
             median_masked_fraction=float(meta["masked_fraction"].median()) if "masked_fraction" in meta else np.nan,
             p95_delta_e_mask=float(meta["delta_e_mask"].quantile(0.95)) if "delta_e_mask" in meta else np.nan)
    # Wings of companions sharing the target footprint, among the kept pairs (those above dataset1.
    # max_shared_light_frac are counted in the rejected reasons as shared_footprint_light).
    if "shared_light_frac" in meta:
        s = meta["shared_light_frac"].astype(float)
        q.update(frac_with_shared_companion=float((meta["n_shared_companions"].astype(float) > 0).mean()),
                 shared_light_frac_quantiles={str(p): float(s.quantile(p)) for p in (0.5, 0.9, 0.99)})
    # Contaminants inside 3" are expected (the recipe isolates at the ellipse overlap, not at 3"): this is a warning
    # that tells how many, not a failure.
    return dict(status=_status(False, q["frac_with_contaminant"] > 0.2), numbers=q,
                worst=worst(meta, "nearest_euclid_arcsec", largest=False))


def describe(s):
    s = s.astype(float)
    s = s[np.isfinite(s)]
    if not len(s):
        return {}
    return {k: float(v) for k, v in s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).items()}


def check_distributions(meta):
    cols = ["i_cModel_mag", "euclid_vis_mag", "snr_i", "snr_vis_aper", "sigma_hsm", "sigma_vis_px", "e1", "e2",
            "e1_vis", "e2_vis", "resolution_i", "psf_sigma_i"]
    return dict(status="pass", numbers={c: describe(meta[c]) for c in cols if c in meta})


def check_resolution_and_orientation(meta, pa_offset_deg=0.0):
    ok = meta["hsm_vis_ok"].astype(bool) & np.isfinite(meta["e1"].astype(float)) & (meta["snr_i"] > 20)
    m = meta[ok]
    out = dict(n=int(len(m)))
    if len(m) < 30:
        return dict(status="warn", numbers=out)
    # Size in arcsec: VIS sigma (0.1"/px) against Rubin sigma (0.2"/px).
    out["median_sigma_vis_arcsec"] = float((m["sigma_vis_px"] * 0.1).median())
    out["median_sigma_rubin_arcsec"] = float((m["sigma_hsm"] * 0.2).median())
    # Same galaxy seen in two frames: e1 and e2 must correlate positively. A negative e2 slope means a flipped axis,
    # a negative e1 slope a 90 degree rotation. The slopes are below 1 (different PSF and band), never negative.
    for c in ("e1", "e2"):
        x, y = m[f"{c}_vis"].astype(float), m[c].astype(float)
        out[f"slope_{c}"] = float(np.polyfit(x, y, 1)[0])
        out[f"corr_{c}"] = float(np.corrcoef(x, y)[0, 1])
    fail = out["corr_e1"] < 0.2 or out["corr_e2"] < 0.2
    if "euclid_position_angle" in m:
        out["position_angle"] = pa_check(m, pa_offset_deg)
        # The ellipses built by the code (copaings overlap tests, companion masks, D2) must follow the galaxies.
        used = out["position_angle"].get("used_by_code_median_abs_diff_deg")
        fail = fail or (used is not None and used > PA_MAX_MEDIAN_DIFF_DEG)
    return dict(status=_status(fail), numbers=out)


def angle_diff(a, b):
    # Smallest difference between two orientations (degrees, modulo 180), in [0, 90].
    d = np.abs((np.asarray(a) - np.asarray(b)) % 180.0)
    return np.minimum(d, 180.0 - d)


PA_MAX_MEDIAN_DIFF_DEG = 15.0


def pa_check(m, pa_offset_deg=0.0, e_min=0.2):
    # Orientation of the catalogue ellipse against the orientation measured on the VIS stamp (frame x = column
    # towards west, y = row towards north). A convention that is right gives a median difference of a few degrees;
    # a wrong one gives ~45 (unrelated) or ~90 (rotated). "PA" is euclid_position_angle, the position_angle_true of
    # the copaings Euclid table (which depends on euclid.pa_convention of the stamp pipeline); the code builds its
    # ellipses with theta = 270 - (PA + euclid.pa_offset_deg) (common/ellipse.py), reported as used_by_code.
    e = np.hypot(m["e1_vis"].astype(float), m["e2_vis"].astype(float))
    s = m[(e > e_min) & np.isfinite(m["euclid_position_angle"].astype(float))]
    if len(s) < 20:
        return dict(n=int(len(s)))
    measured = 0.5 * np.degrees(np.arctan2(s["e2_vis"].astype(float), s["e1_vis"].astype(float))) % 180.0
    pa = s["euclid_position_angle"].astype(float).to_numpy()
    conventions = {"copaings: 270 - PA": (270.0 - pa) % 180.0, "-PA": (-pa) % 180.0,
                   "PA + 90": (pa + 90.0) % 180.0, "PA": pa % 180.0}
    used = float(np.median(angle_diff((270.0 - (pa + float(pa_offset_deg))) % 180.0, measured)))
    return dict(n=int(len(s)), used_by_code_median_abs_diff_deg=used, pa_offset_deg=float(pa_offset_deg),
                median_abs_diff_deg={k: float(np.median(angle_diff(v, measured)))
                                                    for k, v in conventions.items()})


def check_splits(meta, cfg):
    counts = meta["split"].value_counts().to_dict()
    patches = meta.groupby("split")[["tract", "patch"]].apply(lambda g: len(g.drop_duplicates())).to_dict()
    cross = splits.cross_split_neighbours(meta["ra"], meta["dec"], meta["split"],
                                          float(cfg["splits"]["leakage_radius_arcsec"]))
    shared = meta.groupby(["tract", "patch"])["split"].nunique()
    return dict(status=_status(bool((shared > 1).any()), cross.mean() > 0.02),
                numbers=dict(counts=counts, patches=patches, frac_cross_split_neighbours=float(cross.mean()),
                             patches_in_two_splits=int((shared > 1).sum())))


def run(cfg, st, name, sample_size=5000, seed=0):
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(st), size=min(sample_size, len(st)), replace=False))
    meta = st.meta
    report = dict(dataset=name, n=len(st), rejected=st.rejected()["reason"].value_counts().to_dict(),
                  checks=dict(
                      astrometry=check_astrometry(meta),
                      invalid_values=check_invalid(st, sample),
                      ivar=check_ivar(st, sample),
                      shapes_and_bands=check_shapes(st, cfg),
                      duplicates=check_duplicates(meta),
                      contamination=check_contamination(meta, cfg),
                      distributions=check_distributions(meta),
                      resolution_and_orientation=check_resolution_and_orientation(
                          meta, float(cfg["euclid"].get("pa_offset_deg", 0.0))),
                      splits=check_splits(meta, cfg)))
    report["status"] = ("fail" if any(c["status"] == "fail" for c in report["checks"].values())
                        else "warn" if any(c["status"] == "warn" for c in report["checks"].values()) else "pass")
    out = st.path / "quality_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"{name}: {report['status']} -> {out}")
    for k, c in report["checks"].items():
        print(f"  {k:28s} {c['status']}")
    return report


def run_d2(cfg, st, sample_size=5000, seed=0):
    # D2: the same pixel checks as D1, plus what is specific to resolved blends (separation test, scenes, partners).
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(st), size=min(sample_size, len(st)), replace=False))
    meta = st.meta
    dup = int(meta["euclid_object_id"].duplicated().sum())
    rej = st.rejected()
    sep_cols = ["sep_delta_e", "sep_centroid_shift_arcsec", "sep_rel_size_change", "sep_rel_flux_change",
                "euclid_masked_fraction_in_target", "partner_sep_arcsec", "sb_rubin", "euclid_vis_mag",
                "partner_vis_mag", "snr_i"]
    checks = dict(
        invalid_values=check_invalid(st, sample),
        ivar=check_ivar(st, sample),
        duplicates=dict(status=_status(dup > 0), numbers=dict(duplicate_euclid_ids=dup)),
        separation=dict(status="pass", numbers={c: describe(meta[c]) for c in sep_cols if c in meta}),
        scenes=dict(status=_status(False, meta["scene_complete"].mean() < 0.3),
                    numbers=dict(frac_scene_complete=float(meta["scene_complete"].mean()),
                                 frac_partner_kept=float(meta["partner_kept"].mean()),
                                 groups=int(meta["group_id"].nunique()))),
        splits=check_splits(meta.assign(ra=meta["euclid_ra"], dec=meta["euclid_dec"]), cfg),
        euclid_side=dict(status="pass", numbers=dict(
            frac_wcs_assumed=float(meta["euclid_wcs_assumed"].mean()),
            separation_reasons=rej.loc[rej["reason"] == "euclid_separation", "detail"].value_counts().to_dict()
            if "detail" in rej else {})),
    )
    report = dict(dataset="d2", n=len(st), rejected=rej["reason"].value_counts().to_dict(), checks=checks)
    report["status"] = ("fail" if any(c["status"] == "fail" for c in checks.values())
                        else "warn" if any(c["status"] == "warn" for c in checks.values()) else "pass")
    out = st.path / "quality_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"d2: {report['status']} -> {out}")
    for k, c in checks.items():
        print(f"  {k:28s} {c['status']}")
    return report


def summary_table(report):
    return pd.DataFrame([dict(check=k, status=c["status"]) for k, c in report["checks"].items()])


def _id_column(meta):
    return "object_id" if "object_id" in meta else "euclid_object_id"
