# stamps.py
# Anna Preto
# AstroDeep Project
# Cuts the stamps from the six images of a region, applies the pixel cuts, masks companions and
# fills them with empty sky, and takes the masking decision. Each cut records its continuous quantity next to its
# condition; no target is removed here.
#
# Two things not to get wrong:
# - the WCS returns (x, y) = (column, row), whereas an array is indexed [row, column];
# - astropy returns 0-based positions (the first FITS pixel is 1): nothing is added or subtracted.
# The integer centre of a stamp is floor(x + 0.5); the fractional position is kept for measurements.

import json

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, label

from ..common import masks
from ..common.ellipse import (FWHM2SIGMA, euclid_ellipse, moments_to_ellipse, psf_ellipse,
                              radius_map)
from ..common.shapes import adaptive_moments, circular_weight, delta_e
from ..sources.rubin_soda import tract_origin
from .catalog_table import copaings_dir
from ..config import RUBIN_DIR, TABLES_DIR

NJY_ZEROPOINT = 31.4  # AB magnitude of a 1 nJy flux


# ----------------------------------------------------------------------------- images
def load_images(cfg, tract=None, patch=None):
    if cfg["pixels"].get("source", "soda") == "butler":
        from ..sources import rubin_butler
        return rubin_butler.load_patch(cfg, tract, patch)
    
    return _load_images_soda(cfg)


def _load_images_soda(cfg):
    img_dir = cfg.data(RUBIN_DIR, "images", "x").parent
    meta = json.load(open(img_dir / "images.json"))
    flux, var, mask, wcs, origins, bits = [], [], [], [], [], []
    for b in cfg["bands"]:
        with fits.open(img_dir / meta["files"][b]) as h:
            flux.append(h["IMAGE"].data.astype(np.float32))
            var.append(h["VARIANCE"].data.astype(np.float32))
            mask.append(h["MASK"].data.astype(np.int32))
            wcs.append(WCS(h["IMAGE"].header))
            origins.append(tract_origin(h["IMAGE"].header))
            bits.append(masks.plane_bits(h["MASK"].header))
    if len({f.shape for f in flux}) != 1 or len(set(origins)) != 1 or any(bt != bits[0] for bt in bits):
        raise RuntimeError("the six images do not share the same pixel grid or the same mask planes")
    mask = np.stack(mask)
    # Pixels detected in at least one band: a companion may be detected in some cfg["bands"] only.
    detected = np.any((mask & masks.bitmask(bits[0], ["DETECTED"])) != 0, axis=0)
    return dict(flux=np.stack(flux), var=np.stack(var), mask=mask, detected=detected, wcs=wcs, x0=origins[0][0],
                y0=origins[0][1], bits=bits[0], meta=meta)


# ----------------------------------------------------------------------------- companions
def companions(cfg, img, cat):
    # Every known object in the region, without any cut: Rubin (including point-like objects and those without
    # i_extendedness) and Euclid (including stars), each with its ellipse in the region pixels.
    w = img["wcs"][cfg["bands"].index("i")]
    raw = pd.read_parquet(cfg.data(RUBIN_DIR, "catalog", "object_box.parquet"))
    a, b, th = moments_to_ellipse(raw["i_ixx"], raw["i_iyy"], raw["i_ixy"])
    ap, bp, thp = psf_ellipse(raw["i_ixxPSF"], raw["i_iyyPSF"])
    bad = ~(np.isfinite(a) & np.isfinite(b) & (b > 0))
    a, b, th = np.where(bad, ap, a), np.where(bad, bp, b), np.where(bad, thp, th)
    x, y = w.world_to_pixel_values(raw["coord_ra"].values, raw["coord_dec"].values)
    rubin = pd.DataFrame(dict(survey="rubin", id=raw["objectId"].astype("int64"), x=x, y=y, a=a, b=b, theta=th))

    # Euclid ellipses: same construction as the copaings overlap test (catalogs.build_euclid then true_params),
    # read from the tiles that copaings read, hence with the same angle convention.
    from .run_copaings import _import_copaings, make_case
    catalogs, _, _ = _import_copaings(cfg)
    case = make_case(cfg)
    lsst_cop = pd.read_parquet(copaings_dir(cfg) / "catalogs" / "lsst.parquet")
    eu = catalogs.build_euclid(case, lsst_cop)
    # PSF: median Rubin i-band PSF over the region (copaings uses that of the first group member).
    fwhm = float(np.nanmedian(np.asarray(cat["psf_sigma_i"], float))) * FWHM2SIGMA
    ea, eb, eth = euclid_ellipse(eu["size_true"], eu["size_minor_true"], eu["position_angle_true"], fwhm)
    ex, ey = w.world_to_pixel_values(eu["ra"].values, eu["dec"].values)
    euclid = pd.DataFrame(dict(survey="euclid", ra=eu["ra"].values, dec=eu["dec"].values, x=ex, y=ey,
                               a=ea, b=eb, theta=eth))
    ok = np.isfinite(euclid["a"]) & np.isfinite(euclid["b"])
    euclid = euclid[ok].reset_index(drop=True)
    return rubin, euclid


def companion_mask(shape, ox, oy, comp, pad):
    # "ellipse" method of the recipe: union of the companions' half-light ellipses, box with origin (ox, oy).
    h, w = shape
    sel = ((comp["x"] > ox - pad) & (comp["x"] < ox + w + pad) & (comp["y"] > oy - pad) & (comp["y"] < oy + h + pad))
    m = np.zeros(shape, bool)
    for r in comp[sel].itertuples():
        m |= radius_map(shape, r.x - ox, r.y - oy, r.a, r.b, r.theta) <= 1.0
    return m


def companion_mask_footprint(detected, shape, ox, oy, target, comp, pad, grow_px, shared_k):
    # "footprint" method (validated deviation from the recipe): the half-light ellipse left the bright wings of
    # companions around sky-filled holes. It relies on the detection footprints (DETECTED plane, in at least one
    # band, connected components):
    # - the footprint that contains the target belongs to it: left untouched, its wings stay whole;
    # - every other footprint in the box is another source: masked whole, grown by grow_px pixels without
    #   eating into the target footprint;
    # - a companion whose centre falls inside the target footprint (shared footprint): shared_k times its
    #   ellipse, within that footprint;
    # - an undetected companion (seen by Euclid only): its half-light ellipse.
    h, w = shape
    det = detected[oy:oy + h, ox:ox + w]
    labels, _ = label(det, structure=np.ones((3, 3), int))
    xt, yt = target[0], target[1]
    it, jt = int(np.floor(yt + 0.5)), int(np.floor(xt + 0.5))
    lt = labels[it, jt] if (0 <= it < h and 0 <= jt < w) else 0
    target_fp = labels == lt if lt > 0 else np.zeros(shape, bool)

    m = det & ~target_fp
    if grow_px > 0 and m.any():
        m = binary_dilation(m, iterations=int(grow_px)) & ~target_fp
    sel = ((comp["x"] > ox - pad) & (comp["x"] < ox + w + pad) & (comp["y"] > oy - pad) & (comp["y"] < oy + h + pad))
    for r in comp[sel].itertuples():
        x, y = r.x - ox, r.y - oy
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        inside = 0 <= ix < w and 0 <= iy < h
        if inside and target_fp[iy, ix]:
            m |= (radius_map(shape, x, y, r.a, r.b, r.theta) <= shared_k) & target_fp
        elif not (inside and det[iy, ix]):
            m |= radius_map(shape, x, y, r.a, r.b, r.theta) <= 1.0
    return m

# ----------------------------------------------------------------------------- empty sky
class SkyBank:
    # Sky windows of the region with no detected, damaged or unexposed pixel, in any band.
    def __init__(self, img, unusable_mask, size, stride=4):
        self.unusable = np.any((img["mask"] & unusable_mask) != 0, axis=0)
        H, W = self.unusable.shape
        rr, cc = np.meshgrid(np.arange(0, H - size + 1, stride), np.arange(0, W - size + 1, stride), indexing="ij")
        self.corners = np.column_stack([rr.ravel(), cc.ravel()])
        self.size = size

    def find(self, need, rng, tries):
        # Window whose usable pixels cover every pixel to fill; failing that, the least bad one.
        n = self.size
        order = rng.permutation(len(self.corners))[:tries]
        best, best_bad = None, np.inf
        for k in order:
            r, c = self.corners[k]
            nbad = int(np.count_nonzero(self.unusable[r:r + n, c:c + n] & need))
            if nbad < best_bad:
                best, best_bad = (r, c), nbad
                if nbad == 0:
                    break
        return best, best_bad


def fill(img, ox, oy, size, need, window):
    # Copies flux, variance and bits of the chosen sky onto the pixels to fill, and sets the FILLED_SKY bit.
    sl = np.s_[:, oy:oy + size, ox:ox + size]
    flux, var, mbits = img["flux"][sl].copy(), img["var"][sl].copy(), img["mask"][sl].copy()
    if need.any():
        r, c = window
        ws = np.s_[:, r:r + size, c:c + size]
        flux[:, need] = img["flux"][ws][:, need]
        var[:, need] = img["var"][ws][:, need]
        mbits[:, need] = img["mask"][ws][:, need] | masks.FILLED_BIT
    return flux, var, mbits


# ----------------------------------------------------------------------------- one target
def process_target(row, img, cfg, rubin, euclid, banks, rng):
    cc = cfg["cuts"]
    half, dhalf = cc["stamp_px"] // 2, cc["diag_px"] // 2
    H, W = img["flux"].shape[1:]
    bits = img["bits"]
    nodata = masks.bitmask(bits, ["NO_DATA"])
    bad = masks.bitmask(bits, cc["bad_planes"])
    out = dict(object_id=row.object_id)

    # Positions in each band; cut 7: same integer pixel everywhere.
    pos = np.array([w.world_to_pixel_values(row.ra, row.dec) for w in img["wcs"]], float)
    ib = cfg["bands"].index(cc["shape_band"])
    x, y = pos[cfg["bands"].index("i")]
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    centers = np.floor(pos + 0.5).astype(int)
    out["x_tract"], out["y_tract"] = img["x0"] + x, img["y0"] + y
    out["cut7_value"] = float(np.max(np.hypot(*(pos - pos[cfg["bands"].index("i")]).T)))
    out["cut7_pass"] = bool(np.all(centers == centers[cfg["bands"].index("i")]))

    # Local demo: the stamp must fit inside the downloaded region (on the platform, it is the whole patch).
    out["in_region"] = bool(half <= ix < W - half and half <= iy < H - half)
    if not out["in_region"]:
        return out, None

    # Cut 3: integer margin inside the patch bounding box, tract frame.
    pb = img["meta"]["patch_bbox"]
    tx, ty = img["x0"] + ix, img["y0"] + iy
    margin = min(tx - pb["x_min"], pb["x_max"] - tx, ty - pb["y_min"], pb["y_max"] - ty)
    out["cut3_value"] = int(margin)
    out["cut3_pass"] = bool(margin >= cc["margin_px"])

    n = cc["stamp_px"]
    ox, oy = ix - half, iy - half
    xc, yc = x - ox, y - oy  # fractional position in the stamp (column, row)
    box = np.s_[:, oy:oy + n, ox:ox + n]
    mbox = img["mask"][box]

    # Optionally fetch the PSF
    if "psf" in img:
        from ..sources.rubin_butler import psf_stamp
        from ..common.psf import fwhm_from_stamp
        size = int(cfg["psf_px"])
        psf = np.stack([psf_stamp(img, k, img["x0"]+x, img["y0"]+y, size)
                                  for k in range(len(cfg["bands"]))])
        for b in cfg["bands"]:
            fwhm = fwhm_from_stamp(psf, cfg["pixel_scale"], oversample=4)
            out[f"psf_fwhm_{b}"] = float(fwhm)
    else:
        psf = None

    # Cut 4: no unexposed pixel in the box, in any band.
    nod = (mbox & nodata) != 0
    out["cut4_value"] = int(nod.sum())
    out["cut4_pass"] = out["cut4_value"] == 0

    # Cut 5: reduced radius of the nearest damaged pixel, in units of the target ellipse.
    a, b, th = row.ellipse_a, row.ellipse_b, row.ellipse_theta
    if not (np.isfinite(a) and np.isfinite(b) and b > 0):
        a = b = float(np.sqrt(2 * np.log(2)) * row.psf_sigma_i / cfg["pixel_scale"])
        th = 0.0
    rmap = radius_map((n, n), xc, yc, a, b, th)
    badpix = np.any((mbox & bad) != 0, axis=0)
    out["cut5_value"] = float(rmap[badpix].min()) if badpix.any() else np.inf
    out["cut5_n_bad_in_k"] = int(np.count_nonzero(badpix & (rmap <= cc["k_ellipse"])))
    out["cut5_pass"] = bool(out["cut5_value"] > cc["k_ellipse"])

    # Companions: every object in the box outside the target group (the Rubin target and its Euclid counterpart).
    rc = rubin[rubin["id"] != row.object_id]
    ec = euclid
    if pd.notna(row.euclid_ra):
        ec = euclid[~((euclid["ra"] == row.euclid_ra) & (euclid["dec"] == row.euclid_dec))]
    comp = pd.concat([rc[["x", "y", "a", "b", "theta"]], ec[["x", "y", "a", "b", "theta"]]], ignore_index=True)
    pad = float(np.nanmax(comp["a"])) if len(comp) else 0.0
    pad = min(pad, 50.0)
    mc = cfg["masking"]

    def make_need(size, bx, by):
        if mc["method"] == "ellipse":
            return companion_mask((size, size), bx, by, comp, pad)
        if mc["method"] == "footprint":
            return companion_mask_footprint(img["detected"], (size, size), bx, by, (x - bx, y - by, a, b, th),
                                            comp, pad, mc["grow_px"], mc["shared_k"])
        raise ValueError('masking.method must be "ellipse" or "footprint"')

    need = make_need(n, ox, oy)
    out["masked_fraction"] = float(need.mean())
    out["masked_fraction_in_ellipse"] = float(need[rmap <= 1].mean()) if (rmap <= 1).any() else np.nan
    out["n_companions"] = int(((comp["x"] - ox + 0.5).between(0, n) & (comp["y"] - oy + 0.5).between(0, n)).sum())

    window, nbad = banks[n].find(need, rng, cc["sky_tries"]) if need.any() else (None, 0)
    flux, var, mbits = fill(img, ox, oy, n, need, window)
    out["sky_fill_bad_frac"] = float(nbad / need.sum()) if need.any() else 0.0
    if window is not None:
        out["sky_window_x_tract"], out["sky_window_y_tract"] = img["x0"] + window[1], img["y0"] + window[0]

    # Masking decision: ellipticity offset before / after, in the shape band.
    weight = (~np.any(nod, axis=0)).astype(np.int32)
    m0 = adaptive_moments(img["flux"][box][ib], weight, (xc, yc))
    m1 = adaptive_moments(flux[ib], weight, (xc, yc))
    out.update(e1_before=m0["e1"], e2_before=m0["e2"], e1=m1["e1"], e2=m1["e2"], sigma_hsm=m1["sigma"],
               shape_ok=bool(m0["ok"] and m1["ok"]))
    out["delta_e_mask"] = delta_e(m0, m1) if out["shape_ok"] else np.nan
    # Shape measurement never selects: a failed measurement does not reject the stamp.
    out["mask_pass"] = bool(not np.isfinite(out["delta_e_mask"]) or out["delta_e_mask"] <= cc["delta_e_max"])

    # Local depth: median sky variance (pixels neither detected nor unexposed), 5-sigma limiting magnitude for a
    # point source the size of the PSF.
    det = masks.bitmask(bits, ["DETECTED", "NO_DATA"])
    for k, band in enumerate(cfg["bands"]):
        sky = (mbox[k] & det) == 0
        v = float(np.median(img["var"][box][k][sky])) if sky.any() else np.nan
        s_pix = float(getattr(row, f"psf_sigma_{band}")) / cfg["pixel_scale"]
        out[f"sky_var_{band}"] = v
        out[f"depth5_{band}"] = NJY_ZEROPOINT - 2.5 * np.log10(5.0 * np.sqrt(4 * np.pi * s_pix ** 2 * v))

    # Cut 8: diagnostic stamp twice as large, masked the same way, shape measured in two circular apertures.
    nd = cc["diag_px"]
    dox, doy = ix - dhalf, iy - dhalf
    diag = None
    out["cut8_evaluated"] = False
    if 0 <= dox and dox + nd <= W and 0 <= doy and doy + nd <= H:
        dneed = make_need(nd, dox, doy)
        dwin, dbad = banks[nd].find(dneed, rng, cc["sky_tries"]) if dneed.any() else (None, 0)
        dflux, _, dbits = fill(img, dox, doy, nd, dneed, dwin)
        dxc, dyc = x - dox, y - doy
        dok = ((dbits[ib] & nodata) == 0)
        r_in, r_out = cc["aperture_radii"]
        ma = adaptive_moments(dflux[ib], circular_weight((nd, nd), dxc, dyc, r_in) * dok, (dxc, dyc))
        mb = adaptive_moments(dflux[ib], circular_weight((nd, nd), dxc, dyc, r_out) * dok, (dxc, dyc))
        out["cut8_value"] = delta_e(ma, mb) if (ma["ok"] and mb["ok"]) else np.nan
        out["cut8_evaluated"] = bool(np.isfinite(out["cut8_value"]))
        out["diag_sky_fill_bad_frac"] = float(dbad / dneed.sum()) if dneed.any() else 0.0
        diag = dict(diag_flux=dflux[ib].astype(np.float32).ravel(), diag_need=dneed.ravel())
    out["cut8_pass"] = bool(not out["cut8_evaluated"] or out["cut8_value"] <= cc["delta_e_max"])

    with np.errstate(divide="ignore"):
        ivar = np.where((var > 0) & ((mbits & nodata) == 0), 1.0 / var, 0.0).astype(np.float32)
    arrays = dict(object_id=row.object_id,
                  flux_raw=img["flux"][box].astype(np.float32).ravel(),
                  flux=flux.astype(np.float32).ravel(), 
                  ivar=ivar.ravel(), 
                  maskbits=mbits.astype(np.int32).ravel(),
                  psf=psf.ravel() if psf is not None else None,
                  variance = var.astype(np.float32).ravel(),
                  companion_mask=need.ravel())
    if diag:
        arrays.update(diag)
    return out, arrays


def run(cfg):
    cc = cfg["cuts"]
    img = load_images(cfg)
    cat = pd.read_parquet(cfg.data(TABLES_DIR, "catalog.parquet"))
    patches = cfg["pixels"].get("patches") or sorted(
        {(int(t), int(p)) for t, p in zip(cat["tract"], cat["patch"])})
    
    for tract, patch in patches:
        done = cfg.data(TABLES_DIR, "shards", f"stamps_{tract}_{patch}.parquet")
        if done.exists():
            continue

        img = load_images(cfg, tract, patch)
        table, stamps_df = run_patch(cfg, img, cat, tract, patch)
        table.to_parquet(cfg.data(TABLES_DIR, "shards", f"table_{tract}_{patch}.parquet"))
        stamps_df.to_parquet(cfg.data(TABLES_DIR, "shards", f"stamps_{tract}_{patch}.parquet"))
        done.touch()

        del img


def run_patch(cfg, img, cat, tract, patch):
    cc = cfg["cuts"]

    if cfg["stamps"].get("targets", "isolated") == "all":
        targets = cat[cat[(cat["tract"] == tract) & (cat["patch"] == patch)]]
    else: 
        targets = cat[cat["cut1_pass"] & cat["cut2_pass"] & (cat["tract"] == tract)
                  & (cat["patch"] == patch)]
        
    rubin, euclid = companions(cfg, img, cat)

    unusable = masks.bitmask(img["bits"], ["DETECTED", "NO_DATA"] + list(cc["bad_planes"]))
    banks = {n: SkyBank(img, unusable, n) for n in (cc["stamp_px"], cc["diag_px"])}
    rows, arrays = [], []
    for row in targets.itertuples(index=False):
        rng = np.random.default_rng([int(cc["seed"]), int(row.object_id) & 0x7FFFFFFF])
        out, arr = process_target(row, img, cfg, rubin, euclid, banks, rng)
        rows.append(out)
        if arr is not None:
            arrays.append(arr)
    res = pd.DataFrame(rows)

    # Full table: every catalogue row, pixel columns empty outside targets.
    table = cat.merge(res, on="object_id", how="left")
    table["pixel_target"] = table["object_id"].isin(res["object_id"])
    table["in_region"] = table["in_region"].astype("boolean").fillna(False).astype(bool)
    table.to_parquet(cfg.data(TABLES_DIR, "table.parquet"), index=False)

    st = pd.DataFrame(arrays)
    st.attrs = {}
    st.to_parquet(cfg.data(TABLES_DIR, "stamps.parquet"), index=False)
    with open(cfg.data(TABLES_DIR, "stamps.json"), "w") as f:
        json.dump(dict(bands=list(cfg["bands"]), stamp_px=cc["stamp_px"], diag_px=cc["diag_px"], plane_bits=img["bits"],
                       filled_bit=masks.FILLED_BIT, tract=tract, patch=patch), f, indent=2)

    t = table[table["in_region"]]
    msg = {k: int(t[f"{k}_pass"].sum()) for k in ("cut3", "cut4", "cut5", "cut6", "cut7", "cut8", "mask")}
    print(f"patch targets {len(targets):,}; in region {len(t):,}; passing: {msg}")
    return table

def split_of(tract, patch, split_cfg):
    """
    Determine the train / val / test split of a patch, based on its patch number only.
    0 train, 1 validation, 2 test. 
    """
    import hashlib
    h = hashlib.md5(f"{split_cfg['seed']}:{tract}:{patch}".encode()).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    return 0 if u < split_cfg["train"] else 1 if u < split_cfg["train"] + split_cfg["val"] else 2
    
