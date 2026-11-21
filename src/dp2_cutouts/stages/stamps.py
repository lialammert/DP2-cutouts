# stamps.py
# Anna Preto
# AstroDeep Project
# Cuts the stamps from the six images of a region, applies the pixel cuts (3, 4, 5, 7, 8), masks companions and
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
from ..common.ellipse import (FWHM2SIGMA, elliptical_radius, euclid_ellipse, moments_to_ellipse, psf_ellipse,
                              radius_map)
from ..common.shapes import adaptive_moments, circular_weight, delta_e
from ..sources.rubin_soda import BANDS, tract_origin
from .catalog_table import copaings_dir

NJY_ZEROPOINT = 31.4  # AB magnitude of a 1 nJy flux


# ----------------------------------------------------------------------------- images
def load_images(cfg):
    img_dir = cfg.data("10_rubin", "images", "x").parent
    meta = json.load(open(img_dir / "images.json"))
    flux, var, mask, wcs, origins, bits = [], [], [], [], [], []
    for b in BANDS:
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
    # Pixels detected in at least one band: a companion may be detected in some bands only.
    detected = np.any((mask & masks.bitmask(bits[0], ["DETECTED"])) != 0, axis=0)
    return dict(flux=np.stack(flux), var=np.stack(var), mask=mask, detected=detected, wcs=wcs, x0=origins[0][0],
                y0=origins[0][1], bits=bits[0], meta=meta)


# ----------------------------------------------------------------------------- companions
def companions(cfg, img, cat):
    # Every known object in the region, without any cut: Rubin (including point-like objects and those without
    # i_extendedness) and Euclid (including stars), each with its ellipse in the region pixels.
    w = img["wcs"][BANDS.index("i")]
    raw = pd.read_parquet(cfg.data("10_rubin", "catalog", "object_box.parquet"))
    a, b, th = moments_to_ellipse(raw["i_ixx"], raw["i_iyy"], raw["i_ixy"])
    ap, bp, thp = psf_ellipse(raw["i_ixxPSF"], raw["i_iyyPSF"])
    bad = ~(np.isfinite(a) & np.isfinite(b) & (b > 0))
    a, b, th = np.where(bad, ap, a), np.where(bad, bp, b), np.where(bad, thp, th)
    x, y = w.world_to_pixel_values(raw["coord_ra"].values, raw["coord_dec"].values)
    rubin = pd.DataFrame(dict(survey="rubin", id=raw["objectId"].astype("int64"), x=x, y=y, a=a, b=b, theta=th,
                              flux_i=raw["i_cModelFlux"].astype(float).values, flux_vis=np.nan))

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
                               a=ea, b=eb, theta=eth, flux_i=np.nan, flux_vis=eu["vis_flux"].astype(float).values))
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
    #   ellipse, within that footprint (its wings beyond stay: companion_light_ratio measures what reaches the
    #   target);
    # - an undetected companion (seen by Euclid only): its half-light ellipse.
    # target: (x, y, a, b, theta, flux); comp: x, y, a, b, theta, flux (same unit as the target's, NaN unknown).
    # Also returns the number of shared companions and the flux of the brightest one over the target's.
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
    n_shared, ratio_max = 0, 0.0
    sel = ((comp["x"] > ox - pad) & (comp["x"] < ox + w + pad) & (comp["y"] > oy - pad) & (comp["y"] < oy + h + pad))
    for r in comp[sel].itertuples():
        x, y = r.x - ox, r.y - oy
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        inside = 0 <= ix < w and 0 <= iy < h
        if inside and target_fp[iy, ix]:
            m |= (radius_map(shape, x, y, r.a, r.b, r.theta) <= shared_k) & target_fp
            n_shared += 1
            if np.isfinite(r.flux) and target[5] > 0:
                ratio_max = max(ratio_max, float(r.flux) / float(target[5]))
        elif not (inside and det[iy, ix]):
            m |= radius_map(shape, x, y, r.a, r.b, r.theta) <= 1.0
    return m, dict(n_shared=n_shared, flux_ratio_max=ratio_max, target_fp=target_fp)


def companion_mask_segmentation(detected, seg, shape, ox, oy, target, grow_px, seg_grow_px, counterpart=None,
                                search_px=2):
    # "segmentation" method: the pixels Euclid assigns to each source (MER segmentation map, sampled on the Rubin
    # grid at sub-pixel points, see sources/euclid_seg.py) separate what the merged Rubin DETECTED plane cannot:
    # a companion touching the target is masked over its own pixels, however large, instead of a multiple of its
    # ellipse.
    # - target segments: the one at the catalogue position of the target's Euclid counterpart (counterpart: (x, y)
    #   in the box frame; it lies in its own segment by construction), the one under the Rubin centre, and any
    #   whose centroid lies inside the target's Rubin ellipse (Euclid often cuts one galaxy into several segments:
    #   they stay). VIS segments are tight (a few Euclid pixels for a faint galaxy): when none of these points falls
    #   in a segment, the nearest segment within search_px Rubin pixels of the counterpart (or centre) is the
    #   target's. Without this, a faint target whose segment sat one pixel off the centre was masked as a neighbour;
    # - every other segment touching a pixel (at any of its sub-pixel points) is masked, grown by seg_grow_px
    #   Rubin pixels for the wider Rubin PSF (its light spreads past the VIS segment), never into the target's
    #   segments;
    # - Rubin footprints that do not contain the target are masked as in the footprint method (sources that
    #   Euclid did not segment, e.g. outside its coverage).
    h, w = shape
    det = detected[oy:oy + h, ox:ox + w]
    labels, _ = label(det, structure=np.ones((3, 3), int))
    xt, yt, a, b, th = target[:5]
    it, jt = int(np.floor(yt + 0.5)), int(np.floor(xt + 0.5))
    lt = labels[it, jt] if (0 <= it < h and 0 <= jt < w) else 0
    target_fp = labels == lt if lt > 0 else np.zeros(shape, bool)
    m = det & ~target_fp
    if grow_px > 0 and m.any():
        m = binary_dilation(m, iterations=int(grow_px)) & ~target_fp

    L = seg[:, oy:oy + h, ox:ox + w]
    ids, inv = np.unique(L, return_inverse=True)
    inv = inv.reshape(L.shape)
    rows, cols = np.indices(shape)
    k = L.shape[0]
    cnt = np.bincount(inv.ravel(), minlength=len(ids)).astype(float)
    cx = np.bincount(inv.ravel(), weights=np.broadcast_to(cols, L.shape).ravel(), minlength=len(ids)) / cnt
    cy = np.bincount(inv.ravel(), weights=np.broadcast_to(rows, L.shape).ravel(), minlength=len(ids)) / cnt
    is_target = (elliptical_radius(cx - xt, cy - yt, a, b, th) <= 1.0) & (ids != 0)
    anchors = [(xt, yt)] if counterpart is None else [tuple(counterpart), (xt, yt)]
    found = False
    for ax_, ay_ in anchors:
        ia, ja = int(np.floor(ay_ + 0.5)), int(np.floor(ax_ + 0.5))
        if 0 <= ia < h and 0 <= ja < w and (L[:, ia, ja] != 0).any():
            is_target |= np.isin(ids, L[:, ia, ja]) & (ids != 0)
            found = True
    if not found:
        ax_, ay_ = anchors[0]
        d = np.hypot(cols - ax_, rows - ay_)
        near = (d <= search_px)[None] & (L != 0)
        if near.any():
            dd = np.broadcast_to(d, L.shape)
            is_target |= ids == L[near][np.argmin(dd[near])]
    t_pix = is_target[inv]
    n_pix = (L != 0) & ~t_pix
    target_core = t_pix.any(axis=0)
    neigh = n_pix.any(axis=0)
    if seg_grow_px > 0 and neigh.any():
        neigh = binary_dilation(neigh, iterations=int(seg_grow_px))
    neigh &= ~target_core
    info = dict(n_target_segments=int(is_target.sum()), n_other_segments=int(((ids != 0) & ~is_target).sum()),
                target_segment_px=int(target_core.sum()), sub_samples=int(k))
    return m | neigh, info


def companion_mask_segmentation_catalog(detected, seg, shape, ox, oy, target, comp, pad, grow_px, seg_grow_px,
                                        counterpart=None, keep_k=2.0, dedup_px=2.0):
    # "segmentation_catalog" method: the Euclid segmentation decides where the edges are, the catalogues decide who
    # owns what. Every catalogued object (Rubin Object table, Euclid MER catalogue; a Euclid source within dedup_px
    # of a Rubin object or of the target is the same object) is placed in the segment found at its position.
    # - a segment holding several catalogued objects (the segmentation merged them, e.g. VIS + NIR combination) is
    #   shared between them: each pixel goes to the object predicted brightest there (exponential profile from its
    #   catalogue ellipse and flux, as companion_light_ratio), so that a faint source next to a bright galaxy takes
    #   only its own surroundings, not the galaxy's wings; without a flux for every owner, to the object of smallest
    #   reduced radius; the target keeps its share;
    # - a segment holding no catalogued object (a clump or arm cut out by Euclid, or a faint uncatalogued source)
    #   stays with the target when it reaches within keep_k target ellipses, and is masked otherwise;
    # - a catalogued object with no segment at its position (missed by the segmentation) is masked over its
    #   half-light ellipse;
    # - Rubin footprints that do not contain the target are masked as in the footprint method.
    # Neighbour pixels are grown by seg_grow_px (Rubin PSF), never into the target's pixels.
    # Objects too faint for Rubin (comp column visible = False, the depth rule) are not masked: a segment owned only
    # by such objects is left as it is, and they take no share of a segment they share with visible objects.
    # target: (x, y, a, b, theta[, flux]) in the box frame; comp: x, y (region frame), a, b, theta, survey[, visible,
    # flux], fluxes in the same unit.
    h, w = shape
    det = detected[oy:oy + h, ox:ox + w]
    labels, _ = label(det, structure=np.ones((3, 3), int))
    xt, yt, a, b, th = target[:5]
    it, jt = int(np.floor(yt + 0.5)), int(np.floor(xt + 0.5))
    lt = labels[it, jt] if (0 <= it < h and 0 <= jt < w) else 0
    target_fp = labels == lt if lt > 0 else np.zeros(shape, bool)
    m = det & ~target_fp
    if grow_px > 0 and m.any():
        m = binary_dilation(m, iterations=int(grow_px)) & ~target_fp

    L = seg[:, oy:oy + h, ox:ox + w]
    lab = L[0].copy()
    for k in range(1, L.shape[0]):
        lab = np.where(lab == 0, L[k], lab)

    def label_at(x_, y_):
        i_, j_ = int(np.floor(y_ + 0.5)), int(np.floor(x_ + 0.5))
        if not (0 <= i_ < h and 0 <= j_ < w):
            return None
        v = L[:, i_, j_]
        v = v[v != 0]
        if not len(v):
            return 0
        vals, counts = np.unique(v, return_counts=True)       # the segment most of the pixel's points fall in
        return int(vals[np.argmax(counts)])

    # Catalogued objects: the target first (its counterpart position too), then the others, deduplicated.
    f_t = float(target[5]) if len(target) > 5 else np.nan
    objs = [dict(x=xt, y=yt, a=a, b=b, th=th, target=True, visible=True, flux=f_t)]
    anchors = [(xt, yt)]
    if counterpart is not None:
        objs.append(dict(x=counterpart[0], y=counterpart[1], a=a, b=b, th=th, target=True, visible=True, flux=f_t))
        anchors.append(tuple(counterpart))
    sel = ((comp["x"] > ox - pad) & (comp["x"] < ox + w + pad) & (comp["y"] > oy - pad) & (comp["y"] < oy + h + pad))
    near = comp[sel]
    rub = near[near["survey"] == "rubin"]
    rub_pts = np.array([[r.x - ox, r.y - oy] for r in rub.itertuples()] + list(anchors), float).reshape(-1, 2)
    for r in near.itertuples():
        x_, y_ = r.x - ox, r.y - oy
        if r.survey == "euclid" and len(rub_pts) and np.min(np.hypot(rub_pts[:, 0] - x_, rub_pts[:, 1] - y_)) < dedup_px:
            continue
        objs.append(dict(x=x_, y=y_, a=r.a, b=r.b, th=r.theta, target=False,
                         visible=bool(getattr(r, "visible", True)), flux=float(getattr(r, "flux", np.nan))))
    for o in objs:
        o["label"] = label_at(o["x"], o["y"])

    target_pix = np.zeros(shape, bool)
    neigh = np.zeros(shape, bool)
    r_target = radius_map(shape, xt, yt, a, b, th)
    n_split = n_kept_orphan = 0
    for v in np.unique(lab[lab != 0]):
        pix = lab == v
        owners = [o for o in objs if o["label"] == v]
        if owners and not any(o["target"] or o["visible"] for o in owners):
            continue                                        # only sources Rubin cannot see: left untouched
        owners = [o for o in owners if o["target"] or o["visible"]]
        if not owners:
            if (r_target[pix] <= keep_k).any():
                target_pix |= pix
                n_kept_orphan += 1
            else:
                neigh |= pix
            continue
        if all(o["target"] for o in owners):
            target_pix |= pix
            continue
        if not any(o["target"] for o in owners):
            neigh |= pix
            continue
        n_split += 1
        rr = np.stack([radius_map(shape, o["x"], o["y"], o["a"], o["b"], o["th"]) for o in owners])
        fl = np.array([o["flux"] for o in owners], float)
        ab = np.array([o["a"] * o["b"] for o in owners], float)
        if np.all(np.isfinite(fl) & (fl > 0) & (ab > 0)):
            # log surface brightness of each owner's exponential profile (see exp_profile)
            lsb = (np.log(fl * EXP_HALF_LIGHT ** 2 / (2.0 * np.pi * ab)))[:, None, None] - EXP_HALF_LIGHT * rr
            win = np.argmax(np.where(np.isfinite(lsb), lsb, -np.inf), axis=0)
        else:
            win = np.argmin(np.where(np.isfinite(rr), rr, np.inf), axis=0)
        is_t = np.array([o["target"] for o in owners])[win]
        target_pix |= pix & is_t
        neigh |= pix & ~is_t
    for o in objs:
        if not o["target"] and o["visible"] and o["label"] == 0:
            neigh |= radius_map(shape, o["x"], o["y"], o["a"], o["b"], o["th"]) <= 1.0
    if seg_grow_px > 0 and neigh.any():
        neigh = binary_dilation(neigh, iterations=int(seg_grow_px))
    neigh &= ~target_pix
    return m | neigh, dict(n_split_segments=n_split, n_orphan_segments_kept=n_kept_orphan,
                           target_px=int(target_pix.sum()))


def halo_ring(flux, var, need, usable, r_target, r_out=4, k_target=2.0):
    # Light left just outside the masked regions, measured on the raw image (not predicted): the pixels 1 to r_out
    # pixels outside the mask, usable, and away from the target (reduced radius > k_target, so that its own wings do
    # not count). Each pixel in units of its noise (flux / sqrt(variance)); a neighbour's wing left past its mask
    # makes the ring brighter than the sky. Returns (mean in sigma, number of ring pixels, significance
    # sum / sqrt(n)); NaN without a mask or a ring. The coadd noise is somewhat correlated: the significance is for
    # comparing maskings and setting a cut, not a calibrated probability.
    if not need.any():
        return np.nan, 0, np.nan
    ring = binary_dilation(need, iterations=int(r_out)) & ~need & usable & (r_target > k_target)
    ring &= np.isfinite(flux) & np.isfinite(var) & (var > 0)
    n = int(ring.sum())
    if n == 0:
        return np.nan, 0, np.nan
    s = flux[ring] / np.sqrt(var[ring])
    return float(s.mean()), n, float(s.sum() / np.sqrt(n))


EXP_HALF_LIGHT = 1.678  # half-light radius of an exponential profile, in scale lengths


def exp_profile(r_reduced, a, b, flux):
    # Surface brightness of an exponential profile whose half-light ellipse has semi-axes a, b and total flux flux,
    # at reduced radius r_reduced (1 on that ellipse). Unknown or non-positive flux or axes: 0.
    if not (np.isfinite(flux) and flux > 0 and a > 0 and b > 0):
        return np.zeros(np.shape(r_reduced))
    return flux * EXP_HALF_LIGHT ** 2 / (2.0 * np.pi * a * b) * np.exp(-EXP_HALF_LIGHT * np.asarray(r_reduced))


def companion_light_ratio(shape, target, comp, keep, k_region=3.0, dedup_px=2.0):
    # Light that the companions leave on the target, predicted rather than measured (the measured light of a faint
    # target is mostly noise): exponential profiles from the catalogue ellipses and fluxes, summed over the pixels
    # kept after masking (keep) within k_region times the target ellipse, where its shape is measured; divided by
    # the target's own light there. 0 for an isolated target; ~1 when a neighbour's halo is as bright as the target
    # where it is measured. NaN without a target flux.
    # target: (x, y, a, b, theta, flux) in the box frame; comp: x, y (box frame), a, b, theta, flux, survey. A
    # Euclid source within dedup_px of a Rubin companion or of the target is the same object: counted once.
    xt, yt, a, b, th, ft = target
    if not (np.isfinite(ft) and ft > 0):
        return np.nan
    rt = radius_map(shape, xt, yt, a, b, th)
    region = (rt <= k_region) & keep
    if not region.any():
        return np.nan
    t_light = float(exp_profile(rt[region], a, b, ft).sum())
    rub = comp[comp["survey"] == "rubin"]
    pts = np.vstack([rub[["x", "y"]].to_numpy(float), [[xt, yt]]])
    c_light = 0.0
    for r in comp.itertuples():
        if r.survey == "euclid" and np.min(np.hypot(pts[:, 0] - r.x, pts[:, 1] - r.y)) < dedup_px:
            continue
        rc = radius_map(shape, r.x, r.y, r.a, r.b, r.theta)
        c_light += float(exp_profile(rc[region], r.a, r.b, r.flux).sum())
    return c_light / t_light if t_light > 0 else np.nan


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
    ib = BANDS.index(cc["shape_band"])
    x, y = pos[BANDS.index("i")]
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    centers = np.floor(pos + 0.5).astype(int)
    out["x_tract"], out["y_tract"] = img["x0"] + x, img["y0"] + y
    out["cut7_value"] = float(np.max(np.hypot(*(pos - pos[BANDS.index("i")]).T)))
    out["cut7_pass"] = bool(np.all(centers == centers[BANDS.index("i")]))

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

    # Cut 4: no unexposed pixel in the box, in any band.
    nod = (mbox & nodata) != 0
    out["cut4_value"] = int(nod.sum())
    out["cut4_pass"] = out["cut4_value"] == 0

    # Cut 5: reduced radius of the nearest damaged pixel, in units of the target ellipse.
    a, b, th = row.ellipse_a, row.ellipse_b, row.ellipse_theta
    if not (np.isfinite(a) and np.isfinite(b) and b > 0):
        a = b = float(np.sqrt(2 * np.log(2)) * row.psf_sigma_i / 0.2)
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
    # Fluxes for the light the companions leave on the target (companion_light_ratio): i-band cModel for Rubin
    # companions; Euclid VIS fluxes brought to the i band with the target's own i / VIS ratio (1 without VIS flux).
    f_target = float(getattr(row, "i_cModelFlux", np.nan))
    vis_mag = float(getattr(row, "euclid_vis_mag", np.nan))
    f_target_vis = 10.0 ** (-0.4 * (vis_mag - NJY_ZEROPOINT)) if np.isfinite(vis_mag) else np.nan
    i_over_vis = f_target / f_target_vis if (np.isfinite(f_target_vis) and f_target_vis > 0
                                             and np.isfinite(f_target) and f_target > 0) else 1.0
    rc = rc.assign(flux=rc["flux_i"])
    ec = ec.assign(flux=ec["flux_vis"] * i_over_vis)
    cols = ["x", "y", "a", "b", "theta", "flux", "survey"]
    comp = pd.concat([rc[cols], ec[cols]], ignore_index=True)
    pad = float(np.nanmax(comp["a"])) if len(comp) else 0.0
    pad = min(pad, 50.0)
    mc = cfg["masking"]

    # Depth rule (masking.euclid_depth_margin, magnitudes; null = off): a source seen by Euclid only is masked in the
    # Rubin stamp only if Rubin could see it, i.e. its magnitude (VIS brought to i with the target's colour, as for
    # companion_light_ratio) is brighter than the local Rubin 5-sigma depth in i of this stamp + the margin. Euclid
    # is deeper: masking what Rubin cannot see only replaces Rubin noise, sometimes right next to the target.
    # Rubin companions are detected by Rubin by definition and are always masked.
    det_nd = masks.bitmask(bits, ["DETECTED", "NO_DATA"])
    i_band = BANDS.index("i")
    sky_i = (mbox[i_band] & det_nd) == 0
    v_i = float(np.median(img["var"][box][i_band][sky_i])) if sky_i.any() else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_i = NJY_ZEROPOINT - 2.5 * np.log10(5.0 * np.sqrt(4 * np.pi * (float(row.psf_sigma_i) / 0.2) ** 2 * v_i))
        mag_c = NJY_ZEROPOINT - 2.5 * np.log10(comp["flux"].to_numpy(float))
    margin = mc.get("euclid_depth_margin")
    below = np.zeros(len(comp), bool)
    if margin is not None and np.isfinite(depth_i):
        below = (comp["survey"].to_numpy() == "euclid") & np.isfinite(mag_c) & (mag_c > depth_i + float(margin))
    comp = comp.assign(visible=~below)
    comp_mask = comp[~below]
    in_box_c = ((comp["x"] - ox + 0.5).between(0, n) & (comp["y"] - oy + 0.5).between(0, n)).to_numpy()
    out["n_euclid_below_depth"] = int((below & in_box_c).sum())

    def make_need(size, bx, by):
        # Pixels to fill, and the shared-footprint diagnostics (None for the ellipse method).
        if mc["method"] == "ellipse":
            return companion_mask((size, size), bx, by, comp_mask, pad), None
        if mc["method"] == "footprint":
            return companion_mask_footprint(img["detected"], (size, size), bx, by,
                                            (x - bx, y - by, a, b, th, f_target), comp_mask, pad, mc["grow_px"],
                                            mc["shared_k"])
        if mc["method"] == "segmentation":
            if "seg" not in img:
                raise RuntimeError('masking.method "segmentation" needs the Euclid segmentation on the Rubin grid '
                                   '(img["seg"], sources/euclid_seg.seg_on_grid)')
            cp = None
            if pd.notna(row.euclid_ra):
                ex, ey = img["wcs"][ib].world_to_pixel_values(row.euclid_ra, row.euclid_dec)
                cp = (float(ex) - bx, float(ey) - by)
            need_s, _ = companion_mask_segmentation(img["detected"], img["seg"], (size, size), bx, by,
                                                    (x - bx, y - by, a, b, th), mc["grow_px"], mc["seg_grow_px"],
                                                    counterpart=cp)
            return need_s, None
        if mc["method"] in ("segmentation_catalog", "hybrid"):
            if "seg" not in img:
                raise RuntimeError(f'masking.method "{mc["method"]}" needs img["seg"] (sources/euclid_seg.seg_on_grid)')
            cp = None
            if pd.notna(row.euclid_ra):
                ex, ey = img["wcs"][ib].world_to_pixel_values(row.euclid_ra, row.euclid_dec)
                cp = (float(ex) - bx, float(ey) - by)
            need_s, _ = companion_mask_segmentation_catalog(img["detected"], img["seg"], (size, size), bx, by,
                                                            (x - bx, y - by, a, b, th, f_target), comp, pad,
                                                            mc["grow_px"],
                                                            mc["seg_grow_px"], counterpart=cp)
            if mc["method"] == "segmentation_catalog":
                return need_s, None
            # "hybrid": the footprint masking (wide: whole Rubin footprints of separate neighbours, no ring of
            # left-over wings), plus, inside the target's own footprint only, the neighbours' share of the Euclid
            # segments (instead of shared_k times their ellipse, which left the halos of bright neighbours that
            # share the target footprint). Measured on 5063_34: tight Euclid masks alone left rings around
            # separate neighbours, footprint masks alone left large halos of bright shared neighbours.
            need_f, info_f = companion_mask_footprint(img["detected"], (size, size), bx, by,
                                                      (x - bx, y - by, a, b, th, f_target), comp_mask, pad,
                                                      mc["grow_px"], mc["shared_k"])
            return need_f | (need_s & info_f["target_fp"]), info_f
        raise ValueError('masking.method must be "ellipse", "footprint", "segmentation", "segmentation_catalog" or '
                         '"hybrid"')

    need, shared_info = make_need(n, ox, oy)
    out["masked_fraction"] = float(need.mean())
    out["masked_fraction_in_ellipse"] = float(need[rmap <= 1].mean()) if (rmap <= 1).any() else np.nan
    out["n_companions"] = int(((comp["x"] - ox + 0.5).between(0, n) & (comp["y"] - oy + 0.5).between(0, n)).sum())
    if shared_info is not None:
        out["n_shared_companions"] = int(shared_info["n_shared"])
        out["shared_flux_ratio_max"] = float(shared_info["flux_ratio_max"])
    else:
        out["n_shared_companions"], out["shared_flux_ratio_max"] = -1, np.nan
    near = comp[(comp["x"] > ox - pad) & (comp["x"] < ox + n + pad) & (comp["y"] > oy - pad) & (comp["y"] < oy + n + pad)]
    out["companion_light_ratio"] = companion_light_ratio(
        (n, n), (xc, yc, a, b, th, f_target), near.assign(x=near["x"] - ox, y=near["y"] - oy),
        ~need & ~np.any(nod, axis=0), float(mc.get("light_region_k", 3.0)))
    # Measured counterpart of companion_light_ratio: excess light in a ring just outside the masks (shape band).
    out["halo_ring_mean_sigma"], out["halo_ring_npix"], out["halo_ring_z"] = halo_ring(
        img["flux"][box][ib], img["var"][box][ib], need, ~np.any(nod, axis=0), rmap,
        int(mc.get("halo_ring_px", 4)), float(mc.get("halo_target_k", 2.0)))

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
    for k, band in enumerate(BANDS):
        sky = (mbox[k] & det) == 0
        v = float(np.median(img["var"][box][k][sky])) if sky.any() else np.nan
        s_pix = float(getattr(row, f"psf_sigma_{band}")) / 0.2
        out[f"sky_var_{band}"] = v
        out[f"depth5_{band}"] = NJY_ZEROPOINT - 2.5 * np.log10(5.0 * np.sqrt(4 * np.pi * s_pix ** 2 * v))

    # Cut 8: diagnostic stamp twice as large, masked the same way, shape measured in two circular apertures.
    nd = cc["diag_px"]
    dox, doy = ix - dhalf, iy - dhalf
    diag = None
    out["cut8_evaluated"] = False
    if 0 <= dox and dox + nd <= W and 0 <= doy and doy + nd <= H:
        dneed, _ = make_need(nd, dox, doy)
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
                  flux=flux.astype(np.float32).ravel(), ivar=ivar.ravel(), maskbits=mbits.astype(np.int32).ravel(),
                  companion_mask=need.ravel())
    if diag:
        arrays.update(diag)
    return out, arrays


def run(cfg):
    cc = cfg["cuts"]
    img = load_images(cfg)
    cat = pd.read_parquet(cfg.data("30_processed", "catalog.parquet"))
    meta = img["meta"]
    targets = cat[cat["cut1_pass"] & cat["cut2_pass"] & (cat["tract"] == meta["tract"])
                  & (cat["patch"] == meta["patch"])]
    rubin, euclid = companions(cfg, img, cat)
    mc = cfg["masking"]
    if mc["method"] in ("segmentation", "segmentation_catalog", "hybrid"):
        # Euclid segmentation sampled on the Rubin grid of the region (tiles downloaded by fetch).
        from ..sources import euclid_seg
        tiles = euclid_seg.cached_tiles(euclid_seg.cache_dir(mc))
        if not tiles:
            raise RuntimeError(f"masking.method {mc['method']!r}: no segmentation tile in "
                               f"{euclid_seg.cache_dir(mc)} (run fetch with this configuration)")
        img["seg"] = euclid_seg.seg_on_grid(tiles, img["wcs"][BANDS.index("i")], img["flux"].shape[1:])
        if not (img["seg"] != 0).any():
            raise RuntimeError("the Euclid segmentation tiles in the cache do not cover this region")

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
    table.to_parquet(cfg.data("30_processed", "table.parquet"), index=False)

    st = pd.DataFrame(arrays)
    st.attrs = {}
    st.to_parquet(cfg.data("30_processed", "stamps.parquet"), index=False)
    with open(cfg.data("30_processed", "stamps.json"), "w") as f:
        json.dump(dict(bands=list(BANDS), stamp_px=cc["stamp_px"], diag_px=cc["diag_px"], plane_bits=img["bits"],
                       filled_bit=masks.FILLED_BIT, tract=meta["tract"], patch=meta["patch"]), f, indent=2)

    t = table[table["in_region"]]
    msg = {k: int(t[f"{k}_pass"].sum()) for k in ("cut3", "cut4", "cut5", "cut6", "cut7", "cut8", "mask")}
    print(f"patch targets {len(targets):,}; in region {len(t):,}; passing: {msg}")
    return table
