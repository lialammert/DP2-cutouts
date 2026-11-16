# adapters.py
# Anna Preto
# AstroDeep Project
# Data-source adapters, chosen in the "sources" section of the configuration. Each adapter fills the files of a case
# in the layout that the stages read, so that the stages never know where the data came from:
#   rubin_catalog  -> 10_rubin/catalog/object_box.parquet      (tap)
#   euclid_catalog -> 00_euclid/euclid_box.parquet + tiles     (hf)
#   rubin_pixels   -> 10_rubin/images/*.fits + images.json     (soda, butler)
# The adapters take no scientific decision.

import hashlib
import io
import json
import time

import numpy as np
from astropy.io import fits

from .sources import euclid_hf, rubin_soda, rubin_tap

BANDS = rubin_soda.BANDS


def _kind(cfg, source):
    kind = cfg["sources"][source]
    return kind["kind"] if isinstance(kind, dict) else kind


def fetch_rubin_catalog(cfg):
    kind = _kind(cfg, "rubin_catalog")
    if kind == "tap":
        return rubin_tap.run(cfg)
    raise ValueError(f"unknown rubin_catalog source: {kind}")


def fetch_euclid_catalog(cfg):
    kind = _kind(cfg, "euclid_catalog")
    if kind == "hf":
        return euclid_hf.run(cfg)
    raise ValueError(f"unknown euclid_catalog source: {kind}")


def fetch_rubin_pixels(cfg, patch):
    # patch: dict with tract, patch, s_ra, s_dec, s_xel1, s_xel2 (one row of plan/patches.parquet).
    kind = _kind(cfg, "rubin_pixels")
    if kind == "soda":
        return soda_full_patch(cfg, patch)
    if kind == "butler":
        return butler_full_patch(cfg, patch)
    raise ValueError(f"unknown rubin_pixels source: {kind}")


# ----------------------------------------------------------------------------- SODA
def _retry(fn, attempts=4, wait=10.0):
    # Transient network errors: retried with an increasing wait; the last error is raised.
    for k in range(attempts):
        try:
            return fn()
        except Exception:
            if k == attempts - 1:
                raise
            time.sleep(wait * 2 ** k)


def _write_images(cfg, patch, data_by_band, source_ids):
    # Writes the six FITS files and images.json of a full patch; checks that every band covers the whole patch.
    out_dir = cfg.data("10_rubin", "images", "x").parent
    tract, p = int(patch["tract"]), int(patch["patch"])
    files, sha, origins, shapes = {}, {}, set(), set()
    for b in BANDS:
        path = out_dir / f"deep_coadd_{tract}_{p}_{b}.fits"
        with open(path, "wb") as f:
            f.write(data_by_band[b])
        files[b] = path.name
        sha[b] = hashlib.sha256(data_by_band[b]).hexdigest()
        with fits.open(io.BytesIO(data_by_band[b])) as h:
            origins.add(rubin_soda.tract_origin(h["IMAGE"].header))
            shapes.add(h["IMAGE"].data.shape)
    if len(origins) != 1 or len(shapes) != 1:
        raise RuntimeError(f"patch {tract}_{p}: the six bands do not share the same pixel grid")
    (x0, y0), (ny, nx) = origins.pop(), shapes.pop()
    if (nx, ny) != (int(patch["s_xel1"]), int(patch["s_xel2"])):
        raise RuntimeError(f"patch {tract}_{p}: cutout {nx} x {ny} px is not the full patch "
                           f"{patch['s_xel1']} x {patch['s_xel2']}")
    meta = dict(tract=tract, patch=p, center=[float(patch["s_ra"]), float(patch["s_dec"])], size_px=nx,
                files=files, sha256=sha, source_ids=source_ids,
                patch_bbox=dict(x_min=x0, x_max=x0 + nx - 1, y_min=y0, y_max=y0 + ny - 1))
    with open(out_dir / "images.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def soda_full_patch(cfg, patch):
    # The whole patch (outer bounding box) in six bands: a SODA circle that encloses the patch is clipped by the
    # service to the image, which returns the full patch.
    s = rubin_soda.session(cfg["rubin"]["token_file"])
    tract, p = int(patch["tract"]), int(patch["patch"])
    ra, dec = float(patch["s_ra"]), float(patch["s_dec"])
    rows = _retry(lambda: rubin_soda.find_patch_images(s, ra, dec, tract, p))
    radius = 0.5 * max(int(patch["s_xel1"]), int(patch["s_xel2"])) * 0.2 * np.sqrt(2.0) / 3600.0 + 0.002
    data, ids = {}, {}
    for b in BANDS:
        data[b] = _retry(lambda: rubin_soda.soda_cutout(s, rows[b]["access_url"], ra, dec, radius))
        ids[b] = str(rows[b]["obs_publisher_did"])
    return _write_images(cfg, patch, data, ids)


# ----------------------------------------------------------------------------- butler
def butler_full_patch(cfg, patch):
    # For the Rubin Science Platform or a Rubin data facility (needs the LSST software stack). Writes the same files
    # as SODA, so that the SODA / butler cross-check can compare them bit for bit. Not tested yet.
    from lsst.daf.butler import Butler

    bc = cfg["sources"]["rubin_pixels"]
    butler = Butler(bc["repo"], collections=bc["collections"])
    tract, p = int(patch["tract"]), int(patch["patch"])
    data, ids = {}, {}
    for b in BANDS:
        refs = butler.query_datasets("deep_coadd", where=f"tract = {tract} AND patch = {p} AND band = '{b}'",
                                     limit=None)
        if len(refs) != 1:
            raise RuntimeError(f"patch {tract}_{p} band {b}: {len(refs)} deep_coadd datasets")
        exp = butler.get(refs[0], storageClass="ExposureF")
        data[b] = _exposure_to_fits(exp)
        ids[b] = str(refs[0].id)
    return _write_images(cfg, patch, data, ids)


def _exposure_to_fits(exp):
    # Same layout as the SODA output: IMAGE / MASK / VARIANCE extensions, tract WCS, "A" WCS giving the tract pixel
    # of the first array pixel, mask planes in MSKNnnnn / MSKMnnnn keywords.
    x0, y0 = exp.getXY0()
    wcs_hdr = fits.Header()
    for k, v in exp.getWcs().getFitsMetadata().toDict().items():
        wcs_hdr[k] = v
    wcs_hdr.update(CTYPE1A="LINEAR", CTYPE2A="LINEAR", CRPIX1A=1.0, CRPIX2A=1.0, CRVAL1A=float(x0),
                   CRVAL2A=float(y0), CUNIT1A="PIXEL", CUNIT2A="PIXEL")
    mask_hdr = wcs_hdr.copy()
    for k, (name, bit) in enumerate(sorted(exp.mask.getMaskPlaneDict().items(), key=lambda kv: kv[1])):
        mask_hdr[f"MSKN{k:04d}"] = name
        mask_hdr[f"MSKM{k:04d}"] = 1 << bit
    hdus = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(exp.image.array.astype(np.float32), header=wcs_hdr, name="IMAGE"),
                         fits.ImageHDU(exp.mask.array.astype(np.int32), header=mask_hdr, name="MASK"),
                         fits.ImageHDU(exp.variance.array.astype(np.float32), header=wcs_hdr, name="VARIANCE")])
    buf = io.BytesIO()
    hdus.writeto(buf)
    return buf.getvalue()
