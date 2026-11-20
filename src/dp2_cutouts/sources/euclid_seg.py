# euclid_seg.py
# Anna Preto
# AstroDeep Project
# Euclid Q1 MER segmentation maps (MER_SEG), for masking companions in the Rubin stamps by the pixels Euclid assigns
# to each source. One FITS file per MER tile (about 2.7 GB, 0.1 arcsec pixels), in IRSA's browsable directories:
#   https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/MER_SEG/<tile>/EUC_MER_FINAL-SEGMAP_TILE<tile>-*.fits
# The tiles covering a position come from IRSA's SIA2 service (MER mosaics, collection euclid_DpdMerBksMosaic).
# Pixel values are segment numbers (SEGMENTATION_MAP_ID of the MER catalogue, not OBJECT_ID; 0 = sky): the code
# never needs the catalogue link, segments are identified by position.
# Network needed for find_tiles / download (prepost on Jean Zay); seg_on_grid reads the files only.

import re
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

IRSA_SIA = "https://irsa.ipac.caltech.edu/SIA"
MER_SEG_ROOT = "https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/MER_SEG"
MOSAIC_COLLECTION = "euclid_DpdMerBksMosaic"


def find_tiles(ra, dec, radius_deg):
    # MER tile numbers whose mosaics touch the circle (any band; the tile number is in the file name).
    from pyvo.dal import SIA2Service
    res = SIA2Service(IRSA_SIA).search(pos=(float(ra), float(dec), float(radius_deg)), collection=MOSAIC_COLLECTION)
    t = res.to_table()
    tiles = set()
    for col in ("access_url", "obs_id", "obs_publisher_did"):
        if col in t.colnames:
            for v in t[col]:
                m = re.search(r"TILE(\d{6,})", str(v))
                if m:
                    tiles.add(m.group(1))
    if not tiles and "obs_id" in t.colnames:
        tiles = {str(v)[:9] for v in t["obs_id"] if str(v)[:9].isdigit()}
    return sorted(tiles)


def seg_url(tile, session=None):
    # The file name carries a hash and a timestamp: read it from the tile directory listing.
    s = session or requests
    r = s.get(f"{MER_SEG_ROOT}/{tile}/", timeout=60)
    r.raise_for_status()
    names = sorted(set(re.findall(r'(EUC_MER_FINAL-SEGMAP_TILE\d+[^"<>\s]*?\.fits(?:\.gz)?)', r.text)))
    if not names:
        raise FileNotFoundError(f"no segmentation map listed in {MER_SEG_ROOT}/{tile}/")
    return f"{MER_SEG_ROOT}/{tile}/{names[-1]}"


def download(url, dest_dir, chunk=8 << 20):
    # Into dest_dir/<file name>; skipped when a file of the advertised size is already there.
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / url.rsplit("/", 1)[1]
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        size = int(r.headers.get("Content-Length", -1))
        if out.exists() and out.stat().st_size == size:
            return out
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk):
                f.write(block)
    if size >= 0 and tmp.stat().st_size != size:
        raise IOError(f"{url}: got {tmp.stat().st_size} bytes, expected {size}")
    tmp.rename(out)
    return out


def _image_hdu(h):
    for hdu in h:
        if getattr(hdu, "is_image", False) and hdu.header.get("NAXIS", 0) == 2:
            return hdu
    raise ValueError("no 2-D image in the segmentation file")


def seg_on_grid(seg_paths, wcs, shape, sub=2, rows_per_chunk=256):
    # Euclid segment numbers seen by each pixel of a Rubin grid (wcs, shape = (H, W)), sampled at sub x sub points
    # inside the pixel: a Rubin pixel (0.2") covers 2 x 2 Euclid pixels (0.1"), and a pixel touched by a
    # neighbour's segment at any of its points must be masked. Returns int64 (sub*sub, H, W); 0 = sky or no tile.
    H, W = shape
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    dx, dy = [a.ravel() for a in np.meshgrid(offs, offs)]
    out = np.zeros((sub * sub, H, W), np.int64)
    for path in seg_paths:
        with fits.open(path, memmap=True) as h:
            hdu = _image_hdu(h)
            we = WCS(hdu.header)
            ny, nx = hdu.header["NAXIS2"], hdu.header["NAXIS1"]
            # Bounding box of the grid in the tile, read once.
            cy, cx = np.array([0, 0, H - 1, H - 1]), np.array([0, W - 1, 0, W - 1])
            ex, ey = we.world_to_pixel_values(*wcs.pixel_to_world_values(cx, cy))
            x0, x1 = max(int(np.floor(ex.min())) - 4, 0), min(int(np.ceil(ex.max())) + 5, nx)
            y0, y1 = max(int(np.floor(ey.min())) - 4, 0), min(int(np.ceil(ey.max())) + 5, ny)
            if x0 >= x1 or y0 >= y1:
                continue
            block = np.asarray(hdu.section[y0:y1, x0:x1]).astype(np.int64)
        for r0 in range(0, H, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, H)
            yy, xx = np.mgrid[r0:r1, 0:W]
            for k in range(sub * sub):
                ra, dec = wcs.pixel_to_world_values(xx + dx[k], yy + dy[k])
                px, py = we.world_to_pixel_values(ra, dec)
                ix, iy = np.floor(px + 0.5).astype(np.int64) - x0, np.floor(py + 0.5).astype(np.int64) - y0
                ok = (ix >= 0) & (ix < x1 - x0) & (iy >= 0) & (iy < y1 - y0)
                lab = np.zeros(ix.shape, np.int64)
                lab[ok] = block[iy[ok], ix[ok]]
                cur = out[k, r0:r1]
                take = (cur == 0) & (lab != 0)  # overlapping tiles: first non-zero wins
                cur[take] = lab[take]
    return out


def fetch_for_box(ra, dec, radius_deg, cache_dir):
    # Tiles touching the circle, downloaded into cache_dir/<tile>/; returns the local paths.
    paths = []
    for tile in find_tiles(ra, dec, radius_deg):
        paths.append(download(seg_url(tile), Path(cache_dir) / tile))
    return paths
