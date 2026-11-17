# euclid_images.py
# Anna Preto
# AstroDeep Project
# Euclid Q1 image cutouts around given objects, put on one common footing: stamp_px x stamp_px at 0.1"/px, centred
# on the integer pixel nearest to the catalogue position, bands on the last axis, with
#   flux  (S, S, B) float32, ivar (S, S, B) float32 (0 where unknown), mask (S, S, B) bool (True = do not trust),
#   wcs   astropy WCS of the stamp, and wcs_assumed = True when the source gave none,
#   dx, dy  position of the catalogue centre relative to the stamp centre (pixels, x = column, y = row).
# Two sources:
#   mmu   the "image" column of the MultimodalUniverse Euclid Q1 HATS catalogue (Hugging Face). Its cutouts carry no
#         WCS here: they are taken as centred on the object, north up and east left, at their stated pixel scale.
#         Both assumptions are checked downstream (centroid offsets, e1/e2 correlation with Rubin), not trusted.
#   irsa  cutouts of the MER mosaics through the IRSA image service, with their WCS; tiles tried in order of
#         distance, since a tile that does not cover the position answers HTTP 500.
# Bands absent from a source are absent from `bands`; callers decide whether that rejects the object.

import gzip
import io
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

MMU_BANDS = {"VIS": ("VIS", "vis", "EUCLID-VIS"), "Y": ("Y", "NIR_Y", "EUCLID-NISP-Y", "y"),
             "J": ("J", "NIR_J", "EUCLID-NISP-J", "j"), "H": ("H", "NIR_H", "EUCLID-NISP-H", "h")}
IRSA_BANDS = {"VIS": "VIS", "Y": "NISP_Y", "J": "NISP_J", "H": "NISP_H"}


def tan_wcs(ra, dec, n, scale_arcsec, xc=None, yc=None):
    # North up, east left: x (column) towards west, y (row) towards north, like the Rubin tract frame.
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [float(ra), float(dec)]
    c = (n - 1) / 2.0
    w.wcs.crpix = [(c if xc is None else xc) + 1.0, (c if yc is None else yc) + 1.0]
    s = scale_arcsec / 3600.0
    w.wcs.cd = [[-s, 0.0], [0.0, s]]
    return w


def crop_center(arr, cx, cy, n, fill):
    # n x n window of arr (..., rows, cols) centred on the integer pixel nearest to (cx, cy); outside -> fill.
    ix, iy = int(np.floor(cx + 0.5)), int(np.floor(cy + 0.5))
    h = n // 2
    out = np.full(arr.shape[:-2] + (n, n), fill, dtype=arr.dtype)
    H, W = arr.shape[-2:]
    y0, x0 = iy - h, ix - h
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + n, H), min(x0 + n, W)
    if sy1 > sy0 and sx1 > sx0:
        out[..., sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[..., sy0:sy1, sx0:sx1]
    return out, cx - ix, cy - iy, (x0, y0)


def _as_records(image):
    # The nested "image" value of one row, whatever container nested-pandas / pyarrow returned, as a list of dicts.
    if hasattr(image, "to_dict") and hasattr(image, "columns"):
        return image.to_dict("records")
    if isinstance(image, dict):
        keys = list(image)
        return [dict(zip(keys, vals)) for vals in zip(*[image[k] for k in keys])]
    return list(image)


def _band_key(name):
    name = str(name)
    for key, aliases in MMU_BANDS.items():
        if name in aliases or name.upper() == key:
            return key
    return None


def _stack2d(v):
    return np.asarray([np.asarray(r, dtype=np.float64) for r in v])


def parse_mmu_image(image, want_bands, n, scale_expected):
    recs = _as_records(image)
    planes, found, scales = {}, [], []
    for r in recs:
        key = _band_key(r.get("band"))
        if key is None or key not in want_bands:
            continue
        flux = _stack2d(r["flux"])
        ivar = _stack2d(r["ivar"]) if r.get("ivar") is not None else np.zeros_like(flux)
        mask = np.asarray([np.asarray(m, bool) for m in r["mask"]]) if r.get("mask") is not None \
            else np.zeros(flux.shape, bool)
        planes[key] = (flux, ivar, mask)
        found.append(key)
        scales.append(float(r.get("scale", np.nan)))
    bands = [b for b in want_bands if b in planes]
    if not bands:
        return None
    shapes = {planes[b][0].shape for b in bands}
    if len(shapes) != 1:
        raise ValueError(f"MMU bands of one object differ in shape: {shapes}")
    src_n = planes[bands[0]][0].shape[0]
    c = (src_n - 1) / 2.0
    flux = np.stack([planes[b][0] for b in bands])
    ivar = np.stack([planes[b][1] for b in bands])
    mask = np.stack([planes[b][2] for b in bands])
    f, dx, dy, _ = crop_center(flux, c, c, n, 0.0)
    iv, _, _, _ = crop_center(ivar, c, c, n, 0.0)
    m, _, _, _ = crop_center(mask, c, c, n, True)
    scale = np.nanmedian(scales) if np.isfinite(scales).any() else np.nan
    return dict(bands=bands, flux=np.moveaxis(f, 0, -1).astype(np.float32),
                ivar=np.moveaxis(iv, 0, -1).astype(np.float32), mask=np.moveaxis(m, 0, -1),
                dx=float(dx), dy=float(dy), source_px=int(src_n), source_scale=float(scale),
                scale_ok=bool(np.isfinite(scale) and abs(scale - scale_expected) < 1e-3))


TRANSIENT_ERRORS = {"ProxyError", "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
                    "ReadError", "RemoteProtocolError", "ConnectionError", "ConnectionResetError", "TimeoutError",
                    "ClientConnectorError", "ServerDisconnectedError", "ClientPayloadError", "IncompleteRead"}
TRANSIENT_STATUS = ("429", "500", "502", "503", "504")


def is_transient(exc):
    # Network hiccups worth retrying: the Jean Zay proxy answers "500 Internal Server Error" now and then, and
    # huggingface_hub's own backoff does not retry a ProxyError. Walks the cause chain.
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ in TRANSIENT_ERRORS:
            return True
        msg = str(exc)
        if any(f"{c} " in msg or msg.endswith(c) or f"status={c}" in msg for c in TRANSIENT_STATUS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def with_retries(fn, what, tries=6, wait=10.0, max_wait=300.0, sleep=time.sleep):
    # fn() retried on transient network errors, waiting 10, 20, 40 ... s (capped); anything else is raised at once.
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - filtered by is_transient
            if k == tries - 1 or not is_transient(e):
                raise
            delay = min(wait * 2 ** k, max_wait)
            print(f"[mmu] {what}: {type(e).__name__}: {str(e)[:120]} -> retry {k + 1}/{tries - 1} in {delay:.0f} s",
                  flush=True)
            sleep(delay)


def mmu_partition_files(hats_url, box=None):
    # Parquet files of the MMU HATS partitions that overlap the box (catalogue metadata only, no data read).
    import lsdb
    from hats.io import paths
    cat = with_retries(lambda: lsdb.open_catalog(hats_url, columns=["object_id"]), "open catalogue")
    base = cat.hc_structure.catalog_base_dir
    suffix = getattr(cat.hc_structure.catalog_info, "npix_suffix", None) or ".parquet"
    if box is not None:
        cat = cat.box_search(ra=(box[0], box[1]), dec=(box[2], box[3]))
    files = []
    for px in cat.get_healpix_pixels():
        p = paths.pixel_catalog_file(base, px, npix_suffix=suffix)
        files.extend(sorted(p.rglob("*.parquet")) if suffix in ("/", "\\") else [p])
    return files


def _footer(f):
    import pyarrow.parquet as pq
    with f.open("rb") as fh:
        return pq.ParquetFile(fh).metadata


def _column_bytes(md, g, name):
    rg = md.row_group(g)
    return sum(rg.column(c).total_compressed_size for c in range(rg.num_columns)
               if rg.column(c).path_in_schema.split(".")[0] == name)


def mmu_layout(files, image_column="image"):
    # Row groups of each file and the compressed size of the image column per group, from the parquet footers.
    out = []
    for f in files:
        md = with_retries(lambda: _footer(f), f"footer of {f.name}")
        groups = range(md.num_row_groups)
        out.append(dict(file=str(f), rows=md.num_rows, row_groups=md.num_row_groups,
                        max_rows_per_group=max((md.row_group(g).num_rows for g in groups), default=0),
                        max_image_mb_per_group=max((_column_bytes(md, g, image_column) for g in groups),
                                                   default=0) / 1e6))
    return out


def read_mmu_rows(hats_url, ids, columns, box=None, max_group_mb=4000.0):
    # Rows of the MMU catalogue whose object_id is in ids, read row group by row group: the small object_id column
    # first, then the requested columns of the groups holding a target only. Only those column chunks are fetched
    # (range requests), and memory never holds more than one row group. lsdb's compute() instead loads whole
    # partitions in parallel before filtering (37 GB for the 4 partitions of one patch: OOM on prepost).
    # MMU stores object_id as a string, the pipeline's Euclid catalogue as int64: compared as strings.
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = pa.array(sorted({str(int(v)) for v in ids}), pa.string())
    cols = list(dict.fromkeys(["object_id", *columns]))
    found = {c: [] for c in cols}

    def read_group(f, g):
        # One row group on a fresh handle, so that a retry never resumes a half-broken connection. None when the
        # group holds no target.
        with f.open("rb") as fh:
            pf = pq.ParquetFile(fh)
            oid = pf.read_row_group(g, columns=["object_id"]).column("object_id")
            hit = pc.is_in(pc.cast(oid, pa.string()), value_set=want)
            if not pc.any(hit).as_py():
                return None
            mb = _column_bytes(pf.metadata, g, "image") / 1e6
            if "image" in cols and mb > max_group_mb:
                raise MemoryError(f"{f}: row group {g} holds {mb:.0f} MB of images (compressed), above "
                                  f"euclid.mmu_max_group_mb={max_group_mb:.0f}")
            return pf.read_row_group(g, columns=cols).filter(hit)

    for f in mmu_partition_files(hats_url, box):
        n_groups = with_retries(lambda: _footer(f), f"footer of {f.name}").num_row_groups
        for g in range(n_groups):
            t = with_retries(lambda: read_group(f, g), f"{f.name} row group {g}/{n_groups}")
            if t is None:
                continue
            for c in cols:
                found[c].extend(t.column(c).to_pylist())
            del t
    df = pd.DataFrame(found)
    df["object_id"] = df["object_id"].astype(str)
    return df


def fetch_mmu(hats_url, objects, want_bands, n, scale, box=None, max_group_mb=4000.0):
    # objects: DataFrame with object_id, ra, dec. Returns {object_id: cutout dict}. Rows are filtered on object_id
    # while reading (pyarrow filter), so memory holds the requested rows, not the partitions.
    ids = [int(v) for v in objects["object_id"]]
    df = read_mmu_rows(hats_url, ids, ["object_id", "image"], box, max_group_mb)
    pos = objects.set_index(objects["object_id"].astype("int64"))
    out = {}
    for i in range(len(df)):
        oid = int(str(df["object_id"].iloc[i]))
        cut = parse_mmu_image(df["image"].iloc[i], want_bands, n, scale)
        if cut is None:
            continue
        ra, dec = float(pos.loc[oid, "ra"]), float(pos.loc[oid, "dec"])
        cut["wcs"] = tan_wcs(ra, dec, n, scale, (n - 1) / 2.0 + cut["dx"], (n - 1) / 2.0 + cut["dy"])
        cut["wcs_assumed"] = True
        cut["source"] = "mmu"
        out[oid] = cut
    return out


class IrsaCutouts:
    def __init__(self, sia_url, collection, want_bands, n, scale, timeout=120):
        self.sia_url, self.collection = sia_url, collection
        self.want_bands, self.n, self.scale, self.timeout = list(want_bands), n, scale, timeout
        self.tiles = {}

    def _tiles(self, ra, dec, radius_deg=0.5):
        import pyvo
        key = (round(ra, 1), round(dec, 1))
        if key not in self.tiles:
            r = pyvo.dal.SIA2Service(self.sia_url).search(pos=(ra, dec, radius_deg),
                                                          collection=self.collection).to_table()
            rows = []
            for t in r:
                rows.append(dict(band=str(t["energy_bandpassname"]), subtype=str(t["dataproduct_subtype"]),
                                 ra=float(t["s_ra"]), dec=float(t["s_dec"]),
                                 url=str(t["access_url"]).replace("irsacloud.", "irsa.")))
            self.tiles[key] = rows
        return self.tiles[key]

    def _get(self, url, ra, dec, size_arcsec):
        import requests
        for k in range(4):
            try:
                rq = requests.get(url, params=dict(center=f"{ra},{dec}", size=f"{size_arcsec}arcsec"),
                                  timeout=self.timeout)
            except requests.RequestException:
                time.sleep(2 ** k)
                continue
            if rq.status_code != 200:
                return None
            b = rq.content
            return fits.open(io.BytesIO(gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b))[0]
        return None

    def _plane(self, band, subtype, ra, dec, size):
        cand = [t for t in self._tiles(ra, dec) if t["band"] == IRSA_BANDS[band] and t["subtype"] == subtype]
        cand.sort(key=lambda t: np.hypot((t["ra"] - ra) * np.cos(np.radians(dec)), t["dec"] - dec))
        for t in cand:
            h = self._get(t["url"], ra, dec, size)
            if h is not None:
                return h
        return None

    def cutout(self, ra, dec):
        size = self.n * self.scale + 2.0
        flux, ivar, mask, bands, wcs, est = [], [], [], [], None, False
        for b in self.want_bands:
            h = self._plane(b, "science", ra, dec, size)
            if h is None:
                continue
            w = WCS(h.header)
            x, y = w.world_to_pixel_values(ra, dec)
            data = np.asarray(h.data, np.float64)
            f, dx, dy, (x0, y0) = crop_center(data, float(x), float(y), self.n, np.nan)
            rms = self._plane(b, "noise", ra, dec, size)
            if rms is not None and np.asarray(rms.data).shape == data.shape:
                r, _, _, _ = crop_center(np.asarray(rms.data, np.float64), float(x), float(y), self.n, np.nan)
                iv = np.where(np.isfinite(r) & (r > 0), 1.0 / np.maximum(r, 1e-30) ** 2, 0.0)
            else:
                good = f[np.isfinite(f)]
                sig = 1.4826 * np.median(np.abs(good - np.median(good))) if good.size else np.nan
                iv = np.where(np.isfinite(f), 1.0 / sig ** 2 if np.isfinite(sig) and sig > 0 else 0.0, 0.0)
                est = True
            m = ~np.isfinite(f) | (iv <= 0)
            flux.append(np.where(m, 0.0, f))
            ivar.append(np.where(m, 0.0, iv))
            mask.append(m)
            bands.append(b)
            if wcs is None:
                wcs = w.deepcopy()
                wcs.wcs.crpix = [w.wcs.crpix[0] - x0, w.wcs.crpix[1] - y0]
        if not bands:
            return None
        return dict(bands=bands, flux=np.stack(flux, -1).astype(np.float32), ivar=np.stack(ivar, -1).astype(np.float32),
                    mask=np.stack(mask, -1), dx=float(dx), dy=float(dy), wcs=wcs, wcs_assumed=False,
                    ivar_estimated=est, source="irsa", source_px=self.n, source_scale=self.scale, scale_ok=True)

    def fetch(self, objects):
        out = {}
        for r in objects.itertuples(index=False):
            c = self.cutout(float(r.ra), float(r.dec))
            if c is not None:
                out[int(r.object_id)] = c
        return out


def fetch(ec, objects, box=None):
    # ec: the "euclid" section of the ML configuration.
    bands, n, scale = list(ec["bands"]), int(ec["stamp_px"]), float(ec["pixel_scale"])
    if ec["image_source"] == "mmu":
        return fetch_mmu(ec["hats_url"], objects, bands, n, scale, box, float(ec.get("mmu_max_group_mb", 4000)))
    if ec["image_source"] == "irsa":
        return IrsaCutouts(ec["irsa_sia_url"], ec["irsa_collection"], bands, n, scale).fetch(objects)
    raise ValueError(f"unknown euclid.image_source {ec['image_source']!r} (mmu | irsa)")


def to_band_array(cut, bands):
    # Cutout on a fixed band list: missing bands -> flux 0, ivar 0, mask True. Returns (flux, ivar, mask, present).
    n = cut["flux"].shape[0]
    flux = np.zeros((n, n, len(bands)), np.float32)
    ivar = np.zeros((n, n, len(bands)), np.float32)
    mask = np.ones((n, n, len(bands)), bool)
    present = np.zeros(len(bands), bool)
    for k, b in enumerate(bands):
        if b in cut["bands"]:
            j = cut["bands"].index(b)
            flux[..., k], ivar[..., k], mask[..., k] = cut["flux"][..., j], cut["ivar"][..., j], cut["mask"][..., j]
            present[k] = True
    bad = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar < 0)
    mask |= bad
    flux[bad], ivar[bad] = 0.0, 0.0
    return flux, ivar, mask, present
