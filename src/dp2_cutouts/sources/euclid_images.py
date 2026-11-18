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
import json
import os
import time
from pathlib import Path

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


def parse_mmu_image(image, want_bands, n, scale_expected, mask_true_is_valid=True):
    # mask_true_is_valid: MMU Euclid Q1 marks VALID pixels with True (probe 2026-11-18: background pixels with
    # ivar > 0 are True; only the saturated core of a star is False). The returned mask is ours: True = do not trust,
    # which also covers non-finite flux and, when ivar is given, ivar <= 0.
    recs = _as_records(image)
    planes, found, scales = {}, [], []
    for r in recs:
        key = _band_key(r.get("band"))
        if key is None or key not in want_bands:
            continue
        flux = _stack2d(r["flux"])
        ivar = _stack2d(r["ivar"]) if r.get("ivar") is not None else np.zeros_like(flux)
        if r.get("mask") is not None:
            m = np.asarray([np.asarray(v, bool) for v in r["mask"]])
            mask = ~m if mask_true_is_valid else m
        else:
            mask = np.zeros(flux.shape, bool)
        mask = mask | ~np.isfinite(flux)
        if r.get("ivar") is not None:
            mask = mask | ~(np.isfinite(ivar) & (ivar > 0))
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


# Local mirror of the MMU partitions. Each partition is ONE parquet row group of ~30 GB of compressed images
# (probe 2026-11-18: 77 684 rows, 27 GB, in Npix=143742): nothing smaller than a partition can be fetched, and a patch
# needs 2 to 4 of them. Rather than stream ~100 GB through the Jean Zay proxy for every patch and every dataset (D1
# and D2 read the same partitions), `dp2-ml mmu-mirror` downloads the partitions of all patches once (resumable)
# into euclid.mmu_cache, and the readers then stream the local files.

MIRROR_MANIFEST = "mmu_mirror.json"


def mmu_cache_dir(ec):
    # euclid.mmu_cache: a folder, "auto" ($SCRATCH/mmu_euclid_q1, else $DP2_ML_ROOT/mmu_euclid_q1), or null to read
    # Hugging Face directly (streamed: slow, but bounded memory).
    v = ec.get("mmu_cache", "auto")
    if v in (None, False, "", "none", "null"):
        return None
    if v == "auto":
        base = os.environ.get("SCRATCH") or os.environ.get("DP2_ML_ROOT")
        if not base:
            raise KeyError("euclid.mmu_cache: auto needs $SCRATCH or $DP2_ML_ROOT")
        return Path(base) / "mmu_euclid_q1"
    return Path(v)


def hf_repo_file(remote):
    # "hf://datasets/<org>/<repo>[@revision]/<path in repo>" -> (repo_id, revision or None, path in repo)
    s, prefix = str(remote), "hf://datasets/"
    if not s.startswith(prefix):
        raise ValueError(f"not a Hugging Face dataset path: {s}")
    org, repo, path = s[len(prefix):].split("/", 2)
    repo, _, rev = repo.partition("@")
    return f"{org}/{repo}", rev or None, path


def _same_box(a, b, tol=1e-9):
    return a is not None and b is not None and np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


def mirror_mmu(hats_url, cache, boxes, dry_run=False):
    # boxes {patch name: (ra_min, ra_max, dec_min, dec_max)}. Downloads every partition overlapping a box into
    # cache/<path in repo> (huggingface_hub: resumable, files already complete are skipped) and writes the manifest
    # the readers use. With dry_run, only lists the files and their sizes.
    from huggingface_hub import HfFileSystem, hf_hub_download
    cache = Path(cache)
    need = {}
    for name, box in sorted(boxes.items()):
        for f in mmu_partition_files(hats_url, box):
            need.setdefault(str(f), []).append(name)
    fs = HfFileSystem()
    files = []
    for remote, patches in sorted(need.items()):
        repo, rev, path = hf_repo_file(remote)
        size = int(with_retries(lambda: fs.info(remote)["size"], f"size of {path}"))
        files.append(dict(remote=remote, repo=repo, revision=rev, path=path, local=str(cache / path), size=size,
                          patches=sorted(patches)))
    total = sum(f["size"] for f in files)
    for f in files:
        print(f"[mmu-mirror] {f['path']}  {f['size'] / 1e9:7.1f} GB  patches {' '.join(f['patches'])}", flush=True)
    print(f"[mmu-mirror] {len(files)} partitions, {total / 1e9:.1f} GB -> {cache}", flush=True)
    plan = dict(hats_url=str(hats_url), cache=str(cache), boxes={k: list(map(float, v)) for k, v in boxes.items()},
                files=files, total_bytes=total)
    if dry_run:
        return plan
    cache.mkdir(parents=True, exist_ok=True)
    for f in files:
        local = Path(f["local"])
        if local.exists() and local.stat().st_size == f["size"]:
            print(f"[mmu-mirror] already there: {f['path']}", flush=True)
            continue
        t0 = time.time()
        got = Path(with_retries(lambda: hf_hub_download(f["repo"], f["path"], repo_type="dataset",
                                                        revision=f["revision"], local_dir=cache),
                                f"download {f['path']}"))
        if got.resolve() != local.resolve() or got.stat().st_size != f["size"]:
            raise RuntimeError(f"{f['path']}: got {got} ({got.stat().st_size} B), expected {local} ({f['size']} B)")
        dt = time.time() - t0
        print(f"[mmu-mirror] {f['path']}: {f['size'] / 1e9:.1f} GB in {dt / 60:.1f} min "
              f"({f['size'] / 1e6 / max(dt, 1e-3):.0f} MB/s)", flush=True)
    plan["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = cache / (MIRROR_MANIFEST + ".tmp")
    tmp.write_text(json.dumps(plan, indent=2))
    tmp.replace(cache / MIRROR_MANIFEST)
    return plan


def mirrored_files(hats_url, cache, box=None):
    # Local partitions of the mirror for one patch box (all of them when box is None). Fails loudly when the mirror
    # is missing, belongs to another catalogue, does not cover the box or lost a file (purged $SCRATCH).
    man_path = Path(cache) / MIRROR_MANIFEST
    if not man_path.exists():
        raise FileNotFoundError(f"no MMU mirror in {cache}: run `dp2-ml <cfg> mmu-mirror` first "
                                "(submit_ml.sh mmu-mirror), or set euclid.mmu_cache: null to read Hugging Face directly")
    man = json.loads(man_path.read_text())
    if man["hats_url"] != str(hats_url):
        raise ValueError(f"MMU mirror {cache} is of {man['hats_url']}, not {hats_url}")
    names = None
    if box is not None:
        names = {k for k, b in man["boxes"].items() if _same_box(b, box)}
        if not names:
            raise ValueError(f"sky box {tuple(box)} is not one of the patches mirrored in {cache}: rerun mmu-mirror")
    files = [f for f in man["files"] if names is None or names & set(f["patches"])]
    for f in files:
        p = Path(f["local"])
        if not p.exists() or p.stat().st_size != f["size"]:
            raise FileNotFoundError(f"MMU mirror incomplete: {p} missing or truncated (purged?): rerun mmu-mirror")
    return [Path(f["local"]) for f in files]


def _is_remote(f):
    return getattr(f, "protocol", "") not in ("", "file", "local")


def _scan_partition(f, want, cols, batch_rows, sink, buffer_bytes=8 << 20):
    # Rows of one partition file whose object_id is in want, handed to sink(arrow table) batch by batch; returns the
    # number of rows found. The object_id column is read whole (a few MB); the other columns are streamed (buffered
    # reads: memory holds one batch, never the 30 GB column chunk), and reading stops after the last target. HATS
    # sorts each partition by HEALPix index, so the targets of one patch sit in a contiguous stretch of the file.
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    fh = f.open("rb") if _is_remote(f) else None
    try:
        pf = pq.ParquetFile(fh if fh is not None else str(f), buffer_size=buffer_bytes, pre_buffer=False)
        found = 0
        for g in range(pf.num_row_groups):
            oid = pf.read_row_group(g, columns=["object_id"]).column("object_id")
            hit = pc.fill_null(pc.is_in(pc.cast(oid, pa.string()), value_set=want), False)
            idx = np.flatnonzero(hit.to_numpy())
            if not len(idx):
                continue
            found += len(idx)
            if cols == ["object_id"]:
                sink(pa.table(dict(object_id=oid.take(pa.array(idx)))))
                continue
            start = 0
            for b in pf.iter_batches(batch_size=batch_rows, row_groups=[g], columns=cols, use_threads=False):
                stop = start + b.num_rows
                sel = idx[(idx >= start) & (idx < stop)] - start
                if len(sel):
                    sink(pa.Table.from_batches([b.take(pa.array(sel))]))
                start = stop
                if start > idx[-1]:
                    break
        return found
    finally:
        if fh is not None:
            fh.close()


def scan_mmu(hats_url, ids, columns, sink, box=None, cache=None, batch_rows=64):
    # Every row of the MMU catalogue whose object_id is in ids goes to sink(arrow table of a few rows), from the local
    # mirror (cache) or, without one, streamed from Hugging Face. lsdb's compute() loads whole partitions before
    # filtering (37 GB for the 4 partitions of one patch: OOM on prepost); here memory holds one batch of batch_rows
    # rows. A partition retried after a network error hands its rows to sink again: sinks must be idempotent.
    # MMU stores object_id as a string, the pipeline's Euclid catalogue as int64: compared as strings.
    import pyarrow as pa
    want = pa.array(sorted({str(int(v)) for v in ids}), pa.string())
    cols = list(dict.fromkeys(["object_id", *columns]))
    files = mirrored_files(hats_url, cache, box) if cache is not None else mmu_partition_files(hats_url, box)
    for f in files:
        t0 = time.time()
        n = with_retries(lambda: _scan_partition(f, want, cols, int(batch_rows), sink), f"read {f.name}")
        print(f"[mmu] {f.name}: {n} target rows in {time.time() - t0:.0f} s", flush=True)


def read_mmu_rows(hats_url, ids, columns, box=None, cache=None, batch_rows=64):
    # The rows as a DataFrame of Python objects: for a handful of objects (probe, tests). The images of a whole patch
    # go through fetch_mmu, which crops each row as soon as it is read.
    import pandas as pd
    cols = list(dict.fromkeys(["object_id", *columns]))
    rows = {}

    def sink(t):
        for r in t.to_pylist():
            rows[str(r["object_id"])] = r

    scan_mmu(hats_url, ids, columns, sink, box, cache, batch_rows)
    df = pd.DataFrame([rows[k] for k in sorted(rows)], columns=cols)
    df["object_id"] = df["object_id"].astype(str)
    return df


def fetch_mmu(hats_url, objects, want_bands, n, scale, box=None, cache=None, batch_rows=64,
              mask_true_is_valid=True):
    # objects: DataFrame with object_id, ra, dec. Returns {object_id: cutout dict}. Each row is converted to Python
    # and cropped to n x n as soon as it is read: the source cutouts (several MB each as Python lists) never pile up.
    pos = objects.set_index(objects["object_id"].astype("int64"))
    out = {}

    def sink(t):
        ids_, images = t.column("object_id"), t.column("image")
        for i in range(t.num_rows):
            oid = int(str(ids_[i].as_py()))
            cut = mmu_cutout(images.slice(i, 1).to_pylist()[0], pos.loc[oid], want_bands, n, scale,
                             mask_true_is_valid)
            if cut is not None:
                out[oid] = cut

    scan_mmu(hats_url, [int(v) for v in objects["object_id"]], ["image"], sink, box, cache, batch_rows)
    return out


def mmu_cutout(image, row, want_bands, n, scale, mask_true_is_valid=True):
    # One MMU image value -> cutout dict with its (assumed) WCS, or None without a wanted band.
    cut = parse_mmu_image(image, want_bands, n, scale, mask_true_is_valid)
    if cut is None:
        return None
    ra, dec = float(row["ra"]), float(row["dec"])
    cut["wcs"] = tan_wcs(ra, dec, n, scale, (n - 1) / 2.0 + cut["dx"], (n - 1) / 2.0 + cut["dy"])
    cut["wcs_assumed"] = True
    cut["source"] = "mmu"
    return cut


def mmu_cutouts(df, objects, want_bands, n, scale, mask_true_is_valid=True):
    # Rows read by read_mmu_rows (object_id, image) -> {object_id: cutout dict}.
    pos = objects.set_index(objects["object_id"].astype("int64"))
    out = {}
    for oid, image in zip(df["object_id"], df["image"]):
        cut = mmu_cutout(image, pos.loc[int(str(oid))], want_bands, n, scale, mask_true_is_valid)
        if cut is not None:
            out[int(str(oid))] = cut
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
        return fetch_mmu(ec["hats_url"], objects, bands, n, scale, box, mmu_cache_dir(ec),
                         int(ec.get("mmu_batch_rows", 64)), bool(ec.get("mmu_mask_true_is_valid", True)))
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
