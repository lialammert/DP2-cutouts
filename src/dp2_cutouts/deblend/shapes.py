# shapes.py
# Anna Preto
# AstroDeep Project
# Shape measurements of the evaluations (galsim HSM): adaptive moments and REGAUSS PSF-corrected distortions, with a
# Gaussian PSF model built from the catalogue PSF moments (the PSF images are not in the datasets).
# The evaluations measure every posterior sample (5 000 galaxies x 128 samples x 3 runs, in D1 and D2): one core
# would take hours, so shapes_many spreads the measurements over the job's cores (2026-11-28) with a pool of worker
# processes started with spawn (a fork of a process running JAX may deadlock). This module imports neither keras nor
# JAX, so the workers start fast. The results are the same as one by one, in the same order.

import os

import numpy as np

from ..pairs import common


def gaussian_psf_image(sigma_px, e1, e2, n=25):
    import galsim
    psf = galsim.Gaussian(sigma=max(float(sigma_px), 0.3)).shear(e1=float(np.clip(e1, -0.5, 0.5)),
                                                                 e2=float(np.clip(e2, -0.5, 0.5)))
    return psf.drawImage(nx=n, ny=n, scale=1.0).array


def shapes(img, valid, psf_sigma_px, psf_e1=0.0, psf_e2=0.0):
    # Adaptive moments (observed, PSF-convolved) and REGAUSS PSF-corrected distortion. NaN on failure.
    import galsim
    out = dict(e1=np.nan, e2=np.nan, sigma=np.nan, ok=False, e1_rg=np.nan, e2_rg=np.nan, ok_rg=False)
    n = img.shape[0]
    c = (n - 1) / 2.0
    m = common.moments(img, valid.astype(np.int32), (c, c))
    out.update(e1=m["e1"], e2=m["e2"], sigma=m["sigma"], ok=bool(m["ok"]))
    if not np.isfinite(psf_sigma_px):
        return out
    try:
        gi = galsim.Image(np.ascontiguousarray(img, np.float64), scale=1.0)
        bad = galsim.Image((~valid).astype(np.int32), scale=1.0)
        pi = galsim.Image(gaussian_psf_image(psf_sigma_px, psf_e1, psf_e2), scale=1.0)
        r = galsim.hsm.EstimateShear(gi, pi, badpix=bad, shear_est="REGAUSS", strict=False)
        if not r.error_message:
            out.update(e1_rg=float(r.corrected_e1), e2_rg=float(r.corrected_e2), ok_rg=True)
    except Exception:  # noqa: BLE001 - a failed measurement is recorded as such
        pass
    return out


def _chunk(args):
    imgs, valids, sig, e1, e2 = args
    return [shapes(imgs[k], valids[k], float(sig[k]), float(e1[k]), float(e2[k])) for k in range(len(imgs))]


def n_workers():
    # DP2_SHAPE_WORKERS, else the job's cores (SLURM_CPUS_PER_TASK), else 1 (no pool).
    return max(int(os.environ.get("DP2_SHAPE_WORKERS") or os.environ.get("SLURM_CPUS_PER_TASK") or 1), 1)


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        _POOL = ProcessPoolExecutor(n_workers(), mp_context=mp.get_context("spawn"))
    return _POOL


def shapes_many(imgs, valids, sig, e1=None, e2=None, min_parallel=256):
    # imgs, valids (N, n, n) one band; sig, e1, e2 (N,) PSF per image (pixels; e1, e2 default 0) -> list of N dicts.
    imgs = np.ascontiguousarray(imgs, np.float32)
    valids = np.ascontiguousarray(valids, bool)
    n = len(imgs)
    sig = np.broadcast_to(np.asarray(sig, float), (n,))
    e1 = np.zeros(n) if e1 is None else np.broadcast_to(np.asarray(e1, float), (n,))
    e2 = np.zeros(n) if e2 is None else np.broadcast_to(np.asarray(e2, float), (n,))
    w = n_workers()
    if w <= 1 or n < min_parallel:
        return _chunk((imgs, valids, sig, e1, e2))
    size = int(np.ceil(n / (4 * w)))
    jobs = [(imgs[s:s + size], valids[s:s + size], sig[s:s + size], e1[s:s + size], e2[s:s + size])
            for s in range(0, n, size)]
    out = []
    for part in _pool().map(_chunk, jobs):
        out.extend(part)
    return out
