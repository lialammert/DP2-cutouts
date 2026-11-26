# test_stamps.py
# Anna Preto
# AstroDeep Project
# The two pitfalls of the recipe, on a synthetic image: the (column, row) order of the WCS against the
# [row, column] indexing of arrays, and the FITS origin 1 against the array origin 0.

from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from dp2_cutouts.common.shapes import adaptive_moments


def tract_like_wcs():
    # Same form as the deep_coadd WCS: TAN, x towards west (PC1_1 < 0), y towards north, 0.2 arcsec.
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [53.1, -27.8]
    w.wcs.crpix = [101.0, 81.0]  # FITS: reference pixel as 1-based indices
    s = 0.2 / 3600
    w.wcs.cd = [[-s, 0.0], [0.0, s]]
    return w


def gaussian(shape, xc, yc, sx, sy):
    rows, cols = np.indices(shape)
    return np.exp(-0.5 * (((cols - xc) / sx) ** 2 + ((rows - yc) / sy) ** 2))


def test_world_to_pixel_is_column_row_zero_based():
    w = tract_like_wcs()
    x, y = w.world_to_pixel_values(53.1, -27.8)
    # The FITS reference pixel (101, 81) is array index [row 80, column 100].
    assert np.isclose(x, 100.0) and np.isclose(y, 80.0)


def test_stamp_centered_and_not_transposed():
    w = tract_like_wcs()
    shape = (160, 200)  # row, column: not square so that a transposition shows
    ra, dec = w.pixel_to_world_values(130.3, 60.7)
    img = gaussian(shape, 130.3, 60.7, 4.0, 1.5)  # elongated along x (columns): e1 > 0
    x, y = w.world_to_pixel_values(ra, dec)
    ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
    stamp = img[iy - 29:iy + 30, ix - 29:ix + 30]
    peak = np.unravel_index(np.argmax(stamp), stamp.shape)
    assert peak == (29, 29)
    m = adaptive_moments(stamp, center=(x - (ix - 29), y - (iy - 29)))
    assert m["ok"] and m["e1"] > 0.5 and abs(m["e2"]) < 0.01
    # The same galaxy, stamp transposed by mistake: e1 changes sign.
    mt = adaptive_moments(stamp.T, center=(29, 29))
    assert mt["e1"] < -0.5


def test_companion_light_left_on_the_target():
    # Footprint masking keeps the wings of a companion sharing the target footprint (only shared_k times its
    # ellipse is filled). companion_light_ratio predicts that light on the kept pixels around the target: large for
    # a bright neighbour, small for a faint one, 0 when alone; a Euclid duplicate of a Rubin companion counts once.
    import pandas as pd

    from dp2_cutouts.stages.stamps import companion_light_ratio, companion_mask_footprint

    shape = (41, 41)
    usable = np.ones(shape, bool)
    target = (14.0, 20.0, 2.0, 2.0, 0.0, 25.0)

    def comp_at(xc, flux, survey="rubin"):
        return pd.DataFrame(dict(x=[xc], y=[20.0], a=[4.0], b=[4.0], theta=[0.0], flux=[flux], survey=[survey]))

    flux = gaussian(shape, 14.0, 20.0, 2.0, 2.0) + 20.0 * gaussian(shape, 26.0, 20.0, 4.0, 4.0)
    bright = comp_at(26.0, 2000.0)
    need, info = companion_mask_footprint(flux > 0.02, shape, 0, 0, target, bright, 10.0, 2, 2.0)
    assert info["n_shared"] == 1 and info["flux_ratio_max"] == 80.0
    assert need[20, 26] and not need[20, 14]                      # companion core masked, target kept
    r_bright = companion_light_ratio(shape, target, bright, ~need & usable)
    assert r_bright > 0.3

    faint = comp_at(26.0, 2.0)
    r_faint = companion_light_ratio(shape, target, faint, ~need & usable)
    assert r_faint < 0.01 and r_faint == pytest.approx(r_bright / 1000.0)

    twice = pd.concat([bright, comp_at(26.5, 2000.0, "euclid")], ignore_index=True)
    assert companion_light_ratio(shape, target, twice, ~need & usable) == pytest.approx(r_bright)

    alone = comp_at(0.0, 0.0).iloc[:0]
    assert companion_light_ratio(shape, target, alone, usable) == 0.0
    assert np.isnan(companion_light_ratio(shape, (*target[:5], np.nan), bright, usable))


def test_segmentation_masks_the_neighbour_segment_not_the_target_fragments():
    # Euclid cuts the target into two segments (1 under the centre, 3 inside its ellipse) and a neighbour (2)
    # touches it. Only the neighbour is masked, grown by seg_grow_px, never into the target's segments.
    from dp2_cutouts.stages.stamps import companion_mask_segmentation

    shape = (21, 21)
    lab = np.zeros(shape, np.int64)
    lab[8:13, 6:11] = 1      # target, centre (10, 8) in (row, column) = (y, x) ... x 6-10
    lab[8:13, 11:13] = 3     # fragment of the target, centroid x 11.5, inside its ellipse
    lab[8:13, 13:19] = 2     # neighbour touching the fragment
    seg = np.broadcast_to(lab, (4,) + shape).copy()
    detected = lab > 0       # one merged Rubin footprint
    target = (8.0, 10.0, 4.0, 3.0, 0.0)
    need, info = companion_mask_segmentation(detected, seg, shape, 0, 0, target, 2, 0)
    assert info["n_target_segments"] == 2 and info["n_other_segments"] == 1
    assert need[10, 15] and not need[10, 8] and not need[10, 12]
    grown, _ = companion_mask_segmentation(detected, seg, shape, 0, 0, target, 2, 2)
    assert grown[10, 19] and grown[10, 20] and not grown[10, 12]   # grows outwards, not into the fragment
    assert (grown & (lab == 1)).sum() == 0 and (grown & (lab == 3)).sum() == 0


def test_euclid_segmentation_is_sampled_on_the_rubin_grid(tmp_path):
    # A 0.1" Euclid tile and a 0.2" Rubin grid on the same sky: every Rubin pixel sees the 2 x 2 Euclid pixels it
    # covers, so a one-Euclid-pixel segment reaches exactly one Rubin pixel.
    from astropy.io import fits

    from dp2_cutouts.sources.euclid_seg import seg_on_grid

    def tan(scale, crpix):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [53.1, -27.8]
        w.wcs.crpix = crpix
        w.wcs.cd = [[-scale / 3600, 0.0], [0.0, scale / 3600]]
        return w

    lab = np.zeros((200, 200), np.int64)
    lab[100, 100] = 7                      # Euclid pixel (x 100, y 100) = Rubin pixel (50, 50) below
    lab[40:60, 140:160] = 9
    path = tmp_path / "seg.fits"
    fits.PrimaryHDU(lab, header=tan(0.1, [101.0, 101.0]).to_header()).writeto(path)
    wr = tan(0.2, [50.75, 50.75])          # Rubin pixel 50 spans Euclid pixels 100 and 101 (0-based)
    out = seg_on_grid([path], wr, (100, 100))
    assert out.shape == (4, 100, 100)
    seen7 = np.argwhere((out == 7).any(axis=0))
    assert seen7.tolist() == [[50, 50]]
    assert (out == 7).sum() == 1
    nine = (out == 9).any(axis=0)
    assert nine[25, 75] and nine.sum() == 100


def test_segmentation_finds_a_tight_target_segment_off_centre():
    # A faint target whose VIS segment is tiny and one pixel off the Rubin centre, outside its small ellipse: the
    # Euclid counterpart position (or the nearest segment) identifies it, so it is not masked as a neighbour.
    from dp2_cutouts.stages.stamps import companion_mask_segmentation

    shape = (21, 21)
    lab = np.zeros(shape, np.int64)
    lab[10, 12] = 4                       # target segment: one pixel, 2 px right of the centre (10, 10)
    lab[10, 16:19] = 5                    # a real neighbour
    seg = np.broadcast_to(lab, (4,) + shape).copy()
    detected = np.zeros(shape, bool)
    target = (10.0, 10.0, 1.0, 1.0, 0.0)  # small ellipse: the segment centroid (12, 10) is outside it
    need, info = companion_mask_segmentation(detected, seg, shape, 0, 0, target, 2, 0)
    assert not need[10, 12] and need[10, 17]                  # nearest segment within 2 px is the target's
    need, info = companion_mask_segmentation(detected, seg, shape, 0, 0, target, 2, 0, counterpart=(12.2, 9.9))
    assert not need[10, 12] and need[10, 17] and info["n_target_segments"] == 1


def test_overlapping_euclid_tiles_are_not_mixed(tmp_path):
    # Two MER tiles cover the same galaxy with different segment numbers and sizes (7, 4 px radius, in tile A whose
    # core holds the galaxy; 9, 6 px radius, in tile B where it sits near the edge). Mixing them pixel by pixel left a
    # ring of 9 around 7, masked as a neighbour. Each point must take the tile where it is farthest from the edge.
    from astropy.io import fits

    from dp2_cutouts.sources.euclid_seg import seg_on_grid

    def tan(scale, crpix):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [53.1, -27.8]
        w.wcs.crpix = crpix
        w.wcs.cd = [[-scale / 3600, 0.0], [0.0, scale / 3600]]
        return w

    yy, xx = np.indices((200, 200))
    a = np.where(np.hypot(xx - 100, yy - 100) <= 4, 7, 0).astype(np.int64)       # galaxy at the centre of A
    pa = tmp_path / "a.fits"
    fits.PrimaryHDU(a, header=tan(0.1, [101.0, 101.0]).to_header()).writeto(pa)
    # Tile B: the same sky shifted so that the galaxy sits 15 px from B's left edge, with a larger segment.
    b = np.where(np.hypot(xx - 15, yy - 100) <= 6, 9, 0).astype(np.int64)
    pb = tmp_path / "b.fits"
    fits.PrimaryHDU(b, header=tan(0.1, [16.0, 101.0]).to_header()).writeto(pb)
    wr = tan(0.2, [50.75, 50.75])
    for order in ([pa, pb], [pb, pa]):
        out = seg_on_grid(order, wr, (100, 100))
        assert (out == 7).any() and not (out == 9).any()        # A's core wins everywhere around the galaxy


def test_segmentation_checked_by_the_catalogue():
    # Segment 1 holds the target and a catalogued source the segmentation merged with it: shared between them by
    # reduced radius, the source's share masked. Segment 2, uncatalogued, touches the target: a piece of it, kept.
    # Segment 3, uncatalogued and far: masked. A catalogued source in the sky (no segment): masked over its ellipse.
    import pandas as pd

    from dp2_cutouts.stages.stamps import companion_mask_segmentation_catalog

    shape = (31, 31)
    yy, xx = np.indices(shape)
    lab = np.zeros(shape, np.int64)
    lab[12:19, 6:24] = 1                                  # target (10, 15) + merged source (20, 15)
    lab[(np.hypot(xx - 12, yy - 20) <= 1.5)] = 2          # uncatalogued clump next to the target
    lab[(np.hypot(xx - 27, yy - 27) <= 1.5)] = 3          # uncatalogued, far
    seg = np.broadcast_to(lab, (4,) + shape).copy()
    comp = pd.DataFrame(dict(x=[20.0, 3.0], y=[15.0, 3.0], a=[1.5, 1.5], b=[1.5, 1.5], theta=[0.0, 0.0],
                             survey=["rubin", "euclid"]))
    target = (10.0, 15.0, 3.0, 3.0, 0.0)
    need, info = companion_mask_segmentation_catalog(np.zeros(shape, bool), seg, shape, 0, 0, target, comp, 10.0,
                                                     2, 0)
    assert info["n_split_segments"] == 1 and info["n_orphan_segments_kept"] == 1
    assert need[15, 20] and not need[15, 10] and not need[15, 13]   # merged source masked, target kept
    assert not need[20, 12]                                         # the clump stays with the target
    assert need[27, 27] and need[3, 3]                              # far orphan, catalogued source in the sky


def test_segmentation_catalog_leaves_sources_rubin_cannot_see():
    # Depth rule: a segment owned only by a Euclid source too faint for Rubin (visible = False) is left as it is,
    # and such a source in the sky is not masked over its ellipse; a visible one still is.
    import pandas as pd

    from dp2_cutouts.stages.stamps import companion_mask_segmentation_catalog

    shape = (31, 31)
    yy, xx = np.indices(shape)
    lab = np.zeros(shape, np.int64)
    lab[(np.hypot(xx - 10, yy - 15) <= 3)] = 1            # target
    lab[(np.hypot(xx - 24, yy - 24) <= 2)] = 2            # faint Euclid source, segmented
    lab[(np.hypot(xx - 24, yy - 6) <= 2)] = 3             # bright Euclid source, segmented
    seg = np.broadcast_to(lab, (4,) + shape).copy()
    comp = pd.DataFrame(dict(x=[24.0, 24.0, 4.0, 4.0], y=[24.0, 6.0, 4.0, 26.0], a=1.5, b=1.5, theta=0.0,
                             survey="euclid", visible=[False, True, False, True]))
    need, _ = companion_mask_segmentation_catalog(np.zeros(shape, bool), seg, shape, 0, 0, (10.0, 15.0, 3.0, 3.0, 0.0),
                                                  comp, 10.0, 2, 0)
    assert not need[24, 24] and need[6, 24]              # faint segment left, bright segment masked
    assert not need[4, 4] and need[26, 4]                # faint source in the sky left, bright one masked


def test_shared_segment_is_split_by_predicted_brightness():
    # A segment holding a bright target and a catalogued neighbour near its edge. Split by reduced radius, a faint
    # neighbour took a band of the target's wings; split by predicted brightness it takes nothing (the target is
    # brighter everywhere, even at the neighbour's centre), while a neighbour of comparable flux keeps its core.
    import pandas as pd

    from dp2_cutouts.stages.stamps import companion_mask_segmentation_catalog

    shape = (31, 31)
    lab = np.zeros(shape, np.int64)
    lab[10:21, 4:20] = 1
    seg = np.broadcast_to(lab, (4,) + shape).copy()
    target = (10.0, 15.0, 3.0, 3.0, 0.0, 1.0e4)

    def run(flux):
        comp = pd.DataFrame(dict(x=[16.0], y=[15.0], a=[1.5], b=[1.5], theta=[0.0], survey=["rubin"], flux=[flux]))
        need, _ = companion_mask_segmentation_catalog(np.zeros(shape, bool), seg, shape, 0, 0, target, comp, 10.0, 2, 0)
        return need

    faint = run(10.0)
    assert not faint[15, 16] and not faint[15, 18] and not faint[15, 10]
    comparable = run(3.0e3)
    assert comparable[15, 16] and not comparable[15, 10] and not comparable[15, 12]


def test_halo_ring_sees_the_wing_left_past_a_mask():
    # A companion masked over its core only: its wing in the ring just outside the mask makes the ring brighter
    # than the sky; masked far enough, the ring is plain sky. The target's own region never counts.
    from scipy.ndimage import binary_dilation

    from dp2_cutouts.stages.stamps import halo_ring

    rng = np.random.default_rng(0)
    shape = (41, 41)
    yy, xx = np.indices(shape)
    r_comp = np.hypot(xx - 30, yy - 20)
    flux = rng.normal(0.0, 1.0, shape) + 40.0 * np.exp(-r_comp / 3.0)
    var = np.ones(shape)
    usable = np.ones(shape, bool)
    r_target = np.hypot(xx - 10, yy - 20) / 3.0
    tight = r_comp <= 3
    loose = r_comp <= 14
    mean_t, n_t, z_t = halo_ring(flux, var, tight, usable, r_target)
    mean_l, n_l, z_l = halo_ring(flux, var, loose, usable, r_target)
    assert n_t > 0 and n_l > 0 and z_t > 10 and abs(z_l) < 4 and mean_t > mean_l
    assert np.isnan(halo_ring(flux, var, np.zeros(shape, bool), usable, r_target)[0])
    near_target = binary_dilation(np.hypot(xx - 10, yy - 20) <= 1, iterations=1)
    assert halo_ring(flux, var, near_target, usable, r_target)[1] == 0      # ring inside 2 target ellipses only


def test_two_tasks_downloading_the_same_seg_tile_do_not_clash(tmp_path, monkeypatch):
    # Neighbouring patches fetched at the same time share MER tiles: the second download must not fail on, or
    # corrupt, the first one's temporary file.
    import threading

    from dp2_cutouts.sources import euclid_seg
    payload = b"x" * 1000
    started, release = threading.Barrier(2), threading.Event()

    class Response:
        headers = {"Content-Length": str(len(payload))}
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk):
            started.wait()  # both tasks are mid-download
            yield payload[:500]
            release.wait(5)
            yield payload[500:]

    monkeypatch.setattr(euclid_seg.requests, "get", lambda *a, **k: Response())
    import os
    monkeypatch.setattr(os, "getpid", lambda: threading.get_ident())
    results, errors = [], []

    def task():
        try:
            results.append(euclid_seg.download("https://x/MER_SEG/1/EUC_MER_FINAL-SEGMAP_TILE1-A.fits", tmp_path))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=task) for _ in range(2)]
    for t in threads:
        t.start()
    release.set()
    for t in threads:
        t.join()
    assert not errors
    assert results[0] == results[1] and results[0].read_bytes() == payload
    assert sorted(p.name for p in tmp_path.iterdir()) == ["EUC_MER_FINAL-SEGMAP_TILE1-A.fits"]


def test_seg_tile_search_retries_irsa_outages(tmp_path, monkeypatch):
    # IRSA's SIA service fails now and then (pyvo DALServiceError): the tile search is retried, not the patch lost.
    from dp2_cutouts.sources import euclid_images, euclid_seg

    class DALServiceError(Exception):
        pass

    calls = []

    def find_tiles(*a):
        calls.append(1)
        if len(calls) < 3:
            raise DALServiceError("Unable to access the capabilities endpoint")
        return ["102018211"]

    monkeypatch.setattr(euclid_seg, "find_tiles", find_tiles)
    monkeypatch.setattr(euclid_seg, "seg_url", lambda tile: f"https://x/{tile}/T.fits")
    monkeypatch.setattr(euclid_seg, "download", lambda url, d: Path(d) / "T.fits")
    monkeypatch.setattr(euclid_images.time, "sleep", lambda s: None)
    monkeypatch.setattr(euclid_images.with_retries, "__defaults__",
                        (6, 10.0, 300.0, lambda s: None, None))
    assert euclid_seg.fetch_for_box(52.9, -28.1, 0.1, tmp_path) == [tmp_path / "102018211" / "T.fits"]
    assert len(calls) == 3


def test_euclid_arbiter_masks_only_what_euclid_sees():
    # masking.euclid_arbiter: a Rubin object with no Euclid source nearby (a deblender piece of the target's halo)
    # and a Rubin footprint holding no Euclid source are left unmasked; a Rubin neighbour Euclid also sees is
    # masked as before.
    import pandas as pd

    from dp2_cutouts.stages.stamps import companion_mask_segmentation_catalog

    shape = (41, 41)
    lab = np.zeros(shape, np.int64)
    lab[5:36, 5:36] = 1                                 # one Euclid segment: the target and its halo
    seg = np.broadcast_to(lab, (4,) + shape).copy()
    det = np.zeros(shape, bool)
    det[2:8, 33:39] = True                              # a separate Rubin footprint, nothing in Euclid
    det[30:36, 2:8] = True                              # a separate Rubin footprint with a Euclid source
    target = (20.0, 20.0, 3.0, 3.0, 0.0, 1.0e5)
    comp = pd.DataFrame(dict(x=[31.0, 5.0, 5.2], y=[20.0, 33.0, 33.1], a=[4.0, 1.0, 1.0], b=[4.0, 1.0, 1.0],
                             theta=0.0, survey=["rubin", "rubin", "euclid"], flux=[3.0e4, 50.0, 50.0]))

    def run(arbiter):
        need, _ = companion_mask_segmentation_catalog(det, seg, shape, 0, 0, target, comp, 10.0, 1, 0,
                                                      euclid_arbiter=arbiter)
        return need

    old, new = run(False), run(True)
    assert old[20, 31] and old[4, 35]                   # halo piece and Euclid-less footprint masked before
    assert not new[20, 31] and not new[4, 35]           # not any more
    assert old[33, 5] and new[33, 5]                    # the neighbour both surveys see stays masked
