# test_stamps.py
# Anna Preto
# AstroDeep Project
# The two pitfalls of the recipe, on a synthetic image: the (column, row) order of the WCS against the
# [row, column] indexing of arrays, and the FITS origin 1 against the array origin 0.

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
