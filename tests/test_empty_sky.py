import numpy as np

from dp2_cutouts.common import masks
from dp2_cutouts.stages.empty_sky import fill_window, rank_windows
from dp2_cutouts.stages.export_debvader import bad_mask
from dp2_cutouts.stages.stamps import SkyBank

BITS = {"NO_DATA": 1, "SATURATED": 2, "DETECTED": 4}
SIZE = 9


def fake_patch(seed=0):
    """3 bands, 36 x 36 pixels = a 4 x 4 grid of 9 x 9 windows, with known detected fractions."""
    rng = np.random.default_rng(seed)
    C, H, W = 3, 36, 36
    mask = np.zeros((C, H, W), np.int32)
    # Windows (row, col of the first pixel), 81 pixels each:
    #   (0, 0) clean;  (0, 9) 9 detected pixels, band 1 only;  (0, 18) 27 detected, band 0 only;
    #   (0, 27) 54 detected (2/3, above a 0.5 cap);  (9, 0) no detected pixel but one SATURATED;  the rest clean.
    mask[1, 0:1, 9:18] |= BITS["DETECTED"]
    mask[0, 0:3, 18:27] |= BITS["DETECTED"]
    mask[2, 0:6, 27:36] |= BITS["DETECTED"]
    mask[0, 13, 4] |= BITS["SATURATED"]
    detected = np.any((mask & BITS["DETECTED"]) != 0, axis=0)
    flux = rng.normal(size=(C, H, W)).astype(np.float32)
    flux[:, detected] += 100.0                   # sources
    var = np.ones((C, H, W), np.float32) + rng.uniform(0, 0.1, size=(C, H, W)).astype(np.float32)
    return dict(flux=flux, var=var, mask=mask, detected=detected, bits=BITS, x0=0, y0=0, meta={})


def test_rank_windows_order_and_hard_cuts():
    img = fake_patch()
    corners, frac, counts = rank_windows(img, ["SATURATED"], SIZE, 0.5, np.random.default_rng(1))
    det = img["detected"]
    expected = {(r, c): det[r:r + SIZE, c:c + SIZE].mean() for r in range(0, 36, 9) for c in range(0, 36, 9)}
    assert counts["n_windows"] == 16
    # the window with a SATURATED pixel is out; so is any window above the cap
    assert (9, 0) not in {tuple(x) for x in corners}
    assert all(expected[tuple(x)] <= 0.5 for x in corners)
    assert counts["n_candidates"] == sum(1 for k, v in expected.items() if v <= 0.5 and k != (9, 0))
    # emptiest first, and the reported fractions are the true ones
    assert np.all(np.diff(frac) >= 0)
    assert np.allclose(frac, [expected[tuple(x)] for x in corners])
    assert counts["n_clean"] == int((frac == 0).sum())


def test_fill_window_replaces_detected_pixels_with_clean_sky():
    img = fake_patch()
    unusable = masks.bitmask(BITS, ["DETECTED", "NO_DATA", "SATURATED"])
    bank = SkyBank(img, unusable, SIZE, stride=1)
    r, c = 0, 18                                            # 27 detected pixels, in band 0 only
    need = img["detected"][r:r + SIZE, c:c + SIZE]
    flux, var, mbits, bad = fill_window(img, bank, r, c, SIZE, np.random.default_rng(2), tries=5000)
    assert bad == 0.0
    # filled pixels: the source is gone in every band, FILLED_SKY set, DETECTED cleared, variance copied (not 0)
    assert np.all(np.abs(flux[:, need]) < 10)
    assert np.all((mbits[:, need] & masks.FILLED_BIT) != 0)
    assert np.all((mbits[:, need] & BITS["DETECTED"]) == 0)
    assert np.all(var[:, need] >= 1.0)
    # untouched pixels are the original ones
    assert np.array_equal(flux[:, ~need], img["flux"][:, r:r + SIZE, c:c + SIZE][:, ~need])
    # export: nothing masked once every detected pixel is filled
    m = bad_mask(mbits, var, BITS, ["NO_DATA", "SATURATED", "DETECTED"])
    assert not m.any()


def test_fill_window_clean_window_is_a_plain_copy():
    img = fake_patch()
    bank = SkyBank(img, masks.bitmask(BITS, ["DETECTED"]), SIZE)
    flux, var, mbits, bad = fill_window(img, bank, 27, 27, SIZE, np.random.default_rng(0), tries=10)
    assert bad == 0.0 and np.array_equal(flux, img["flux"][:, 27:36, 27:36])
    assert not np.any(mbits & masks.FILLED_BIT)


def test_unfilled_detected_pixels_are_masked_at_export():
    # every candidate sky window is unusable: the fill cannot find clean sky, the pixels keep DETECTED
    img = fake_patch()
    img["mask"][:] |= BITS["DETECTED"]
    img["detected"][:] = True
    bank = SkyBank(img, masks.bitmask(BITS, ["DETECTED"]), SIZE)
    flux, var, mbits, bad = fill_window(img, bank, 0, 0, SIZE, np.random.default_rng(0), tries=20)
    assert bad == 1.0
    assert bad_mask(mbits, var, BITS, ["NO_DATA", "DETECTED"]).all()
