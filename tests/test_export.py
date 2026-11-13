import numpy as np
import pytest

from dp2_cutouts.sources.rubin_butler import psf_stamp
from dp2_cutouts.stages.export_debvader import bad_mask
from dp2_cutouts.stages.stamps import split_of

SPLIT = dict(train=0.8, val=0.1, test=0.1, seed=0)


def test_split_by_patch_is_a_function_of_the_patch():
    a = split_of(5063, 34, SPLIT, 100, 200)
    assert all(split_of(5063, 34, SPLIT, x, y) == a for x, y in [(0, 0), (5000, 9000)])
    fractions = np.bincount([split_of(5063, p, SPLIT) for p in range(2000)], minlength=3) / 2000
    assert np.allclose(fractions, [0.8, 0.1, 0.1], atol=0.03)


def test_split_by_block():
    cfg = dict(SPLIT, unit="block", block_px=600)
    assert split_of(5063, 34, cfg, 12000, 9000) == split_of(5063, 34, cfg, 12599, 9599)   # same block
    assert len({split_of(5063, 34, cfg, x, y) for x in range(0, 30000, 600) for y in range(0, 6000, 600)}) == 3
    with pytest.raises(ValueError):
        split_of(5063, 34, cfg)


class FakeImage:
    def __init__(self, a):
        self.array = a


class BoundsError(Exception):
    pass


class FakePsf:
    def __init__(self, n=35, missing=False):
        y, x = np.mgrid[:n, :n] - n // 2
        self.k, self.missing = np.exp(-0.5 * (x * x + y * y) / 4.0), missing

    def compute_kernel_image(self, x, y):
        if self.missing:
            raise BoundsError("Cell is missing for this PSF.")
        return FakeImage(self.k)


def test_psf_stamp_centre_and_crop():
    img = {"psf": [FakePsf()]}
    for size in (35, 25, 41):
        p = psf_stamp(img, 0, 0.0, 0.0, size)
        assert p.shape == (size, size) and abs(p.sum() - 1) < 1e-6
        assert np.unravel_index(p.argmax(), p.shape) == (size // 2, size // 2)
    with pytest.raises(ValueError):
        psf_stamp(img, 0, 0.0, 0.0, 34)


def test_psf_stamp_missing_cell_returns_none():
    assert psf_stamp({"psf": [FakePsf(missing=True)]}, 0, 0.0, 0.0, 35) is None


def test_bad_mask():
    bits = {"NO_DATA": 1, "SATURATED": 8, "DETECTED": 128}
    maskbits = np.array([0, 1, 8, 128, 128 | (1 << 16), 0])
    var = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    m = bad_mask(maskbits, var, bits, ["NO_DATA", "SATURATED"])
    # DETECTED and FILLED_SKY (bit 16) are valid pixels; zero variance is not
    assert list(m) == [False, True, True, False, False, True]
