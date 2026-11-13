# KL placeholder - export the products generated in these stages as debvader-compatible products, 
# based on  the follostatuswing
### 4.1 The dataset contract

# One format is shared by DP2-cutouts (writer), debvader-keras (reader, blend builder) and the pipeline stages. It is debvader's existing "real build" directory, extended:

# ```
# <dataset>/
#   images.npy      (N, H, W, C) float32   flux, nJy                         [exists today]
#   variance.npy    (N, H, W, C) float32   variance plane, nJy^2             [new]
#   mask.npy        (N, H, W, C) bool      True = ignore this pixel          [new]
#   psf.npy         (N, P, P, C) float32   PSF stamp, each band sums to 1    [exists today]
#   targets.npy     (N, H, W, C) float32   blends only: what to recover      [exists today]
#   meta.npz        per-object columns (below)                               [exists, extended]
#   stats.json      {"nonlinear": {"scale_per_band": [...]}}                 [exists today]
#   dataset.json    instrument, bands, pixel_scale, stamp_arcsec, stamp_px,
#                   psf_px, unit, kind, provenance                           [new]
# ```

# `meta.npz` columns:

# | Key | Meaning | Needed by |
# |---|---|---|
# | `split` | 0 train, 1 validation, 2 test; assigned by patch | loader, blend builder |
# | `metadata_id` | Rubin `objectId` | everything |
# | `ra`, `dec`, `tract`, `patch` | position | QA, evaluation |
# | `offset_x`, `offset_y` | sub-pixel offset of the object from the stamp centre | evaluation |
# | `metadata_magnitude` | i-band cModel magnitude | blend builder |
# | `metadata_e1`, `metadata_e2` | shape of the object on the stamp (HSM) | blend builder |
# | `hsm_sigma`, `hsm_resolution` | size and resolution factor | blend builder |
# | `masked_fraction` | fraction of pixels filled with empty sky | QA |
# | `sb_euclid`, `has_euclid_counterpart`, `cut*_pass`, `cut*_value` | isolation and blend flags | evaluation |
# | `psf_fwhm` | (N, C) FWHM per band, arcsec | `MeasurePsf` |

# `dataset.json` example:

# ```json
# {
#   "instrument": "rubin",
#   "bands": ["u", "g", "r", "i", "z", "y"],
#   "pixel_scale": 0.2,
#   "stamp_arcsec": 11.8,
#   "stamp_px": 59,
#   "psf_px": 35,
#   "unit": "nJy",
#   "kind": "isolated",
#   "source": {"repo": "dp2", "collection": "dp2", "skymap": "lsst_cells_v2"}
# }
# ```

# Conventions that everything depends on:

# - Arrays are channel-last, indexed `[row, column, band]`.
# - The object sits on the centre pixel `(H // 2, W // 2)` to within half a pixel; `offset_x`, `offset_y` record the remainder.
# - The PSF stamp is centred on pixel `(P // 2, P // 2)`.
# - `mask` is True for `NO_DATA`, `SATURATED`, `COSMIC_RAY`, `DETECTION_EDGE`, and any pixel with non-finite or non-positive variance. Pixels filled with empty sky (`FILLED_SKY`) are **valid**: they hold real sky with the correct variance.
# - For the joint test, a dataset becomes a directory with one such sub-directory per instrument (`rubin/`, `euclid/`), row-aligned.

# Stamp size rule (put this function in both libraries; the pipeline passes the result):

# ```python
# def odd_pixels(size_arcsec: float, pixel_scale: float) -> int:
#     """Pixel count for an angular stamp size, always odd so that a centre pixel exists."""
#     n = int(round(size_arcsec / pixel_scale))
#     return n if n % 2 == 1 else n + 1

# # odd_pixels(11.8, 0.2) -> 59;  odd_pixels(11.8, 0.1) -> 119
# ```

# ---