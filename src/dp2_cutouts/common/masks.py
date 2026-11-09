# masks.py
# Anna Preto
# AstroDeep Project
# Mask planes of the DP2 deep_coadd images, read from the header of each cutout (MSKNnnnn / MSKMnnnn keywords),
# and the pipeline's own bit for pixels filled with empty sky during companion masking.

import numpy as np

# Bit added by the pipeline, beyond the nine DP2 planes: pixel replaced by empty sky.
FILLED_BIT = 1 << 16
FILLED_NAME = "FILLED_SKY"


def plane_bits(header):
    # Plane name -> bit mask, from the FITS header of the MASK plane.
    bits = {}
    for key in header:
        if key.startswith("MSKN"):
            n = key[4:]
            bits[str(header[key]).strip()] = int(header[f"MSKM{n}"])
    if not bits:
        raise KeyError("no mask plane in the header (MSKN/MSKM)")
    return bits


def bitmask(bits, names):
    # Bit mask combining the requested planes; a missing plane is an error, not a silent omission.
    missing = [n for n in names if n not in bits and n != FILLED_NAME]
    if missing:
        raise KeyError(f"missing mask planes: {missing}")
    m = 0
    for n in names:
        m |= FILLED_BIT if n == FILLED_NAME else bits[n]
    return m


def any_set(maskbits, mask):
    return (np.asarray(maskbits) & mask) != 0
