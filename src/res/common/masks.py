# masks.py
# Anna Preto
# AstroDeep Project
# Plans de masque des deep_coadd de DP2, lus dans l'en-tête de chaque découpe (mots-clés MSKNnnnn / MSKMnnnn),
# et bit propre à la chaîne pour les pixels remplis par du ciel vide au masquage des compagnes.

import numpy as np

# Bit ajouté par la chaîne, au-delà des neuf plans de DP2 : pixel remplacé par du ciel vide.
FILLED_BIT = 1 << 16
FILLED_NAME = "FILLED_SKY"


def plane_bits(header):
    # Nom du plan -> masque binaire, d'après l'en-tête FITS du plan MASK.
    bits = {}
    for key in header:
        if key.startswith("MSKN"):
            n = key[4:]
            bits[str(header[key]).strip()] = int(header[f"MSKM{n}"])
    if not bits:
        raise KeyError("aucun plan de masque dans l'en-tête (MSKN/MSKM)")
    return bits


def bitmask(bits, names):
    # Masque binaire réunissant les plans demandés ; un plan absent est une erreur, pas un oubli silencieux.
    missing = [n for n in names if n not in bits and n != FILLED_NAME]
    if missing:
        raise KeyError(f"plans de masque absents : {missing}")
    m = 0
    for n in names:
        m |= FILLED_BIT if n == FILLED_NAME else bits[n]
    return m


def any_set(maskbits, mask):
    return (np.asarray(maskbits) & mask) != 0
