import numpy as np
import pandas as pd

from .stamps import load_images, SkyBank
from ..config import TABLES_DIR
from ..common import masks

# KL Incomplete-- the output is certanily not right here


def run(cfg):
    """ run the empty sky cutout stage."""

    n_cutouts = int(cfg["export"]["n_empty_per_patch"])
    size = int(cfg["cuts"]["stamp_px"])

    cat = pd.read_parquet(cfg.data(TABLES_DIR, "catalog.parquet"))
    patches = cfg["pixels"].get("patches") or sorted(
            {(int(t), int(p)) for t, p in zip(cat["tract"], cat["patch"])})


    for tract, patch in patches:
        img = load_images(cfg, tract, patch) 
        bad = masks.bitmask(img['bits'], ['DETECTED', 'NO_DATA'] + list(cfg["cuts"]["bad_planes"]))
        bank = SkyBank(img, bad, size, stride=size)
        clean = [c for c in bank.corners
                 if not bank.unusable[c[0]:c[0]+size, c[1]:c[1]+size].any()]

        # Log number of clean cutouts available for this patch
        cfg.log.info(f"Empty-sky cutouts: tract {tract}, patch {patch}, {len(clean)} clean cutouts.")
        
        rng = np.random.default_rng([int(cfg["cuts"]["seed"]), tract, patch])
        pick = rng.permutation(len(clean))[:n_cutouts]

        # for each picked corner: flux, var, mask bits, PSF stamp at the window centre, ra/dec
        # store the quantities in a table which will be exported as an npz per patch
        table = []
        for i in pick:
            c = clean[i]

            flux = bank.img["flux"][:, c[0]:c[0]+size, c[1]:c[1]+size]
            var = bank.img["var"][:, c[0]:c[0]+size, c[1]:c[1]+size]
            mask = bank.img["mask"][:, c[0]:c[0]+size, c[1]:c[1]+size]
            psf = bank.img["psf"][:, c[0]+size//2, c[1]+size//2]

            # where are ra and dec stored?
            ra = bank.img["ra"][c[0]+size//2, c[1]+size//2]
            dec = bank.img["dec"][c[0]+size//2, c[1]+size//2]

            table.append({
                "tract": tract,
                "patch": patch,
                "corner_x": c[1],
                "corner_y": c[0],
                "flux": flux,
                "var": var,
                "mask": mask,
                "psf": psf,
                "ra": ra,
                "dec": dec
            })

        # write one shard per patch: empty_tract_patch.npz
        np.savez_compressed(f"empty_{tract}_{patch}.npz",
                             **{k: [row[k] for row in table] for k in table[0].keys()})