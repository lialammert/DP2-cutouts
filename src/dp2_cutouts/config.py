# config.py
# Anna Preto
# AstroDeep Project
# Reads a YAML configuration file and lays out the data folders of a case.

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Subfolders of data_root. Every name says what the folder holds; use these constants, not string literals.
EUCLID_DIR = "euclid_catalog"       # Euclid MER catalogue in the sky box, and the tiles that copaings reads
RUBIN_DIR = "rubin_data"            # Rubin object table (catalog/) and, for the SODA path, the coadd images (images/)
COPAINGS_DIR = "copaings_matching"  # copaings working files: matching groups, blending entropy S_b
TABLES_DIR = "tables_and_stamps"    # catalog.parquet (one row per object + catalogue cuts), table.parquet
                                    # (+ pixel cuts), stamps.parquet (pixel arrays), stamps.json
PACKAGE_DIR = "final_dataset"       # packaged dataset (MultimodalUniverse format)


@dataclass
class Config:
    raw: dict
    path: Path

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path) as f:
            return cls(raw=yaml.safe_load(f), path=path)

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def name(self):
        return self.raw["name"]

    @property
    def sky_box(self):
        return list(self.raw["sky_box"])

    @property
    def data_root(self):
        return Path(os.path.expanduser(self.raw["data_root"]))

    def data(self, *parts, makedirs=True):
        # Path under data_root; creates the parent folder if needed.
        p = self.data_root.joinpath(*parts)
        if makedirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p
