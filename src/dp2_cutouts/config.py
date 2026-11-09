# config.py
# Anna Preto
# AstroDeep Project
# Reads a YAML configuration file and lays out the data folders of a case.

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Subfolders of data_root, one per pipeline step.
EUCLID_DIR = "00_euclid"
RUBIN_DIR = "10_rubin"
COPAINGS_DIR = "20_copaings"


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
