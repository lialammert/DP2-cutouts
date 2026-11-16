# config.py
# Anna Preto
# AstroDeep Project
# Reads a YAML configuration file and lays out the data folders of a case.

import copy
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

    def derive(self, name, data_root, sky_box, **sections):
        # Configuration of a sub-case (one patch): same science parameters, its own name, folder and sky box.
        # `sections` overrides whole sections or keys of sections, e.g. rubin={"tracts": [5063]}.
        raw = copy.deepcopy(self.raw)
        raw.update(name=name, data_root=str(data_root), sky_box=[float(v) for v in sky_box])
        raw.pop("reference", None)
        for key, value in sections.items():
            if isinstance(value, dict) and isinstance(raw.get(key), dict):
                raw[key].update(value)
            else:
                raw[key] = value
        return Config(raw=raw, path=self.path)

    def data(self, *parts, makedirs=True):
        # Path under data_root; creates the parent folder if needed.
        p = self.data_root.joinpath(*parts)
        if makedirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p
