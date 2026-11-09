# config.py
# Anna Preto
# AstroDeep Project
# Lecture d'un fichier de configuration YAML et disposition des dossiers de données d'un cas.

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Sous-dossiers de data_root, un par étape de la chaîne.
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
        # Chemin sous data_root ; crée le dossier parent au besoin.
        p = self.data_root.joinpath(*parts)
        if makedirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p
