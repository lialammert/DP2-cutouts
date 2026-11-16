# mlconfig.py
# Anna Preto
# AstroDeep Project
# Configuration of the cross-instrument work (config/ml_jeanzay.yaml): environment variables expanded, command-line
# overrides (section.key=value), and the folder layout under ml_root. Kept apart from config.py, which describes one
# case of the stamp pipeline.

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _expand(value):
    if isinstance(value, str):
        def repl(m):
            name = m.group(1) or m.group(2)
            if name not in os.environ:
                raise KeyError(f"environment variable {name} is not set (needed by the ML configuration)")
            return os.environ[name]
        return os.path.expanduser(_ENV.sub(repl, value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _parse_value(text):
    # YAML parsing of a command-line value: 0.1 -> float, true -> bool, [a, b] -> list, anything else -> str.
    return yaml.safe_load(text)


def apply_overrides(raw, overrides):
    raw = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form section.key=value")
        key, value = item.split("=", 1)
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"override {item!r}: no section {p!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"override {item!r}: no key {parts[-1]!r} (a typo would otherwise be ignored silently)")
        node[parts[-1]] = _parse_value(value)
    return raw


@dataclass
class MLConfig:
    raw: dict
    path: Path

    @classmethod
    def load(cls, path, overrides=None):
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        raw = apply_overrides(raw, overrides)
        return cls(raw=_expand(raw), path=path)

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def root(self):
        return Path(self.raw["ml_root"])

    def path_of(self, *parts, makedirs=True):
        p = self.root.joinpath(*parts)
        if makedirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # Folder layout under ml_root.
    def dataset_dir(self, name):
        return self.root / "datasets" / name

    def run_dir(self, name):
        return self.root / "runs" / name

    def eval_dir(self, name):
        return self.root / "evaluations" / name

    def pipeline(self):
        from .config import Config
        return Config.load(self.raw["pipeline_config"])

    def fingerprint(self, sections):
        # Short hash of the configuration sections a product depends on, written next to it.
        sub = {k: self.raw.get(k) for k in sections}
        return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def snapshot(self, out_path):
        with open(out_path, "w") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False)
