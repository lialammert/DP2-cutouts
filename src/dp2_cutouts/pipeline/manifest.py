# manifest.py
# Anna Preto
# AstroDeep Project
# One JSON manifest per patch: status of each phase (done / skipped / failed), the configuration fingerprint it was
# run with, its counts and outputs, the last error. A phase is skipped on relaunch when it is done with the same
# configuration fingerprint; a change of configuration that matters to a phase reruns it and every later phase.

import hashlib
import json
import platform
import subprocess
import time
import traceback
from importlib import metadata
from pathlib import Path

PHASES = ("plan", "fetch", "process")

# Configuration sections that each phase depends on (a later phase also depends on the earlier ones).
PHASE_SECTIONS = {
    "plan": ("sources.rubin_catalog", "sources.euclid_catalog", "rubin", "euclid", "copaings", "cuts.snr_min",
             "patches.catalog_margin_arcsec"),
    # The masking method decides whether fetch also downloads the Euclid MER segmentation tiles (network).
    "fetch": ("sources.rubin_pixels", "masking.method", "masking.seg_cache"),
    "process": ("cuts", "masking"),
}


def _get(raw, dotted):
    v = raw
    for k in dotted.split("."):
        v = v.get(k) if isinstance(v, dict) else None
    return v


def config_hash(cfg, phase):
    sections = []
    for ph in PHASES[:PHASES.index(phase) + 1]:
        sections += PHASE_SECTIONS[ph]
    payload = json.dumps({s: _get(cfg.raw, s) for s in sections}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def software():
    def version(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    return dict(dp2_cutouts_commit=commit, dp2_cutouts=version("dp2-cutouts"), copaings=version("copaings"),
                python=platform.python_version())


def path(cfg, patch):
    return cfg.data("manifests", f"{patch['name']}.json")


def load(cfg, patch):
    p = path(cfg, patch)
    if p.exists():
        return json.loads(p.read_text())
    return dict(tract=int(patch["tract"]), patch=int(patch["patch"]), name=patch["name"], phases={})


def save(cfg, patch, man):
    p = path(cfg, patch)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=2, default=str))
    tmp.replace(p)  # atomic: an interrupted job never leaves a half-written manifest


def is_done(cfg, man, phase):
    st = man["phases"].get(phase, {})
    return st.get("status") in ("done", "skipped") and st.get("config_hash") == config_hash(cfg, phase)


def run_phase(cfg, patch, phase, fn, force=False):
    # Runs fn(cfg, patch) -> (status, info) unless the phase is already done with the same configuration.
    man = load(cfg, patch)
    if not force and is_done(cfg, man, phase):
        print(f"[{patch['name']}] {phase}: already done")
        return man["phases"][phase]
    for prev in PHASES[:PHASES.index(phase)]:
        if not is_done(cfg, man, prev):
            raise RuntimeError(f"[{patch['name']}] {phase}: phase {prev} is not done with the current configuration")
    t0 = time.time()
    try:
        status, info = fn(cfg, patch)
        entry = dict(status=status, **info)
    except Exception as e:
        entry = dict(status="failed", error_type=type(e).__name__, error=str(e)[:2000],
                     traceback=traceback.format_exc()[-4000:])
    entry.update(config_hash=config_hash(cfg, phase), seconds=round(time.time() - t0, 1),
                 finished=time.strftime("%Y-%m-%dT%H:%M:%S"), software=software())
    man["phases"][phase] = entry
    # A rerun phase invalidates the later ones.
    for later in PHASES[PHASES.index(phase) + 1:]:
        man["phases"].pop(later, None)
    save(cfg, patch, man)
    print(f"[{patch['name']}] {phase}: {entry['status']} ({entry['seconds']} s)"
          + (f" -- {entry['error_type']}: {entry['error']}" if entry["status"] == "failed" else ""))
    return entry
