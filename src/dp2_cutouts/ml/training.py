# training.py
# Anna Preto
# AstroDeep Project
# Pieces shared by the Stage 1 and Stage 2 trainings: run folder, seeds, callbacks (best checkpoint of the trainable
# part only, resumable backups for the 20 h walltime of the A100 partition), the proof that frozen parts did not move,
# and the nonlinear normalisation written with keras.ops.

import json
import platform
import subprocess
import time
from pathlib import Path

import keras
import numpy as np
from keras import ops

from .debvader_bridge import weights_fingerprint


def set_seeds(seed):
    keras.utils.set_random_seed(int(seed))


def git_commit(repo=None):
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo or Path(__file__).parent,
                              capture_output=True, text=True, timeout=10).stdout.strip() or "n/a"
    except Exception:  # noqa: BLE001 - no git on compute nodes
        return "n/a"


def environment():
    out = dict(python=platform.python_version(), keras=keras.__version__, backend=keras.backend.backend(),
               numpy=np.__version__, host=platform.node(), commit=git_commit())
    try:
        import jax
        out["jax"] = jax.__version__
        out["devices"] = [str(d) for d in jax.devices()]
    except Exception:  # noqa: BLE001
        pass
    return out


def denormalise_ops(x, scale_per_band, beta):
    # Inverse of DebVAder's nonlinear normalisation (same clipping as NonLinearNorm.inverse), keras.ops.
    x = ops.clip(x, -1 + 1e-3, 1 - 1e-3)
    return ops.sinh(ops.arctanh(x)) / beta * ops.convert_to_tensor(np.asarray(scale_per_band, np.float32))


class FrozenGuard:
    # Fingerprints of the parts that must not move; check() raises if one did.
    def __init__(self, **parts):
        self.parts = parts
        self.before = {k: weights_fingerprint(v) for k, v in parts.items()}

    def check(self):
        after = {k: weights_fingerprint(v) for k, v in self.parts.items()}
        moved = [k for k in after if after[k] != self.before[k]]
        if moved:
            raise RuntimeError(f"frozen parts changed during training: {moved}")
        return after


class BestSubmodels(keras.callbacks.Callback):
    # Saves the trainable sub-models (not the frozen VAE) when the monitored validation metric improves, and the
    # last state every epoch.
    def __init__(self, run_dir, submodels, monitor="val_total_loss"):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.submodels = submodels
        self.monitor = monitor
        self.best_path = self.run_dir / "best.json"
        self.best = json.loads(self.best_path.read_text())["value"] if self.best_path.exists() else np.inf

    def _save(self, tag):
        for name, m in self.submodels.items():
            m.save_weights(str(self.run_dir / f"{name}_{tag}.weights.h5"))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._save("last")
        v = logs.get(self.monitor)
        if v is not None and np.isfinite(v) and v < self.best:
            self.best = float(v)
            self._save("best")
            self.best_path.write_text(json.dumps(dict(value=self.best, epoch=int(epoch) + 1,
                                                      time=time.strftime("%Y-%m-%d %H:%M:%S"))))


class StopOnNaN(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        v = (logs or {}).get("total_loss")
        if v is not None and not np.isfinite(v):
            raise RuntimeError(f"non-finite loss at batch {batch}: {v}")


def callbacks(run_dir, submodels, patience, resume, backup_every_steps=None):
    # backup_every_steps: resumable backup every that many steps instead of every epoch (an epoch of the full D1 is
    # thousands of steps; a job cut by its walltime resumes from the last backup and restarts that epoch from its
    # first batch, the weights keeping what the cut part had learnt: keras 3 BackupAndRestore, tested).
    run_dir = Path(run_dir)
    cbs = [BestSubmodels(run_dir, submodels), StopOnNaN(),
           keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
           keras.callbacks.EarlyStopping(monitor="val_total_loss", mode="min", patience=int(patience))]
    if resume:
        freq = int(backup_every_steps) if backup_every_steps else "epoch"
        cbs.insert(0, keras.callbacks.BackupAndRestore(str(run_dir / "backup"), save_freq=freq,
                                                       delete_checkpoint=False))
    return cbs


def schedule(settings, n_train_batches):
    # Number of epochs: settings.train_steps (a budget in optimisation steps, the same whatever the size of the
    # training set) when given, settings.epochs otherwise.
    steps = settings.get("train_steps")
    if steps:
        return max(1, int(np.ceil(int(steps) / max(int(n_train_batches), 1))))
    return int(settings["epochs"])


def validation_indices(meta, split, max_examples, seed):
    # Rows of `split` used for validation: all of them, or a fixed random subset of max_examples (validating on the
    # whole val split of the full run would take as long as a training epoch). None: let the dataset take the split.
    if not max_examples:
        return None
    rows = np.flatnonzero((meta["split"] == split).to_numpy())
    if len(rows) <= int(max_examples):
        return None
    return np.sort(np.random.default_rng(seed).choice(rows, size=int(max_examples), replace=False))


def optimizer(lr, clipnorm):
    kw = dict(learning_rate=float(lr))
    if clipnorm and float(clipnorm) > 0:
        kw["clipnorm"] = float(clipnorm)
    return keras.optimizers.Adam(**kw)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def prepare_run_dir(run_dir, resume, force):
    # False when the run is finished and must be left alone. A fresh start (no resume, or force) removes what a
    # previous run left: its best value would otherwise stop the new run from saving its own best weights, and
    # BackupAndRestore would resume it.
    import shutil
    run_dir = Path(run_dir)
    if (run_dir / "DONE").exists() and not force:
        print(f"{run_dir.name}: already finished (DONE); --force to train again from scratch")
        return False
    if force or not resume:
        for p in ["best.json", "history.csv", "DONE", "frozen_check.json"] + \
                 [f.name for f in run_dir.glob("*_best.weights.h5")] + [f.name for f in run_dir.glob("*_last.weights.h5")]:
            (run_dir / p).unlink(missing_ok=True)
        shutil.rmtree(run_dir / "backup", ignore_errors=True)
    return True
