# tracking.py
# Anna Preto
# AstroDeep Project
# Optional Weights & Biases tracking of the deblending trainings (2026-11-28), to follow the campaign from a phone.
# On only when DP2_WANDB=1 (scripts/jeanzay/ml/env.sh); otherwise nothing is imported and nothing changes.
#
# Jean Zay compute nodes have no internet access: the runs are written offline (WANDB_MODE=offline) under
# WANDB_DIR ($SCRATCH/wandb), and scripts/jeanzay/ml/wandb_sync.slurm (prepost partition, which has internet access)
# uploads them every few minutes. wandb is not installed in the venv ($WORK must not grow): DP2_WANDB_LIB points to
# a `pip install --target` folder on $SCRATCH, appended to sys.path (after the venv, so the venv's packages win).
#
# What is sent: scalars (losses and metrics of the trainings every `every` updates, the epoch logs with the
# validation, gradient-norm quantiles and the learning rate) and the stage settings. Images only with
# DP2_WANDB_IMAGES=1 (Anna's choice, 2026-11-28, private W&B project; off by default because DP2 pixels then leave
# Jean Zay for W&B's servers): at every epoch, for 4 fixed validation examples, observed | model | residual of the
# stages that reconstruct images (Rubin VAE, K1, K2, C1: Rubin i band and / or VIS; C2: VIS through C1's decoder),
# arcsinh-stretched per example. No catalogue row is sent.
# One W&B run per run folder: its id is kept in <run>/wandb_id, so the jobs that resume a training append to the
# same W&B run; the step axis is the number of updates.

import os
import sys
import uuid
from pathlib import Path

import keras
import numpy as np


def enabled():
    return str(os.environ.get("DP2_WANDB", "")).lower() not in ("", "0", "false", "no")


def _import_wandb():
    lib = os.environ.get("DP2_WANDB_LIB")
    if lib and lib not in sys.path:
        sys.path.append(lib)
    try:
        import wandb
        return wandb
    except ImportError as e:
        print(f"[wandb] DP2_WANDB is set but wandb cannot be imported ({e}): no tracking", flush=True)
        return None


def images_enabled():
    return enabled() and str(os.environ.get("DP2_WANDB_IMAGES", "")).lower() not in ("", "0", "false", "no")


# (observed key, model output key, channel, title) of the reconstructions shown, when both keys exist.
PANELS = (("r_flux", "mean", 3, "Rubin i"), ("e_flux", "mean", 0, "VIS"), ("r_flux", "mean_r", 3, "Rubin i"),
          ("e_flux", "mean_e", 0, "VIS"), ("d2_e_flux", "mean_e", 0, "VIS of the D2 target"))


def panels(x, out, rubin_scale=None):
    # -> list of (title, uint8 image (H, 3 W)) per example and panel: observed | model | residual, one arcsinh
    # stretch per example (from the observed image), residual on the same stretch, centred on grey.
    res = []
    for ok, mk, ch, title in PANELS:
        # A panel when the observation and the model output exist and have the same channels (K2's "mean" is a
        # Rubin image: shown against r_flux, not against e_flux).
        if ok not in x or mk not in out or out[mk].shape[-1] != x[ok].shape[-1]:
            continue
        obs = np.asarray(x[ok])[..., ch]
        mod = np.asarray(keras.ops.convert_to_numpy(out[mk]))[..., ch]
        for i in range(len(obs)):
            s = float(np.percentile(np.abs(obs[i]), 99)) or 1.0
            st = lambda a: np.arcsinh(a / (0.1 * s)) / np.arcsinh(10.0)  # noqa: E731
            a, b = st(obs[i]), st(mod[i])
            r = 0.5 + 0.5 * np.clip(st(obs[i] - mod[i]), -1, 1)
            img = np.concatenate([np.clip(a, 0, 1), np.clip(b, 0, 1), r], axis=1)
            res.append((f"{title} #{i}", (255 * img).astype(np.uint8)))
    return res


def define_axes(run):
    # The epoch curves and the images plotted against the epoch (the updates stay the axis of train/*).
    try:
        run.define_metric("epoch")
        run.define_metric("epoch/*", step_metric="epoch")
        run.define_metric("images/*", step_metric="epoch")
    except Exception as e:  # noqa: BLE001 - cosmetic
        print(f"[wandb] define_metric skipped ({e})", flush=True)


def preview_outputs(model, x):
    # The model's outputs on the small preview batch, computed on the CPU from copies of the weights (JAX): on the GPU
    # every new input shape starts a cuDNN autotuning, which failed at every epoch on the V100 (2026-11-28, K1: no
    # image). Other backends: a plain call.
    if keras.backend.backend() != "jax":
        out = model(x, training=False)
        return {k: np.asarray(keras.ops.convert_to_numpy(v)) for k, v in out.items()}
    import jax
    cpu = jax.devices("cpu")[0]
    tv = [jax.device_put(v.value, cpu) for v in model.trainable_variables]
    ntv = [jax.device_put(v.value, cpu) for v in model.non_trainable_variables]
    xc = {k: jax.device_put(np.asarray(v), cpu) for k, v in x.items()}
    with jax.default_device(cpu):
        out, _ = model.stateless_call(tv, ntv, xc, training=False)
    return {k: np.asarray(v) for k, v in out.items()}


def finished_vae_panels(cfg, stage, n=4, split="val"):
    # Panels of a finished VAE stage from its best weights (scripts/inspect/wandb_vae_images.py), on the first n
    # examples of the split: rubin_vae (Rubin i), k1 (VIS), k2 (Rubin i from the K1 latent), c1 (Rubin i and VIS).
    # The same observed | model | residual images as the live tracking.
    from . import data, evaluate
    camp = evaluate.Campaign(cfg, students=False)
    part = dict(rubin_vae="rvae", k1="k1", k2="k2", c1="c1")[stage]
    if part not in camp.parts or (stage == "k2" and "k1" not in camp.parts):
        return []
    x, _ = data.D1Batches(cfg, split, n, False)[0]
    cond = x.get("r_psf")
    if stage == "rubin_vae":
        enc, dec = camp.parts["rvae"]
        out = dict(mean=camp.decode(dec, camp.mu(enc, x["r_in"], cond=cond), camp.sr, cond))
    elif stage == "k1":
        enc, dec = camp.parts["k1"]
        out = dict(mean=camp.decode(dec, camp.mu(enc, x["e_in"]), camp.se))
    elif stage == "k2":
        out = dict(mean=camp.decode(camp.parts["k2"], camp.mu(camp.parts["k1"][0], x["e_in"]), camp.sr, cond))
    else:
        enc, dec_r, dec_e = camp.parts["c1"]
        mu = camp.mu(enc, x["r_in"], x["e_in"], cond=cond)
        out = dict(mean_r=camp.decode(dec_r, mu, camp.sr, cond), mean_e=camp.decode(dec_e, mu, camp.se))
    return panels(x, out)


def _scalars(logs, prefix):
    out = {}
    for k, v in (logs or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out[f"{prefix}{k}"] = f
    return out


class WandbLogger(keras.callbacks.Callback):
    def __init__(self, run_dir, name, group, config, every=50, preview=None):
        super().__init__()
        self.run, self.every, self.step = None, int(every), 0
        self.preview = preview if images_enabled() else None      # a small validation batch for the images
        self.wandb = None
        if not enabled():
            return
        wandb = _import_wandb()
        if wandb is None:
            return
        self.wandb = wandb
        os.environ.setdefault("WANDB_MODE", "offline")
        # Everything below may fail with a wandb version other than the one tested: tracking is then off and the
        # training goes on (2026-11-28: wandb 0.31 has no wandb.util.generate_id, and the first jobs stopped).
        try:
            idf = Path(run_dir) / "wandb_id"
            rid = idf.read_text().strip() if idf.exists() else ""
            if not rid:
                # Our own id (8 lowercase hex characters, like W&B's): no wandb internal is needed for it.
                rid = uuid.uuid4().hex[:8]
                idf.write_text(rid + "\n")
            self.run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dp2-deblend"),
                                  entity=os.environ.get("WANDB_ENTITY") or None, name=name, group=group, id=rid,
                                  resume="allow", config=config, dir=os.environ.get("WANDB_DIR") or None)
            print(f"[wandb] run {name} (id {rid}, mode {os.environ.get('WANDB_MODE')})", flush=True)
            define_axes(self.run)
        except Exception as e:  # noqa: BLE001 - tracking must never stop a training
            print(f"[wandb] init failed ({e}): no tracking", flush=True)
            self.run = None

    def on_train_begin(self, logs=None):
        opt = getattr(self.model, "optimizer", None)
        if opt is not None and getattr(opt, "built", False):
            self.step = int(keras.ops.convert_to_numpy(opt.iterations))

    def _log(self, d):
        if self.run is not None and d:
            try:
                self.run.log(d, step=self.step)
            except Exception as e:  # noqa: BLE001
                print(f"[wandb] log failed ({e})", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.run is not None and self.step % self.every == 0:
            self._log(_scalars(logs, "train/"))

    def on_epoch_end(self, epoch, logs=None):
        d = _scalars(logs, "epoch/")
        d["epoch"] = int(epoch) + 1
        if self.run is not None and self.preview is not None:
            try:
                out = preview_outputs(self.model, self.preview)
                for title, img in panels(self.preview, out):
                    d.setdefault(f"images/{title.split(' #')[0]}", []).append(
                        self.wandb.Image(img, caption=f"{title}: observed | model | residual"))
            except Exception as e:  # noqa: BLE001 - images are a convenience
                print(f"[wandb] images skipped ({e})", flush=True)
        self._log(d)

    def on_train_end(self, logs=None):
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:  # noqa: BLE001
                print(f"[wandb] finish failed ({e})", flush=True)
            self.run = None
