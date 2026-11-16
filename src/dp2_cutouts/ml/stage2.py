# stage2.py
# Anna Preto
# AstroDeep Project
# Stage 2: a Rubin blend encoder E_S (the DebVAder encoder architecture) trained so that, on a Rubin blend centred
# on a target that Euclid resolves, it returns the latent the frozen Stage 1 teacher E_E gives from the Euclid view
# of that target alone. The Rubin decoder stays frozen; E_E stays frozen.
#   latent  L(mu_S, mu_T)                         mu_T = mu(E_E(x_E)), computed online on the same augmentation
#   infonce symmetric InfoNCE between projections of mu_S and mu_T; negatives whose teacher latents are closer than
#           the negative_mask_quantile of the train-split distances are dropped (morphological twins)
#   scene   for complete two-target scenes: D_R(mu_S) + D_R(mu_T of the partner), placed at its offset, must explain
#           the observed Rubin blend (chi^2 in VAE flux units)
# Warm start of E_S: "stage2" = the DebVAder deblender (paper_baseline stage 2), "stage1" = the reference encoder of
# isolated galaxies, "random" = the same architecture freshly initialised.

import json

import keras
import numpy as np
from keras import ops

from . import losses
from .data import Stage2Batches
from .debvader_bridge import FrozenRubinVAE
from .models import ProjectionHead, add_temperature, inverse_temperature
from .stage1 import load_euclid_encoder
from .training import (FrozenGuard, callbacks, denormalise_ops, environment, optimizer, prepare_run_dir, set_seeds,
                       write_json)


def student_encoder(debvader_run, init, stats_path=None):
    if init in ("stage1", "stage2"):
        enc = FrozenRubinVAE(debvader_run, stage=init, stats_path=stats_path).encoder
    elif init == "random":
        from debvader.core.encoder import Encoder
        ref = FrozenRubinVAE(debvader_run, stage="stage1", stats_path=stats_path)
        enc = Encoder(ref.config.model)
        enc(ops.zeros((1, *ref.input_shape)), training=False)
    else:
        raise ValueError(f"unknown stage2.init {init!r} (stage2 | stage1 | random)")
    enc.trainable = True
    return enc


class Stage2Model(keras.Model):
    def __init__(self, vae, student, teacher, s2, whitening, twin_threshold_sq, **kwargs):
        super().__init__(**kwargs)
        self.D, self.C = vae.latent_dim, vae.n_bands
        self.student = student
        self.teacher = teacher
        self.rubin_decoder = vae.decoder
        self.teacher.trainable = False
        self.rubin_decoder.trainable = False
        self.weights_ = {k: float(v) for k, v in s2["weights"].items()}
        self.weights_.setdefault("scene", float(s2["scene"]["weight"]) if s2["scene"]["enabled"] else 0.0)
        self.latent_kind = s2["latent_loss"]
        if self.latent_kind == "gaussian_kl":
            raise ValueError("stage2.latent_loss gaussian_kl is not supported (the teacher posterior is only "
                             "calibrated through its mean); use mu_mse or mu_whitened")
        self.whitening = np.asarray(whitening, np.float32)
        self.twin_threshold_sq = float(twin_threshold_sq)
        self.scale_per_band, self.beta = vae.scale_per_band, vae.beta
        self.use_nce = self.weights_.get("infonce", 0.0) > 0
        if self.use_nce:
            ct = s2["contrastive"]
            self.proj_student = ProjectionHead(ct["projection_dims"], name="proj_student")
            self.proj_teacher = ProjectionHead(ct["projection_dims"], name="proj_teacher")
            self.log_inv_tau = add_temperature(self, ct["temperature"], ct["learn_temperature"])
        self.use_scene = self.weights_.get("scene", 0.0) > 0
        self.trackers = {k: keras.metrics.Mean(name=k) for k in
                         ("total_loss", "latent_loss", "infonce_loss", "scene_loss", "latent_mse")}

    @property
    def metrics(self):
        return list(self.trackers.values())

    def call(self, x, training=False):
        params_s = self.student(x["rubin_norm"], training=training)
        mu_t = ops.stop_gradient(self.teacher([x["euclid"], x["cond"]], training=False)[:, :self.D])
        out = dict(mu_s=params_s[:, :self.D], mu_t=mu_t)
        if self.use_nce:
            out["proj_s"] = self.proj_student(out["mu_s"])
            out["proj_t"] = self.proj_teacher(mu_t)
        if self.use_scene and "partner_euclid" in x:
            mu_p = ops.stop_gradient(self.teacher([x["partner_euclid"], x["partner_cond"]], training=False)[:, :self.D])
            rec_s = denormalise_ops(self.rubin_decoder(out["mu_s"], training=False)[..., :self.C],
                                    self.scale_per_band, self.beta)
            rec_p = denormalise_ops(self.rubin_decoder(mu_p, training=False)[..., :self.C],
                                    self.scale_per_band, self.beta)
            B, H, W = ops.shape(rec_p)[0], ops.shape(rec_p)[1], ops.shape(rec_p)[2]
            flat = ops.reshape(rec_p, (B, H * W, self.C))
            idx = ops.expand_dims(ops.cast(x["partner_index"], "int32"), -1)
            placed = ops.take_along_axis(flat, ops.repeat(idx, self.C, axis=-1), axis=1)
            placed = placed * ops.expand_dims(x["partner_valid"], -1)
            out["scene_model"] = rec_s + ops.reshape(placed, (B, H, W, self.C))
        return out

    def loss_terms(self, x, out):
        t = dict(latent_loss=losses.latent_loss(self.latent_kind, out["mu_s"], out["mu_t"], self.whitening),
                 latent_mse=losses.latent_mu_mse(out["mu_s"], out["mu_t"]))
        if self.use_nce:
            mask = losses.twin_negative_mask(out["mu_t"], self.twin_threshold_sq)
            t["infonce_loss"] = losses.info_nce(out["proj_s"], out["proj_t"], inverse_temperature(self.log_inv_tau),
                                                mask)
        else:
            t["infonce_loss"] = ops.zeros(())
        if "scene_model" in out:
            w = x["rubin_weight"] * ops.reshape(x["scene_ok"], (-1, 1, 1, 1))
            chi2 = ops.square(out["scene_model"] - x["rubin_flux"]) * x["rubin_ivar"] * w
            t["scene_loss"] = ops.sum(chi2) / (ops.sum(w) + 1e-8)
        else:
            t["scene_loss"] = ops.zeros(())
        w = self.weights_
        t["total_loss"] = (w.get("latent", 0.0) * t["latent_loss"] + w.get("infonce", 0.0) * t["infonce_loss"]
                           + w.get("scene", 0.0) * t["scene_loss"])
        return t

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        t = self.loss_terms(x, y_pred)
        for k, v in t.items():
            self.trackers[k].update_state(v)
        return t["total_loss"]


def settings(cfg, ablation=None):
    s2 = json.loads(json.dumps(cfg["stage2"]))
    if ablation:
        if ablation not in s2["ablations"]:
            raise KeyError(f"unknown Stage 2 ablation {ablation!r}; known: {sorted(s2['ablations'])}")
        w = dict(s2["ablations"][ablation])
        s2["scene"]["enabled"] = w.get("scene", 0.0) > 0
        if s2["scene"]["enabled"]:
            s2["scene"]["weight"] = w["scene"]
        s2["weights"] = w
    return s2


def twin_threshold(cfg, vae, teacher, s2, max_n=4000):
    # Quantile of the squared distances between teacher latents of the D2 train split.
    c2 = json.loads(json.dumps(cfg.raw))
    c2["stage2"]["scene"]["enabled"] = False
    from ..mlconfig import MLConfig
    ds = Stage2Batches(MLConfig(raw=c2, path=cfg.path), vae, "train", 256, train=False)
    rows = ds.idx[:max_n]
    ds = Stage2Batches(MLConfig(raw=c2, path=cfg.path), vae, "train", 256, train=False, indices=rows)
    mus = []
    for i in range(len(ds)):
        x, _ = ds[i]
        mus.append(np.asarray(ops.convert_to_numpy(teacher([ops.convert_to_tensor(x["euclid"]),
                                                            ops.convert_to_tensor(x["cond"])], training=False)))
                   [:, :vae.latent_dim])
    mu = np.concatenate(mus)
    d = losses.pairwise_sq_dist(mu, mu)[np.triu_indices(len(mu), 1)]
    return float(np.quantile(d, float(s2["contrastive"]["negative_mask_quantile"])))


def train(cfg, debvader_run, stage1_run, run_name, ablation=None, stats_path=None, resume=True, epochs=None,
          force=False):
    set_seeds(cfg["seed"])
    run_dir = cfg.run_dir(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not prepare_run_dir(run_dir, resume, force):
        return run_dir
    s2 = settings(cfg, ablation)
    if epochs is not None:
        s2["epochs"] = int(epochs)
    raw = json.loads(json.dumps(cfg.raw))
    raw["stage2"] = s2
    from ..mlconfig import MLConfig
    cfg2 = MLConfig(raw=raw, path=cfg.path)

    vae = FrozenRubinVAE(debvader_run, stage="stage1", stats_path=stats_path)
    s1_dir = cfg.run_dir(stage1_run)
    if not (s1_dir / "DONE").exists():
        raise FileNotFoundError(f"Stage 1 run {s1_dir} is not finished (no DONE file)")
    teacher = load_euclid_encoder(cfg, s1_dir, vae.latent_dim)
    ref = dict(np.load(s1_dir / "latent_reference.npz"))
    student = student_encoder(debvader_run, s2["init"], stats_path)
    thr = twin_threshold(cfg2, vae, teacher, s2) if s2["weights"].get("infonce", 0) > 0 else 0.0
    model = Stage2Model(vae, student, teacher, s2, ref["W"], thr)
    probe = Stage2Batches(cfg2, vae, "val", 2, train=False)
    x, _ = probe[0]
    model({k: v[:2] for k, v in x.items()}, training=False)
    guard = FrozenGuard(teacher=teacher, rubin_decoder=vae.decoder)

    cfg2.snapshot(run_dir / "ml_config.yaml")
    write_json(run_dir / "run.json", dict(stage=2, run_name=run_name, ablation=ablation, stage1_run=stage1_run,
                                          settings=s2, twin_threshold_sq=thr, vae=vae.describe(),
                                          environment=environment(),
                                          d2=json.loads((cfg.dataset_dir("d2") / "info.json").read_text())))
    train_ds = Stage2Batches(cfg2, vae, "train", s2["batch_size"], train=True, seed=int(cfg["seed"]),
                             workers=4, use_multiprocessing=False, max_queue_size=8)
    val_ds = Stage2Batches(cfg2, vae, "val", s2["batch_size"], train=False)
    model.compile(optimizer=optimizer(s2["learning_rate"], s2["clipnorm"]))
    subs = dict(student_encoder=model.student)
    if model.use_nce:
        subs.update(proj_student=model.proj_student, proj_teacher=model.proj_teacher)
    model.fit(train_ds, validation_data=val_ds, epochs=int(s2["epochs"]),
              callbacks=callbacks(run_dir, subs, s2["early_stopping_patience"], resume), verbose=2)
    after = guard.check()
    write_json(run_dir / "frozen_check.json", dict(before=guard.before, after=after, unchanged=True))
    (run_dir / "DONE").write_text("stage 2 training finished\n")
    print(f"Stage 2 run {run_name} -> {run_dir}")
    return run_dir


def load_student(debvader_run, run_dir, stats_path=None, tag="best"):
    # The weights hold everything; "random" only builds the architecture.
    enc = student_encoder(debvader_run, "random", stats_path)
    enc.load_weights(str(run_dir / f"student_encoder_{tag}.weights.h5"))
    enc.trainable = False
    return enc
