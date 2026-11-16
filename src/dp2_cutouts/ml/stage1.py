# stage1.py
# Anna Preto
# AstroDeep Project
# Stage 1: a Euclid encoder E_E trained to place a Euclid image of a galaxy where the frozen DebVAder encoder E_R
# places the Rubin image of the same galaxy, so that the frozen Rubin decoder D_R turns it into the Rubin view.
#   latent  L(mu_E, mu_R)                       (mu_mse | mu_whitened | gaussian_kl)
#   cross   L(D_R(mu_E), target)                target = observed Rubin stamp or D_R(mu_R)
#   infonce symmetric InfoNCE between projections of mu_E and mu_R (optional)
# E_R and D_R never move (checked by fingerprints before / after). The reference latents are computed online on the
# same augmented Rubin input as the Euclid one, never cached, so that flips do not desynchronise them.

import json

import keras
import numpy as np
from keras import ops

from . import losses
from .data import Stage1Batches
from .debvader_bridge import FrozenRubinVAE, cholesky_from_params
from .models import EuclidEncoder, ProjectionHead, add_temperature, inverse_temperature
from .training import (FrozenGuard, callbacks, denormalise_ops, environment, optimizer, prepare_run_dir, set_seeds,
                       write_json)


class Stage1Model(keras.Model):
    def __init__(self, vae, s1, whitening, **kwargs):
        super().__init__(**kwargs)
        self.D, self.C = vae.latent_dim, vae.n_bands
        self.rubin_encoder = vae.encoder
        self.rubin_decoder = vae.decoder
        self.rubin_encoder.trainable = False
        self.rubin_decoder.trainable = False
        ee = s1["euclid_encoder"]
        self.euclid_encoder = EuclidEncoder(self.D, ee["filters"], ee["cond_embedding"], ee["dense"],
                                            name="euclid_encoder")
        self.weights_ = {k: float(v) for k, v in s1["weights"].items()}
        self.latent_kind, self.cross_kind = s1["latent_loss"], s1["cross_loss"]
        self.cross_target = s1["cross_target"]
        if self.cross_target not in ("observation", "reference_reconstruction"):
            raise ValueError(f"unknown stage1.cross_target {self.cross_target!r}")
        self.band_w = np.asarray(s1["band_weights"], np.float32)
        if len(self.band_w) != self.C:
            raise ValueError(f"stage1.band_weights has {len(self.band_w)} entries, the VAE {self.C} bands")
        self.whitening = np.asarray(whitening, np.float32)
        self.scale_per_band, self.beta = vae.scale_per_band, vae.beta
        self.use_nce = self.weights_.get("infonce", 0.0) > 0
        if self.use_nce:
            ct = s1["contrastive"]
            if ct.get("space", "mu") != "mu":
                raise ValueError("stage1.contrastive.space: only 'mu' is implemented")
            self.proj_euclid = ProjectionHead(ct["projection_dims"], name="proj_euclid")
            self.proj_rubin = ProjectionHead(ct["projection_dims"], name="proj_rubin")
            self.log_inv_tau = add_temperature(self, ct["temperature"], ct["learn_temperature"])
        self.trackers = {k: keras.metrics.Mean(name=k) for k in
                         ("total_loss", "latent_loss", "cross_loss", "infonce_loss", "latent_mse")}

    @property
    def metrics(self):
        return list(self.trackers.values())

    def call(self, x, training=False):
        params_e = self.euclid_encoder([x["euclid"], x["cond"]], training=training)
        params_r = ops.stop_gradient(self.rubin_encoder(x["rubin_norm"], training=False))
        out = dict(params_e=params_e, params_r=params_r, mu_e=params_e[:, :self.D], mu_r=params_r[:, :self.D])
        if self.weights_.get("cross", 0.0) > 0 or not training:
            dec = self.rubin_decoder(out["mu_e"], training=False)
            out["mean_e"], out["scale_e"] = dec[..., :self.C], 1e-4 + dec[..., self.C:]
            if self.cross_target == "reference_reconstruction" or not training:
                out["mean_r"] = ops.stop_gradient(self.rubin_decoder(out["mu_r"], training=False)[..., :self.C])
        if self.use_nce:
            out["proj_e"] = self.proj_euclid(out["mu_e"])
            out["proj_r"] = self.proj_rubin(out["mu_r"])
        return out

    def loss_terms(self, x, out):
        t = {}
        L_e = L_r = None
        if self.latent_kind == "gaussian_kl":
            L_e = cholesky_from_params(out["params_e"], self.D)
            L_r = cholesky_from_params(out["params_r"], self.D)
        t["latent_loss"] = losses.latent_loss(self.latent_kind, out["mu_e"], out["mu_r"], self.whitening, L_e, L_r)
        t["latent_mse"] = losses.latent_mu_mse(out["mu_e"], out["mu_r"])
        if "mean_e" in out:
            bw = ops.convert_to_tensor(self.band_w)
            if self.cross_target == "observation":
                target, w = x["rubin_norm"], x["rubin_weight"]
            else:
                target, w = out["mean_r"], ops.ones_like(out["mean_r"])
            if self.cross_kind == "normalised_mse":
                t["cross_loss"] = losses.recon_normalised_mse(out["mean_e"], target, w, bw)
            elif self.cross_kind == "gaussian_nll":
                t["cross_loss"] = losses.recon_gaussian_nll(out["mean_e"], out["scale_e"], target, w, bw)
            elif self.cross_kind == "chi2_physical":
                if self.cross_target != "observation":
                    raise ValueError("chi2_physical needs cross_target = observation (it uses the observed ivar)")
                pred = denormalise_ops(out["mean_e"], self.scale_per_band, self.beta)
                t["cross_loss"] = losses.recon_chi2_physical(pred, x["rubin_flux"], x["rubin_ivar"], w, bw)
            else:
                raise ValueError(f"unknown stage1.cross_loss {self.cross_kind!r}")
        else:
            t["cross_loss"] = ops.zeros(())
        if self.use_nce:
            t["infonce_loss"] = losses.info_nce(out["proj_e"], out["proj_r"], inverse_temperature(self.log_inv_tau))
        else:
            t["infonce_loss"] = ops.zeros(())
        w = self.weights_
        t["total_loss"] = (w.get("latent", 0.0) * t["latent_loss"] + w.get("cross", 0.0) * t["cross_loss"]
                           + w.get("infonce", 0.0) * t["infonce_loss"])
        return t

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        t = self.loss_terms(x, y_pred)
        for k, v in t.items():
            self.trackers[k].update_state(v)
        return t["total_loss"]


def latent_reference(vae, cfg, max_n=20000, batch=256):
    # Covariance of the Rubin latents of the D1 train split (no augmentation): whitening of the latent loss and of
    # the evaluation distances.
    ds = Stage1Batches(cfg, vae, "train", batch, train=False)
    rows = ds.idx
    if len(rows) > max_n:
        rows = np.sort(np.random.default_rng(0).choice(rows, max_n, replace=False))
    ds = Stage1Batches(cfg, vae, "train", batch, train=False, indices=rows)
    mus = []
    for i in range(len(ds)):
        x, _ = ds[i]
        mus.append(np.asarray(ops.convert_to_numpy(vae.mu(vae.encode_params(ops.convert_to_tensor(x["rubin_norm"]))))))
    mu = np.concatenate(mus)
    W, mean, cov = losses.whitening_from_latents(mu)
    return dict(W=W, mean=mean, cov=cov, n=len(mu))


def settings(cfg, ablation=None):
    s1 = json.loads(json.dumps(cfg["stage1"]))
    if ablation:
        if ablation not in s1["ablations"]:
            raise KeyError(f"unknown Stage 1 ablation {ablation!r}; known: {sorted(s1['ablations'])}")
        s1["weights"] = dict(s1["ablations"][ablation])
    return s1


def build(cfg, vae, s1, whitening):
    model = Stage1Model(vae, s1, whitening)
    ds = Stage1Batches(cfg, vae, "val", 2, train=False)
    x, _ = ds[0]
    model({k: v[:2] for k, v in x.items()}, training=False)
    return model


def train(cfg, debvader_run, run_name, ablation=None, stats_path=None, resume=True, epochs=None, force=False):
    set_seeds(cfg["seed"])
    run_dir = cfg.run_dir(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not prepare_run_dir(run_dir, resume, force):
        return run_dir
    s1 = settings(cfg, ablation)
    if epochs is not None:
        s1["epochs"] = int(epochs)
    vae = FrozenRubinVAE(debvader_run, stage="stage1", stats_path=stats_path)

    ref_path = run_dir / "latent_reference.npz"
    if ref_path.exists():
        ref = dict(np.load(ref_path))
    else:
        ref = latent_reference(vae, cfg)
        np.savez(ref_path, **ref)
    model = build(cfg, vae, s1, ref["W"])
    guard = FrozenGuard(rubin_encoder=vae.encoder, rubin_decoder=vae.decoder)

    cfg.snapshot(run_dir / "ml_config.yaml")
    write_json(run_dir / "run.json", dict(stage=1, run_name=run_name, ablation=ablation, settings=s1,
                                          vae=vae.describe(), environment=environment(),
                                          d1=json.loads((cfg.dataset_dir("d1") / "info.json").read_text())))
    train_ds = Stage1Batches(cfg, vae, "train", s1["batch_size"], train=True, seed=int(cfg["seed"]),
                             workers=4, use_multiprocessing=False, max_queue_size=8)
    val_ds = Stage1Batches(cfg, vae, "val", s1["batch_size"], train=False)
    model.compile(optimizer=optimizer(s1["learning_rate"], s1["clipnorm"]))
    subs = dict(euclid_encoder=model.euclid_encoder)
    if model.use_nce:
        subs.update(proj_euclid=model.proj_euclid, proj_rubin=model.proj_rubin)
    model.fit(train_ds, validation_data=val_ds, epochs=int(s1["epochs"]),
              callbacks=callbacks(run_dir, subs, s1["early_stopping_patience"], resume), verbose=2)
    after = guard.check()
    if model.use_nce:
        np.save(run_dir / "log_inv_tau.npy", np.asarray(ops.convert_to_numpy(model.log_inv_tau)))
    write_json(run_dir / "frozen_check.json", dict(before=guard.before, after=after, unchanged=True))
    (run_dir / "DONE").write_text("stage 1 training finished\n")
    print(f"Stage 1 run {run_name} -> {run_dir}")
    return run_dir


def load_euclid_encoder(cfg, run_dir, latent_dim, tag="best"):
    run = json.loads((run_dir / "run.json").read_text())
    ee = run["settings"]["euclid_encoder"]
    enc = EuclidEncoder(latent_dim, ee["filters"], ee["cond_embedding"], ee["dense"], name="euclid_encoder")
    n_e = int(cfg["euclid"]["stamp_px"])
    enc([ops.zeros((1, n_e, n_e, len(cfg["euclid"]["encoder_bands"]))),
         ops.zeros((1, len(cfg["dataset1"]["cond_features"])))])
    enc.load_weights(str(run_dir / f"euclid_encoder_{tag}.weights.h5"))
    enc.trainable = False
    return enc
