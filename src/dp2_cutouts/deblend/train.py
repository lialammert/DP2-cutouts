# train.py
# Anna Preto
# AstroDeep Project
# One entry point per stage: dp2-ml <cfg> deblend-train <stage> [--seed S]. Stages and what they need:
#   rubin_vae  D1                     reference Rubin VAE (classical, Rubin in, Rubin out), for the D1 comparisons
#   k1         D1 (euclid_fill)       Euclid VIS VAE, frozen afterwards (shared by Runs A and B)
#   k2         D1, k1                 Rubin decoder from the Euclid latent, frozen afterwards (shared by A and B)
#   A          D2, k1                 MC-dropout regressor of sampled K1 latents
#   B          D2, k1                 CFM student of sampled K1 latents
#   c1         D1                     joint Rubin-Euclid VAE, frozen afterwards
#   c2         D2 + D1 replay, c1     conditional teacher r(z | X, E_c) and Gaussian student s(z | X)
#   c3         D2 + D1 replay, c1, c2 CFM student of the C2 teacher (and of q_A on D1)
# Run folders: <ml_root>/deblend/runs/<deblend.runs[stage]>[_s<seed>]. Each holds run.json (settings, the
# dependencies' run folders and weight fingerprints, environment), the config snapshot, history.csv,
# <submodel>_best / _last weights, backup/ (resume after a walltime cut), frozen_check.json, DONE.
# Hyperparameters: deblend.train (common) overridden by deblend.<stage>.train. Checkpoints are selected on the
# validation loss of the stage only (never on test or holdout).

import json
import time
from pathlib import Path

import keras
import numpy as np
from keras import ops

from ..ml.training import (FrozenGuard, callbacks, environment, optimizer, prepare_run_dir, schedule, set_seeds,
                           validation_indices, write_json)
from . import data
from .campaign import DEPENDS, STAGES, run_dir, run_name  # noqa: F401  (names shared with the campaign)
from .layers import ConvDecoder, sample_diag, sample_tril
from .models import (C2Branches, C2Model, C3Model, CFMStudent, ImageEncoder, ImageVAE, JointEncoder, JointVAE,
                     K2Model, RubinBlendRegressor, RunAModel, RunBModel)


# ----------------------------------------------------------------------------- settings and folders
def settings(cfg, stage):
    db = json.loads(json.dumps(cfg["deblend"]))
    st = dict(db.get(stage, db.get(stage.lower(), {})) or {})
    tr = dict(db["train"])
    tr.update(st.pop("train", {}) or {})
    return dict(stage=stage, d=int(db["latent_dim"]), train=tr, model=st)


def dep_dir(cfg, stage, seed=None):
    # Dependencies are shared across seeds of a later stage unless deblend.replicate_dependencies is true.
    rep = bool(cfg["deblend"].get("replicate_dependencies", False))
    d = run_dir(cfg, stage, seed if rep else None)
    if not (d / "DONE").exists():
        raise RuntimeError(f"{stage} run {d} is not finished (no DONE): train it first")
    return d


def sizes(cfg):
    return int(cfg["rubin"]["stamp_px"]), len(cfg["rubin"]["bands"]), int(cfg["euclid"]["stamp_px"])


# ----------------------------------------------------------------------------- loading frozen parts
def _load(model, path, *dummy):
    model(*[ops.zeros(s) for s in dummy])
    model.load_weights(str(path))
    model.trainable = False
    return model


def load_k1_encoder(cfg, d_k1, d):
    _, _, n_e = sizes(cfg)
    return _load(ImageEncoder(d, name="encoder"), d_k1 / "encoder_best.weights.h5", (1, n_e, n_e, 2))


def load_decoder(path, n, channels, d):
    dec = ConvDecoder(n, channels, name="decoder")
    return _load(dec, path, (1, d))


def load_joint_encoder(cfg, d_c1, d, s):
    n_r, b_r, n_e = sizes(cfg)
    enc = JointEncoder(d, s["model"].get("embed", 128), tuple(s["model"].get("fusion", [256])), name="encoder")
    enc(ops.zeros((1, n_r, n_r, b_r)), ops.zeros((1, n_e, n_e, 2)))
    enc.load_weights(str(d_c1 / "encoder_best.weights.h5"))
    enc.trainable = False
    return enc


def load_c2_branches(cfg, d_c2, d, s_c2):
    n_r, b_r, n_e = sizes(cfg)
    m = s_c2["model"]
    br = C2Branches(d, m.get("embed", 128), tuple(m.get("fusion", [256])), name="branches")
    br(ops.zeros((1, n_r, n_r, b_r)), ops.zeros((1, n_e, n_e, 2)))
    br.load_weights(str(d_c2 / "branches_best.weights.h5"))
    br.trainable = False
    return br


def load_student(cfg, stage, d_run, d, s):
    # Trained student of Run A / B / C (C = stage c3) for inference.
    n_r, b_r, _ = sizes(cfg)
    m = s["model"]
    if stage == "A":
        st = RubinBlendRegressor(d, m.get("embed", 128), float(m.get("dropout", 0.1)), name="student")
    else:
        stats = json.loads((d_run / "latent_stats.json").read_text()) if (d_run / "latent_stats.json").exists() \
            else None
        st = CFMStudent(d, m.get("context", 128), tuple(m.get("hidden", [128, 128])), stats, name="student")
    st(ops.zeros((1, n_r, n_r, b_r)))
    if stage != "A":
        st.field(ops.zeros((1, d)), ops.zeros((1, 1)), ops.zeros((1, m.get("context", 128))))
    st.load_weights(str(d_run / "student_best.weights.h5"))
    st.trainable = False
    return st


# ----------------------------------------------------------------------------- latent standardisation
def latent_stats(sampler, ds, max_batches):
    zs = []
    for i in range(min(len(ds), max_batches)):
        x, _ = ds[i]
        zs.append(np.asarray(ops.convert_to_numpy(sampler(x))))
    z = np.concatenate(zs)
    return dict(mean=z.mean(0).tolist(), std=np.maximum(z.std(0), 1e-3).tolist(), n=int(len(z)))


# ----------------------------------------------------------------------------- lambda_D1 of C2 (JAX)
def calibrate_lambda(model, ds, n_batches=8, fallback=1.0):
    # Ratio of the gradient norms of the D2 objective and of the D1 preservation term on the shared Rubin backbone,
    # median over a few batches, at initialisation. JAX backend only (diagnostic; other backends: set a value).
    # Batches whose norms are not finite and positive are skipped; with none left (the pilot of 2026-11-26 found
    # none), the fallback value is used and the per-batch numbers are printed and returned for the diagnosis.
    if keras.backend.backend() != "jax":
        raise RuntimeError("deblend.c2.lambda_d1 = auto needs the JAX backend: give a number instead")
    import jax
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    shared = {id(v) for v in model.branches.r_embed.trainable_variables}
    sel = [k for k, v in enumerate(model.trainable_variables) if id(v) in shared]

    def term(tv_, x, which):
        out, _ = model.stateless_call(tv_, ntv, x, training=True)
        t = model.loss_terms(x, out)
        if which == "d2":
            return t["d2_nll_e"] + t["d2_kl_rs"]
        return 0.5 * t["d1_kl_s"] + 0.5 * t["d1_kl_r"]

    ratios, seen = [], []
    for i in range(min(n_batches, len(ds))):
        x, _ = ds[i]
        x = {k: jax.numpy.asarray(v) for k, v in x.items()}
        rec = {}
        for which in ("d2", "d1"):
            rec[f"{which}_loss"] = float(term(tv, x, which))
            g = jax.grad(lambda p: term(p, x, which))(tv)
            rec[f"{which}_grad"] = float(np.sqrt(sum(float(jax.numpy.sum(g[k] ** 2)) for k in sel)))
        seen.append(rec)
        if all(np.isfinite(v) for v in rec.values()) and rec["d1_grad"] > 0:
            ratios.append(rec["d2_grad"] / rec["d1_grad"])
    print(f"lambda_d1 calibration: {len(sel)} shared variables; per batch {seen}", flush=True)
    if not ratios:
        print(f"lambda_d1 calibration: no usable batch, lambda_d1 = {fallback} (fallback)", flush=True)
        return float(fallback), [], seen
    return float(np.median(ratios)), [float(r) for r in ratios], seen


# ----------------------------------------------------------------------------- stage builders
def _loaders(cfg, s, which, seed, workers=None):
    tr = s["train"]
    bs = int(tr["batch_size"])
    workers = int(workers or tr.get("workers", 4))
    kw = dict(workers=workers, use_multiprocessing=False, max_queue_size=8, block=int(tr.get("read_block", 1)))
    if which == "d1":
        st = data.Store(cfg.dataset_dir("d1"))
        vidx = validation_indices(st.meta, "val", tr.get("val_max_examples"), seed)
        fields = s.get("fields", ("rubin", "euclid"))
        pool = s.get("mask_pool")
        prob = float(s["model"].get("mask_augment_prob", 0.0))
        return (data.D1Batches(cfg, "train", bs, True, seed, fields=fields, mask_pool=pool, mask_prob=prob, **kw),
                data.D1Batches(cfg, "val", bs, False, seed, indices=vidx, fields=fields))
    if which == "d2":
        st = data.Store(cfg.dataset_dir("d2"))
        keep, info = data.d2_keep(cfg, st.meta)
        print(f"D2 selection (deblend.d2_selection): {info or 'none'}")
        rows = data.d2_rows(st.meta, "val", cfg["deblend"]["d2_centering"], keep)
        mx = tr.get("val_max_examples")
        if mx and len(rows) > int(mx):
            rows = np.sort(np.random.default_rng(seed).choice(rows, int(mx), replace=False))
        return (data.D2Batches(cfg, "train", bs, True, seed, **kw), data.D2Batches(cfg, "val", bs, False, seed,
                                                                                    indices=rows))
    if which == "mixed":
        # Balanced D2 / D1 batches (deblend.<stage>.d1_batch_size, default = batch_size).
        b1 = int(s["model"].get("d1_batch_size", bs))
        d2_tr, d2_va = _loaders(cfg, s, "d2", seed, workers)
        s1 = dict(s, train=dict(tr, batch_size=b1))
        d1_tr, d1_va = _loaders(cfg, s1, "d1", seed + 1, workers)
        mkw = {k: v for k, v in kw.items() if k != "block"}
        return data.Mixed(d2_tr, d1_tr, **mkw), data.Mixed(d2_va, d1_va)
    raise ValueError(which)


def build(cfg, stage, s, seed, deps):
    n_r, b_r, n_e = sizes(cfg)
    stats = data.load_norm_stats(cfg)
    sr, se = stats["rubin"]["scale"], stats["euclid"]["scale"]
    d, m = s["d"], s["model"]
    frozen, subs = {}, {}
    if stage == "rubin_vae":
        s["fields"] = ("rubin",)
        model = ImageVAE("rubin", n_r, b_r, sr, d, float(m.get("beta", 1.0)), seed, name="rubin_vae")
        subs = dict(encoder=model.encoder, decoder=model.decoder)
        loaders = _loaders(cfg, s, "d1", seed)
    elif stage == "k1":
        s["fields"] = ("euclid",)
        if float(m.get("mask_augment_prob", 0.0)) > 0:
            s["mask_pool"] = data.d2_mask_pool(cfg, int(m.get("mask_pool_size", 5000)), seed)
        model = ImageVAE("euclid", n_e, 1, se, d, float(m.get("beta", 1.0)), seed, name="k1")
        subs = dict(encoder=model.encoder, decoder=model.decoder)
        loaders = _loaders(cfg, s, "d1", seed)
    elif stage == "k2":
        enc = load_k1_encoder(cfg, deps["k1"], d)
        model = K2Model(enc, n_r, b_r, sr, seed, name="k2")
        frozen, subs = dict(k1_encoder=enc), dict(decoder=model.decoder)
        loaders = _loaders(cfg, s, "d1", seed)
    elif stage == "A":
        enc = load_k1_encoder(cfg, deps["k1"], d)
        model = RunAModel(enc, d, int(m.get("embed", 128)), float(m.get("dropout", 0.1)), seed, name="run_a")
        frozen, subs = dict(k1_encoder=enc), dict(student=model.student)
        loaders = _loaders(cfg, s, "d2", seed)
    elif stage == "B":
        enc = load_k1_encoder(cfg, deps["k1"], d)
        loaders = _loaders(cfg, s, "d2", seed)
        st_ = None
        if m.get("standardise", True):
            gen = keras.random.SeedGenerator(seed)
            st_ = latent_stats(lambda x: sample_diag(*enc(x["e_in"]), gen), loaders[0],
                               int(m.get("stats_batches", 100)))
        model = RunBModel(enc, d, int(m.get("context", 128)), tuple(m.get("hidden", [128, 128])), st_, seed,
                          name="run_b")
        frozen, subs = dict(k1_encoder=enc), dict(student=model.student)
        s["latent_stats"] = st_
    elif stage == "c1":
        model = JointVAE(n_r, b_r, n_e, sr, se, d, int(m.get("embed", 128)), tuple(m.get("fusion", [256])),
                         float(m.get("beta", 1.0)), float(m.get("weight_r", 1.0)), float(m.get("weight_e", 1.0)),
                         seed, name="c1")
        subs = dict(encoder=model.encoder, dec_r=model.dec_r, dec_e=model.dec_e)
        loaders = _loaders(cfg, s, "d1", seed)
    elif stage == "c2":
        s_c1 = json.loads((deps["c1"] / "run.json").read_text())["settings"]
        enc = load_joint_encoder(cfg, deps["c1"], d, s_c1)
        dec_e = load_decoder(deps["c1"] / "dec_e_best.weights.h5", n_e, 1, d)
        lam = m.get("lambda_d1", 1.0)
        model = C2Model(enc, dec_e, se, d, int(m.get("embed", 128)), tuple(m.get("fusion", [256])),
                        1.0 if lam == "auto" else float(lam), seed, name="c2")
        loaders = _loaders(cfg, s, "mixed", seed)
        x0, _ = loaders[1][0]
        model({k: v[:2] for k, v in x0.items()})
        # Backbones initialised from copies of the C1 embedders (the C1 encoder itself stays frozen).
        model.branches.r_embed.set_weights(enc.r_embed.get_weights())
        model.branches.e_embed.set_weights(enc.e_embed.get_weights())
        frozen, subs = dict(c1_encoder=enc, c1_dec_e=dec_e), dict(branches=model.branches)
        if lam == "auto":
            val, ratios, seen = calibrate_lambda(model, loaders[0], int(m.get("lambda_batches", 8)),
                                                 float(m.get("lambda_fallback", 1.0)))
            model.lambda_d1 = val
            s["lambda_d1_calibration"] = dict(value=val, ratios=ratios, fallback=not ratios, batches=seen)
    elif stage == "c3":
        s_c1 = json.loads((deps["c1"] / "run.json").read_text())["settings"]
        s_c2 = json.loads((deps["c2"] / "run.json").read_text())["settings"]
        enc = load_joint_encoder(cfg, deps["c1"], d, s_c1)
        br = load_c2_branches(cfg, deps["c2"], d, s_c2)
        loaders = _loaders(cfg, s, "mixed", seed)
        st_ = None
        if m.get("standardise", True):
            gen = keras.random.SeedGenerator(seed)
            st_ = latent_stats(lambda x: sample_tril(*br.teacher(x["d2_x_in"], x["d2_e_in"]), gen), loaders[0],
                               int(m.get("stats_batches", 100)))
        model = C3Model(br, enc, d, int(m.get("context", 128)), tuple(m.get("hidden", [128, 128])), st_,
                        float(m.get("lambda_replay", 1.0)), seed, name="c3")
        x0, _ = loaders[1][0]
        model({k: v[:2] for k, v in x0.items()})
        if m.get("init_context_from_c2", True):
            model.student.context.set_weights(br.r_embed.get_weights())
        frozen, subs = dict(c2_branches=br, c1_encoder=enc), dict(student=model.student)
        s["latent_stats"] = st_
    else:
        raise ValueError(f"unknown deblend stage {stage!r}; known: {STAGES}")
    x0, _ = loaders[1][0]
    model({k: v[:2] for k, v in x0.items()})
    return model, loaders, frozen, subs


def _selection(loader):
    # deblend.d2_selection as applied by the D2 loader of this stage (None for D1-only stages).
    for d in (loader, getattr(loader, "a", None)):
        if d is not None and hasattr(d, "selection"):
            return d.selection
    return None


class TimeBudget(keras.callbacks.Callback):
    # Stops training after `seconds` (counted from `t0`) at the end of a batch, so that a job of the dev QoS ends
    # on its own before its walltime. The backup is then written with the epoch in progress, which a resumed run
    # restarts (keras 3 resumes at epoch boundaries), so no completed step is lost to the cut.
    def __init__(self, seconds, backup, t0):
        super().__init__()
        self.seconds, self.backup, self.t0, self.hit = float(seconds), backup, float(t0), False
        self._epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if not self.hit and time.time() - self.t0 > self.seconds:
            self.hit = True
            self.model.stop_training = True
            print(f"time budget of {self.seconds / 60:.0f} min reached at epoch {self._epoch} batch {batch}: "
                  "saving the backup and stopping (the next job resumes)")

    def on_epoch_end(self, epoch, logs=None):
        # After BackupAndRestore.on_epoch_end (earlier in the list), which counts the cut epoch as completed.
        if self.hit and self.backup is not None:
            self.backup._current_epoch = epoch
            self.backup._last_batch_seen = 0
            self.backup._save_model()


def train(cfg, stage, seed=None, resume=True, force=False, max_steps=None, time_budget_s=None):
    # time_budget_s: stop after that many seconds (from this call) without writing DONE; the same command resumes
    # from <run>/backup. Ignored with max_steps (pilot runs are short and not resumable).
    t0 = time.time()
    seed_v = int(cfg["deblend"]["seed"]) if seed is None else int(seed)
    set_seeds(seed_v)
    rd = run_dir(cfg, stage, seed)
    rd.mkdir(parents=True, exist_ok=True)
    if not prepare_run_dir(rd, resume, force):
        return rd
    s = settings(cfg, stage)
    if max_steps:
        s["train"].update(train_steps=int(max_steps), backup_every_steps=None)
    deps = {k: dep_dir(cfg, k, seed) for k in DEPENDS[stage]}
    model, (tr, va), frozen, subs = build(cfg, stage, s, seed_v, deps)
    guard = FrozenGuard(**frozen) if frozen else None
    from ..ml.debvader_bridge import weights_fingerprint
    if s.get("latent_stats") is not None:
        write_json(rd / "latent_stats.json", s["latent_stats"])
    cfg.snapshot(rd / "ml_config.yaml")
    record = {k: v for k, v in s.items() if k not in ("mask_pool",)}
    write_json(rd / "run.json", dict(stage=stage, run_name=rd.name, seed=seed_v, settings=record,
                                     dependencies={k: str(v) for k, v in deps.items()},
                                     dependency_fingerprints={k: weights_fingerprint(v) for k, v in frozen.items()},
                                     n_params={k: int(v.count_params()) for k, v in subs.items()},
                                     n_train_batches=len(tr), d2_selection=_selection(tr),
                                     environment=environment()))
    model.compile(optimizer=optimizer(s["train"]["learning_rate"], s["train"].get("clipnorm")))
    resumable = resume and not max_steps
    cbs = callbacks(rd, subs, s["train"]["early_stopping_patience"], resumable, s["train"].get("backup_every_steps"))
    budget = None
    if time_budget_s and resumable:
        backup = next(c for c in cbs if isinstance(c, keras.callbacks.BackupAndRestore))
        budget = TimeBudget(time_budget_s, backup, t0)
        cbs.append(budget)
    model.fit(tr, validation_data=va, epochs=schedule(s["train"], len(tr)),
              steps_per_epoch=min(len(tr), int(max_steps)) if max_steps else None, callbacks=cbs, verbose=2)
    if budget is not None and budget.hit:
        print(f"deblend {stage}: stopped on the time budget, not finished (no DONE); the same command resumes")
        return rd
    if guard is not None:
        after = guard.check()
        write_json(rd / "frozen_check.json", dict(before=guard.before, after=after, unchanged=True))
    (rd / "DONE").write_text(f"deblend stage {stage} finished\n")
    print(f"deblend {stage} -> {rd}")
    return rd
