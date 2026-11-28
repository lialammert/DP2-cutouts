# train.py
# Anna Preto
# AstroDeep Project
# One entry point per stage: dp2-ml <cfg> deblend-train <stage> [--seed S]. Stages and what they need:
#   rubin_vae  D1                     reference Rubin VAE (classical, Rubin in, Rubin out), for the D1 comparisons
#   k1         D1 (euclid_fill)       Euclid VIS VAE, frozen afterwards (shared by Runs A and B)
#   k2         D1, k1                 Rubin decoder from the Euclid latent, frozen afterwards (shared by A and B)
#   A          D2 + D1 replay, k1     MC-dropout student of K1 latents, energy score
#   B          D2 + D1 replay, k1     CFM student of K1 latents
#   c1         D1                     joint Rubin-Euclid VAE, frozen afterwards
#   c2         D2 + D1 replay, c1     conditional teacher r(z | X, E_c) and Gaussian student s(z | X)
#   c3         D2 + D1 replay, c1, c2 CFM student of the C2 teacher (and of q_A on D1)
# Run folders: <ml_root>/deblend/runs/<deblend.runs[stage]>[_s<seed>]. Each holds run.json (settings, the
# dependencies' run folders and weight fingerprints, environment), the config snapshot, history.csv,
# <submodel>_best / _last weights, backup/ (resume after a walltime cut), frozen_check.json, DONE.
# Hyperparameters: deblend.train (common) overridden by deblend.<stage>.train. Checkpoints are selected on the
# validation loss of the stage only (never on test or holdout): val_total_loss, or the D2 term of the students that
# replay D1 (A: val_es_d2, B and C3: val_cfm_d2).
#
# Recipe of 2026-11-28 (deblend.train): a budget in epochs (20 on D1, 60 on D2; an epoch of a mixed stage is one
# pass over D2 plus as many D1 examples, new ones at every epoch), linear warm-up over the first 2 % of the updates
# then cosine decay to 1 % of the peak (optim.lr_schedule), Adam with the global gradient norm clipped at 1
# (optim.DeblendAdam), early stopping after 10 epochs without improvement as a safety net, the best validation
# checkpoint kept. VAEs: linear KL warm-up from beta = 0 over the first 10 % of the updates (validation always at the
# full beta). CFM students: EMA of the weights (0.999), used for validation and saved as the best weights.
#
# Latent residual block and warm start (2026-11-28, after the amortisation-gap diagnostics of the beta = 1 Rubin VAE,
# GPT's decision adopted by Anna), for the VAEs (rubin_vae, k1, c1):
#   deblend.<stage>.latent_res_hidden: 256   a LatentResBlock (Dense 256 -> PReLU -> Dense, last one at zero) between
#                                            the embedding (after the PSF) or the fusion and the variational head;
#   deblend.<stage>.init_from: <run name>    every weight of that finished run of the same stage copied in (the
#                                            encoder's backbone and head, the decoders); the new block starts at zero,
#                                            so the posteriors are exactly the source's at the first update;
#   deblend.<stage>.freeze_decoders: true    the decoders stay fixed (FrozenGuard), the whole encoder trains.
# The block's width is read back from each run's run.json wherever an encoder is reloaded (load_*_encoder).

import json
import os
import time
from pathlib import Path

import keras
import numpy as np
from keras import ops

from ..ml.training import (FrozenGuard, callbacks, environment, prepare_run_dir, schedule, set_seeds,
                           validation_indices, write_json)
from . import data
from .campaign import DEPENDS, STAGES, run_dir, run_name  # noqa: F401  (names shared with the campaign)
from .layers import ConvDecoder
from .models import (LAMBDA_PROTOCOL, C2Branches, C2Model, C3Model, CFMStudent, ImageEncoder, ImageVAE,
                     JointEncoder, JointVAE, K2Model, RubinBlendRegressor, RunAModel, RunBModel)
from .optim import DeblendAdam, EMAValidation, EMAWeights, GradNormLogger, LambdaD1Protocol, lr_schedule
from .tracking import WandbLogger, images_enabled

# Metric each stage's checkpoints are selected (and early-stopped) on, unless deblend.<stage>.monitor says otherwise.
MONITOR = dict(A="val_es_d2", B="val_cfm_d2", c3="val_cfm_d2")


# ----------------------------------------------------------------------------- settings and folders
def settings(cfg, stage):
    db = json.loads(json.dumps(cfg["deblend"]))
    st = dict(db.get(stage, db.get(stage.lower(), {})) or {})
    tr = dict(db["train"])
    tr.update(st.pop("train", {}) or {})
    g = int(tr.get("gpus", 1) or 1)
    if g > 1:
        # Data parallel: batch_size examples per GPU, the optimiser sees g x batch_size per step.
        tr["batch_size_per_device"] = int(tr["batch_size"])
        tr["batch_size"] = int(tr["batch_size"]) * g
    return dict(stage=stage, d=int(db["latent_dim"]), train=scale_for_batch(tr), model=st)


def data_parallel(n):
    # Replicates the model on the first n devices and splits every batch between them (keras.distribution
    # DataParallel, JAX: gradients averaged across devices at every step). Set before the model is built.
    devs = keras.distribution.list_devices()
    if len(devs) < n:
        raise RuntimeError(f"deblend.train.gpus = {n} but only {len(devs)} devices: {devs} (sbatch --gres=gpu:{n})")
    keras.distribution.set_distribution(keras.distribution.DataParallel(devices=devs[:n]))
    print(f"data parallel on {n} devices: {devs[:n]}", flush=True)


def _trim(rows, g):
    # Validation rows: a multiple of the number of devices, so that every batch splits evenly.
    return rows if rows is None or g <= 1 else rows[:len(rows) - len(rows) % g]


def scale_for_batch(tr):
    # Batch sizes larger than the recipe's: the learning rate follows lr_scaling (sqrt: x sqrt(B /
    # lr_reference_batch), linear, or none). train_examples (optional): a budget in examples that sets the number of
    # steps (train_steps); without it the budget is in epochs. At B = lr_reference_batch nothing changes.
    tr = dict(tr)
    b = int(tr["batch_size"])
    ref = tr.get("lr_reference_batch")
    mode = str(tr.get("lr_scaling") or "none")
    if ref and mode != "none":
        f = {"sqrt": (b / int(ref)) ** 0.5, "linear": b / int(ref)}.get(mode)
        if f is None:
            raise ValueError(f"deblend.train.lr_scaling must be sqrt, linear or none, not {mode!r}")
        tr["learning_rate_reference"] = float(tr["learning_rate"])
        tr["learning_rate"] = float(tr["learning_rate"]) * f
    if tr.get("train_examples"):
        tr["train_steps"] = int(np.ceil(int(tr["train_examples"]) / b))
    return tr


def dep_dir(cfg, stage, seed=None):
    # Dependencies are shared across seeds of a later stage unless deblend.replicate_dependencies is true.
    rep = bool(cfg["deblend"].get("replicate_dependencies", False))
    d = run_dir(cfg, stage, seed if rep else None)
    if not (d / "DONE").exists():
        raise RuntimeError(f"{stage} run {d} is not finished (no DONE): train it first")
    return d


def sizes(cfg):
    return int(cfg["rubin"]["stamp_px"]), len(cfg["rubin"]["bands"]), int(cfg["euclid"]["stamp_px"])


def _c0(cfg):
    # Dummy Rubin conditioning for building a part (None without PSF conditioning).
    nc = data.n_cond(cfg)
    return ops.zeros((1, nc)) if nc else None


# ----------------------------------------------------------------------------- loading frozen parts
def _load(model, path, *dummy, **kw):
    model(*[ops.zeros(s) for s in dummy], **kw)
    model.load_weights(str(path))
    model.trainable = False
    return model


def res_hidden_of(d_run):
    # Width of the latent residual block of a finished run (0: none, as every run before 2026-11-28).
    p = Path(d_run) / "run.json"
    if not p.exists():
        return 0
    return int((json.loads(p.read_text()).get("settings", {}).get("model", {}) or {}).get("latent_res_hidden") or 0)


def load_k1_encoder(cfg, d_k1, d):
    _, _, n_e = sizes(cfg)
    return _load(ImageEncoder(d, res_hidden=res_hidden_of(d_k1), name="encoder"), d_k1 / "encoder_best.weights.h5",
                 (1, n_e, n_e, 2))


def load_rubin_encoder(cfg, d_run, d):
    n_r, b_r, _ = sizes(cfg)
    return _load(ImageEncoder(d, res_hidden=res_hidden_of(d_run), name="encoder"),
                 d_run / "encoder_best.weights.h5", (1, n_r, n_r, b_r), cond=_c0(cfg))


def load_decoder(path, n, channels, d, cfg=None):
    # cfg given: a Rubin decoder (conditioned on the Rubin PSF when deblend.psf_condition); None: no conditioning.
    dec = ConvDecoder(n, channels, name="decoder")
    return _load(dec, path, (1, d), cond=_c0(cfg) if cfg is not None else None)


def load_joint_encoder(cfg, d_c1, d, s):
    n_r, b_r, n_e = sizes(cfg)
    enc = JointEncoder(d, s["model"].get("embed", 128), tuple(s["model"].get("fusion", [256])),
                       int(s["model"].get("latent_res_hidden") or 0), name="encoder")
    enc(ops.zeros((1, n_r, n_r, b_r)), ops.zeros((1, n_e, n_e, 2)), cond=_c0(cfg))
    enc.load_weights(str(d_c1 / "encoder_best.weights.h5"))
    enc.trainable = False
    return enc


def load_c2_branches(cfg, d_c2, d, s_c2):
    n_r, b_r, n_e = sizes(cfg)
    m = s_c2["model"]
    br = C2Branches(d, m.get("embed", 128), tuple(m.get("fusion", [256])), name="branches")
    br(ops.zeros((1, n_r, n_r, b_r)), ops.zeros((1, n_e, n_e, 2)), cond=_c0(cfg))
    br.load_weights(str(d_c2 / "branches_best.weights.h5"))
    br.trainable = False
    return br


def _cfm_student(m, d, stats):
    return CFMStudent(d, m.get("context", 128), tuple(m.get("hidden", [128, 128])), stats,
                      int(m.get("time_dim", 0)), name="student")


def load_student(cfg, stage, d_run, d, s):
    # Trained student of Run A / B / C (C = stage c3) for inference, with its latent standardisation.
    n_r, b_r, _ = sizes(cfg)
    m = s["model"]
    stats = json.loads((d_run / "latent_stats.json").read_text()) if (d_run / "latent_stats.json").exists() \
        else None
    if stage == "A":
        st = RubinBlendRegressor(d, m.get("embed", 128), float(m.get("dropout", 0.1)), name="student")
        st.stats = stats
    else:
        st = _cfm_student(m, d, stats)
    st(ops.zeros((1, n_r, n_r, b_r)), cond=_c0(cfg))
    st.load_weights(str(d_run / "student_best.weights.h5"))
    st.trainable = False
    return st


# ----------------------------------------------------------------------------- warm start of a VAE
def init_dir(cfg, stage, s):
    # Finished run named by deblend.<stage>.init_from (None without it).
    name = s["model"].get("init_from")
    if not name:
        return None
    d = data.deblend_dir(cfg) / "runs" / str(name)
    if not (d / "DONE").exists():
        raise RuntimeError(f"deblend.{stage}.init_from: {d} is not a finished run (no DONE)")
    return d


def warm_start(cfg, stage, model, src, freeze_decoders):
    # Copies every weight of the finished VAE run `src` (same stage) into the built `model`: the encoder's parts by
    # name (its new LatentResBlock, absent from the source, keeps its zero start), the decoders whole. Returns the
    # decoders, frozen when freeze_decoders.
    d = int(model.encoder.head.d)
    if stage == "c1":
        enc = load_joint_encoder(cfg, src, d, json.loads((src / "run.json").read_text())["settings"])
        parts = ("r_embed", "e_embed", "fusion", "res", "head")
        decs = dict(dec_r=(model.dec_r, "dec_r", cfg), dec_e=(model.dec_e, "dec_e", None))
        n_r, b_r, n_e = sizes(cfg)
        shapes = dict(dec_r=(n_r, b_r), dec_e=(n_e, 1))
    else:
        enc = (load_rubin_encoder if stage == "rubin_vae" else load_k1_encoder)(cfg, src, d)
        parts = ("embed", "res", "head")
        decs = dict(decoder=(model.decoder, "decoder", cfg if stage == "rubin_vae" else None))
        shapes = dict(decoder=(model.decoder.n, model.decoder.channels))
    copied = []
    for p in parts:
        a, b = getattr(enc, p, None), getattr(model.encoder, p, None)
        if a is not None and b is not None:
            b.set_weights(a.get_weights())
            copied.append(p)
    out = {}
    for k, (dec, f, c) in decs.items():
        src_dec = load_decoder(src / f"{f}_best.weights.h5", *shapes[k], d, c)
        dec.set_weights(src_dec.get_weights())
        if freeze_decoders:
            dec.trainable = False
        out[k] = dec
    print(f"deblend {stage}: warm start from {src} (encoder parts {copied}, decoders {sorted(out)}"
          + (", frozen" if freeze_decoders else "") + ")", flush=True)
    return out


# ----------------------------------------------------------------------------- latent standardisation
def moments(mus, variances):
    # Mean and standard deviation per dimension of draws of Gaussian posteriors N(mu_i, diag var_i), computed
    # exactly (law of total variance: E[var] + Var[mu]) rather than from random draws, so that every run that needs
    # them gets the same numbers.
    mu, var = np.concatenate(mus), np.concatenate(variances)
    std = np.sqrt(var.mean(0) + mu.var(0))
    return dict(mean=mu.mean(0).tolist(), std=np.maximum(std, 1e-3).tolist(), n=int(len(mu)))


def _n(a):
    return np.asarray(ops.convert_to_numpy(a))


def shared_k1_stats(cfg, enc, d_k1, ds, n_batches=100):
    # Standardisation of the K1 latent shared by Runs A and B and all their seeds: posterior moments of K1 on the
    # first n_batches training batches of the stage (balanced D2 / D1 with the replay), computed by the first run
    # that needs them and written next to the campaign (deblend/latent_stats/), keyed by the K1 run and its
    # weights; every later run reads that file. (The batches come from the stage's own loader: a second D2 loader
    # would hold a second in-memory copy of D2.)
    from ..ml.debvader_bridge import weights_fingerprint
    # (the number of batches is in the key: deblend-bench asks for one batch and must not set the campaign's file)
    p = data.deblend_dir(cfg) / "latent_stats" / \
        f"k1_{Path(d_k1).name}_{weights_fingerprint(enc)}_b{int(n_batches)}.json"
    if p.exists():
        return json.loads(p.read_text())
    mus, vs = [], []
    for i in range(min(len(ds), int(n_batches))):
        x, _ = ds[i]
        for key in ("d2_e_in", "d1_e_in", "e_in"):
            if key in x:
                mu, lv = enc(x[key], training=False)
                mus.append(_n(mu))
                vs.append(np.exp(_n(lv)))
    s = dict(moments(mus, vs), source=dict(k1_run=str(d_k1), batches=int(min(len(ds), int(n_batches))),
                                           what="K1 posterior moments on the training batches of the first "
                                                "student (D2 and D1 replay)"))
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        data._write_json_atomic(p, s)
    # Two runs started together: both use the file that stayed.
    return json.loads(p.read_text())


def c3_stats(br, enc, ds, n_batches):
    # Moments of the two targets of C3 on mixed train batches: the C2 teacher r (full covariance: its diagonal) on
    # D2, q_A of C1 on D1.
    mus, vs = [], []
    for i in range(min(len(ds), int(n_batches))):
        x, _ = ds[i]
        mu, L = br.teacher(x["d2_x_in"], x["d2_e_in"], cond=x.get("d2_x_psf"))
        mus.append(_n(mu))
        vs.append((_n(L) ** 2).sum(-1))
        if "d1_r_in" in x:
            mu, lv = enc(x["d1_r_in"], x["d1_e_in"], cond=x.get("d1_r_psf"))
            mus.append(_n(mu))
            vs.append(np.exp(_n(lv)))
    return dict(moments(mus, vs), what="C2 teacher on D2 and C1 q_A on D1, mixed train batches")


# ----------------------------------------------------------------------------- fixed validation draws
def val_noise(stage, s):
    # Draws of the validation batches, fixed per object (data.fixed_draws), by half of a batch: {"d2": spec,
    # "d1": spec} (a D1-only stage uses "d1").
    d, m = s["d"], s["model"]
    if stage in ("rubin_vae", "k1", "k2", "c1"):
        return dict(d1=dict(eps_z=("normal", (d,))))
    if stage == "A":
        sp = dict(eps_t=("normal", (d,)), a_masks=("uniform", (int(m.get("val_draws", 32)), int(m.get("embed", 128)))))
        return dict(d2=sp, d1=sp)
    if stage in ("B", "c3"):
        M = int(m.get("val_triplets", 8))
        sp = dict(cfm_eps=("normal", (M, d)), cfm_u=("uniform", (M,)), cfm_z0=("normal", (M, d)))
        return dict(d2=sp, d1=sp)
    if stage == "c2":
        return dict(d2=dict(eps_r=("normal", (d,))))
    return {}


# ----------------------------------------------------------------------------- stage builders
def _loaders(cfg, s, which, seed, workers=None, noise=None):
    tr = s["train"]
    bs = int(tr["batch_size"])
    w = workers or tr.get("workers", 4)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or 8) if w == "auto" else int(w)
    g = int(tr.get("gpus", 1) or 1)
    blk = tr.get("read_block", 1)
    kw = dict(workers=workers, use_multiprocessing=False, max_queue_size=8,
              block=blk if blk == "auto" else int(blk or 1))
    noise = noise or {}
    if which == "d1":
        cols = data.train_columns(cfg, "d1")
        st = data.Store(cfg.dataset_dir("d1"), cols, shared=True)
        keep, info = data.d1_keep(cfg, st.meta)
        if info:
            print(f"D1 selection (deblend.d1_selection): {info}")
        vidx = validation_indices(st.meta, "val", tr.get("val_max_examples"), seed)
        if vidx is not None:
            vidx = vidx[keep[vidx]]
        elif g > 1:
            vidx = np.flatnonzero((st.meta["split"] == "val").to_numpy() & keep)
        vidx = _trim(vidx, g)
        fields = s.get("fields", ("rubin", "euclid"))
        pool = s.get("mask_pool")
        prob = float(s["model"].get("mask_augment_prob", 0.0))
        return (data.D1Batches(cfg, "train", bs, True, seed, fields=fields, mask_pool=pool, mask_prob=prob,
                               meta_columns=cols, **kw),
                data.D1Batches(cfg, "val", bs, False, seed, indices=vidx, fields=fields, meta_columns=cols,
                               val_noise=noise.get("d1")))
    if which == "d2":
        cols = data.train_columns(cfg, "d2")
        st = data.Store(cfg.dataset_dir("d2"), cols, shared=True)
        keep, info = data.d2_keep(cfg, st.meta)
        print(f"D2 selection (deblend.d2_selection): {info or 'none'}")
        rows = data.d2_rows(st.meta, "val", cfg["deblend"]["d2_centering"], keep)
        mx = tr.get("val_max_examples")
        if mx and len(rows) > int(mx):
            rows = np.sort(np.random.default_rng(seed).choice(rows, int(mx), replace=False))
        rows = _trim(rows, g)
        return (data.D2Batches(cfg, "train", bs, True, seed, meta_columns=cols, **kw),
                data.D2Batches(cfg, "val", bs, False, seed, indices=rows, meta_columns=cols,
                               val_noise=noise.get("d2")))
    if which == "mixed":
        # Balanced D2 / D1 batches (deblend.<stage>.d1_batch_size, default = batch_size): an epoch is one pass
        # over D2; the D1 batch of step i is batch i of D1's own order, reshuffled at every epoch.
        b1 = int(s["model"].get("d1_batch_size", bs))
        d2_tr, d2_va = _loaders(cfg, s, "d2", seed, workers, noise)
        s1 = dict(s, train=dict(tr, batch_size=b1))
        d1_tr, d1_va = _loaders(cfg, s1, "d1", seed + 1, workers, noise)
        mkw = {k: v for k, v in kw.items() if k != "block"}
        return data.Mixed(d2_tr, d1_tr, **mkw), data.Mixed(d2_va, d1_va)
    raise ValueError(which)


def build(cfg, stage, s, seed, deps):
    n_r, b_r, n_e = sizes(cfg)
    stats = data.load_norm_stats(cfg)
    sr, se = stats["rubin"]["scale"], stats["euclid"]["scale"]
    d, m = s["d"], s["model"]
    frozen, subs = {}, {}
    noise = val_noise(stage, s)
    replay = "mixed" if m.get("replay", True) else "d2"
    if stage == "rubin_vae":
        s["fields"] = ("rubin",)
        model = ImageVAE("rubin", n_r, b_r, sr, d, float(m.get("beta", 1.0)), seed,
                         int(m.get("latent_res_hidden") or 0), name="rubin_vae")
        subs = dict(encoder=model.encoder, decoder=model.decoder)
        loaders = _loaders(cfg, s, "d1", seed, noise=noise)
    elif stage == "k1":
        s["fields"] = ("euclid",)
        if float(m.get("mask_augment_prob", 0.0)) > 0:
            s["mask_pool"] = data.d2_mask_pool(cfg, int(m.get("mask_pool_size", 5000)), seed)
        model = ImageVAE("euclid", n_e, 1, se, d, float(m.get("beta", 1.0)), seed,
                         int(m.get("latent_res_hidden") or 0), name="k1")
        subs = dict(encoder=model.encoder, decoder=model.decoder)
        loaders = _loaders(cfg, s, "d1", seed, noise=noise)
    elif stage == "k2":
        enc = load_k1_encoder(cfg, deps["k1"], d)
        model = K2Model(enc, n_r, b_r, sr, seed, name="k2")
        frozen, subs = dict(k1_encoder=enc), dict(decoder=model.decoder)
        loaders = _loaders(cfg, s, "d1", seed, noise=noise)
    elif stage in ("A", "B"):
        enc = load_k1_encoder(cfg, deps["k1"], d)
        loaders = _loaders(cfg, s, replay, seed, noise=noise)
        st_ = shared_k1_stats(cfg, enc, deps["k1"], loaders[0], int(m.get("stats_batches", 100))) \
            if m.get("standardise", True) else None
        lam = float(m.get("lambda_replay", 1.0))
        if stage == "A":
            model = RunAModel(enc, d, int(m.get("embed", 128)), float(m.get("dropout", 0.1)),
                              int(m.get("draws", 8)), st_, lam, seed, name="run_a")
        else:
            model = RunBModel(enc, d, int(m.get("context", 128)), tuple(m.get("hidden", [128, 128])), st_, seed,
                              int(m.get("time_dim", 0)), lam, name="run_b")
        frozen, subs = dict(k1_encoder=enc), dict(student=model.student)
        s["latent_stats"] = st_
    elif stage == "c1":
        model = JointVAE(n_r, b_r, n_e, sr, se, d, int(m.get("embed", 128)), tuple(m.get("fusion", [256])),
                         float(m.get("beta", 1.0)), float(m.get("weight_r", 1.0)), float(m.get("weight_e", 1.0)),
                         seed, int(m.get("latent_res_hidden") or 0), name="c1")
        subs = dict(encoder=model.encoder, dec_r=model.dec_r, dec_e=model.dec_e)
        loaders = _loaders(cfg, s, "d1", seed, noise=noise)
    elif stage == "c2":
        s_c1 = json.loads((deps["c1"] / "run.json").read_text())["settings"]
        enc = load_joint_encoder(cfg, deps["c1"], d, s_c1)
        dec_e = load_decoder(deps["c1"] / "dec_e_best.weights.h5", n_e, 1, d)
        lam = m.get("lambda_d1", 1.0)
        auto = lam == "auto"
        model = C2Model(enc, dec_e, se, d, int(m.get("embed", 128)), tuple(m.get("fusion", [256])),
                        1.0 if auto else float(lam), seed, lambda_auto=auto,
                        protocol=dict(LAMBDA_PROTOCOL, **(m.get("lambda_protocol") or {})), name="c2")
        loaders = _loaders(cfg, s, "mixed", seed, noise=noise)
        x0, _ = loaders[1][0]
        model({k: v[:2] for k, v in x0.items()})
        # Backbones initialised from copies of the C1 embedders (the C1 encoder itself stays frozen).
        model.branches.r_embed.set_weights(enc.r_embed.get_weights())
        model.branches.e_embed.set_weights(enc.e_embed.get_weights())
        frozen, subs = dict(c1_encoder=enc, c1_dec_e=dec_e), dict(branches=model.branches)
    elif stage == "c3":
        s_c1 = json.loads((deps["c1"] / "run.json").read_text())["settings"]
        s_c2 = json.loads((deps["c2"] / "run.json").read_text())["settings"]
        enc = load_joint_encoder(cfg, deps["c1"], d, s_c1)
        br = load_c2_branches(cfg, deps["c2"], d, s_c2)
        loaders = _loaders(cfg, s, "mixed", seed, noise=noise)
        st_ = c3_stats(br, enc, loaders[0], int(m.get("stats_batches", 100))) if m.get("standardise", True) \
            else None
        model = C3Model(br, enc, d, int(m.get("context", 128)), tuple(m.get("hidden", [128, 128])), st_,
                        float(m.get("lambda_replay", 1.0)), seed, int(m.get("time_dim", 0)), name="c3")
        x0, _ = loaders[1][0]
        model({k: v[:2] for k, v in x0.items()})
        if m.get("init_context_from_c2", True):
            # The context starts as a copy of C2's Rubin backbone and is trained.
            model.student.context.set_weights(br.r_embed.get_weights())
        frozen, subs = dict(c2_branches=br, c1_encoder=enc), dict(student=model.student)
        s["latent_stats"] = st_
    else:
        raise ValueError(f"unknown deblend stage {stage!r}; known: {STAGES}")
    x0, _ = loaders[1][0]
    model({k: v[:2] for k, v in x0.items()})
    src = init_dir(cfg, stage, s) if stage in ("rubin_vae", "k1", "c1") else None
    if src is not None:
        decs = warm_start(cfg, stage, model, src, bool(m.get("freeze_decoders", False)))
        if m.get("freeze_decoders"):
            frozen = dict(frozen, **decs)        # still saved with the run (subs): later stages read them there
        s["init_from_run"] = str(src)
    return model, loaders, frozen, subs


def _selection(loader):
    # deblend.d2_selection as applied by the D2 loader of this stage (None for D1-only stages).
    for d in (loader, getattr(loader, "a", None)):
        if d is not None and hasattr(d, "selection"):
            return d.selection
    return None


def nonfinite_report(model, x):
    # Which inputs, outputs and loss terms of one batch are not finite: printed, nothing changed.
    bad_in = {k: int((~np.isfinite(np.asarray(v))).sum()) for k, v in x.items()
              if np.issubdtype(np.asarray(v).dtype, np.floating)}
    out = model(x, training=False)
    t = model.loss_terms(x, out)
    bad_out = {k: int((~np.isfinite(np.asarray(v))).sum()) for k, v in out.items()}
    terms = {k: float(v) for k, v in t.items()}
    print(f"nonfinite inputs {bad_in}", flush=True)
    print(f"nonfinite outputs {bad_out}", flush=True)
    print(f"loss terms {terms}", flush=True)


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
    gpus = int(s["train"].get("gpus", 1) or 1)
    if gpus > 1:
        data_parallel(gpus)
    try:
        return _train(cfg, stage, seed, s, deps, rd, seed_v, t0, resume, max_steps, time_budget_s)
    finally:
        if gpus > 1:
            keras.distribution.set_distribution(None)


def plan(s, n_batches, max_steps=None):
    # Epochs, updates per epoch and total updates of a stage: train_steps (a budget in updates) or epochs; a pilot
    # (max_steps) runs that many updates.
    tr = s["train"]
    spe = min(int(n_batches), int(max_steps)) if max_steps else int(n_batches)
    epochs = schedule(tr, n_batches)
    return epochs, spe, epochs * spe


def scaled_protocol(p, total):
    # The lambda_D1 protocol must end before 60 % of the run (pilots of a few hundred updates): shrunk otherwise.
    p = dict(p)
    end = p["warm"] + p["measure"] + p["ramp"]
    if end > 0.6 * total:
        f = 0.6 * total / end
        for k in ("warm", "measure", "ramp"):
            p[k] = max(int(p[k] * f), 1)
        p["every"] = max(1, min(int(p["every"]), p["measure"] // 4 or 1))
    return p


def _train(cfg, stage, seed, s, deps, rd, seed_v, t0, resume, max_steps, time_budget_s):
    model, (tr, va), frozen, subs = build(cfg, stage, s, seed_v, deps)
    guard = FrozenGuard(**frozen) if frozen else None
    from ..ml.debvader_bridge import weights_fingerprint
    t = s["train"]
    epochs, spe, total = plan(s, len(tr), max_steps)
    sched, warm = lr_schedule(t["learning_rate"], total, t.get("warmup_frac", 0.02), t.get("final_lr_frac", 0.01))
    m = s["model"]
    if hasattr(model, "kl_warmup"):
        model.kl_warmup = int(round(float(m.get("kl_warmup_frac", 0.0)) * total))
    start_epoch, extra = 0, []
    if getattr(model, "lambda_auto", False):
        model.protocol = scaled_protocol(model.protocol, total)
        p = model.protocol
        # Best checkpoint and early stopping count only once lambda_D1 is fixed.
        start_epoch = min(int(np.ceil((p["warm"] + p["measure"] + p["ramp"]) / spe)), max(epochs - 1, 0))
        extra.append(LambdaD1Protocol(tr, rd, p))
    ema = float(m["ema"]) if m.get("ema") else None
    # Steady-state gradient norms (the clipping threshold is read from them): after the warm-up, and for C2 once
    # lambda_D1 is fixed.
    stats_from = warm
    if getattr(model, "lambda_auto", False):
        p = model.protocol
        stats_from = max(warm, int(p["warm"] + p["measure"] + p["ramp"]))
    opt = DeblendAdam(sched, t.get("global_clipnorm"), ema=ema, ema_start=warm, stats_from=stats_from)
    monitor = m.get("monitor") or MONITOR.get(stage, "val_total_loss")
    s["schedule"] = dict(epochs=epochs, updates_per_epoch=spe, total_updates=total, warmup_updates=warm,
                         peak_lr=float(t["learning_rate"]), final_lr=float(t["learning_rate"]) *
                         float(t.get("final_lr_frac", 0.01)), kl_warmup_updates=getattr(model, "kl_warmup", None),
                         lambda_protocol=getattr(model, "protocol", None) if getattr(model, "lambda_auto", False)
                         else None, monitor=monitor, select_from_epoch=start_epoch, ema=ema,
                         global_clipnorm=t.get("global_clipnorm"), grad_stats_from=stats_from)
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
    print(f"deblend {stage}: {epochs} epochs x {spe} updates = {total}, warm-up {warm}, peak lr "
          f"{t['learning_rate']:.3g}, monitor {monitor}, global clipnorm {t.get('global_clipnorm')}"
          + (f", EMA {ema}" if ema else ""), flush=True)
    model.compile(optimizer=opt)
    resumable = resume and not max_steps
    # W&B (DP2_WANDB=1, offline on the compute nodes): after GradNormLogger, so that its entries are in the epoch logs.
    wb = WandbLogger(rd, rd.name, str(cfg["deblend"].get("run_suffix") or "") or "campaign",
                     dict(stage=stage, seed=seed_v, **{k: v for k, v in record.items() if k in ("train", "model",
                                                                                                "schedule", "d")}),
                     preview=({k: v[:4] for k, v in va[0][0].items()} if images_enabled() else None))
    first = [GradNormLogger(rd), wb] + ([EMAValidation()] if ema else []) + extra
    cbs = callbacks(rd, subs, t["early_stopping_patience"], resumable, t.get("backup_every_steps"),
                    monitor=monitor, start_epoch=start_epoch, ema=(lambda: EMAWeights(opt)) if ema else None,
                    first=first)
    budget = None
    if time_budget_s and resumable:
        backup = next(c for c in cbs if isinstance(c, keras.callbacks.BackupAndRestore))
        budget = TimeBudget(time_budget_s, backup, t0)
        cbs.append(budget)
    model.fit(tr, validation_data=va, epochs=epochs, steps_per_epoch=spe if max_steps else None, callbacks=cbs,
              verbose=2)
    if budget is not None and budget.hit:
        print(f"deblend {stage}: stopped on the time budget, not finished (no DONE); the same command resumes")
        return rd
    if guard is not None:
        after = guard.check()
        write_json(rd / "frozen_check.json", dict(before=guard.before, after=after, unchanged=True))
    (rd / "DONE").write_text(f"deblend stage {stage} finished\n")
    print(f"deblend {stage} -> {rd}")
    return rd


# Former name, kept for the scripts that compile a stage outside a training (bench, speed_probe).
def optimizer(lr, clipnorm=None, warmup_steps=0):
    return DeblendAdam(float(lr), clipnorm)
