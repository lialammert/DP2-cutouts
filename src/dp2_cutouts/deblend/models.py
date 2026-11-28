# models.py
# Anna Preto
# AstroDeep Project
# Trainable models of every stage. Each is a keras.Model whose call() returns a dict and whose compute_loss() turns it
# into the training objective (backend agnostic: no custom train_step). Image likelihoods are observational
# Gaussians in physical units (likelihood.gaussian_nll): decoders output the expected signal in units of the
# per-band scale s_b of norm_stats.json, multiplied back before comparison with the observed flux.
#
#   ImageVAE        one instrument: Rubin reference VAE (6 bands) or K1 (Euclid VIS, mask-aware input); diagonal
#                   Gaussian q(z | image), loss  E_q[-log p(image | z)] + beta KL(q || N(0, I)).
#   K2Model         frozen K1 encoder, z_E ~ q_E(z | E), new Rubin decoder; loss E_z[-log p_R(R | z_E)].
#   RunAModel       Rubin blend CNN, M MC-dropout draws z_B; target one draw of q_E(z | E_c, M_E) (K1 frozen), both
#                   in the standardised K1 latent space; loss: energy score of the draws against the target.
#   CFMStudent      Rubin blend CNN context h(X) + velocity field; targets: samples of a frozen teacher (K1 for
#                   Run B; the C2 teacher r(z | X, E_c) plus a D1 replay with q_A(z | R, E) for C3).
#   JointVAE        C1: Rubin and Euclid embedders, fusion MLP, diagonal Gaussian q_A(z | R, E), two decoders;
#                   loss E_q[-log p_R(R | z) - log p_E(E | z)] + beta KL(q_A || N(0, I)).
#   C2Model         Gaussian student s(z | X) and teacher r(z | X, E_c), full covariance, sharing one Rubin
#                   backbone (initialised from C1's); frozen C1 Euclid decoder; loss on D2
#                   E_r[-log p_E(E_c | z)] + KL(r || s), plus lambda_D1 (1/2 KL(q_A || s(R)) + 1/2 KL(q_A || r(R, E)))
#                   on D1 examples, q_A from the frozen C1 encoder.
#
# Changes of 2026-11-28:
# - PSF conditioning (deblend.psf_condition): the standardised Rubin PSF moments of the example (18 scalars, keys
#   r_psf on D1, x_psf on D2) are appended to every Rubin embedding (before the latent heads and the fusion) and to
#   the latent at the input of every Rubin decoder. Euclid parts get none (only a constant VIS FWHM is known).
# - Runs A and B replay D1 like C2 and C3 (mixed batches, keys d2_* and d1_*): the same objective on the isolated
#   pairs (Rubin isolated R as input, K1 on E as teacher), 50 / 50 with D2.
# - Training step counter n_updates (a non-trainable variable, so a resumed run keeps it): linear KL warm-up of the
#   VAEs, and the lambda_D1 protocol of C2.
# - Validation on fixed draws given by the loader (keys eps_z, eps_t, a_masks, cfm_u / cfm_z0 / cfm_eps, eps_r):
#   the same reparametrisation noise, dropout masks and flow triplets for every checkpoint, so that validation
#   losses compare checkpoints and not draws. The latents themselves move with the encoders.

import keras
import numpy as np
from keras import layers, ops

from . import cfm
from .layers import (MLP, ConvDecoder, ConvEmbedder, DiagGaussianHead, TriLGaussianHead, sample_diag, sample_tril,
                     with_cond)
from .likelihood import chi2_per_pixel, diag_to_tril, energy_score, gaussian_nll, kl_diag_std, kl_tril_tril


def _t(a):
    return ops.convert_to_tensor(np.asarray(a, np.float32))


def ramp(step, n):
    # min(1, step / n), 1 when n <= 0.
    if not n or float(n) <= 0:
        return 1.0
    return ops.clip(ops.cast(step, "float32") / float(n), 0.0, 1.0)


def active_dims(mu, threshold=0.01):
    # Latent dimensions in use (Burda et al. 2016): variance over the batch of the posterior mean above threshold.
    return ops.sum(ops.cast(ops.var(mu, axis=0) > threshold, "float32"))


def halves(x):
    # (name, key prefix, image side) of each part of a batch: mixed D2 / D1 batches (keys d2_*, d1_*; D2 input x_*,
    # D1 input r_*) or a plain D2 batch.
    if "d2_x_in" in x:
        out = [("d2", "d2_", "x")]
        if "d1_r_in" in x:
            out.append(("d1", "d1_", "r"))
        return out
    return [("d2", "", "x")]


def draw_diag(mu, log_var, eps, seed, training):
    # One draw of a diagonal Gaussian: the loader's fixed eps in validation when given, a new one otherwise.
    if not training and eps is not None:
        return mu + ops.exp(0.5 * log_var) * eps
    return sample_diag(mu, log_var, seed)


def draws_diag(mu, log_var, eps):
    # (B, d), eps (B, M, d) -> (B, M, d).
    return mu[:, None, :] + ops.exp(0.5 * log_var)[:, None, :] * eps


def draws_tril(mu, L, eps):
    return mu[:, None, :] + ops.einsum("bij,bkj->bki", L, eps)


class Tracked(keras.Model):
    # Mean trackers for every loss term (shown in logs and history.csv), and the count of training updates.
    def setup_trackers(self, names):
        self.trackers = {k: keras.metrics.Mean(name=k) for k in names}
        self.n_updates = self.add_weight(shape=(), initializer="zeros", trainable=False, name="n_updates")

    @property
    def metrics(self):
        return list(self.trackers.values())

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        t = self.loss_terms(x, y_pred, training=training)
        for k, v in t.items():
            if k in self.trackers:
                self.trackers[k].update_state(v)
        if training:
            self.n_updates.assign(self.n_updates + 1.0)
        return t["total_loss"]


class ImageEncoder(keras.Model):
    # image -> (mu, log_var): ConvEmbedder + diagonal head; cond: Rubin PSF (Rubin encoders only).
    def __init__(self, d, embed=256, filters=(32, 64, 128, 256), **kw):
        super().__init__(**kw)
        self.embed = ConvEmbedder(embed, filters, name="embed")
        self.head = DiagGaussianHead(d, name="head")

    def call(self, x, cond=None, training=False):
        return self.head(self.embed(x, cond=cond, training=training))


class ImageVAE(Tracked):
    def __init__(self, side, n, out_channels, scale, d=32, beta=1.0, seed=0, **kw):
        super().__init__(**kw)
        self.side = side                          # "rubin" (keys r_*) or "euclid" (keys e_*)
        self.k = "r" if side == "rubin" else "e"
        self.encoder = ImageEncoder(d, name="encoder")
        self.decoder = ConvDecoder(n, out_channels, name="decoder")
        self.scale = _t(scale)
        self.beta = float(beta)
        self.kl_warmup = 0                        # steps of the linear KL warm-up (train.build)
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "nll", "kl", "chi2", "beta_eff", "active_dims"])

    def call(self, x, training=False):
        c = x.get("r_psf") if self.k == "r" else None
        mu, lv = self.encoder(x[f"{self.k}_in"], cond=c, training=training)
        eps = x.get("eps_z")
        z = draw_diag(mu, lv, eps, self.seed_gen, training) if (training or eps is not None) else mu
        return dict(mu=mu, log_var=lv, mean=self.decoder(z, cond=c) * self.scale)

    def loss_terms(self, x, out, training=False):
        nll = ops.mean(gaussian_nll(x[f"{self.k}_flux"], out["mean"], x[f"{self.k}_w"]))
        kl = ops.mean(kl_diag_std(out["mu"], out["log_var"]))
        # Validation always at the full beta (the ELBO at beta = 1), also during the warm-up.
        beta = self.beta * ramp(self.n_updates, self.kl_warmup) if training else self.beta
        return dict(total_loss=nll + beta * kl, nll=nll, kl=kl, beta_eff=ops.convert_to_tensor(beta, "float32"),
                    chi2=ops.mean(chi2_per_pixel(x[f"{self.k}_flux"], out["mean"], x[f"{self.k}_w"])),
                    active_dims=active_dims(out["mu"]))


class K2Model(Tracked):
    # Rubin decoder trained from the frozen Euclid latent; it never sees a Rubin image as input (only the Rubin PSF).
    def __init__(self, k1_encoder, n_r, bands_r, scale_r, seed=0, **kw):
        super().__init__(**kw)
        self.k1_encoder = k1_encoder
        self.k1_encoder.trainable = False
        self.decoder = ConvDecoder(n_r, bands_r, name="rubin_decoder")
        self.scale = _t(scale_r)
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "nll", "chi2"])

    def call(self, x, training=False):
        mu, lv = self.k1_encoder(x["e_in"], training=False)
        mu, lv = ops.stop_gradient(mu), ops.stop_gradient(lv)
        eps = x.get("eps_z")
        z = draw_diag(mu, lv, eps, self.seed_gen, training) if (training or eps is not None) else mu
        return dict(mean=self.decoder(z, cond=x.get("r_psf")) * self.scale)

    def loss_terms(self, x, out, training=False):
        nll = ops.mean(gaussian_nll(x["r_flux"], out["mean"], x["r_w"]))
        return dict(total_loss=nll, nll=nll, chi2=ops.mean(chi2_per_pixel(x["r_flux"], out["mean"], x["r_w"])))


class RubinBlendRegressor(keras.Model):
    # Run A student: CNN embedding h(X) computed once per blend; each draw applies its own dropout mask to it (rate
    # `dropout`, inverted scaling), appends the PSF and maps linearly to the (standardised K1) latent. The masks come
    # from uniform draws u (B, M, embed): kept where u >= rate. Kept active at inference: the draws are the samples.
    def __init__(self, d=32, embed=128, dropout=0.1, **kw):
        super().__init__(**kw)
        self.embed_dim = int(embed)
        self.embed = ConvEmbedder(embed, name="embed")
        self.out = layers.Dense(d)
        self.rate = float(dropout)

    def call(self, x_in, cond=None, training=False):
        # The draw without dropout (all units kept); builds the layers.
        return self.out(with_cond(self.embed(x_in, training=training), cond))

    def draws(self, x_in, cond, u, training=False):
        h = self.embed(x_in, training=training)
        keep = ops.cast(u >= self.rate, h.dtype) / (1.0 - self.rate)
        hm = h[:, None, :] * keep
        if cond is not None:
            c = ops.cast(cond, hm.dtype)[:, None, :]
            hm = ops.concatenate([hm, ops.repeat(c, int(u.shape[1]), axis=1)], axis=-1)
        return self.out(hm)

    def sample(self, x_in, cond, k, seed):
        u = keras.random.uniform((ops.shape(x_in)[0], int(k), self.embed_dim), seed=seed)
        return self.draws(x_in, cond, u)


class RunAModel(Tracked):
    # Energy score of M dropout draws of the student against one draw of the frozen K1 posterior, per object and per
    # step, both standardised with the K1 latent statistics shared with Run B (train.shared_k1_stats). D2 and, with
    # mixed batches, the D1 replay (weight lambda_replay). Checkpoints are chosen on the D2 term (val_es_d2).
    def __init__(self, k1_encoder, d=32, embed=128, dropout=0.1, draws=8, stats=None, lambda_replay=1.0, seed=0,
                 **kw):
        super().__init__(**kw)
        self.teacher = k1_encoder
        self.teacher.trainable = False
        self.student = RubinBlendRegressor(d, embed, dropout, name="student")
        self.n_draws = int(draws)
        self.stats = stats
        self.lambda_replay = float(lambda_replay)
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "es_d2", "err_d2", "spread_d2", "es_d1", "err_d1", "spread_d1"])

    def call(self, x, training=False):
        out = {}
        for name, p, side in halves(x):
            mu, lv = self.teacher(x[p + "e_in"], training=False)
            mu, lv = ops.stop_gradient(mu), ops.stop_gradient(lv)
            y = draw_diag(mu, lv, x.get(p + "eps_t"), self.seed_gen, training)
            u = None if training else x.get(p + "a_masks")
            if u is None:
                u = keras.random.uniform((ops.shape(mu)[0], self.n_draws, self.student.embed_dim), seed=self.seed_gen)
            xin, c = x[p + side + "_in"], x.get(p + side + "_psf")
            if not self.student.built:
                self.student(xin, cond=c)            # builds the student (draws() alone does not mark it built)
            out[f"z_{name}"] = self.student.draws(xin, c, u, training=training)
            out[f"y_{name}"] = cfm.standardise(y, self.stats)
        return out

    def loss_terms(self, x, out, training=False):
        t, total = {}, 0.0
        for name, w in (("d2", 1.0), ("d1", self.lambda_replay)):
            if f"z_{name}" not in out:
                continue
            es, err, spread = energy_score(out[f"z_{name}"], out[f"y_{name}"])
            t[f"es_{name}"], t[f"err_{name}"], t[f"spread_{name}"] = ops.mean(es), ops.mean(err), ops.mean(spread)
            total = total + w * t[f"es_{name}"]
        t["total_loss"] = total
        return t


class CFMStudent(keras.Model):
    # Rubin blend (+ PSF) -> context h(X) -> velocity field. Inference: sample(x_in, cond, k, nfe, method, seed).
    def __init__(self, d=32, context=128, hidden=(128, 128), stats=None, time_dim=0, **kw):
        super().__init__(**kw)
        self.context = ConvEmbedder(context, name="context")
        self.field = cfm.VelocityField(d, hidden, time_dim, name="field")
        self.stats = stats

    def call(self, x_in, cond=None, training=False):
        h = self.context(x_in, cond=cond, training=training)
        if not self.field.built:
            # The field is otherwise first called inside the loss: its variables must exist before compilation.
            n = ops.shape(h)[0]
            self.field(ops.zeros((n, self.field.d)), ops.zeros((n, 1)), h)
        return h

    def sample(self, x_in, cond, k, nfe, method, seed):
        h = self.context(x_in, cond=cond, training=False)
        return cfm.sample(self.field, h, k, nfe, method, seed, self.stats)


def _cfm_targets(out, name, x, p, training, stats, seed, mu, scale, tril=False):
    # Teacher targets of one half: one random draw (training, or validation without fixed draws), or the M fixed
    # triplets of the loader (validation: p + cfm_eps teacher noise, p + cfm_u, p + cfm_z0).
    eps = None if training else x.get(p + "cfm_eps")
    if eps is not None:
        z1s = draws_tril(mu, scale, eps) if tril else draws_diag(mu, scale, eps)
        out.update({f"z1s_{name}": cfm.standardise(z1s, stats), f"u_{name}": x[p + "cfm_u"],
                    f"z0_{name}": x[p + "cfm_z0"]})
    else:
        z1 = sample_tril(mu, scale, seed) if tril else sample_diag(mu, scale, seed)
        out[f"z1_{name}"] = cfm.standardise(z1, stats)


def _cfm_loss(field, out, name, seed):
    if f"z1s_{name}" in out:
        return cfm.fixed_loss(field, out[f"z1s_{name}"], out[f"h_{name}"], out[f"u_{name}"], out[f"z0_{name}"])
    return cfm.cfm_loss(field, out[f"z1_{name}"], out[f"h_{name}"], seed)


class RunBModel(Tracked):
    # CFM student on draws of the frozen K1 teacher: D2 and, with mixed batches, the D1 replay (weight
    # lambda_replay). Checkpoints are chosen on the D2 term (val_cfm_d2).
    def __init__(self, k1_encoder, d=32, context=128, hidden=(128, 128), stats=None, seed=0, time_dim=0,
                 lambda_replay=1.0, **kw):
        super().__init__(**kw)
        self.teacher = k1_encoder
        self.teacher.trainable = False
        self.student = CFMStudent(d, context, hidden, stats, time_dim, name="student")
        self.lambda_replay = float(lambda_replay)
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "cfm_d2", "cfm_d1"])

    def call(self, x, training=False):
        out = {}
        for name, p, side in halves(x):
            mu, lv = self.teacher(x[p + "e_in"], training=False)
            mu, lv = ops.stop_gradient(mu), ops.stop_gradient(lv)
            out[f"h_{name}"] = self.student(x[p + side + "_in"], cond=x.get(p + side + "_psf"), training=training)
            _cfm_targets(out, name, x, p, training, self.student.stats, self.seed_gen, mu, lv)
        return out

    def loss_terms(self, x, out, training=False):
        t, total = {}, 0.0
        for name, w in (("d2", 1.0), ("d1", self.lambda_replay)):
            if f"h_{name}" in out:
                t[f"cfm_{name}"] = _cfm_loss(self.student.field, out, name, self.seed_gen)
                total = total + w * t[f"cfm_{name}"]
        t["total_loss"] = total
        return t


class JointEncoder(keras.Model):
    # (R, E) -> embeddings (the Rubin one with the PSF) -> fusion MLP -> diagonal Gaussian (C1's q_A).
    def __init__(self, d=32, embed=128, fusion=(256,), **kw):
        super().__init__(**kw)
        self.r_embed = ConvEmbedder(embed, name="r_embed")
        self.e_embed = ConvEmbedder(embed, name="e_embed")
        self.fusion = MLP(fusion, fusion[-1], name="fusion")
        self.head = DiagGaussianHead(d, name="head")

    def call(self, r_in, e_in, cond=None, training=False):
        h = ops.concatenate([self.r_embed(r_in, cond=cond, training=training),
                             self.e_embed(e_in, training=training)], -1)
        return self.head(ops.silu(self.fusion(h)))


class JointVAE(Tracked):
    def __init__(self, n_r, bands_r, n_e, scale_r, scale_e, d=32, embed=128, fusion=(256,), beta=1.0,
                 weight_r=1.0, weight_e=1.0, seed=0, **kw):
        super().__init__(**kw)
        self.encoder = JointEncoder(d, embed, fusion, name="encoder")
        self.dec_r = ConvDecoder(n_r, bands_r, name="dec_r")
        self.dec_e = ConvDecoder(n_e, 1, name="dec_e")
        self.scale_r, self.scale_e = _t(scale_r), _t(scale_e)
        self.beta, self.w_r, self.w_e = float(beta), float(weight_r), float(weight_e)
        self.kl_warmup = 0
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "nll_r", "nll_e", "kl", "chi2_r", "chi2_e", "beta_eff", "active_dims"])

    def call(self, x, training=False):
        c = x.get("r_psf")
        mu, lv = self.encoder(x["r_in"], x["e_in"], cond=c, training=training)
        eps = x.get("eps_z")
        z = draw_diag(mu, lv, eps, self.seed_gen, training) if (training or eps is not None) else mu
        return dict(mu=mu, log_var=lv, mean_r=self.dec_r(z, cond=c) * self.scale_r,
                    mean_e=self.dec_e(z) * self.scale_e)

    def loss_terms(self, x, out, training=False):
        nr = ops.mean(gaussian_nll(x["r_flux"], out["mean_r"], x["r_w"]))
        ne = ops.mean(gaussian_nll(x["e_flux"], out["mean_e"], x["e_w"]))
        kl = ops.mean(kl_diag_std(out["mu"], out["log_var"]))
        beta = self.beta * ramp(self.n_updates, self.kl_warmup) if training else self.beta
        return dict(total_loss=self.w_r * nr + self.w_e * ne + beta * kl, nll_r=nr, nll_e=ne, kl=kl,
                    beta_eff=ops.convert_to_tensor(beta, "float32"),
                    chi2_r=ops.mean(chi2_per_pixel(x["r_flux"], out["mean_r"], x["r_w"])),
                    chi2_e=ops.mean(chi2_per_pixel(x["e_flux"], out["mean_e"], x["e_w"])),
                    active_dims=active_dims(out["mu"]))


class C2Branches(keras.Model):
    # Shared Rubin backbone (with the PSF); student head s(z | X); teacher = Euclid embedder + fusion + head
    # r(z | X, E_c).
    def __init__(self, d=32, embed=128, fusion=(256,), **kw):
        super().__init__(**kw)
        self.r_embed = ConvEmbedder(embed, name="r_embed")
        self.e_embed = ConvEmbedder(embed, name="e_embed")
        self.s_head = TriLGaussianHead(d, name="s_head")
        self.fusion = MLP(fusion, fusion[-1], name="fusion")
        self.r_head = TriLGaussianHead(d, name="r_head")

    def student(self, x_in, cond=None, training=False):
        return self.s_head(self.r_embed(x_in, cond=cond, training=training))

    def teacher(self, x_in, e_in, cond=None, training=False):
        h = ops.concatenate([self.r_embed(x_in, cond=cond, training=training),
                             self.e_embed(e_in, training=training)], -1)
        return self.r_head(ops.silu(self.fusion(h)))

    def call(self, x_in, e_in, cond=None, training=False):
        return self.student(x_in, cond, training), self.teacher(x_in, e_in, cond, training)


# lambda_D1 protocol of C2 (deblend.c2.lambda_d1 = auto, 2026-11-28), in training updates: lambda_D1 = 1 for `warm`
# updates; the gradient norms of the two objectives on the shared backbone are measured during the next `measure`
# (train.LambdaD1Protocol); lambda_D1 then moves linearly to the median ratio over `ramp` updates and stays there.
LAMBDA_PROTOCOL = dict(warm=500, measure=200, ramp=200, every=10, batches=2)


class C2Model(Tracked):
    def __init__(self, c1_encoder, c1_dec_e, scale_e, d=32, embed=128, fusion=(256,), lambda_d1=1.0, seed=0,
                 lambda_auto=False, protocol=None, **kw):
        super().__init__(**kw)
        self.c1_encoder = c1_encoder
        self.c1_encoder.trainable = False
        self.dec_e = c1_dec_e
        self.dec_e.trainable = False
        self.branches = C2Branches(d, embed, fusion, name="branches")
        self.scale_e = _t(scale_e)
        self.lambda_d1 = float(lambda_d1)
        self.lambda_auto = bool(lambda_auto)
        self.protocol = dict(LAMBDA_PROTOCOL, **(protocol or {}))
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "d2_nll_e", "d2_kl_rs", "d1_kl_s", "d1_kl_r", "d2_chi2_e", "lambda_d1"])
        # Target of the protocol, set once by train.LambdaD1Protocol (kept in the backups: a resumed run keeps it).
        self.lambda_target = self.add_weight(shape=(), initializer="ones", trainable=False, name="lambda_target")

    def current_lambda(self):
        if not self.lambda_auto:
            return ops.convert_to_tensor(self.lambda_d1, "float32")
        p = self.protocol
        frac = ops.clip((self.n_updates - float(p["warm"] + p["measure"])) / float(max(p["ramp"], 1)), 0.0, 1.0)
        return 1.0 + (self.lambda_target - 1.0) * frac

    def call(self, x, training=False):
        out = {}
        (mu_s, L_s), (mu_r, L_r) = self.branches(x["d2_x_in"], x["d2_e_in"], cond=x.get("d2_x_psf"),
                                                 training=training)
        eps = None if training else x.get("d2_eps_r")
        z = mu_r + ops.squeeze(ops.matmul(L_r, eps[..., None]), -1) if eps is not None else \
            sample_tril(mu_r, L_r, self.seed_gen)
        out.update(mu_s=mu_s, L_s=L_s, mu_r=mu_r, L_r=L_r, mean_e=self.dec_e(z) * self.scale_e)
        if "d1_r_in" in x:
            c1 = x.get("d1_r_psf")
            mu_a, lv_a = self.c1_encoder(x["d1_r_in"], x["d1_e_in"], cond=c1, training=False)
            out.update(mu_a=ops.stop_gradient(mu_a), L_a=ops.stop_gradient(diag_to_tril(lv_a)))
            (out["mu_s1"], out["L_s1"]), (out["mu_r1"], out["L_r1"]) = \
                self.branches(x["d1_r_in"], x["d1_e_in"], cond=c1, training=training)
        return out

    def loss_terms(self, x, out, training=False):
        nll = ops.mean(gaussian_nll(x["d2_e_flux"], out["mean_e"], x["d2_e_w"]))
        kl_rs = ops.mean(kl_tril_tril(out["mu_r"], out["L_r"], out["mu_s"], out["L_s"]))
        lam = self.current_lambda()
        t = dict(d2_nll_e=nll, d2_kl_rs=kl_rs, lambda_d1=lam,
                 d2_chi2_e=ops.mean(chi2_per_pixel(x["d2_e_flux"], out["mean_e"], x["d2_e_w"])))
        total = nll + kl_rs
        if "mu_a" in out:
            ks = ops.mean(kl_tril_tril(out["mu_a"], out["L_a"], out["mu_s1"], out["L_s1"]))
            kr = ops.mean(kl_tril_tril(out["mu_a"], out["L_a"], out["mu_r1"], out["L_r1"]))
            t.update(d1_kl_s=ks, d1_kl_r=kr)
            total = total + lam * (0.5 * ks + 0.5 * kr)
        t["total_loss"] = total
        return t


class C3Model(Tracked):
    # CFM student on draws of the frozen C2 teacher r(z | X, E_c) (D2) and of the frozen C1 q_A(z | R, E) (D1
    # replay). Checkpoints are chosen on the D2 term (val_cfm_d2).
    def __init__(self, c2_branches, c1_encoder, d=32, context=128, hidden=(128, 128), stats=None,
                 lambda_replay=1.0, seed=0, time_dim=0, **kw):
        super().__init__(**kw)
        self.teacher = c2_branches
        self.teacher.trainable = False
        self.c1_encoder = c1_encoder
        self.c1_encoder.trainable = False
        self.student = CFMStudent(d, context, hidden, stats, time_dim, name="student")
        self.lambda_replay = float(lambda_replay)
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "cfm_d2", "cfm_d1"])

    def call(self, x, training=False):
        out = {}
        for name, p, side in halves(x):
            c = x.get(p + side + "_psf")
            if name == "d2":
                mu, L = self.teacher.teacher(x[p + "x_in"], x[p + "e_in"], cond=c, training=False)
                tril = True
            else:
                mu, L = self.c1_encoder(x[p + "r_in"], x[p + "e_in"], cond=c, training=False)
                tril = False
            mu, L = ops.stop_gradient(mu), ops.stop_gradient(L)
            out[f"h_{name}"] = self.student(x[p + side + "_in"], cond=c, training=training)
            _cfm_targets(out, name, x, p, training, self.student.stats, self.seed_gen, mu, L, tril=tril)
        return out

    def loss_terms(self, x, out, training=False):
        t, total = {}, 0.0
        for name, w in (("d2", 1.0), ("d1", self.lambda_replay)):
            if f"h_{name}" in out:
                t[f"cfm_{name}"] = _cfm_loss(self.student.field, out, name, self.seed_gen)
                total = total + w * t[f"cfm_{name}"]
        t["total_loss"] = total
        return t
