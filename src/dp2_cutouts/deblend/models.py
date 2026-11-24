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
#   RunAModel       Rubin blend CNN + MC dropout -> z_B; target z_E ~ q_E(z | E_c, M_E) (K1 frozen);
#                   loss mean_{batch, dims} (z_B - z_E)^2.
#   CFMStudent      Rubin blend CNN context h(X) + velocity field; targets: samples of a frozen teacher (K1 for
#                   Run B; the C2 teacher r(z | X, E_c) plus a D1 replay with q_A(z | R, E) for C3).
#   JointVAE        C1: Rubin and Euclid embedders, fusion MLP, diagonal Gaussian q_A(z | R, E), two decoders;
#                   loss E_q[-log p_R(R | z) - log p_E(E | z)] + beta KL(q_A || N(0, I)).
#   C2Model         Gaussian student s(z | X) and teacher r(z | X, E_c), full covariance, sharing one Rubin
#                   backbone (initialised from C1's); frozen C1 Euclid decoder; loss on D2
#                   E_r[-log p_E(E_c | z)] + KL(r || s), plus lambda_D1 (1/2 KL(q_A || s(R)) + 1/2 KL(q_A || r(R, E)))
#                   on D1 examples, q_A from the frozen C1 encoder.

import keras
import numpy as np
from keras import ops

from . import cfm
from .layers import (MLP, ConvDecoder, ConvEmbedder, DiagGaussianHead, TriLGaussianHead, sample_diag, sample_tril)
from .likelihood import chi2_per_pixel, diag_to_tril, gaussian_nll, kl_diag_std, kl_tril_tril


def _t(a):
    return ops.convert_to_tensor(np.asarray(a, np.float32))


class Tracked(keras.Model):
    # Mean trackers for every loss term (shown in logs and history.csv).
    def setup_trackers(self, names):
        self.trackers = {k: keras.metrics.Mean(name=k) for k in names}

    @property
    def metrics(self):
        return list(self.trackers.values())

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        t = self.loss_terms(x, y_pred)
        for k, v in t.items():
            if k in self.trackers:
                self.trackers[k].update_state(v)
        return t["total_loss"]


class ImageEncoder(keras.Model):
    # image -> (mu, log_var): ConvEmbedder + diagonal head.
    def __init__(self, d, embed=256, filters=(32, 64, 128, 256), **kw):
        super().__init__(**kw)
        self.embed = ConvEmbedder(embed, filters, name="embed")
        self.head = DiagGaussianHead(d, name="head")

    def call(self, x, training=False):
        return self.head(self.embed(x, training=training))


class ImageVAE(Tracked):
    def __init__(self, side, n, out_channels, scale, d=32, beta=1.0, seed=0, **kw):
        super().__init__(**kw)
        self.side = side                          # "rubin" (keys r_*) or "euclid" (keys e_*)
        self.k = "r" if side == "rubin" else "e"
        self.encoder = ImageEncoder(d, name="encoder")
        self.decoder = ConvDecoder(n, out_channels, name="decoder")
        self.scale = _t(scale)
        self.beta = float(beta)
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "nll", "kl", "chi2"])

    def call(self, x, training=False):
        mu, lv = self.encoder(x[f"{self.k}_in"], training=training)
        z = sample_diag(mu, lv, self.seed_gen) if training else mu
        return dict(mu=mu, log_var=lv, mean=self.decoder(z) * self.scale)

    def loss_terms(self, x, out):
        nll = ops.mean(gaussian_nll(x[f"{self.k}_flux"], out["mean"], x[f"{self.k}_w"]))
        kl = ops.mean(kl_diag_std(out["mu"], out["log_var"]))
        return dict(total_loss=nll + self.beta * kl, nll=nll, kl=kl,
                    chi2=ops.mean(chi2_per_pixel(x[f"{self.k}_flux"], out["mean"], x[f"{self.k}_w"])))


class K2Model(Tracked):
    # Rubin decoder trained from the frozen Euclid latent; it never sees a Rubin image as input.
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
        z = sample_diag(mu, lv, self.seed_gen) if training else mu
        return dict(mean=self.decoder(z) * self.scale)

    def loss_terms(self, x, out):
        nll = ops.mean(gaussian_nll(x["r_flux"], out["mean"], x["r_w"]))
        return dict(total_loss=nll, nll=nll, chi2=ops.mean(chi2_per_pixel(x["r_flux"], out["mean"], x["r_w"])))


class RubinBlendRegressor(keras.Model):
    # Run A student: CNN (dropout on the embedding, kept active at inference when mc=True) -> linear latent.
    def __init__(self, d=32, embed=128, dropout=0.1, **kw):
        super().__init__(**kw)
        self.embed = ConvEmbedder(embed, dropout=dropout, mc_dropout=False, name="embed")
        self.out = keras.layers.Dense(d)

    def set_mc(self, on):
        self.embed.mc_dropout = bool(on)

    def call(self, x, training=False):
        return self.out(self.embed(x, training=training))


class RunAModel(Tracked):
    def __init__(self, k1_encoder, d=32, embed=128, dropout=0.1, seed=0, **kw):
        super().__init__(**kw)
        self.teacher = k1_encoder
        self.teacher.trainable = False
        self.student = RubinBlendRegressor(d, embed, dropout, name="student")
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "latent_mse", "mean_mse"])

    def call(self, x, training=False):
        mu, lv = self.teacher(x["e_in"], training=False)
        mu, lv = ops.stop_gradient(mu), ops.stop_gradient(lv)
        z_e = sample_diag(mu, lv, self.seed_gen)
        return dict(z_b=self.student(x["x_in"], training=training), z_e=z_e, mu_e=mu)

    def loss_terms(self, x, out):
        mse = ops.mean(ops.square(out["z_b"] - out["z_e"]))
        return dict(total_loss=mse, latent_mse=mse, mean_mse=ops.mean(ops.square(out["z_b"] - out["mu_e"])))


class CFMStudent(keras.Model):
    # Rubin blend -> context h(X) (128) -> velocity field. Inference: sample(x_in, k, nfe, method).
    def __init__(self, d=32, context=128, hidden=(128, 128), stats=None, **kw):
        super().__init__(**kw)
        self.context = ConvEmbedder(context, name="context")
        self.field = cfm.VelocityField(d, hidden, name="field")
        self.stats = stats

    def call(self, x_in, training=False):
        h = self.context(x_in, training=training)
        if not self.field.built:
            # The field is otherwise first called inside the loss: its variables must exist before compilation.
            n = ops.shape(h)[0]
            self.field(ops.zeros((n, self.field.d)), ops.zeros((n, 1)), h)
        return h

    def sample(self, x_in, k, nfe, method, seed):
        h = self.context(x_in, training=False)
        return cfm.sample(self.field, h, k, nfe, method, seed, self.stats)


class RunBModel(Tracked):
    # CFM student on samples of the frozen K1 teacher (D2 only).
    def __init__(self, k1_encoder, d=32, context=128, hidden=(128, 128), stats=None, seed=0, **kw):
        super().__init__(**kw)
        self.teacher = k1_encoder
        self.teacher.trainable = False
        self.student = CFMStudent(d, context, hidden, stats, name="student")
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "cfm"])

    def call(self, x, training=False):
        mu, lv = self.teacher(x["e_in"], training=False)
        z1 = cfm.standardise(sample_diag(ops.stop_gradient(mu), ops.stop_gradient(lv), self.seed_gen),
                             self.student.stats)
        return dict(h=self.student(x["x_in"], training=training), z1=z1)

    def loss_terms(self, x, out):
        loss = cfm.cfm_loss(self.student.field, out["z1"], out["h"], self.seed_gen)
        return dict(total_loss=loss, cfm=loss)


class JointEncoder(keras.Model):
    # (R, E) -> embeddings -> fusion MLP -> diagonal Gaussian (C1's q_A).
    def __init__(self, d=32, embed=128, fusion=(256,), **kw):
        super().__init__(**kw)
        self.r_embed = ConvEmbedder(embed, name="r_embed")
        self.e_embed = ConvEmbedder(embed, name="e_embed")
        self.fusion = MLP(fusion, fusion[-1], name="fusion")
        self.head = DiagGaussianHead(d, name="head")

    def call(self, r_in, e_in, training=False):
        h = ops.concatenate([self.r_embed(r_in, training=training), self.e_embed(e_in, training=training)], -1)
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
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "nll_r", "nll_e", "kl", "chi2_r", "chi2_e"])

    def call(self, x, training=False):
        mu, lv = self.encoder(x["r_in"], x["e_in"], training=training)
        z = sample_diag(mu, lv, self.seed_gen) if training else mu
        return dict(mu=mu, log_var=lv, mean_r=self.dec_r(z) * self.scale_r, mean_e=self.dec_e(z) * self.scale_e)

    def loss_terms(self, x, out):
        nr = ops.mean(gaussian_nll(x["r_flux"], out["mean_r"], x["r_w"]))
        ne = ops.mean(gaussian_nll(x["e_flux"], out["mean_e"], x["e_w"]))
        kl = ops.mean(kl_diag_std(out["mu"], out["log_var"]))
        return dict(total_loss=self.w_r * nr + self.w_e * ne + self.beta * kl, nll_r=nr, nll_e=ne, kl=kl,
                    chi2_r=ops.mean(chi2_per_pixel(x["r_flux"], out["mean_r"], x["r_w"])),
                    chi2_e=ops.mean(chi2_per_pixel(x["e_flux"], out["mean_e"], x["e_w"])))


class C2Branches(keras.Model):
    # Shared Rubin backbone; student head s(z | X); teacher = Euclid embedder + fusion + head r(z | X, E_c).
    def __init__(self, d=32, embed=128, fusion=(256,), **kw):
        super().__init__(**kw)
        self.r_embed = ConvEmbedder(embed, name="r_embed")
        self.e_embed = ConvEmbedder(embed, name="e_embed")
        self.s_head = TriLGaussianHead(d, name="s_head")
        self.fusion = MLP(fusion, fusion[-1], name="fusion")
        self.r_head = TriLGaussianHead(d, name="r_head")

    def student(self, x_in, training=False):
        return self.s_head(self.r_embed(x_in, training=training))

    def teacher(self, x_in, e_in, training=False):
        h = ops.concatenate([self.r_embed(x_in, training=training), self.e_embed(e_in, training=training)], -1)
        return self.r_head(ops.silu(self.fusion(h)))

    def call(self, x_in, e_in, training=False):
        return self.student(x_in, training), self.teacher(x_in, e_in, training)


class C2Model(Tracked):
    def __init__(self, c1_encoder, c1_dec_e, scale_e, d=32, embed=128, fusion=(256,), lambda_d1=1.0, seed=0,
                 **kw):
        super().__init__(**kw)
        self.c1_encoder = c1_encoder
        self.c1_encoder.trainable = False
        self.dec_e = c1_dec_e
        self.dec_e.trainable = False
        self.branches = C2Branches(d, embed, fusion, name="branches")
        self.scale_e = _t(scale_e)
        self.lambda_d1 = float(lambda_d1)
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "d2_nll_e", "d2_kl_rs", "d1_kl_s", "d1_kl_r", "d2_chi2_e"])

    def call(self, x, training=False):
        out = {}
        (mu_s, L_s), (mu_r, L_r) = self.branches(x["d2_x_in"], x["d2_e_in"], training=training)
        z = sample_tril(mu_r, L_r, self.seed_gen)
        out.update(mu_s=mu_s, L_s=L_s, mu_r=mu_r, L_r=L_r, mean_e=self.dec_e(z) * self.scale_e)
        if "d1_r_in" in x:
            mu_a, lv_a = self.c1_encoder(x["d1_r_in"], x["d1_e_in"], training=False)
            out.update(mu_a=ops.stop_gradient(mu_a), L_a=ops.stop_gradient(diag_to_tril(lv_a)))
            (out["mu_s1"], out["L_s1"]), (out["mu_r1"], out["L_r1"]) = \
                self.branches(x["d1_r_in"], x["d1_e_in"], training=training)
        return out

    def loss_terms(self, x, out):
        nll = ops.mean(gaussian_nll(x["d2_e_flux"], out["mean_e"], x["d2_e_w"]))
        kl_rs = ops.mean(kl_tril_tril(out["mu_r"], out["L_r"], out["mu_s"], out["L_s"]))
        t = dict(d2_nll_e=nll, d2_kl_rs=kl_rs,
                 d2_chi2_e=ops.mean(chi2_per_pixel(x["d2_e_flux"], out["mean_e"], x["d2_e_w"])))
        total = nll + kl_rs
        if "mu_a" in out:
            ks = ops.mean(kl_tril_tril(out["mu_a"], out["L_a"], out["mu_s1"], out["L_s1"]))
            kr = ops.mean(kl_tril_tril(out["mu_a"], out["L_a"], out["mu_r1"], out["L_r1"]))
            t.update(d1_kl_s=ks, d1_kl_r=kr)
            total = total + self.lambda_d1 * (0.5 * ks + 0.5 * kr)
        t["total_loss"] = total
        return t


class C3Model(Tracked):
    # CFM student on samples of the frozen C2 teacher r(z | X, E_c) (D2) and of the frozen C1 q_A(z | R, E) (D1 replay).
    def __init__(self, c2_branches, c1_encoder, d=32, context=128, hidden=(128, 128), stats=None,
                 lambda_replay=1.0, seed=0, **kw):
        super().__init__(**kw)
        self.teacher = c2_branches
        self.teacher.trainable = False
        self.c1_encoder = c1_encoder
        self.c1_encoder.trainable = False
        self.student = CFMStudent(d, context, hidden, stats, name="student")
        self.lambda_replay = float(lambda_replay)
        self.seed_gen = keras.random.SeedGenerator(seed)
        self.setup_trackers(["total_loss", "cfm_d2", "cfm_d1"])

    def call(self, x, training=False):
        mu, L = self.teacher.teacher(x["d2_x_in"], x["d2_e_in"], training=False)
        z1 = sample_tril(ops.stop_gradient(mu), ops.stop_gradient(L), self.seed_gen)
        out = dict(h=self.student(x["d2_x_in"], training=training), z1=cfm.standardise(z1, self.student.stats))
        if "d1_r_in" in x:
            mu_a, lv_a = self.c1_encoder(x["d1_r_in"], x["d1_e_in"], training=False)
            za = sample_diag(ops.stop_gradient(mu_a), ops.stop_gradient(lv_a), self.seed_gen)
            out.update(h1=self.student(x["d1_r_in"], training=training), z1_d1=cfm.standardise(za, self.student.stats))
        return out

    def loss_terms(self, x, out):
        l2 = cfm.cfm_loss(self.student.field, out["z1"], out["h"], self.seed_gen)
        t = dict(cfm_d2=l2)
        total = l2
        if "h1" in out:
            l1 = cfm.cfm_loss(self.student.field, out["z1_d1"], out["h1"], self.seed_gen)
            t["cfm_d1"] = l1
            total = total + self.lambda_replay * l1
        t["total_loss"] = total
        return t
