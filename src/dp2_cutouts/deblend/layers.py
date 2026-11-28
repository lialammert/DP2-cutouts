# layers.py
# Anna Preto
# AstroDeep Project
# Building blocks, keras.ops only (backend agnostic). The convolutional blocks follow DebVAder's
# (debvader-keras core/encoder.py: a stride-1 convolution and a stride-2 convolution per level, PReLU), written
# here rather than imported because importing debvader loads its .env file and its two checkouts share a namespace;
# the CFM velocity field reproduces debvader-keras-mm mmcfm.network.VelocityField (untracked there).
# Sizes are kept small on purpose: a 119 x 119 Euclid stamp is reduced to 8 x 8 x 64 before the dense head.

import math

import keras
import numpy as np
from keras import layers, ops


# Input normalisation on the accelerator (deblend.data.normalise_on: gpu). The loaders then hand the encoders the
# flux (masked pixels already filled with sky noise) and every ConvEmbedder applies arcsinh(2.5 f / s_b) itself,
# inside the compiled graph: the same function as the CPU path (data.asinh_norm), moved off the CPU threads that
# prepare the batches. arcsinh alone since 2026-11-28 (DebVAder's tanh on top bounded the input at 1 and flattened
# the cores of bright galaxies; arcsinh is logarithmic there, linear near zero, and never clipped). Set by data.Prepare from the configuration, so that the loaders and the models of
# a process always agree; empty = the inputs arrive normalised. The side is recognised by its channels: the Rubin
# bands, or 2 for Euclid (flux, validity; validity left as is).
INPUT_NORM = {}


def normalise_input(x):
    if not INPUT_NORM:
        return x
    c = x.shape[-1]
    r = INPUT_NORM["rubin"]
    if c == len(r):
        return ops.arcsinh(2.5 * x / ops.convert_to_tensor(r, dtype=x.dtype))
    if c == 2:
        f = ops.arcsinh(2.5 * x[..., :1] / ops.convert_to_tensor(INPUT_NORM["euclid"], dtype=x.dtype))
        return ops.concatenate([f, x[..., 1:]], axis=-1)
    raise ValueError(f"input with {c} channels: neither Rubin ({len(r)} bands) nor Euclid (2)")


def tril_size(d):
    return d * (d + 1) // 2


def tril_from_params(p, d, min_diag=1e-4):
    # (B, d(d+1)/2) -> lower-triangular (B, d, d) with a positive diagonal (softplus + min_diag). Row-major order of
    # the lower triangle, as numpy.tril_indices.
    rows, cols = np.tril_indices(d)
    idx = rows * d + cols
    B = ops.shape(p)[0]
    onehot = np.zeros((len(idx), d * d), np.float32)
    onehot[np.arange(len(idx)), idx] = 1.0
    flat = ops.matmul(p, ops.convert_to_tensor(onehot, dtype=p.dtype))
    L = ops.reshape(flat, (B, d, d))
    diag_mask = ops.convert_to_tensor(np.eye(d, dtype=np.float32), dtype=p.dtype)
    diag = ops.softplus(L) + min_diag
    return L * (1.0 - diag_mask) + diag * diag_mask


class ConvBlock(layers.Layer):
    # DebVAder level: conv 3x3 stride 1 + PReLU, conv 3x3 stride 2 + PReLU.
    def __init__(self, filters, **kw):
        super().__init__(**kw)
        self.c1 = layers.Conv2D(filters, 3, padding="same")
        self.a1 = layers.PReLU(shared_axes=[1, 2])
        self.c2 = layers.Conv2D(filters, 3, strides=2, padding="same")
        self.a2 = layers.PReLU(shared_axes=[1, 2])

    def call(self, x):
        return self.a2(self.c2(self.a1(self.c1(x))))


class UpBlock(layers.Layer):
    # Transposed level: conv-transpose 3x3 stride 2 + PReLU, conv 3x3 + PReLU.
    def __init__(self, filters, **kw):
        super().__init__(**kw)
        self.t = layers.Conv2DTranspose(filters, 3, strides=2, padding="same")
        self.a1 = layers.PReLU(shared_axes=[1, 2])
        self.c = layers.Conv2D(filters, 3, padding="same")
        self.a2 = layers.PReLU(shared_axes=[1, 2])

    def call(self, x):
        return self.a2(self.c(self.a1(self.t(x))))


class ConvEmbedder(keras.Model):
    # Image (B, n, n, C) -> embedding (B, out_dim). Four levels (n -> ceil(n / 16)), a 1x1 convolution to
    # `squeeze` channels, then one dense layer. dropout: rate applied to the embedding; mc_dropout=True keeps it
    # active at inference (BatchNormalization is not used anywhere, so no statistics can move).
    # cond (B, c): conditioning scalars (the standardised Rubin PSF moments, deblend.psf_condition) appended to the
    # embedding, after its dropout: output (B, out_dim + c). No weight depends on it, so the embedders of two
    # stages stay interchangeable (C2 and C3 start from copies of C1's / C2's).
    def __init__(self, out_dim=128, filters=(32, 64, 128, 256), squeeze=64, dropout=0.0, mc_dropout=False, **kw):
        super().__init__(**kw)
        self.blocks = [ConvBlock(f) for f in filters]
        self.squeeze = layers.Conv2D(squeeze, 1)
        self.sq_act = layers.PReLU(shared_axes=[1, 2])
        self.flat = layers.Flatten()
        self.dense = layers.Dense(out_dim)
        self.act = layers.PReLU()
        self.rate = float(dropout)
        self.mc_dropout = bool(mc_dropout)
        self.drop = layers.Dropout(self.rate) if self.rate > 0 else None

    def call(self, x, cond=None, training=False):
        x = normalise_input(x)
        for b in self.blocks:
            x = b(x)
        h = self.act(self.dense(self.flat(self.sq_act(self.squeeze(x)))))
        if self.drop is not None:
            h = self.drop(h, training=bool(training) or self.mc_dropout)
        return with_cond(h, cond)


def with_cond(h, cond):
    # [h, cond] on the last axis; h unchanged without conditioning.
    return h if cond is None else ops.concatenate([h, ops.cast(cond, h.dtype)], axis=-1)


class ConvDecoder(keras.Model):
    # Latent (B, d) -> mean image (B, n, n, C) in units of the per-band output scale (the caller multiplies by it).
    # cond (B, c): the PSF of the instrument this decoder reconstructs (Rubin decoders only), appended to z.
    # Dense to an s x s x filters[0] canvas, four transposed levels (s * 16 >= n), a linear 3x3 convolution to C
    # channels (no positivity constraint in this first campaign), then a centred crop to n x n.
    def __init__(self, n, channels, filters=(256, 128, 64, 32), hidden=256, **kw):
        super().__init__(**kw)
        self.n, self.channels = int(n), int(channels)
        self.s = int(math.ceil(self.n / 2 ** len(filters)))
        self.d1 = layers.Dense(hidden)
        self.a1 = layers.PReLU()
        self.d2 = layers.Dense(self.s * self.s * filters[0])
        self.a2 = layers.PReLU()
        self.reshape = layers.Reshape((self.s, self.s, filters[0]))
        self.ups = [UpBlock(f) for f in filters]
        self.out = layers.Conv2D(self.channels, 3, padding="same")
        canvas = self.s * 2 ** len(filters)
        lo = (canvas - self.n) // 2
        self.crop = layers.Cropping2D(((lo, canvas - self.n - lo), (lo, canvas - self.n - lo)))

    def call(self, z, cond=None):
        x = self.reshape(self.a2(self.d2(self.a1(self.d1(with_cond(z, cond))))))
        for u in self.ups:
            x = u(x)
        return self.crop(self.out(x))


class MLP(keras.Model):
    def __init__(self, widths, out_dim, activation="silu", **kw):
        super().__init__(**kw)
        self.hidden = [layers.Dense(w, activation=activation) for w in widths]
        self.out = layers.Dense(out_dim)

    def call(self, x):
        for h in self.hidden:
            x = h(x)
        return self.out(x)


class DiagGaussianHead(layers.Layer):
    # embedding -> (mu, log_var), diagonal Gaussian. log_var clipped to [-12, 8] for numerical safety.
    def __init__(self, d, **kw):
        super().__init__(**kw)
        self.d = int(d)
        self.dense = layers.Dense(2 * self.d)

    def call(self, h):
        p = self.dense(h)
        return p[:, :self.d], ops.clip(p[:, self.d:], -12.0, 8.0)


class TriLGaussianHead(layers.Layer):
    # embedding -> (mu, L), full covariance Sigma = L L^T, L lower triangular with softplus diagonal.
    def __init__(self, d, **kw):
        super().__init__(**kw)
        self.d = int(d)
        # Starts near L = identity, mu = 0: a tiny kernel (1e-3 of the usual scale; exactly zero would make the
        # gradients towards the backbone vanish at the first step, and the lambda_D1 calibration with them) and a
        # bias putting every diagonal at softplus(0.5413) = 1. With a default initialisation, real-data embeddings
        # gave off-diagonal entries of +-7 over diagonals down to 1e-4; the triangular solves of the KL divergences
        # then amplified by ~(7 / 1e-4)^k and C2 started at KL ~1e35 / inf (pilot of 2026-11-27).
        rows, cols = np.tril_indices(self.d)
        bias = np.zeros(self.d + tril_size(self.d), np.float32)
        bias[self.d + np.flatnonzero(rows == cols)] = np.log(np.e - 1.0)
        self.dense = layers.Dense(self.d + tril_size(self.d),
                                  kernel_initializer=keras.initializers.VarianceScaling(scale=1e-6),
                                  bias_initializer=keras.initializers.Constant(bias))

    def call(self, h):
        p = self.dense(h)
        return p[:, :self.d], tril_from_params(p[:, self.d:], self.d)


def sample_diag(mu, log_var, seed, k=None):
    # z = mu + exp(log_var / 2) eps; k samples -> (B, k, d) else (B, d).
    if k is None:
        eps = keras.random.normal(ops.shape(mu), seed=seed)
        return mu + ops.exp(0.5 * log_var) * eps
    B, d = ops.shape(mu)[0], ops.shape(mu)[1]
    eps = keras.random.normal((B, k, d), seed=seed)
    return mu[:, None, :] + ops.exp(0.5 * log_var)[:, None, :] * eps


def sample_tril(mu, L, seed, k=None):
    if k is None:
        eps = keras.random.normal(ops.shape(mu), seed=seed)
        return mu + ops.squeeze(ops.matmul(L, eps[..., None]), -1)
    B, d = ops.shape(mu)[0], ops.shape(mu)[1]
    eps = keras.random.normal((B, k, d), seed=seed)
    return mu[:, None, :] + ops.einsum("bij,bkj->bki", L, eps)
