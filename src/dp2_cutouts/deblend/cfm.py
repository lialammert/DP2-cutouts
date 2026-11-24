# cfm.py
# Anna Preto
# AstroDeep Project
# Conditional flow matching on the latent space (Runs B and C).
# Path: z_t = (1 - t) z0 + t z1, z0 ~ N(0, I), z1 = a teacher sample, t ~ U(0, 1); target velocity u = z1 - z0;
# loss mean_{batch, dims} ||v(z_t, t, h) - u||^2. No ODE is solved during training.
# Sampling: z0 ~ N(0, I) integrated from t = 0 to 1 with fixed-step Euler (1 evaluation per step) or Heun (2 per
# step, the last step's second evaluation included: NFE = 2 n_steps). The context h(X) is computed once per blend
# and shared by every step and every posterior sample (vectorised over (batch, samples)).
# Latents may be standardised with train-split statistics (mean, std per dimension): the flow then runs in the
# standardised space and samples are mapped back before decoding.

import keras
import numpy as np
from keras import layers, ops


class VelocityField(keras.Model):
    # [z_t (d), t (1), h (c)] -> 128 SiLU -> 128 SiLU -> d (linear). With d = 32, c = 128: 41 376 parameters.
    def __init__(self, latent_dim=32, hidden=(128, 128), **kw):
        super().__init__(**kw)
        self.d = int(latent_dim)
        self.hidden = [layers.Dense(w, activation="silu") for w in hidden]
        self.out = layers.Dense(self.d)

    def call(self, z, t, h):
        x = ops.concatenate([z, t, h], axis=-1)
        for lay in self.hidden:
            x = lay(x)
        return self.out(x)


def cfm_pair(z1, seed):
    # (z_t, t, u) for a batch of targets z1 (B, d).
    B = ops.shape(z1)[0]
    z0 = keras.random.normal(ops.shape(z1), seed=seed)
    t = keras.random.uniform((B, 1), seed=seed)
    zt = (1.0 - t) * z0 + t * z1
    return zt, t, z1 - z0


def cfm_loss(field, z1, h, seed):
    zt, t, u = cfm_pair(z1, seed)
    v = field(zt, t, h)
    return ops.mean(ops.square(v - u))


def steps_for_nfe(nfe, method):
    if method == "euler":
        return int(nfe)
    if method == "heun":
        if nfe % 2:
            raise ValueError("Heun uses 2 evaluations per step: NFE must be even")
        return int(nfe) // 2
    raise ValueError(f"unknown ODE method {method!r}")


def integrate(field, z0, h, n_steps, method="euler"):
    # z0 (N, d), h (N, c) -> z(1). Fixed steps dt = 1 / n_steps.
    dt = 1.0 / int(n_steps)
    z = z0
    N = ops.shape(z0)[0]
    for k in range(int(n_steps)):
        t = ops.full((N, 1), k * dt)
        v = field(z, t, h)
        if method == "euler":
            z = z + dt * v
        elif method == "heun":
            z_pred = z + dt * v
            v2 = field(z_pred, ops.full((N, 1), (k + 1) * dt), h)
            z = z + 0.5 * dt * (v + v2)
        else:
            raise ValueError(f"unknown ODE method {method!r}")
    return z


def sample(field, h, k, nfe, method, seed, stats=None):
    # h (B, c) -> k latent samples per example (B, k, d); stats: dict(mean, std) of the standardisation or None.
    B, c = ops.shape(h)[0], ops.shape(h)[1]
    d = field.d
    z0 = keras.random.normal((B * k, d), seed=seed)
    hh = ops.reshape(ops.repeat(h[:, None, :], k, axis=1), (B * k, c))
    z = integrate(field, z0, hh, steps_for_nfe(nfe, method), method)
    z = ops.reshape(z, (B, k, d))
    if stats is not None:
        z = z * ops.convert_to_tensor(np.asarray(stats["std"], np.float32)) + \
            ops.convert_to_tensor(np.asarray(stats["mean"], np.float32))
    return z


def standardise(z, stats):
    if stats is None:
        return z
    return (z - ops.convert_to_tensor(np.asarray(stats["mean"], np.float32))) / \
        ops.convert_to_tensor(np.asarray(stats["std"], np.float32))
