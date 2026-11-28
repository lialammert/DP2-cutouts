# likelihood.py
# Anna Preto
# AstroDeep Project
# Observational likelihoods and divergences, keras.ops only.
#
# Gaussian likelihood with known per-pixel variance: for an observation x with inverse variance w (0 on pixels that
# are not observations of the target: invalid, or replaced by sky / belonging to another source),
#     -log p(x | mu) = 1/2 sum_pixels w (x - mu)^2  + const,
# summed (not averaged) over the pixels of one example, so that the reconstruction terms of two instruments and the
# KL divergences are on the same footing (a proper ELBO at beta = 1). The constant (log-determinant of the noise)
# does not depend on the parameters and is dropped; per-pixel chi2 values are reported separately as diagnostics.
# Coadd noise is spatially correlated; the per-pixel ivar ignores it, which this likelihood inherits.
#
# Divergences between Gaussians: diagonal to N(0, I), full covariance to N(0, I), and full covariance to full
# covariance (Cholesky factors, triangular solves).

import numpy as np
from keras import ops


def gaussian_nll(x, mu, w):
    # x, mu, w (B, H, W, C) -> (B,): 1/2 sum w (x - mu)^2.
    r = x - mu
    return 0.5 * ops.sum(w * r * r, axis=(1, 2, 3))


def chi2_per_pixel(x, mu, w):
    # Mean of w (x - mu)^2 over the pixels with w > 0, per example (diagnostic).
    r = x - mu
    n = ops.maximum(ops.sum(ops.cast(w > 0, "float32"), axis=(1, 2, 3)), 1.0)
    return ops.sum(w * r * r, axis=(1, 2, 3)) / n


def kl_diag_std(mu, log_var):
    # KL(N(mu, diag exp(log_var)) || N(0, I)), (B,).
    return 0.5 * ops.sum(ops.exp(log_var) + mu * mu - 1.0 - log_var, axis=-1)


def logdet_tril(L):
    d = ops.diagonal(L, axis1=-2, axis2=-1)
    return 2.0 * ops.sum(ops.log(d), axis=-1)


def kl_tril_std(mu, L):
    # KL(N(mu, L L^T) || N(0, I)) = 1/2 (tr(L L^T) + mu^T mu - d - log det(L L^T)), (B,).
    d = ops.shape(mu)[-1]
    tr = ops.sum(L * L, axis=(-2, -1))
    return 0.5 * (tr + ops.sum(mu * mu, axis=-1) - float(d) - logdet_tril(L))


def kl_tril_tril(mu_q, L_q, mu_p, L_p):
    # KL(N(mu_q, Lq Lq^T) || N(mu_p, Lp Lp^T)), (B,):
    #   1/2 ( ||Lp^-1 Lq||_F^2 + ||Lp^-1 (mu_p - mu_q)||^2 - d + log det Sp - log det Sq ).
    d = ops.shape(mu_q)[-1]
    A = ops.solve_triangular(L_p, L_q, lower=True)
    b = ops.solve_triangular(L_p, (mu_p - mu_q)[..., None], lower=True)[..., 0]
    return 0.5 * (ops.sum(A * A, axis=(-2, -1)) + ops.sum(b * b, axis=-1) - float(d)
                  + logdet_tril(L_p) - logdet_tril(L_q))


def kl_tril_tril_numpy(mu_q, Sq, mu_p, Sp):
    # Reference implementation on covariance matrices (tests).
    d = len(mu_q)
    Sp_inv = np.linalg.inv(Sp)
    dm = mu_p - mu_q
    return 0.5 * (np.trace(Sp_inv @ Sq) + dm @ Sp_inv @ dm - d + np.log(np.linalg.det(Sp) / np.linalg.det(Sq)))


def diag_to_tril(log_var):
    # Diagonal Gaussian scale as a Cholesky factor (B, d, d).
    s = ops.exp(0.5 * log_var)
    eye = ops.convert_to_tensor(np.eye(int(log_var.shape[-1]), dtype=np.float32))
    return s[..., None] * eye


# Energy score (Gneiting & Raftery 2007), a strictly proper scoring rule for a distribution known by samples:
#     ES(P, y) = E ||Z - y|| - 1/2 E ||Z - Z'||,  Z, Z' ~ P independent.
# Run A's loss (2026-11-28, as Pacchiardi et al. 2024 for implicit generative networks): Z = M dropout draws of the
# student, y = one draw of the teacher posterior. Unbiased estimator with plain (not squared) Euclidean distances,
# the pairs i < j only (no diagonal term):
#     (1/M) sum_i ||z_i - y||  -  1/(M (M - 1)) sum_{i<j} ||z_i - z_j||
# (the 1/2 is already in the pair count: sum_{i<j} is half of sum_{i != j}).
def safe_norm(v):
    # ||v|| on the last axis, with a zero (sub)gradient where v = 0 (two identical draws): sqrt'(0) is infinite.
    n2 = ops.sum(v * v, axis=-1)
    pos = n2 > 0
    return ops.where(pos, ops.sqrt(ops.where(pos, n2, 1.0)), 0.0)


def energy_score(z, y):
    # z (B, M, d) samples, y (B, d) observation -> (score, mean distance to y, mean distance between two samples),
    # each (B,). M >= 2 (static).
    M = int(z.shape[1])
    if M < 2:
        raise ValueError("the energy score needs at least two samples per example")
    err = ops.mean(safe_norm(z - y[:, None, :]), axis=1)
    i, j = np.triu_indices(M, 1)
    dz = ops.take(z, ops.convert_to_tensor(i), axis=1) - ops.take(z, ops.convert_to_tensor(j), axis=1)
    pair_sum = ops.sum(safe_norm(dz), axis=1)
    return err - pair_sum / float(M * (M - 1)), err, 2.0 * pair_sum / float(M * (M - 1))
