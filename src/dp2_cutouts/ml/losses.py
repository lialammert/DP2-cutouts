# losses.py
# Anna Preto
# AstroDeep Project
# Loss terms of Stage 1 and Stage 2, written with keras.ops only. Shapes: B batch, D latent, images (B, H, W, C).
#
# Latent alignment (student s, teacher t; the teacher side is never differentiated):
#   mu_mse        mean over the batch of |mu_s - mu_t|^2 / D
#   mu_whitened   same after whitening by the covariance of the Rubin latents of D1 train: a direction in which the
#                 Rubin latents hardly vary weighs as much as the main ones
#   gaussian_kl   KL(q_s || q_t) between the full Gaussian posteriors (Cholesky parameterisation of DebVAder)
# Reconstruction through the frozen Rubin decoder (Stage 1 cross term):
#   normalised_mse    weighted squared error in normalised units, masked pixels excluded
#   gaussian_nll      negative log-likelihood under the decoder's own per-pixel scale
#   chi2_physical     chi^2 in VAE flux units with the observed inverse variance
# Contrastive: symmetric InfoNCE on L2-normalised projections; negatives can be masked when the teacher latents of
# the pair are near-identical (morphological twins).

import numpy as np
from keras import ops


def latent_mu_mse(mu_s, mu_t):
    return ops.mean(ops.sum(ops.square(mu_s - mu_t), axis=-1)) / ops.cast(ops.shape(mu_s)[-1], mu_s.dtype)


def latent_mu_whitened(mu_s, mu_t, whitening):
    # whitening W (D, D) with W Sigma W^T = I: |W (mu_s - mu_t)|^2 / D.
    d = ops.matmul(mu_s - mu_t, ops.transpose(whitening))
    return ops.mean(ops.sum(ops.square(d), axis=-1)) / ops.cast(ops.shape(mu_s)[-1], mu_s.dtype)


def gaussian_kl(mu_s, L_s, mu_t, L_t):
    # KL(N(mu_s, L_s L_s^T) || N(mu_t, L_t L_t^T)), batch mean.
    D = ops.cast(ops.shape(mu_s)[-1], mu_s.dtype)
    M = ops.solve(L_t, L_s)                                     # L_t^{-1} L_s
    v = ops.solve(L_t, ops.expand_dims(mu_t - mu_s, -1))[..., 0]
    eye = ops.eye(ops.shape(L_s)[-1], dtype=L_s.dtype)
    logdet_t = 2.0 * ops.sum(ops.log(ops.sum(L_t * eye, axis=-1)), axis=-1)
    logdet_s = 2.0 * ops.sum(ops.log(ops.sum(L_s * eye, axis=-1)), axis=-1)
    kl = 0.5 * (ops.sum(ops.square(M), axis=(-2, -1)) + ops.sum(ops.square(v), axis=-1) - D + logdet_t - logdet_s)
    return ops.mean(kl) / D


def latent_loss(kind, mu_s, mu_t, whitening=None, L_s=None, L_t=None):
    if kind == "mu_mse":
        return latent_mu_mse(mu_s, mu_t)
    if kind == "mu_whitened":
        return latent_mu_whitened(mu_s, mu_t, whitening)
    if kind == "gaussian_kl":
        return gaussian_kl(mu_s, L_s, mu_t, L_t)
    raise ValueError(f"unknown latent loss {kind!r}")


def _weighted_mean(per_pixel, weight, band_weights):
    w = weight * ops.reshape(band_weights, (1, 1, 1, -1))
    return ops.sum(per_pixel * w) / (ops.sum(w) + 1e-8)


def recon_normalised_mse(pred, target, weight, band_weights):
    return _weighted_mean(ops.square(pred - target), weight, band_weights)


def recon_gaussian_nll(pred, scale, target, weight, band_weights):
    nll = 0.5 * (ops.square((target - pred) / scale) + 2.0 * ops.log(scale) + np.log(2 * np.pi))
    return _weighted_mean(nll, weight, band_weights)


def recon_chi2_physical(pred_phys, obs_phys, ivar_phys, weight, band_weights):
    return _weighted_mean(ops.square(pred_phys - obs_phys) * ivar_phys, weight, band_weights)


def info_nce(p_s, p_t, inv_tau, negative_mask=None):
    # p_s, p_t (B, P) unit vectors; positives on the diagonal. negative_mask (B, B): 1 = keep the pair as a
    # negative (the diagonal is always kept as the positive).
    logits = ops.matmul(p_s, ops.transpose(p_t)) * inv_tau
    B = ops.shape(logits)[0]
    eye = ops.eye(B, dtype=logits.dtype)
    if negative_mask is not None:
        keep = ops.maximum(negative_mask, eye)
        logits = logits + (1.0 - keep) * (-1e9)
    l_st = -ops.mean(ops.sum(eye * ops.log_softmax(logits, axis=1), axis=1))
    l_ts = -ops.mean(ops.sum(eye * ops.log_softmax(logits, axis=0), axis=0))
    return 0.5 * (l_st + l_ts)


def twin_negative_mask(mu_t, threshold_sq):
    # 1 where the teacher latents of two examples are further apart than the threshold (squared distance).
    sq = ops.sum(ops.square(ops.expand_dims(mu_t, 1) - ops.expand_dims(mu_t, 0)), axis=-1)
    return ops.cast(sq > threshold_sq, mu_t.dtype)


def retrieval_accuracy(p_s, p_t, k=1):
    # Fraction of examples whose own teacher projection is among the k nearest (numpy, for evaluation).
    p_s, p_t = np.asarray(p_s), np.asarray(p_t)
    sim = p_s @ p_t.T
    rank = (sim > np.diag(sim)[:, None]).sum(axis=1)
    return float((rank < k).mean())


def whitening_from_latents(mu, ridge=1e-2):
    # (W, mean, cov) from a sample of Rubin latents: W = L^{-1} with cov = L L^T. The ridge, a fraction of the mean
    # variance, caps the weight of near-inactive latent directions (a VAE trained with a KL term has some): without
    # it their tiny variance would make sampling noise dominate the whitened loss.
    mu = np.asarray(mu, np.float64)
    cov = np.cov(mu, rowvar=False)
    cov = cov + ridge * max(float(np.trace(cov)) / len(cov), 1e-12) * np.eye(len(cov))
    L = np.linalg.cholesky(cov)
    return np.linalg.inv(L).astype(np.float32), mu.mean(axis=0).astype(np.float32), cov.astype(np.float32)


def pairwise_sq_dist(a, b):
    # (N, M) squared Euclidean distances without the (N, M, D) intermediate (numpy).
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d, 0.0)
