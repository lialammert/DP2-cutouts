# vae_audit.py
# Anna Preto
# AstroDeep Project
# Audit of a trained Rubin VAE (debvader-keras run) whose decoder output looks dead in some bands: are the
# pre-activations of the final ReLU negative, does the gradient still reach them, does the model explain the band as
# noise through sigma, and does it fail even where the catalogue says the galaxy is bright in that band?
# Pure numpy pieces (tested); scripts/inspect/rubin_vae_audit.py reads the run and the data and draws the figures.
#
# Output of the decoder: 2C channels, the C means then the C raw scales, both through ReLU; sigma = 1e-4 + raw scale
# (debvader KerasIndependentNormal). The loss is the per-pixel Gaussian NLL of the normalised stamp.

import numpy as np

from ..common.ellipse import radius_map

LOG_2PI = float(np.log(2.0 * np.pi))


def gaussian_nll(x, mu, sigma):
    return 0.5 * ((x - mu) / sigma) ** 2 + np.log(sigma) + 0.5 * LOG_2PI


def output_gradients(x, mu, sigma, pre_mu, pre_scale, relu_mean=True, relu_scale=True):
    # dL/dmu and dL/dsigma of the per-pixel NLL, and what passes the output activation: through a ReLU,
    # dL/da = dL/d(out) * 1[a > 0]; a linear mean passes it all; a softplus scale too (times its slope, > 0).
    d_mu = (mu - x) / sigma ** 2
    d_sigma = 1.0 / sigma - (x - mu) ** 2 / sigma ** 3
    g_mu = d_mu * (pre_mu > 0) if relu_mean else d_mu
    g_sig = d_sigma * (pre_scale > 0) if relu_scale else d_sigma
    return d_mu, g_mu, d_sigma, g_sig


def regions(shape, centres, ellipses, k_galaxy=2.0, k_sky=4.0):
    # Per stamp: galaxy region (reduced radius <= k_galaxy in the catalogue ellipse of the target) and sky region
    # (> k_sky). centres (N, 2) as (x, y) in pixels; ellipses (N, 3) as (a, b, theta deg). Returns two (N, H, W)
    # boolean arrays. These depend on the catalogue only, never on the VAE.
    gal = np.zeros((len(centres),) + tuple(shape), bool)
    sky = np.zeros_like(gal)
    for i, ((xc, yc), (a, b, th)) in enumerate(zip(centres, ellipses)):
        if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
            a = b = 2.0
            th = 0.0
        r = radius_map(shape, xc, yc, a, b, th)
        gal[i], sky[i] = r <= k_galaxy, r > k_sky
    return gal, sky


def band_stats(x, mu, sigma, pre_mu, pre_scale, region, relu_mean=True, relu_scale=True):
    # Per band, over the pixels of region (N, H, W): what the decoder does there and what gradient reaches it.
    # x, mu, sigma, pre_*: (N, H, W, C), normalised units.
    C = x.shape[-1]
    d_mu, g_mu, d_sig, g_sig = output_gradients(x, mu, sigma, pre_mu, pre_scale, relu_mean, relu_scale)
    nll = gaussian_nll(x, mu, sigma)
    out = []
    for c in range(C):
        m = region
        s_dmu, s_gmu = np.abs(d_mu[..., c][m]), np.abs(g_mu[..., c][m])
        out.append(dict(
            frac_pre_mu_negative=float((pre_mu[..., c][m] <= 0).mean()),
            pre_mu_quantiles={str(q): float(np.quantile(pre_mu[..., c][m], q)) for q in (0.01, 0.5, 0.99)},
            frac_mu_zero=float((mu[..., c][m] == 0).mean()),
            frac_pre_scale_negative=float((pre_scale[..., c][m] <= 0).mean()),
            median_sigma=float(np.median(sigma[..., c][m])),
            median_abs_residual=float(np.median(np.abs(x[..., c][m] - mu[..., c][m]))),
            mean_nll=float(nll[..., c][m].mean()),
            mean_abs_dL_dmu=float(s_dmu.mean()),
            mean_abs_grad_through_relu_mu=float(s_gmu.mean()),
            frac_gradient_blocked_mu=float(1.0 - s_gmu.sum() / s_dmu.sum()) if s_dmu.sum() > 0 else float("nan"),
            # Gradient on the bias of this output channel (sum of dL/da, per pixel of the region).
            bias_gradient_mu=float(g_mu[..., c][m].mean()),
            bias_gradient_mu_without_relu=float(d_mu[..., c][m].mean()),
            mean_abs_grad_through_relu_scale=float(np.abs(g_sig[..., c][m]).mean())))
    return out


def aperture_flux(img, region):
    # img (N, H, W, C), region (N, H, W): summed flux per stamp and band.
    return np.einsum("nhwc,nhw->nc", img, region.astype(img.dtype))


def sky_sigma(img, sky):
    # Robust per-stamp, per-band noise from the sky region (1.4826 MAD).
    out = np.full((img.shape[0], img.shape[-1]), np.nan, np.float32)
    for i in range(img.shape[0]):
        if sky[i].any():
            v = img[i][sky[i]]
            out[i] = 1.4826 * np.median(np.abs(v - np.median(v, axis=0)), axis=0)
    return out


def catalogue_snr(mag, depth5):
    # S/N of a source of magnitude mag where a point source of magnitude depth5 is at 5 sigma (local depth from the
    # variance plane, computed by the stamp stage). Catalogue photometry only: independent of any reconstruction.
    return 5.0 * 10.0 ** (-0.4 * (np.asarray(mag, float) - np.asarray(depth5, float)))
