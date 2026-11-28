# infer.py
# Anna Preto
# AstroDeep Project
# Rubin-only samplers of the three runs: the inputs are the normalised Rubin blend x_in (B, 59, 59, 6) and, when the
# networks are conditioned on it (deblend.psf_condition), the Rubin PSF scalars of the stamp (cond, B x 18: Rubin
# metadata, available at inference). No Euclid array is accepted (tests check it). Each returns latent samples
# (B, K, d) and the decoded centrals (B, K, 59, 59, 6) in nJy (expected signal, convolved by the Rubin PSF, no noise).
#   Run A  K dropout masks on the student's embedding (computed once per blend), mapped back from the standardised
#          K1 latent, decoded by the frozen K2 Rubin decoder;
#   Run B  CFM samples (context computed once per blend) decoded by the frozen K2 Rubin decoder;
#   Run C  CFM samples decoded by the frozen C1 Rubin decoder.

import time

import numpy as np
from keras import ops


# Latents decoded at most this many at a time: 64 galaxies x 128 samples = 8 192 images in one call made a
# convolution tensor of more than 2^31 elements, which cuDNN refuses (pilot of 2026-11-28: "Check failed:
# cudnnSetTensorNdDescriptor ... count: 8192"). 1 024 stays far below and keeps the GPU busy.
DECODE_CHUNK = 1024


def decode_flat(decoder, z, scale, cond=None, chunk=DECODE_CHUNK):
    # z (N, d), cond (N, c) or None -> images (N, n, n, C) x scale, in chunks.
    z = np.asarray(z, np.float32)
    out = []
    for s in range(0, len(z), int(chunk)):
        c = None if cond is None else ops.convert_to_tensor(np.asarray(cond[s:s + chunk], np.float32))
        out.append(np.asarray(ops.convert_to_numpy(decoder(ops.convert_to_tensor(z[s:s + chunk]), cond=c))))
    img = np.concatenate(out) if out else np.zeros((0,), np.float32)
    return img * np.asarray(scale, np.float32)


def _decode(decoder, scale, z, cond=None):
    B, K, d = z.shape
    c = None if cond is None else np.repeat(np.asarray(cond, np.float32), K, axis=0)
    img = decode_flat(decoder, z.reshape(B * K, d), scale, c)
    return img.reshape(B, K, *img.shape[1:])


class RunASampler:
    def __init__(self, student, decoder, scale_r, seed=0):
        import keras
        self.student, self.decoder, self.scale = student, decoder, scale_r
        self.stats = getattr(student, "stats", None)
        self.seed_gen = keras.random.SeedGenerator(seed)

    def latents(self, x_in, k, cond=None, **_):
        z = ops.convert_to_numpy(self.student.sample(ops.convert_to_tensor(np.asarray(x_in)),
                                                     None if cond is None else ops.convert_to_tensor(cond),
                                                     int(k), self.seed_gen))
        if self.stats is not None:
            z = z * np.asarray(self.stats["std"], np.float32) + np.asarray(self.stats["mean"], np.float32)
        return z

    def __call__(self, x_in, k, cond=None, **kw):
        z = self.latents(x_in, k, cond)
        return z, _decode(self.decoder, self.scale, z, cond)


class CFMSampler:
    def __init__(self, student, decoder, scale_r, nfe=40, method="heun", seed=0):
        import keras
        self.student, self.decoder, self.scale = student, decoder, scale_r
        self.nfe, self.method = int(nfe), method
        self.seed_gen = keras.random.SeedGenerator(seed)

    def latents(self, x_in, k, cond=None, nfe=None, method=None):
        z = self.student.sample(np.asarray(x_in), None if cond is None else ops.convert_to_tensor(cond), int(k),
                                int(nfe or self.nfe), method or self.method, self.seed_gen)
        return ops.convert_to_numpy(z)

    def __call__(self, x_in, k, cond=None, nfe=None, method=None):
        z = self.latents(x_in, k, cond, nfe, method)
        return z, _decode(self.decoder, self.scale, z, cond)


def timing(sampler, x_in, k, repeats=3, cond=None, **kw):
    # Wall time of the latent sampling and of the decoding, per galaxy (first call excluded: compilation).
    sampler.latents(x_in[:2], 2, None if cond is None else cond[:2], **kw)
    t_lat, t_dec = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        z = sampler.latents(x_in, k, cond, **kw)
        t1 = time.perf_counter()
        _decode(sampler.decoder, sampler.scale, z, cond)
        t2 = time.perf_counter()
        t_lat.append(t1 - t0)
        t_dec.append(t2 - t1)
    n = len(x_in)
    return dict(n_galaxies=n, k=int(k), latent_s_per_galaxy=float(np.median(t_lat)) / n,
                decode_s_per_galaxy=float(np.median(t_dec)) / n,
                galaxies_per_s=n / float(np.median(np.add(t_lat, t_dec))))
