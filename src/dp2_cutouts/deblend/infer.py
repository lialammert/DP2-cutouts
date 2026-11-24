# infer.py
# Anna Preto
# AstroDeep Project
# Rubin-only samplers of the three runs: the only input is the normalised Rubin blend x_in (B, 59, 59, 6); no
# Euclid array is accepted (tests check it). Each returns latent samples (B, K, d) and the decoded centrals
# (B, K, 59, 59, 6) in nJy (expected signal, convolved by the Rubin PSF, no noise).
#   Run A  K forward passes of the student with dropout active (vectorised: the batch is repeated K times),
#          decoded by the frozen K2 Rubin decoder;
#   Run B  CFM samples (context computed once per blend) decoded by the frozen K2 Rubin decoder;
#   Run C  CFM samples decoded by the frozen C1 Rubin decoder.

import time

import numpy as np
from keras import ops


def _decode(decoder, scale, z):
    B, K, d = z.shape
    img = decoder(ops.reshape(ops.convert_to_tensor(z), (B * K, d)))
    img = ops.convert_to_numpy(img) * np.asarray(scale, np.float32)
    return img.reshape(B, K, *img.shape[1:])


class RunASampler:
    def __init__(self, student, decoder, scale_r):
        self.student, self.decoder, self.scale = student, decoder, scale_r

    def latents(self, x_in, k, **_):
        x = np.asarray(x_in)
        B = len(x)
        self.student.set_mc(True)
        try:
            z = ops.convert_to_numpy(self.student(np.repeat(x, k, axis=0), training=False))
        finally:
            self.student.set_mc(False)
        return z.reshape(B, k, -1)

    def __call__(self, x_in, k, **kw):
        z = self.latents(x_in, k)
        return z, _decode(self.decoder, self.scale, z)


class CFMSampler:
    def __init__(self, student, decoder, scale_r, nfe=20, method="heun", seed=0):
        import keras
        self.student, self.decoder, self.scale = student, decoder, scale_r
        self.nfe, self.method = int(nfe), method
        self.seed_gen = keras.random.SeedGenerator(seed)

    def latents(self, x_in, k, nfe=None, method=None):
        z = self.student.sample(np.asarray(x_in), int(k), int(nfe or self.nfe), method or self.method, self.seed_gen)
        return ops.convert_to_numpy(z)

    def __call__(self, x_in, k, nfe=None, method=None):
        z = self.latents(x_in, k, nfe, method)
        return z, _decode(self.decoder, self.scale, z)


def timing(sampler, x_in, k, repeats=3, **kw):
    # Wall time of the latent sampling and of the decoding, per galaxy (first call excluded: compilation).
    sampler.latents(x_in[:2], 2, **kw)
    t_lat, t_dec = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        z = sampler.latents(x_in, k, **kw)
        t1 = time.perf_counter()
        _decode(sampler.decoder, sampler.scale, z)
        t2 = time.perf_counter()
        t_lat.append(t1 - t0)
        t_dec.append(t2 - t1)
    n = len(x_in)
    return dict(n_galaxies=n, k=int(k), latent_s_per_galaxy=float(np.median(t_lat)) / n,
                decode_s_per_galaxy=float(np.median(t_dec)) / n,
                galaxies_per_s=n / float(np.median(np.add(t_lat, t_dec))))
