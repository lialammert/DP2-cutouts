# debvader_bridge.py
# Anna Preto
# AstroDeep Project
# The frozen DebVAder VAE of a debvader-keras run (paper_baseline): rebuilt with the classes of debvader-keras, not
# re-implemented, loaded from <run>/checkpoints/<stage>_best.weights.h5, and normalised with the stats.json of the
# data set it was trained on.
#
# What the rest of this package uses from it:
#   encoder(x_norm) -> params (B, D + D(D+1)/2): posterior mean mu = params[:, :D], then the Cholesky entries;
#   decoder(z)      -> (B, 59, 59, 12): mean image (first 6 channels) and per-pixel scale (last 6), normalised units.
# The latent used everywhere here is mu, deterministic. DebvaderVAE.call samples z on every forward pass
# (KerasMultivariateNormalTriL has no inference branch), so debvader's own Predictor is stochastic; the bridge never
# goes through call().

import hashlib
import json
from pathlib import Path

import numpy as np

import keras
from keras import ops


def weights_fingerprint(layer):
    # Hash of every weight (trainable or not, BatchNorm moving statistics included): proves that a part moved or not.
    h = hashlib.sha256()
    for v in layer.weights:
        h.update(np.ascontiguousarray(ops.convert_to_numpy(v)).tobytes())
    return h.hexdigest()[:16]


def resolve_stats(run_dir, config, stats_path=None):
    # Order: explicit path, <run>/stats.json, then debvader-keras' own resolution for the run's dataset.
    if stats_path is not None:
        p = Path(stats_path)
    elif (Path(run_dir) / "stats.json").exists():
        p = Path(run_dir) / "stats.json"
    else:
        from debvader.utils.paths import resolve_stats_path
        p = Path(resolve_stats_path(getattr(config.data, "dataset_path", None)))
    if not p.exists():
        raise FileNotFoundError(
            f"normalisation statistics not found ({p}). Pass --debvader-stats with the stats.json of the dataset "
            "paper_baseline was trained on (DATA_DIR/stats.json of the campaign).")
    stats = json.loads(p.read_text())
    if "nonlinear" not in stats:
        raise KeyError(f"{p} has no 'nonlinear' entry: it does not describe a nonlinear-normalised dataset")
    return p, np.asarray(stats["nonlinear"]["scale_per_band"], dtype=np.float32)


class FrozenRubinVAE:
    def __init__(self, run_dir, stage="stage1", stats_path=None, weights_path=None):
        from debvader.core.vae import DebvaderVAE
        from debvader.data.normalization import DEFAULT_NONLINEAR_BETA, NonLinearNorm
        from debvader.utils.config import Config

        self.run_dir = Path(run_dir)
        cfg_path = self.run_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"no config.json in {self.run_dir}: is it a debvader-keras run folder?")
        self.weights_path = Path(weights_path) if weights_path else (
            self.run_dir / "checkpoints" / f"{stage}_best.weights.h5")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.weights_path}")
        self.stage = stage
        self.config = Config.from_json(cfg_path)
        norm = list(self.config.data.normalization)
        if norm != ["nonlinear"]:
            raise ValueError(f"the run is normalised with {norm}; this package handles ['nonlinear'] only")
        if getattr(self.config.model, "use_cvae", False):
            raise ValueError("conditional-VAE runs are not supported: their decoder reads more than the latent")
        self.input_shape = tuple(self.config.data.input_shape)
        self.latent_dim = int(self.config.model.latent_dim)
        self.n_bands = int(self.input_shape[-1])

        self.vae = DebvaderVAE(self.config.model, self.config.training, self.input_shape, normalization="nonlinear")
        self._build()
        fresh = weights_fingerprint(self.vae)
        self.vae.load_weights(str(self.weights_path))
        self.loaded_fingerprint = weights_fingerprint(self.vae)
        if self.loaded_fingerprint == fresh:
            raise RuntimeError("loading the checkpoint changed no weight: the file does not match the model")
        self.vae.trainable = False
        self.encoder, self.decoder = self.vae.encoder, self.vae.decoder

        self.stats_path, self.scale_per_band = resolve_stats(self.run_dir, self.config, stats_path)
        if len(self.scale_per_band) != self.n_bands:
            raise ValueError(f"stats.json has {len(self.scale_per_band)} bands, the model {self.n_bands}")
        self.beta = float(DEFAULT_NONLINEAR_BETA)
        self.norm = NonLinearNorm(beta=self.beta, scale_per_band=self.scale_per_band)

    def _build(self):
        x = ops.zeros((1, *self.input_shape), dtype="float32")
        if getattr(self.config.data, "condition_heads_on_psf", False):
            x = (x, ops.zeros((1, 3), dtype="float32"))
        self.vae(x, training=False)

    # ---------------------------------------------------------------- normalisation (debvader's own code)
    def normalise(self, flux):
        # numpy in, numpy out; flux in the units the VAE was trained in.
        from debvader.data.normalization import normalize_nonlinear_numpy
        return normalize_nonlinear_numpy(flux, self.scale_per_band, self.beta)

    def denormalise(self, x_norm):
        # Backend tensors in and out (keras.ops), debvader's NonLinearNorm.inverse.
        return self.norm.inverse(x_norm)

    def denormalise_np(self, x_norm):
        return np.asarray(ops.convert_to_numpy(self.norm.inverse(ops.convert_to_tensor(
            np.asarray(x_norm, np.float32)))))

    def max_recoverable_flux(self):
        # NonLinearNorm.inverse clips at 1 - 1e-3: above this flux per pixel, the round trip saturates.
        return np.sinh(np.arctanh(1 - 1e-3)) / self.beta * self.scale_per_band

    # ---------------------------------------------------------------- deterministic paths
    def encode_params(self, x_norm, training=False):
        return self.encoder(x_norm, training=training)

    def mu(self, params):
        return params[:, : self.latent_dim]

    def decode(self, z, training=False):
        # Returns (mean, scale) in normalised units; scale = 1e-4 + raw, as in KerasIndependentNormal.
        out = self.decoder(z, training=training)
        c = self.n_bands
        return out[..., :c], 1e-4 + out[..., c:]

    def reconstruct(self, x_norm, batch_size=256):
        # numpy: D_R(mu(E_R(x))) in normalised units, plus mu.
        mus, recs = [], []
        for i in range(0, len(x_norm), batch_size):
            chunk = ops.convert_to_tensor(np.asarray(x_norm[i:i + batch_size], np.float32))
            m = self.mu(self.encode_params(chunk))
            mean, _ = self.decode(m)
            mus.append(ops.convert_to_numpy(m))
            recs.append(ops.convert_to_numpy(mean))
        return np.concatenate(mus), np.concatenate(recs)

    def describe(self):
        return dict(run_dir=str(self.run_dir), stage=self.stage, weights=str(self.weights_path),
                    weights_fingerprint=self.loaded_fingerprint, encoder_fingerprint=weights_fingerprint(self.encoder),
                    decoder_fingerprint=weights_fingerprint(self.decoder), stats=str(self.stats_path),
                    scale_per_band=[float(v) for v in self.scale_per_band], latent_dim=self.latent_dim,
                    input_shape=list(self.input_shape), keras_backend=keras.backend.backend(),
                    keras_version=keras.__version__)


def cholesky_from_params(params, latent_dim):
    # Same construction as KerasMultivariateNormalTriL: lower triangle from the parameters, softplus diagonal + 1e-5.
    d = latent_dim
    rows, cols = np.tril_indices(d)
    flat = rows * d + cols
    perm = np.zeros((len(flat), d * d), np.float32)
    perm[np.arange(len(flat)), flat] = 1.0
    L = ops.reshape(ops.matmul(params[:, d:], ops.convert_to_tensor(perm)), (-1, d, d))
    eye = ops.eye(d)
    diag = ops.softplus(ops.sum(L * eye, axis=2)) + 1e-5
    return L * (1.0 - eye) + ops.expand_dims(diag, 1) * eye
