# models.py
# Anna Preto
# AstroDeep Project
# Trainable parts: the Euclid encoder E_E (Stage 1), projection heads and temperature for the contrastive term.
# E_E ends in the same parameterisation as the DebVAder encoder (mean then Cholesky entries), so that its output
# lives in the latent space of the frozen Rubin decoder and the Gaussian KL loss can compare full posteriors.
# The convolution blocks are those of DebVAder (debvader.modules.layers.DebvaderConvBlock).

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="dp2_ml")
class EuclidEncoder(keras.Model):
    def __init__(self, latent_dim, filters=(32, 64, 128, 256, 256), cond_embedding=64, dense=512, **kwargs):
        super().__init__(**kwargs)
        from debvader.modules.layers import DebvaderConvBlock

        self.latent_dim = int(latent_dim)
        self.filters = [int(f) for f in filters]
        self.cond_embedding = int(cond_embedding)
        self.dense_units = int(dense)
        self.blocks = [DebvaderConvBlock(f, name=f"e_block{i}") for i, f in enumerate(self.filters)]
        self.flatten = layers.Flatten()
        self.cond_dense = layers.Dense(self.cond_embedding, name="cond_dense")
        self.cond_act = layers.PReLU(name="cond_prelu")
        # FiLM: the PSF embedding scales and shifts the image features (a broad PSF should be read differently).
        self.film = layers.Dense(2 * self.filters[-1], name="film", kernel_initializer="zeros")
        self.hidden = layers.Dense(self.dense_units, name="hidden")
        self.hidden_act = layers.PReLU(name="hidden_prelu")
        n_params = self.latent_dim + self.latent_dim * (self.latent_dim + 1) // 2
        self.params = layers.Dense(n_params, name="params")

    def call(self, inputs, training=False):
        img, cond = inputs
        x = img
        for b in self.blocks:
            x = b(x)
        c = self.cond_act(self.cond_dense(cond))
        gb = self.film(c)
        f = self.filters[-1]
        gamma = ops.reshape(gb[:, :f], (-1, 1, 1, f))
        beta = ops.reshape(gb[:, f:], (-1, 1, 1, f))
        x = x * (1.0 + gamma) + beta
        x = self.flatten(x)
        x = ops.concatenate([x, c], axis=-1)
        x = self.hidden_act(self.hidden(x))
        return self.params(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(latent_dim=self.latent_dim, filters=self.filters, cond_embedding=self.cond_embedding,
                   dense=self.dense_units)
        return cfg


@keras.saving.register_keras_serializable(package="dp2_ml")
class ProjectionHead(keras.Model):
    def __init__(self, dims=(256, 128), **kwargs):
        super().__init__(**kwargs)
        self.dims = [int(d) for d in dims]
        self.layers_ = [layers.Dense(d, activation="relu" if i < len(self.dims) - 1 else None, name=f"proj{i}")
                        for i, d in enumerate(self.dims)]

    def call(self, x, training=False):
        for lyr in self.layers_:
            x = lyr(x)
        return x / ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True) + 1e-12)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dims=self.dims)
        return cfg


def add_temperature(model, temperature, learnable):
    # log(1 / tau), CLIP style, as a weight of the model that owns the contrastive term. 1/tau is capped at 100.
    import numpy as np
    return model.add_weight(shape=(), initializer=keras.initializers.Constant(float(np.log(1.0 / temperature))),
                            trainable=bool(learnable), name="log_inv_tau")


def inverse_temperature(log_inv_tau):
    return ops.exp(ops.minimum(log_inv_tau, 4.6052))
