# __init__.py
# Anna Preto
# AstroDeep Project
# Cross-instrument learning on top of the frozen DebVAder VAE: Stage 1 (Euclid encoder into the Rubin latent
# space) and Stage 2 (Rubin blend encoder supervised by the Euclid teacher). Keras 3 throughout; the backend is the
# one debvader-keras uses (JAX) unless KERAS_BACKEND is already set. It must be chosen before keras is imported,
# hence here.

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
