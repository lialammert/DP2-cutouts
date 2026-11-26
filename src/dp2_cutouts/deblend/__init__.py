# deblend
# Anna Preto
# AstroDeep Project
# Probabilistic deblending of real Rubin blends with Euclid as privileged information (training only).
# Three runs are compared on the same D1 / D2 (pairs package), with weak lensing as the priority metric:
#   Run A  Kirill-style sequential: K1 Euclid VIS VAE (frozen), K2 Rubin decoder from the Euclid latent (frozen),
#          Rubin blend CNN with Monte Carlo dropout regressing sampled teacher latents (latent MSE);
#   Run B  same K1 / K2, Rubin blend CNN context + small conditional flow matching (CFM) velocity field;
#   Run C  joint Rubin-Euclid VAE (C1), conditional variational teacher r(z | X, E_c) with a Gaussian student
#          s(z | X) (C2), CFM student on samples of the teacher (C3).
# At inference every run takes the Rubin blend only. Modules:
#   layers      convolutional encoders / decoders, Gaussian heads (diagonal, full covariance), MLP, MC dropout
#   likelihood  observational Gaussian likelihood (ivar, valid pixels), Gaussian KL divergences
#   cfm         velocity field, CFM loss, fixed-step Euler / Heun samplers
#   data        streaming batches of D1 / D2 for every stage
#   models      the trainable models of each stage
#   train       one entry point per stage (checkpoints, resume, frozen-weight checks)
#   infer       Rubin-only samplers of the three runs
#   evaluate    diagnostics and comparisons

import os

# The campaign is written for JAX (the lambda_D1 calibration needs it). Without KERAS_BACKEND, Keras falls back to
# TensorFlow, not installed in the Jean Zay venv: a job submitted from a shell without the variable failed on import.
os.environ.setdefault("KERAS_BACKEND", "jax")
