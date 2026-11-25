# test_deblend.py
# Anna Preto
# AstroDeep Project
# The deblending campaign (Runs A, B, C) on tiny synthetic D1 / D2 stores: numerics of the likelihoods, KL
# divergences and Cholesky heads, flow matching on a toy conditional problem, MC dropout at inference, every stage
# trained for two steps in dependency order with frozen parts checked, and Rubin-only inference.

import json

import numpy as np
import pandas as pd
import pytest

keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

N_R, N_E, B_R = 17, 33, 6


def _blob(n, rng, amp, sig, dx=0.0, dy=0.0):
    yy, xx = np.indices((n, n))
    c = (n - 1) / 2
    return amp * np.exp(-((xx - c - dx) ** 2 + (yy - c - dy) ** 2) / (2 * sig ** 2))


def _write_store(path, meta, arrays):
    from dp2_cutouts.pairs import store
    sh = path / "shards" / "p0"
    store.write_shard(sh, meta, arrays, pd.DataFrame(), info=dict(patch="p0"))
    store.merge_shards([sh], path, dict(name=path.name))


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    from dp2_cutouts.mlconfig import MLConfig
    root = tmp_path_factory.mktemp("ml")
    rng = np.random.default_rng(0)
    n = 24
    split = np.array(["train"] * 16 + ["val"] * 8)
    rf = np.stack([np.stack([_blob(N_R, rng, 50 * (b + 1), 2.0) for b in range(B_R)], -1) for _ in range(n)])
    rf = (rf + rng.normal(0, 1, rf.shape)).astype(np.float32)
    ef = np.stack([np.stack([_blob(N_E, rng, 5.0, 3.0)] * 4, -1) for _ in range(n)])
    ef = (ef + rng.normal(0, 0.05, ef.shape)).astype(np.float32)
    fill = np.zeros((n, N_E, N_E), bool)
    fill[:, :5, :5] = True
    d1 = dict(rubin_flux=rf, rubin_ivar=np.ones_like(rf), rubin_mask=np.zeros(rf.shape, bool), euclid_flux=ef,
              euclid_ivar=np.full_like(ef, 400.0), euclid_mask=np.zeros(ef.shape, bool),
              euclid_present=np.ones((n, 4), bool), euclid_fill=fill)
    common = dict(split=split, tract=np.full(n, 5063), patch=np.arange(n) % 4, psf_sigma_i=np.full(n, 0.3),
                  psf_e1_i=np.zeros(n), psf_e2_i=np.zeros(n), snr_i=np.linspace(5, 80, n))
    _write_store(root / "datasets" / "d1", pd.DataFrame(dict(common, object_id=np.arange(n))), d1)
    # D2: groups of two Euclid targets sharing a Rubin owner; the second one is further from the Rubin centre.
    blend = rf + np.roll(rf, 3, axis=2)
    meta2 = pd.DataFrame(dict(common, group_id=[f"g{i // 2}" for i in range(n)], sb_rubin=np.linspace(0.05, 1, n),
                              euclid_dec=np.where(np.arange(n) % 2, -28.0, -48.0), euclid_vis_mag=np.full(n, 24.0),
                              partner_vis_mag=np.full(n, 23.5), partner_sep_arcsec=np.full(n, 0.8),
                              partner_dx_px=np.full(n, 3), partner_dy_px=np.zeros(n, int),
                              partner_kept=np.ones(n, bool),
                              euclid_object_id=np.arange(100, 100 + n),
                              rc_offset_x_px=[0 if i % 2 == 0 else 3 for i in range(n)],
                              rc_offset_y_px=np.zeros(n, int)))
    d2 = dict(rubin_flux=blend, rubin_ivar=np.ones_like(blend), rubin_mask=np.zeros(blend.shape, bool),
              rubin_flux_rc=blend, rubin_ivar_rc=np.ones_like(blend), rubin_mask_rc=np.zeros(blend.shape, bool),
              euclid_flux=ef, euclid_ivar=np.full_like(ef, 400.0), euclid_mask=np.zeros(ef.shape, bool),
              euclid_present=np.ones((n, 4), bool), euclid_fill=fill)
    _write_store(root / "datasets" / "d2", meta2, d2)
    tr = dict(batch_size=4, learning_rate=1e-3, clipnorm=1.0, train_steps=None, epochs=1,
              early_stopping_patience=5, backup_every_steps=None, val_max_examples=None)
    raw = dict(ml_root=str(root), seed=1, rubin=dict(bands=list("ugrizy"), stamp_px=N_R),
               euclid=dict(bands=["VIS", "Y", "J", "H"], stamp_px=N_E, pixel_scale=0.1, vis_psf_fwhm=0.16),
               deblend=dict(euclid_band="VIS", latent_dim=4, d2_centering="rubin", seed=3, train=tr,
                            k1=dict(beta=1.0), c2=dict(lambda_d1="auto", lambda_batches=2),
                            eval=dict(k_samples=3, max_examples=8, batch_size=4, bootstrap=5, jacobian_n=2,
                                      relation_max_examples=8, latent_max_examples=8, mask_pool=4, nfe=4,
                                      cost_batch=4, cost_k=2),
                            runs={k: f"t_{k}" for k in ("rubin_vae", "k1", "k2", "A", "B", "c1", "c2", "c3")}))
    c = MLConfig(raw=raw, path=root / "cfg.yaml")
    from dp2_cutouts.deblend import data
    data.deblend_dir(c).mkdir(parents=True, exist_ok=True)
    (data.deblend_dir(c) / "norm_stats.json").write_text(json.dumps(data.compute_norm_stats(c)))
    return c


# ----------------------------------------------------------------------------- numerics
def test_kl_divergences_match_numpy():
    from dp2_cutouts.deblend.layers import tril_from_params
    from dp2_cutouts.deblend.likelihood import kl_tril_std, kl_tril_tril, kl_tril_tril_numpy
    rng = np.random.default_rng(1)
    d = 5
    p = rng.normal(size=(2, d * (d + 1) // 2)).astype(np.float32)
    L = np.asarray(ops.convert_to_numpy(tril_from_params(ops.convert_to_tensor(p), d)))
    assert np.all(np.diagonal(L, axis1=1, axis2=2) > 0) and np.allclose(L, np.tril(L))
    mu = rng.normal(size=(2, d)).astype(np.float32)
    got = np.asarray(ops.convert_to_numpy(kl_tril_tril(mu[:1], L[:1], mu[1:], L[1:])))[0]
    ref = kl_tril_tril_numpy(mu[0], L[0] @ L[0].T, mu[1], L[1] @ L[1].T)
    assert got == pytest.approx(ref, rel=1e-3)
    got0 = float(ops.convert_to_numpy(kl_tril_std(mu[:1], L[:1]))[0])
    assert got0 == pytest.approx(kl_tril_tril_numpy(mu[0], L[0] @ L[0].T, np.zeros(d), np.eye(d)), rel=1e-3)


def test_observational_likelihood_ignores_unobserved_pixels():
    from dp2_cutouts.deblend.likelihood import chi2_per_pixel, gaussian_nll
    x = np.zeros((1, 3, 3, 1), np.float32)
    mu = np.ones_like(x)
    w = np.full_like(x, 4.0)
    w[0, 0, 0, 0] = 0.0
    assert float(gaussian_nll(x, mu, w)[0]) == pytest.approx(0.5 * 4.0 * 8)
    assert float(chi2_per_pixel(x, mu, w)[0]) == pytest.approx(4.0)


def test_noise_fill_replaces_only_flagged_pixels():
    from dp2_cutouts.deblend.data import noise_fill
    rng = np.random.default_rng(0)
    f = np.full((2, 8, 8, 1), 100.0, np.float32)
    iv = np.full_like(f, 1 / 4.0)
    bad = np.zeros(f.shape, bool)
    bad[0, :2] = True
    out = noise_fill(f, iv, bad, rng)
    assert np.all(out[~bad] == 100.0) and np.all(np.abs(out[bad]) < 20)


def test_d2_rubin_centring_keeps_one_target_per_group(cfg):
    from dp2_cutouts.deblend.data import d2_rows
    from dp2_cutouts.pairs.store import Store
    meta = Store(cfg.dataset_dir("d2")).meta
    rows = d2_rows(meta, "train", "rubin")
    assert len(rows) == 8 and np.all(meta.loc[rows, "rc_offset_x_px"] == 0)
    assert len(d2_rows(meta, "train", "euclid")) == 16


# ----------------------------------------------------------------------------- flow matching
def test_cfm_integrators_are_exact_on_a_constant_field():
    from dp2_cutouts.deblend.cfm import integrate, steps_for_nfe

    class Const:
        def __call__(self, z, t, h):
            return ops.ones_like(z) * 2.0

    z0 = np.zeros((3, 2), np.float32)
    for m in ("euler", "heun"):
        z = np.asarray(ops.convert_to_numpy(integrate(Const(), z0, np.zeros((3, 1), np.float32),
                                                      steps_for_nfe(10, m), m)))
        assert np.allclose(z, 2.0, atol=1e-5)
    with pytest.raises(ValueError):
        steps_for_nfe(5, "heun")


def test_cfm_learns_a_bimodal_conditional_target():
    # Context +1: z1 = +-2 (two modes); context -1: z1 = -5. The trained flow puts samples near the right modes.
    from dp2_cutouts.deblend import cfm
    keras.utils.set_random_seed(0)
    field = cfm.VelocityField(1, (64, 64))
    gen = keras.random.SeedGenerator(0)
    opt = keras.optimizers.Adam(3e-3)
    rng = np.random.default_rng(0)

    class Wrap(keras.Model):
        def __init__(self):
            super().__init__()
            self.f = field

        def call(self, x):
            return x

        def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
            return cfm.cfm_loss(self.f, x["z1"], x["h"], gen)

    n = 4096
    h = np.where(rng.random(n) < 0.5, 1.0, -1.0)[:, None].astype(np.float32)
    z1 = np.where(h > 0, rng.choice([-2.0, 2.0], (n, 1)), -5.0).astype(np.float32) + \
        rng.normal(0, 0.1, (n, 1)).astype(np.float32)
    w = Wrap()
    field(ops.zeros((1, 1)), ops.zeros((1, 1)), ops.zeros((1, 1)))
    w.compile(optimizer=opt)
    w.fit(dict(z1=z1, h=h), epochs=40, batch_size=256, verbose=0)
    s_pos = np.asarray(ops.convert_to_numpy(cfm.sample(field, np.ones((1, 1), np.float32), 400, 40, "heun", gen)))
    s_neg = np.asarray(ops.convert_to_numpy(cfm.sample(field, -np.ones((1, 1), np.float32), 200, 40, "heun", gen)))
    assert abs(np.median(s_neg) + 5.0) < 0.6
    frac_pos = float(np.mean(s_pos > 0))
    assert 0.25 < frac_pos < 0.75                       # both modes populated
    assert np.mean(np.abs(np.abs(s_pos) - 2.0) < 0.8) > 0.7


# ----------------------------------------------------------------------------- MC dropout
def test_mc_dropout_is_active_only_when_asked():
    from dp2_cutouts.deblend.models import RubinBlendRegressor
    keras.utils.set_random_seed(0)
    m = RubinBlendRegressor(4, 16, 0.5)
    x = np.random.default_rng(0).normal(size=(2, N_R, N_R, B_R)).astype(np.float32)
    a, b = (np.asarray(ops.convert_to_numpy(m(x, training=False))) for _ in range(2))
    assert np.allclose(a, b)
    m.set_mc(True)
    c, d = (np.asarray(ops.convert_to_numpy(m(x, training=False))) for _ in range(2))
    assert not np.allclose(c, d)


# ----------------------------------------------------------------------------- every stage, in order
def test_every_stage_trains_with_frozen_parts_and_samples_from_rubin_only(cfg):
    import inspect

    from dp2_cutouts.deblend import infer, train
    from dp2_cutouts.deblend.data import D2Batches, load_norm_stats
    for stage in ("rubin_vae", "k1", "k2", "A", "B", "c1", "c2", "c3"):
        rd = train.train(cfg, stage, max_steps=2)
        assert (rd / "DONE").exists(), stage
        run = json.loads((rd / "run.json").read_text())
        if train.DEPENDS[stage]:
            assert json.loads((rd / "frozen_check.json").read_text())["unchanged"]
        if stage == "c2":
            assert run["settings"]["lambda_d1_calibration"]["value"] > 0
    # CFM velocity field of the planned size (d = 32, context 128): a few tens of thousands of parameters.
    from dp2_cutouts.deblend.cfm import VelocityField
    f = VelocityField(32)
    f(ops.zeros((1, 32)), ops.zeros((1, 1)), ops.zeros((1, 128)))
    assert f.count_params() == 41376

    # Rubin-only inference: the samplers take the blend input only.
    d = cfg["deblend"]["latent_dim"]
    sr = load_norm_stats(cfg)["rubin"]["scale"]
    x, _ = D2Batches(cfg, "val", 4, False, fields=("blend",))[0]
    assert set(x) == {"x_in", "x_flux", "x_w"}
    s = {st: train.settings(cfg, st) for st in ("A", "B", "c3")}
    dec_k2 = train.load_decoder(train.run_dir(cfg, "k2") / "decoder_best.weights.h5", N_R, B_R, d)
    dec_c1 = train.load_decoder(train.run_dir(cfg, "c1") / "dec_r_best.weights.h5", N_R, B_R, d)
    samplers = dict(
        A=infer.RunASampler(train.load_student(cfg, "A", train.run_dir(cfg, "A"), d, s["A"]), dec_k2, sr),
        B=infer.CFMSampler(train.load_student(cfg, "B", train.run_dir(cfg, "B"), d, s["B"]), dec_k2, sr, 4, "euler"),
        C=infer.CFMSampler(train.load_student(cfg, "c3", train.run_dir(cfg, "c3"), d, s["c3"]), dec_c1, sr, 4,
                           "heun"))
    for name, smp in samplers.items():
        params = set(inspect.signature(smp.__call__).parameters)
        assert not any("e_" in p or "euclid" in p for p in params), name
        z, img = smp(x["x_in"], 3)
        assert z.shape == (4, 3, d) and img.shape == (4, 3, N_R, N_R, B_R) and np.all(np.isfinite(img)), name
    za = samplers["A"].latents(x["x_in"], 5)
    assert np.std(za, axis=1).mean() > 0          # MC dropout gives diverse samples


def test_a_training_stopped_on_its_time_budget_resumes_the_cut_epoch(cfg):
    from dp2_cutouts.deblend import train
    rd = train.train(cfg, "k1", seed=7, time_budget_s=1e-6)
    assert not (rd / "DONE").exists()
    meta = json.loads((rd / "backup" / "training_metadata.json").read_text())
    assert meta["epoch"] == 0                      # the cut epoch is redone, not skipped
    rd = train.train(cfg, "k1", seed=7)
    assert (rd / "DONE").exists()


def test_evaluations_run_on_the_trained_campaign(cfg):
    # After the stage test (same module fixture): every evaluation produces its summary.
    from dp2_cutouts.deblend import evaluate
    for what in ("d1", "masking", "d2", "cost", "latent"):
        summ = evaluate.run(cfg, what, split="val")
        assert summ, what
    d2 = json.loads((evaluate.out_dir(cfg, "d2", "val", None) / "summary.json").read_text())
    assert set(d2["runs"]) == {"A", "B", "C"} and "A_minus_B" in d2["paired"]
    d1 = json.loads((evaluate.out_dir(cfg, "d1", "val", None) / "summary.json").read_text())
    assert {"rvae", "k1k2", "c1"} <= set(d1["paths"]) and set(d1["probabilistic"]) == {"A", "B", "C"}
