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
    common = dict(split=split, tract=np.full(n, 5063), patch=np.arange(n) % 4, snr_i=np.linspace(5, 80, n))
    for b in "ugrizy":
        # The 18 Rubin PSF scalars of the catalogue (deblend.psf_condition).
        common.update({f"psf_sigma_{b}": rng.uniform(0.25, 0.45, n), f"psf_e1_{b}": rng.normal(0.01, 0.03, n),
                       f"psf_e2_{b}": rng.normal(-0.01, 0.03, n)})
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
    tr = dict(batch_size=4, learning_rate=1e-3, global_clipnorm=1.0, train_steps=None, epochs=1,
              early_stopping_patience=5, backup_every_steps=None, val_max_examples=None, warmup_frac=0.02,
              final_lr_frac=0.01)
    raw = dict(ml_root=str(root), seed=1, rubin=dict(bands=list("ugrizy"), stamp_px=N_R),
               euclid=dict(bands=["VIS", "Y", "J", "H"], stamp_px=N_E, pixel_scale=0.1, vis_psf_fwhm=0.16),
               deblend=dict(euclid_band="VIS", latent_dim=4, d2_centering="rubin", seed=3, train=tr,
                            psf_condition=True,
                            rubin_vae=dict(beta=1.0, kl_warmup_frac=0.1), k1=dict(beta=1.0, kl_warmup_frac=0.1),
                            c1=dict(kl_warmup_frac=0.1),
                            A=dict(embed=16, draws=4, val_draws=6, replay=True, stats_batches=2),
                            B=dict(hidden=[16, 16, 16], time_dim=8, replay=True, ema=0.9, val_triplets=3,
                                   stats_batches=2),
                            c2=dict(lambda_d1="auto"),
                            c3=dict(hidden=[16, 16], time_dim=8, ema=0.9, val_triplets=3, stats_batches=2),
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
def test_d2_selection_drops_groups_whose_nearest_target_fails(cfg):
    from dp2_cutouts.deblend.data import d2_keep, d2_rows
    from dp2_cutouts.mlconfig import MLConfig
    meta = pd.DataFrame(dict(split="train", group_id=["g0", "g0", "g1", "g1", "g2", "g2"],
                             rc_offset_x_px=[0, 3, 0, 3, 0, 3], rc_offset_y_px=0, euclid_object_id=range(6),
                             sep_delta_e=[0.0, 0.0, 0.5, 0.0, 0.0, 0.0], sep_centroid_shift_arcsec=0.0,
                             sep_rel_size_change=0.0, sep_rel_flux_change=0.0,
                             euclid_vis_mag=[22.0, 22.0, 22.0, 22.0, 99.0, 22.0]))
    raw = json.loads(json.dumps(cfg.raw))
    raw["deblend"]["d2_selection"] = dict(
        before_after=dict(max_delta_e=0.02, max_centroid_shift_arcsec=0.05, max_rel_size_change=0.05,
                          max_rel_flux_change=0.05), vis_mag_max="auto", vis_mag_quantile=1.0)
    c = MLConfig(raw=raw, path=cfg.path)
    import dp2_cutouts.deblend.data as D
    old = D.d1_vis_mag_limit
    D.d1_vis_mag_limit = lambda cfg_, q: 25.0
    try:
        keep, info = d2_keep(c, meta)
    finally:
        D.d1_vis_mag_limit = old
    assert keep.tolist() == [True, True, False, True, False, True] and info["vis_mag_max"] == 25.0
    # g1's nearest member fails the test and g2's is too faint: those groups are dropped, not given the farther one.
    assert d2_rows(meta, "train", "rubin", keep).tolist() == [0]
    assert d2_keep(cfg, meta)[0].all()          # no deblend.d2_selection: everything
    # D1's cut 1 on the targets (point-like / spurious Euclid objects), from d2/euclid_flags.parquet.
    raw["deblend"]["d2_selection"] = dict(euclid_cut1=True)
    c = MLConfig(raw=raw, path=cfg.path)
    flags = c.dataset_dir("d2") / "euclid_flags.parquet"
    pd.DataFrame(dict(euclid_object_id=np.arange(6, dtype=np.int64),
                      euclid_cut1_pass=[True, False, True, True, True, True])).to_parquet(flags)
    try:
        keep, info = d2_keep(c, meta)
    finally:
        flags.unlink()
    assert keep.tolist() == [True, False, True, True, True, True] and info["pass_euclid_cut1"] == 5 / 6
    with pytest.raises(FileNotFoundError):
        d2_keep(c, meta)


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
def test_dropout_draws_of_run_a_follow_their_masks():
    # The same masks give the same draws, other masks other draws; all units kept = the plain prediction; the PSF
    # is appended to every draw.
    from dp2_cutouts.deblend.models import RubinBlendRegressor
    keras.utils.set_random_seed(0)
    m = RubinBlendRegressor(4, 16, 0.5)
    x = np.random.default_rng(0).normal(size=(2, N_R, N_R, B_R)).astype(np.float32)
    c = np.random.default_rng(1).normal(size=(2, 18)).astype(np.float32)
    plain = np.asarray(ops.convert_to_numpy(m(x, cond=c)))
    u = np.random.default_rng(2).random((2, 5, 16)).astype(np.float32)
    a, b = (np.asarray(ops.convert_to_numpy(m.draws(x, c, u))) for _ in range(2))
    assert a.shape == (2, 5, 4) and np.allclose(a, b) and np.std(a, axis=1).mean() > 0
    m0 = RubinBlendRegressor(4, 16, 0.0)
    p0 = np.asarray(ops.convert_to_numpy(m0(x, cond=c)))
    d0 = np.asarray(ops.convert_to_numpy(m0.draws(x, c, u)))
    np.testing.assert_allclose(d0, np.repeat(p0[:, None], 5, axis=1), rtol=1e-5, atol=1e-5)
    assert plain.shape == (2, 4)


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
        sched = run["settings"]["schedule"]
        assert sched["total_updates"] >= 2 and sched["warmup_updates"] == round(0.02 * sched["total_updates"])
        head = (rd / "history.csv").read_text().splitlines()[0]
        assert "grad_norm_mean" in head and "grad_norm_q95" in head and "grad_norm_nonfinite" in head
        gn = json.loads((rd / "grad_norms.json").read_text())
        assert gn["all"]["n"] == sched["total_updates"] and gn["stats_from"] == sched["grad_stats_from"]
        if stage == "c2":
            lam = json.loads((rd / "lambda_d1.json").read_text())
            assert lam["n_measurements"] >= 1 and lam["target"] > 0 and np.isfinite(lam["target"])
        if stage in ("A", "B", "c3"):
            assert json.loads((rd / "best.json").read_text())["monitor"] == train.MONITOR[stage]
    # CFM velocity field of the planned size (d = 32, context 128): a few tens of thousands of parameters.
    from dp2_cutouts.deblend.cfm import VelocityField
    f = VelocityField(32)
    f(ops.zeros((1, 32)), ops.zeros((1, 1)), ops.zeros((1, 128)))
    assert f.count_params() == 41376

    # Rubin-only inference: the samplers take the blend input only.
    d = cfg["deblend"]["latent_dim"]
    sr = load_norm_stats(cfg)["rubin"]["scale"]
    x, _ = D2Batches(cfg, "val", 4, False, fields=("blend",))[0]
    assert set(x) == {"x_in", "x_flux", "x_w", "x_psf"} and x["x_psf"].shape == (4, 18)
    s = {st: train.settings(cfg, st) for st in ("A", "B", "c3")}
    dec_k2 = train.load_decoder(train.run_dir(cfg, "k2") / "decoder_best.weights.h5", N_R, B_R, d, cfg)
    dec_c1 = train.load_decoder(train.run_dir(cfg, "c1") / "dec_r_best.weights.h5", N_R, B_R, d, cfg)
    # A and B share one K1 latent standardisation (written once, read by the second).
    assert len(list((train.data.deblend_dir(cfg) / "latent_stats").glob("k1_*.json"))) == 1
    assert json.loads((train.run_dir(cfg, "A") / "latent_stats.json").read_text()) == \
        json.loads((train.run_dir(cfg, "B") / "latent_stats.json").read_text())
    samplers = dict(
        A=infer.RunASampler(train.load_student(cfg, "A", train.run_dir(cfg, "A"), d, s["A"]), dec_k2, sr),
        B=infer.CFMSampler(train.load_student(cfg, "B", train.run_dir(cfg, "B"), d, s["B"]), dec_k2, sr, 4, "euler"),
        C=infer.CFMSampler(train.load_student(cfg, "c3", train.run_dir(cfg, "c3"), d, s["c3"]), dec_c1, sr, 4,
                           "heun"))
    for name, smp in samplers.items():
        params = set(inspect.signature(smp.__call__).parameters)
        assert not any("e_" in p or "euclid" in p for p in params), name
        z, img = smp(x["x_in"], 3, cond=x["x_psf"])
        assert z.shape == (4, 3, d) and img.shape == (4, 3, N_R, N_R, B_R) and np.all(np.isfinite(img)), name
    za = samplers["A"].latents(x["x_in"], 5, cond=x["x_psf"])
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
    from dp2_cutouts.mlconfig import MLConfig
    # The synthetic galaxies have no reliable high-S/N shapes: on them the D1 relation guard stops the evaluation
    # (checked below); the end-to-end run relaxes it.
    with pytest.raises(RuntimeError, match="D1 relation"):
        evaluate.run(cfg, "d2", split="val")
    raw = json.loads(json.dumps(cfg.raw))
    raw["deblend"].setdefault("eval", {}).update(relation_min_snr=0, relation_min_pairs=0, relation_max_slope_dev=100,
                                                  relation_max_offset=100)
    loose = MLConfig(raw=raw, path=cfg.path)
    for what in ("d1", "masking", "d2", "cost", "latent", "check_vaes", "check_k2"):
        summ = evaluate.run(loose, what, split="val")
        assert summ, what
    chk = json.loads((evaluate.out_dir(cfg, "check_vaes", "val", None) / "summary.json").read_text())
    assert {"rvae", "c1", "k1k2"} <= set(chk["paths"]) and set(chk["vis_paths"]) == {"k1_vis", "c1_vis"}
    assert set(chk["latent"]) == {"rvae", "k1", "c1"} and not chk["probabilistic"]
    assert len(chk["latent"]["k1"]["kl_per_dim"]) == 4
    d2 = json.loads((evaluate.out_dir(cfg, "d2", "val", None) / "summary.json").read_text())
    assert d2["d1_relation"]["reference"] == "vis" and set(d2["shape_by_snr"]) == {"A", "B", "C"}
    assert set(d2["runs"]) == {"A", "B", "C"} and "A_minus_B" in d2["paired"]
    d1 = json.loads((evaluate.out_dir(cfg, "d1", "val", None) / "summary.json").read_text())
    assert {"rvae", "k1k2", "c1"} <= set(d1["paths"]) and set(d1["probabilistic"]) == {"A", "B", "C"}


def test_side_reads_runs_of_positions_with_pread(tmp_path):
    # compact.Side: any set of positions, dense runs, gaps read through or skipped, long runs cut: same rows as numpy.
    from dp2_cutouts.deblend import compact
    a = np.arange(500 * 3 * 3 * 2, dtype=np.float32).reshape(500, 3, 3, 2)
    np.save(tmp_path / "x.npy", a)
    pos_of = np.arange(500)[::-1].copy()                  # row r is at position 499 - r
    side = compact.Side(tmp_path / "x.npy", pos_of, [], False, "t")
    side.MAX_READ = 7 * side.row_bytes                    # force cut runs
    for rows in ([3], [0, 1, 2, 3], [5, 400, 6, 499, 7], list(range(0, 500, 3)), list(range(100, 160))):
        np.testing.assert_array_equal(side.take(rows), a[pos_of[rows]])
    mem = compact.Side(tmp_path / "x.npy", pos_of, list(range(50, 90)), True, "t")
    np.testing.assert_array_equal(mem.take([60, 50, 89]), a[pos_of[[60, 50, 89]]])


def test_block_order_reads_consecutive_rows():
    # deblend.train.read_block: a training epoch visits every row once, in shuffled blocks of consecutive rows.
    from dp2_cutouts.deblend.data import Batches
    rows = np.arange(5, 101)                            # 96 rows: 12 full blocks of 8
    b = Batches(rows, 16, True, 3, block=8)
    assert sorted(b.order.tolist()) == rows.tolist()
    assert all((np.diff(b.order[k:k + 8]) == 1).all() for k in range(0, 96, 8))
    assert not (b.order == rows).all()                  # shuffled
    one = Batches(rows, 16, True, 3).order              # read_block 1: rows one by one
    assert sorted(one.tolist()) == rows.tolist() and (np.diff(one) != 1).sum() > 48


def test_d1_vis_mag_limit_reads_the_d1_metadata(tmp_path):
    # The D2 selection's VIS limit read on the D1 metadata itself (a missing import once failed only on Jean Zay,
    # where vis_mag_max: auto reaches this function; the selection test above replaces it).
    from dp2_cutouts.deblend.data import d1_vis_mag_limit
    from dp2_cutouts.mlconfig import MLConfig
    d = tmp_path / "datasets" / "d1"
    d.mkdir(parents=True)
    pd.DataFrame(dict(split=["train", "train", "val", "train"], euclid_vis_mag=[22.0, 24.0, 30.0, np.nan],
                      other=1)).to_parquet(d / "meta.parquet")
    c = MLConfig(raw=dict(ml_root=str(tmp_path)), path=tmp_path / "cfg.yaml")
    assert c.dataset_dir("d1") == d
    assert d1_vis_mag_limit(c, 1.0) == 24.0 and d1_vis_mag_limit(c, 0.0) == 22.0


def test_full_covariance_head_starts_well_conditioned():
    # TriLGaussianHead starts near L = identity even on large embeddings, so that the KL divergences between two
    # freshly initialised heads (C2 teacher / student) are finite and moderate (C2 started at KL ~1e35 on real data).
    from dp2_cutouts.deblend.layers import TriLGaussianHead
    from dp2_cutouts.deblend.likelihood import kl_tril_tril
    rng = np.random.default_rng(0)
    h = rng.normal(0, 10, (16, 256)).astype(np.float32)
    a, b = TriLGaussianHead(32), TriLGaussianHead(32)
    (mu_a, La), (mu_b, Lb) = a(h), b(h)
    La, Lb = np.asarray(La), np.asarray(Lb)
    diag = np.diagonal(La, axis1=1, axis2=2)
    off = La - np.einsum("bi,ij->bij", diag, np.eye(32))
    assert np.allclose(diag, 1.0, atol=0.1) and np.abs(off).max() < 0.1
    kl = np.asarray(kl_tril_tril(mu_a, La, mu_b, Lb))
    assert np.isfinite(kl).all() and kl.max() < 10.0


def test_full_covariance_head_passes_gradients_at_start():
    # A tiny but non-zero kernel: the output depends on the embedding from the first step on (the lambda_D1
    # calibration of C2 found zero gradients with an exactly zero kernel).
    import jax
    from dp2_cutouts.deblend.layers import TriLGaussianHead
    h = np.random.default_rng(1).normal(0, 1, (4, 64)).astype(np.float32)
    head = TriLGaussianHead(8)
    head(h)
    g = jax.grad(lambda x: float(0) + jax.numpy.sum(head(x)[1] ** 2) + jax.numpy.sum(head(x)[0] ** 2))(h)
    assert np.abs(np.asarray(g)).max() > 0


def test_lost_flux_selection_of_d1_and_d2(cfg):
    # deblend.d1_selection / d2_selection.max_lost_flux_frac: a pair whose masks hide more than 3 % of the target's
    # light, by more than 3 uncertainties, is left out of every read (D1 rows aligned, D2 by Euclid id).
    from dp2_cutouts.deblend.data import D1Batches, d1_keep, d2_keep
    from dp2_cutouts.mlconfig import MLConfig
    from dp2_cutouts.pairs.store import Store
    raw = json.loads(json.dumps(cfg.raw))
    raw["deblend"]["d1_selection"] = dict(max_lost_flux_frac=0.03, lost_sigma=3.0)
    raw["deblend"]["d2_selection"] = dict(max_lost_flux_frac=0.03, lost_sigma=3.0)
    c = MLConfig(raw=raw, path=cfg.path)
    m1 = Store(cfg.dataset_dir("d1")).meta
    m2 = Store(cfg.dataset_dir("d2")).meta
    f1, f2 = cfg.dataset_dir("d1") / "mask_loss.parquet", cfg.dataset_dir("d2") / "mask_loss.parquet"
    frac = np.zeros(len(m1))
    frac[0], frac[1] = 0.5, 0.05                       # row 0 cut; row 1 above 3 % but within 3 sigma: kept
    pd.DataFrame(dict(object_id=m1["object_id"].astype("int64"), lost_flux_frac=frac,
                      lost_flux_sigma=[0.01, 0.05] + [0.001] * (len(m1) - 2), lost_unknown=0.0)).to_parquet(f1)
    pd.DataFrame(dict(euclid_object_id=m2["euclid_object_id"].astype("int64")[:3], lost_flux_frac=[0.2, 0.0, 0.0],
                      lost_flux_sigma=0.01, lost_unknown=0.0)).to_parquet(f2)
    try:
        keep1, info1 = d1_keep(c, m1)
        assert not keep1[0] and keep1[1:].all() and info1["kept"] == pytest.approx(1 - 1 / len(m1))
        assert 0 not in D1Batches(c, m1["split"].iloc[0], 4, False).rows_all
        keep2, info2 = d2_keep(c, m2)
        assert not keep2[0] and keep2[1:].all()          # rows not measured are kept
        assert d1_keep(cfg, m1)[0].all()                  # no d1_selection: everything
    finally:
        f1.unlink()
        f2.unlink()


def test_compact_copies_give_the_same_batches_up_to_the_fixed_noise_draw(cfg, tmp_path):
    # deblend-compact d1 / d2 (flux and weight, VIS only, masked pixels filled once with sky noise), read memory-mapped
    # or in memory: same weights, same flux and network input wherever the weight is > 0 (masked, NaN and ivar <= 0
    # pixels and the neighbour fill included), finite noise elsewhere; the copy is reproducible.
    import shutil

    from dp2_cutouts.deblend import compact, data
    from dp2_cutouts.mlconfig import MLConfig
    root = tmp_path / "ml"
    shutil.copytree(cfg.root / "datasets", root / "datasets")
    shutil.copytree(data.deblend_dir(cfg), root / "deblend")
    rng = np.random.default_rng(5)
    for which in ("d1", "d2"):
        d = root / "datasets" / which
        for name in [k for k in ("rubin_flux", "rubin_flux_rc", "euclid_flux") if (d / f"{k}.npy").exists()]:
            a = np.load(d / f"{name}.npy", mmap_mode="r+")
            a[rng.random(a.shape) < 0.01] = np.nan
            a.flush()
            m = np.load(d / f"{name.replace('flux', 'mask')}.npy", mmap_mode="r+")
            m[rng.random(m.shape) < 0.02] = True
            m.flush()
            iv = np.load(d / f"{name.replace('flux', 'ivar')}.npy", mmap_mode="r+")
            iv[rng.random(iv.shape) < 0.02] = 0.0
            iv.flush()
    raw = json.loads(json.dumps(cfg.raw))
    raw["ml_root"] = str(root)
    off = MLConfig(raw=json.loads(json.dumps(raw)), path=root / "cfg.yaml")
    raw["deblend"]["data"] = dict(compact=True, in_memory=["d1", "d2"], compact_d1=["euclid", "rubin"])
    on = MLConfig(raw=raw, path=root / "cfg.yaml")
    with pytest.raises(RuntimeError, match="deblend-compact"):
        data.D2Batches(on, "train", 4, True, seed=2)
    info1 = compact.build(on, "d1", workers=1)
    info2 = compact.build(on, "d2", workers=2)                # through the process pool
    assert info1["rows"] == 24 and set(info1["sides"]) == {"euclid", "rubin"}
    # Layout: grouped by split (train first), shuffled within each.
    lay = np.load(root / "datasets" / "d1" / "compact" / "rows.npy")
    assert info1["split_ranges"] == {"train": [0, 16], "val": [16, 24]}
    assert sorted(lay[:16]) == list(range(16)) and sorted(lay[16:]) == list(range(16, 24))
    assert not (lay[:16] == np.arange(16)).all()
    assert info2["rows"] == len(np.unique(np.concatenate([data.D2Batches(off, s, 4, False).rows_all
                                                          for s in ("train", "val")])))
    for train in (True, False):
        for cls, split in ((data.D1Batches, "train"), (data.D2Batches, "train"), (data.D2Batches, "val")):
            a, b = cls(off, split, 4, train, seed=2), cls(on, split, 4, train, seed=2)
            assert not a.sides and set(b.sides) == {"euclid", "rubin"}
            assert (b.sides["euclid"].local is not None) == train
            assert sorted(a.rows_all) == sorted(b.rows_all)
            if train:
                # Training order follows the positions in the (shuffled) copy; same rows, same draws per batch.
                key = compact.position_key(on, "d1" if cls is data.D1Batches else "d2")
                assert (np.diff(key[b.rows_all]) > 0).all()
                a.order = b.order.copy()
            for i in range(len(a)):
                xa, xb = a[i][0], b[i][0]
                assert set(xa) == set(xb)
                for pre in ("r", "x", "e"):
                    if f"{pre}_w" not in xa:
                        continue
                    w = xa[f"{pre}_w"]
                    np.testing.assert_array_equal(w, xb[f"{pre}_w"])
                    ok = w > 0
                    np.testing.assert_array_equal(xa[f"{pre}_flux"][ok], xb[f"{pre}_flux"][ok])
                    nb = xa[f"{pre}_in"].shape[-1] - (1 if pre == "e" else 0)
                    np.testing.assert_array_equal(xa[f"{pre}_in"][..., :nb][ok], xb[f"{pre}_in"][..., :nb][ok])
                    if pre == "e":
                        np.testing.assert_array_equal(xa["e_in"][..., 1], xb["e_in"][..., 1])
                    assert np.isfinite(xb[f"{pre}_flux"]).all() and np.isfinite(xb[f"{pre}_in"]).all()
                    assert (~ok).any()
    # The copy is reproducible (fixed noise draw).
    first = np.load(root / "datasets" / "d2" / "compact" / "rubin.npy")
    compact.build(on, "d2", workers=1)
    np.testing.assert_array_equal(first, np.load(root / "datasets" / "d2" / "compact" / "rubin.npy"))
    # A copy of an older dataset is not used.
    (root / "datasets" / "d2" / "meta.parquet").touch()
    import os
    st = os.stat(root / "datasets" / "d2" / "meta.parquet")
    os.utime(root / "datasets" / "d2" / "meta.parquet", ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    assert compact.read_info(on, "d2") is None


def test_bench_times_every_trained_stage_and_the_loaders(cfg):
    # After the stage test (same module fixture): deblend-bench on two tiny batch sizes.
    from dp2_cutouts.deblend import bench
    res = bench.run(cfg, sizes=[2, 4], loader_batches=2, steps=1)
    assert set(res["stages"]) == {"rubin_vae", "k1", "k2", "A", "B", "c1", "c2", "c3"}
    for st, r in res["stages"].items():
        assert set(r["sizes"]) == {2, 4} and r["sizes"][4]["examples_per_s"] > 0, st
        assert r["saturation_batch"] in (2, 4)
    assert res["loaders"]["d1 original batch 4"]["examples_per_s"] > 0
    assert "skipped" in res["loaders"]["d2 compact, memory-mapped"]
    from dp2_cutouts.deblend import layers
    assert not layers.INPUT_NORM                                  # the GPU part ran with the configured (CPU) path


def test_larger_batches_scale_the_learning_rate_and_keep_the_examples_budget():
    from dp2_cutouts.deblend.train import scale_for_batch
    base = dict(batch_size=128, learning_rate=1e-4, lr_scaling="sqrt", lr_reference_batch=128,
                train_examples=5_120_000, train_steps=1)
    assert scale_for_batch(base)["learning_rate"] == pytest.approx(1e-4)
    assert scale_for_batch(base)["train_steps"] == 40000
    big = scale_for_batch(dict(base, batch_size=1024))
    assert big["learning_rate"] == pytest.approx(1e-4 * 8 ** 0.5) and big["train_steps"] == 5000
    assert scale_for_batch(dict(base, batch_size=1024, lr_scaling="none"))["learning_rate"] == 1e-4


def test_input_normalisation_on_the_gpu_gives_the_same_encodings(cfg):
    # deblend.data.normalise_on: gpu hands the encoders the filled flux and normalises inside them: same latent
    # means as the CPU path, up to float rounding, for a Rubin and a Euclid encoder.
    from dp2_cutouts.deblend import data, layers
    from dp2_cutouts.deblend.models import ImageEncoder
    from dp2_cutouts.mlconfig import MLConfig
    raw = json.loads(json.dumps(cfg.raw))
    raw["deblend"]["data"] = dict(normalise_on="gpu")
    gpu = MLConfig(raw=raw, path=cfg.path)
    try:
        xc = data.D2Batches(cfg, "val", 4, False)[0][0]          # Prepare of cfg: normalised on the CPU
        assert not layers.INPUT_NORM
        enc_r, enc_e = ImageEncoder(4, name="r"), ImageEncoder(4, name="e")
        mr, me = (np.asarray(ops.convert_to_numpy(e(ops.convert_to_tensor(xc[k]))[0]))
                  for e, k in ((enc_r, "x_in"), (enc_e, "e_in")))
        xg = data.D2Batches(gpu, "val", 4, False)[0][0]          # flux in, normalisation in the encoders
        assert layers.INPUT_NORM and np.abs(xg["x_in"]).max() > 2  # raw flux, not tanh outputs
        np.testing.assert_array_equal(xg["e_in"][..., 1], xc["e_in"][..., 1])
        gr, ge = (np.asarray(ops.convert_to_numpy(e(ops.convert_to_tensor(xg[k]))[0]))
                  for e, k in ((enc_r, "x_in"), (enc_e, "e_in")))
        np.testing.assert_allclose(gr, mr, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(ge, me, rtol=1e-4, atol=1e-5)
    finally:
        layers.INPUT_NORM.clear()


def test_a_stage_trains_data_parallel_on_two_devices(cfg, tmp_path):
    # deblend.k1.train.gpus = 2 on two simulated CPU devices (fresh process: the device count is fixed when JAX
    # starts): trained to DONE, global batch = 2 x the per-device batch, learning rate scaled, warm-up accepted.
    import os
    import subprocess
    import sys

    import yaml
    raw = json.loads(json.dumps(cfg.raw))
    raw["deblend"]["k1"] = dict(raw["deblend"].get("k1") or {}, train=dict(gpus=2))
    raw["deblend"]["train"].update(lr_scaling="sqrt", lr_reference_batch=4)
    raw["deblend"]["runs"]["k1"] = "t_k1_dp"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    code = ("from dp2_cutouts.mlconfig import MLConfig\nfrom dp2_cutouts.deblend import train\n"
            f"rd = train.train(MLConfig.load({str(p)!r}), 'k1')\nprint('RUN', rd)\n")
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu",
               KERAS_BACKEND="jax")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, env=env)
    assert "data parallel on 2 devices" in out.stdout, out.stdout[-2000:] + out.stderr[-3000:]
    rd = next(line.split(" ", 1)[1] for line in out.stdout.splitlines() if line.startswith("RUN "))
    from pathlib import Path
    assert (Path(rd) / "DONE").exists()
    tr = json.loads((Path(rd) / "run.json").read_text())["settings"]["train"]
    assert tr["batch_size"] == 8 and tr["batch_size_per_device"] == 4
    assert tr["learning_rate"] == pytest.approx(tr["learning_rate_reference"] * 2 ** 0.5)


def test_cudnn_failure_under_autotuning_restarts_at_level_zero():
    from dp2_cutouts.mlcli import autotune_fallback_env
    err = "XlaRuntimeError: UNKNOWN: CUDNN_STATUS_EXECUTION_FAILED in cuda_dnn.cc"
    env = dict(XLA_FLAGS="--foo --xla_gpu_autotune_level=3", PATH="p")
    new = autotune_fallback_env(err, env)
    assert new["XLA_FLAGS"] == "--foo --xla_gpu_autotune_level=0" and new["DP2_AUTOTUNE_FALLBACK"] == "1"
    assert new["PATH"] == "p"
    assert autotune_fallback_env(err, new) is None                     # never twice
    assert autotune_fallback_env("ValueError: other", env) is None
    assert autotune_fallback_env(err, dict(XLA_FLAGS="--xla_gpu_autotune_level=0")) is None
    assert autotune_fallback_env(err, {})["XLA_FLAGS"] == "--xla_gpu_autotune_level=0"


# ----------------------------------------------------------------------------- recipe of 2026-11-28
def test_energy_score_matches_its_definition_and_stays_finite_on_identical_draws():
    import jax
    from dp2_cutouts.deblend.likelihood import energy_score
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 5, 4)).astype(np.float32)
    y = rng.normal(size=(3, 4)).astype(np.float32)
    es, err, spread = (np.asarray(ops.convert_to_numpy(a)) for a in energy_score(z, y))
    for b in range(3):
        e1 = np.mean([np.linalg.norm(z[b, i] - y[b]) for i in range(5)])
        e2 = np.mean([np.linalg.norm(z[b, i] - z[b, j]) for i in range(5) for j in range(5) if i != j])
        assert es[b] == pytest.approx(e1 - 0.5 * e2, rel=1e-5) and spread[b] == pytest.approx(e2, rel=1e-5)
        assert err[b] == pytest.approx(e1, rel=1e-5)
    # Identical draws (dropout masks that coincide): finite gradients, the pair term contributing zero.
    same = np.repeat(z[:, :1], 5, axis=1)
    g = jax.grad(lambda a: jax.numpy.sum(energy_score(a, y)[0]))(jax.numpy.asarray(same))
    assert np.all(np.isfinite(np.asarray(g)))


def test_learning_rate_warms_up_then_decays_to_one_percent():
    from dp2_cutouts.deblend.optim import lr_schedule
    sched, warm = lr_schedule(2e-4, 1000, 0.02, 0.01)
    lr = lambda s: float(ops.convert_to_numpy(sched(s)))  # noqa: E731
    assert warm == 20
    assert lr(0) == pytest.approx(2e-6, rel=1e-3) and lr(20) == pytest.approx(2e-4, rel=1e-3)
    assert lr(10) == pytest.approx(0.5 * (2e-6 + 2e-4), rel=1e-2)
    assert lr(1000) == pytest.approx(2e-6, rel=1e-2) and lr(510) == pytest.approx(0.5 * (2e-4 + 2e-6), rel=1e-2)
    v = [lr(s) for s in range(20, 1001, 10)]
    assert all(a >= b for a, b in zip(v, v[1:]))


def test_global_clipping_statistics_and_ema_start():
    # The global norm is measured before clipping; the EMA copies the weights until ema_start, then averages.
    from dp2_cutouts.deblend.optim import DeblendAdam
    w = keras.Variable(np.zeros(3, np.float32))
    opt = DeblendAdam(0.1, global_clipnorm=1.0, ema=0.5, ema_start=2)
    opt.build([w])
    for _ in range(4):
        opt.apply([ops.convert_to_tensor(np.array([3.0, 4.0, 0.0], np.float32))], [w])
    st = opt.stats()
    assert st["grad_norm_mean"] == pytest.approx(5.0) and st["clipped_frac"] == 1.0 and st["updates"] == 4
    ema = np.asarray(ops.convert_to_numpy(opt._model_variables_moving_average[0]))
    assert not np.allclose(ema, ops.convert_to_numpy(w))      # averaged after ema_start, lagging the weights
    assert st["grad_norm_q50"] == pytest.approx(5.0, rel=0.07) and st["grad_norm_q99"] == pytest.approx(5.0, rel=0.07)
    h = opt.run_histograms()
    assert h["all"]["n"] == 4 and h["steady"]["n"] == 4 and h["threshold"] == 1.0
    opt.reset_stats()
    assert opt.stats()["updates"] == 0 and opt.run_histograms()["all"]["n"] == 4    # the run's histogram stays
    # Without a threshold (the pilot): nothing clipped, every norm recorded; a non-finite norm is counted apart;
    # the steady histogram starts at stats_from.
    v = keras.Variable(np.zeros(2, np.float32))
    free = DeblendAdam(0.1, None, stats_from=2)
    free.build([v])
    for g in ([1.0, 0.0], [10.0, 0.0], [100.0, 0.0], [np.nan, 0.0]):
        free.apply([ops.convert_to_tensor(np.array(g, np.float32))], [v])
    st = free.stats()
    assert st["clipped_frac"] == 0.0 and st["updates"] == 3 and st["grad_norm_nonfinite"] == 1
    h = free.run_histograms()
    assert h["all"]["n"] == 3 and h["steady"]["n"] == 1 and h["steady"]["q50"] == pytest.approx(100.0, rel=0.07)


def test_lambda_d1_ramps_from_one_to_its_target_after_the_measurements():
    from dp2_cutouts.deblend.models import C2Model
    from dp2_cutouts.deblend.train import scaled_protocol
    m = C2Model(keras.layers.Dense(1), keras.layers.Dense(1), [1.0], d=4, lambda_auto=True,
                protocol=dict(warm=10, measure=5, ramp=10))
    m.lambda_target.assign(3.0)
    got = []
    for step in (0, 15, 20, 25, 40):
        m.n_updates.assign(float(step))
        got.append(float(ops.convert_to_numpy(m.current_lambda())))
    assert got == pytest.approx([1.0, 1.0, 2.0, 3.0, 3.0])
    fixed = C2Model(keras.layers.Dense(1), keras.layers.Dense(1), [1.0], d=4, lambda_d1=0.3)
    assert float(ops.convert_to_numpy(fixed.current_lambda())) == pytest.approx(0.3)
    p = scaled_protocol(dict(warm=500, measure=200, ramp=200, every=10, batches=2), 300)
    assert p["warm"] + p["measure"] + p["ramp"] <= 180 and p["every"] >= 1 and p["measure"] >= 1


def test_psf_ellipticity_turns_with_the_image():
    # An elliptical Gaussian rotated / flipped like the stamps (ml/data.aug_images): its measured distortion changes
    # exactly as data.psf_features transforms the catalogue e1, e2.
    from dp2_cutouts.deblend.data import psf_features
    from dp2_cutouts.ml.data import aug_images
    n = 41
    yy, xx = np.indices((n, n)) - (n - 1) / 2

    def e12(img):
        ixx, iyy, ixy = ((img * xx * xx).sum(), (img * yy * yy).sum(), (img * xx * yy).sum())
        return (ixx - iyy) / (ixx + iyy), 2 * ixy / (ixx + iyy)

    img = np.exp(-(xx ** 2 / (2 * 4.0 ** 2) + yy ** 2 / (2 * 2.5 ** 2) - 0.3 * xx * yy / 6.0))
    e1, e2 = e12(img)
    stats = dict(mean=[0.0] * 3, std=[1.0] * 3)
    for k in range(4):
        for fl in (False, True):
            r = aug_images(img[None, :, :, None], k, fl)[0, :, :, 0]
            want = psf_features(np.array([[[0.3, e1, e2]]], np.float32), [k], [fl], stats)[0]
            assert e12(r) == pytest.approx((want[1], want[2]), abs=1e-6), (k, fl)
            assert want[0] == pytest.approx(0.3)


def test_fixed_validation_draws_depend_on_the_row_only():
    from dp2_cutouts.deblend.data import fixed_draws
    spec = dict(a=("normal", (3,)), b=("uniform", (2, 4)))
    one = fixed_draws(np.array([5, 9, 2]), spec, salt=1)
    two = fixed_draws(np.array([2, 5]), spec, salt=1)
    np.testing.assert_array_equal(one["a"][0], two["a"][1])
    np.testing.assert_array_equal(one["b"][2], two["b"][0])
    assert np.all((one["b"] >= 0) & (one["b"] < 1))
    other = fixed_draws(np.array([5]), spec, salt=2)
    assert not np.allclose(other["a"][0], one["a"][0])


def test_gradient_norm_is_that_of_the_whole_batch_on_two_devices():
    # Data parallel (keras.distribution.DataParallel, two simulated CPU devices): the norm measured before clipping
    # is the norm of the gradient of the batch-mean loss, the same as on one device (no per-device clipping).
    import os
    import subprocess
    import sys
    code = """
import os, numpy as np, keras
from dp2_cutouts.deblend.optim import DeblendAdam
if os.environ["NDEV"] == "2":
    keras.distribution.set_distribution(keras.distribution.DataParallel(devices=keras.distribution.list_devices()[:2]))
keras.utils.set_random_seed(0)
m = keras.Sequential([keras.Input((5,)), keras.layers.Dense(3), keras.layers.Dense(1)])
opt = DeblendAdam(1e-3, None)
m.compile(optimizer=opt, loss="mse")
x = np.random.default_rng(0).normal(size=(8, 5)).astype("float32")
y = np.random.default_rng(1).normal(size=(8, 1)).astype("float32")
m.train_on_batch(x, y)
print("NORM", repr(opt.stats()["grad_norm_mean"]), len(keras.distribution.list_devices()))
"""
    got = {}
    for n in ("1", "2"):
        env = dict(os.environ, XLA_FLAGS=f"--xla_force_host_platform_device_count={n}", JAX_PLATFORMS="cpu",
                   KERAS_BACKEND="jax", NDEV=n)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("NORM ")), None)
        assert line is not None, out.stdout[-2000:] + out.stderr[-3000:]
        _, norm, ndev = line.split()
        assert ndev == n
        got[n] = float(norm)
    assert got["2"] == pytest.approx(got["1"], rel=1e-5) and got["1"] > 0
