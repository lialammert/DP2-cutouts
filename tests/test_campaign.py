# test_campaign.py
# Anna Preto
# AstroDeep Project
# The self-advancing Slurm campaign of the deblending stages, against a fake squeue / sbatch: what is submitted
# when, the cap on queued jobs, refused submissions, failure markers, resubmission of a training stopped on its time
# budget, the shared runs of the seed replicates, and the end of a campaign.

import json

import pytest

from dp2_cutouts.deblend import campaign as C
from dp2_cutouts.mlconfig import MLConfig

QOS = "qos_gpu_a100-dev"


def _cfg(root, suffix=""):
    raw = dict(ml_root=str(root), deblend=dict(run_suffix=suffix, replicate_dependencies=False,
                                               runs={k: f"r_{k}" for k in C.STAGES},
                                               campaign=dict(max_queued=8, max_jobs_train=3, max_jobs_eval=2,
                                                             lock_timeout_s=1, gates={})))
    return MLConfig(raw=raw, path=root / "cfg.yaml")


class FakeSlurm:
    def __init__(self):
        self.queue = []          # (id, name, qos)
        self.submitted = []      # (name, cmd, env)
        self.refuse = False
        self.next_id = 100

    def __call__(self, cmd, env):
        if cmd[0] == "squeue":
            return 0, "".join(f"{i}|{n}|{q}|PENDING\n" for i, n, q in self.queue), ""
        assert cmd[0] == "sbatch"
        if self.refuse:
            return 1, "", "sbatch: error: QOSMaxSubmitJobPerUserLimit"
        name = next(a.split("=", 1)[1] for a in cmd if a.startswith("--job-name="))
        self.next_id += 1
        self.submitted.append((name, cmd, env))
        self.queue.append((str(self.next_id), name, "cpu" if "/s/cpu.slurm" in cmd else QOS))
        return 0, f"{self.next_id}\n", ""

    def names(self):
        return [n for n, _, _ in self.submitted]


ENV = dict(USER="me", DP2_ML_SCRIPTS="/s", DP2_GPU_ACCOUNT="fqn@a100", DP2_CPU_ACCOUNT="fqn@cpu",
           SLURM_JOB_ID="7", SLURM_CPUS_PER_TASK="8")


def _done(cfg, stage, seed=None):
    d = C.run_dir(cfg, stage, seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "DONE").write_text("done")


def _advance(cfg, slurm, env=ENV):
    return C.advance(cfg, runner=slurm, env=env, load=lambda p, s: cfg)


def test_pilot_starts_with_the_statistics_only_and_waits_for_the_datasets(tmp_path):
    cfg = _cfg(tmp_path, "_pilot")
    slurm = FakeSlurm()
    env = {k: v for k, v in ENV.items() if k != "SLURM_JOB_ID"}
    rep = C.start(cfg, tmp_path / "cfg.yaml", ["deblend.run_suffix=_pilot"], "pilot", max_steps=300, split="val",
                  after="afterok:42", runner=slurm, env=env)
    assert slurm.names() == ["dbl.stats"]
    _, cmd, jenv = slurm.submitted[0]
    assert "--dependency=afterok:42" in cmd and cmd[-2:] == ["/s/cpu.slurm", "deblend-stats"]
    assert "--account=fqn@cpu" in cmd and jenv["DEBLEND_SET"] == "--set deblend.run_suffix=_pilot"
    assert jenv["DEBLEND_CAMPAIGN"].endswith("pilot_pilot.json") and "SLURM_CPUS_PER_TASK" not in jenv
    states = rep["pilot_pilot"]
    assert states["dbl.r_k1_pilot"] == "waiting" and states["dbl.eval_d2_val_pilot"] == "waiting"


def test_ready_stages_are_submitted_under_the_queue_cap(tmp_path):
    cfg = _cfg(tmp_path, "_pilot")
    slurm = FakeSlurm()
    C.start(cfg, tmp_path / "cfg.yaml", [], "pilot", max_steps=300, split="val", runner=slurm, env=ENV)
    # The statistics job is still queued: nothing else yet, and `status` shows it as queued.
    slurm.queue = [("101", "dbl.stats", "qos_cpu-t3")]
    _advance(cfg, slurm)
    assert slurm.names() == ["dbl.stats"]
    rep = C.advance(cfg, runner=slurm, env=ENV, load=lambda p, s: cfg, submit=False)
    assert rep["pilot_pilot"]["dbl.stats"] == "queued"
    # Statistics done; 6 other jobs of the user fill the dev QoS: only 2 of the 3 root trainings go in.
    (C.deblend_dir(cfg) / "norm_stats.json").write_text("{}")
    slurm.queue = [(str(i), f"other{i}", QOS) for i in range(6)]
    rep = _advance(cfg, slurm)
    assert slurm.names()[1:] == ["dbl.r_rubin_vae_pilot", "dbl.r_k1_pilot"]
    assert rep["pilot_pilot"]["dbl.r_c1_pilot"] == "deferred (queue full)"
    _, cmd, _ = slurm.submitted[1]
    assert cmd[-5:] == ["/s/deblend.slurm", "deblend-train", "rubin_vae", "--max-steps", "300"]
    assert not any(a.startswith("--dependency") for a in cmd)
    # A refused submission is not counted as an attempt and is retried later.
    slurm.queue = [(str(i), f"other{i}", QOS) for i in range(3)] + [("102", "dbl.r_rubin_vae_pilot", QOS),
                                                                    ("103", "dbl.r_k1_pilot", QOS)]
    slurm.refuse = True
    rep = _advance(cfg, slurm)
    assert rep["pilot_pilot"]["dbl.r_c1_pilot"] == "deferred (sbatch refused)"
    slurm.refuse = False
    _advance(cfg, slurm)
    assert slurm.names()[-1] == "dbl.r_c1_pilot"
    spec = json.loads((C.campaign_dir(cfg) / "pilot_pilot.json").read_text())
    assert spec["attempts"]["dbl.r_c1_pilot"] == 1


def test_dependencies_failures_and_time_budget_resubmission(tmp_path):
    cfg = _cfg(tmp_path)
    slurm = FakeSlurm()
    (C.deblend_dir(cfg)).mkdir(parents=True)
    (C.deblend_dir(cfg) / "norm_stats.json").write_text("{}")
    C.start(cfg, tmp_path / "cfg.yaml", [], "all", split="test", runner=slurm, env=ENV)
    assert set(slurm.names()) == {"dbl.r_rubin_vae", "dbl.r_k1", "dbl.r_c1"}
    # k1 finishes; c1 raised an exception in its job (failure marker).
    _done(cfg, "k1")
    env = dict(ENV, SLURM_JOB_NAME="dbl.r_c1", SLURM_JOB_ID="555")
    import os
    old = dict(os.environ)
    os.environ.update(env)
    try:
        C.mark_failed(cfg, "Traceback: out of memory")
    finally:
        os.environ.clear()
        os.environ.update(old)
    slurm.queue = [q for q in slurm.queue if q[1] == "dbl.r_rubin_vae"]
    assert slurm.queue[0][0] == "101"
    rep = _advance(cfg, slurm)["all"]
    assert rep["dbl.r_c1"] == "FAILED" and rep["dbl.r_c2"] == "blocked" and rep["dbl.r_c3"] == "blocked"
    assert rep["dbl.eval_latent_test"] == "blocked" and rep["dbl.eval_d2_test"] == "blocked"
    assert {"dbl.r_k2", "dbl.r_A", "dbl.r_B", "dbl.eval_masking_test"} <= set(slurm.names())
    # The rubin_vae job stopped on its time budget (no DONE): when it advances the campaign, it resubmits itself.
    n = len(slurm.names())
    _advance(cfg, slurm, dict(ENV, SLURM_JOB_ID="101"))
    assert slurm.names()[n:] == ["dbl.r_rubin_vae"]
    # ... until max_jobs_train (3) jobs have been used (here every job ends without finishing).
    for _ in range(3):
        slurm.queue = []
        rep = _advance(cfg, slurm)["all"]
    assert rep["dbl.r_rubin_vae"] == "gave up after 3 jobs"
    # retry clears the marker and the job budget.
    C.retry(cfg, "dbl.r_c1")
    rep = _advance(cfg, slurm)["all"]
    assert rep["dbl.r_c1"].startswith("submitted")


def test_seed_replicates_use_the_shared_runs_without_training_them(tmp_path):
    cfg = _cfg(tmp_path)
    slurm = FakeSlurm()
    (C.deblend_dir(cfg)).mkdir(parents=True)
    (C.deblend_dir(cfg) / "norm_stats.json").write_text("{}")
    rep = C.start(cfg, tmp_path / "cfg.yaml", [], "seeds", seed=1, runner=slurm, env=ENV)["seeds_s1"]
    # Nothing shared is trained by this campaign: it waits for them.
    assert rep["dbl.r_k1"] == "missing" and rep["dbl.r_A_s1"] == "blocked" and rep["dbl.r_c3_s1"] == "blocked"
    assert slurm.names() == []
    for stage in ("rubin_vae", "k1", "k2", "c1", "c2"):
        _done(cfg, stage)
    rep = _advance(cfg, slurm)["seeds_s1"]
    assert set(slurm.names()) == {"dbl.r_A_s1", "dbl.r_B_s1", "dbl.r_c2_s1", "dbl.r_c3_s1"}
    _, cmd, _ = slurm.submitted[0]
    assert cmd[-4:] == ["deblend-train", "A", "--seed", "1"]
    assert rep["dbl.eval_d2_test_s1"] == "waiting"
    for stage in ("A", "B", "c2", "c3"):
        _done(cfg, stage, 1)
    slurm.queue = []
    _advance(cfg, slurm)
    assert {"dbl.eval_d1_test_s1", "dbl.eval_d2_test_s1", "dbl.eval_cost_test_s1"} <= set(slurm.names())
    _, cmd, _ = slurm.submitted[-1]
    assert cmd[-6:] == ["deblend-eval", "cost", "--split", "test", "--seed", "1"]


def test_a_finished_campaign_becomes_inactive(tmp_path):
    cfg = _cfg(tmp_path)
    slurm = FakeSlurm()
    (C.deblend_dir(cfg)).mkdir(parents=True)
    (C.deblend_dir(cfg) / "norm_stats.json").write_text("{}")
    for stage in C.STAGES:
        _done(cfg, stage)
    C.start(cfg, tmp_path / "cfg.yaml", [], "final", split="holdout", runner=slurm, env=ENV)
    assert set(slurm.names()) == {"dbl.eval_d1_holdout", "dbl.eval_d2_holdout", "dbl.eval_cost_holdout"}
    for what in ("d1", "d2", "cost"):
        d = C.eval_dir(cfg, what, "holdout", None)
        d.mkdir(parents=True)
        (d / "summary.json").write_text("{}")
    _advance(cfg, slurm)
    assert not json.loads((C.campaign_dir(cfg) / "final.json").read_text())["active"]
    assert _advance(cfg, slurm) == {}


def test_unknown_mode_and_comma_in_overrides_are_refused(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(SystemExit):
        C.start(cfg, tmp_path / "cfg.yaml", [], "everything", runner=FakeSlurm(), env=ENV)
    with pytest.raises(SystemExit):
        C.start(cfg, tmp_path / "cfg.yaml", ["deblend.x=[1,2]"], "all", runner=FakeSlurm(), env=ENV)


def test_multi_gpu_stages_ask_for_their_gpus_and_cores(tmp_path):
    # deblend.<stage>.train.gpus: the training job asks for that many GPUs and their share of the cores, after
    # DP2_GPU_EXTRA (whose --cpus-per-task it overrides); the other stages keep the default request.
    cfg = _cfg(tmp_path, "_pilot")
    cfg.raw["deblend"]["c1"] = dict(train=dict(gpus=4))
    slurm = FakeSlurm()
    env = dict(ENV, DP2_GPU_EXTRA="-C v100-32g --qos=qos_gpu-t3 --cpus-per-task=10")
    C.start(cfg, tmp_path / "cfg.yaml", [], "pilot", max_steps=300, split="val", runner=slurm, env=env)
    (C.deblend_dir(cfg) / "norm_stats.json").write_text("{}")
    slurm.queue = []
    _advance(cfg, slurm, env)
    cmds = {n: c for n, c, _ in slurm.submitted}
    c1, vae = cmds["dbl.r_c1_pilot"], cmds["dbl.r_rubin_vae_pilot"]
    assert "--gres=gpu:4" in c1 and c1.index("--cpus-per-task=40") > c1.index("--cpus-per-task=10")
    assert not any(a.startswith("--gres") for a in vae) and "--cpus-per-task=40" not in vae


def test_gated_stages_wait_for_the_approval_of_their_vaes(tmp_path):
    # Mode all: K2, A, B wait for an approved K1 (A, B also for K2), C2 for an approved C1; the checkpoint
    # evaluations run on val once the VAEs are done. Pilots have no gates.
    cfg = _cfg(tmp_path)
    cfg.raw["deblend"]["campaign"]["gates"] = None           # the default gates
    slurm = FakeSlurm()
    (C.deblend_dir(cfg)).mkdir(parents=True, exist_ok=True)
    (C.deblend_dir(cfg) / "norm_stats.json").write_text("{}")
    for st in ("rubin_vae", "k1", "c1"):
        _done(cfg, st)
    rep = C.start(cfg, tmp_path / "cfg.yaml", [], "all", runner=slurm, env=ENV)["all"]
    assert rep["dbl.r_k2"].startswith("awaiting approval of k1") and rep["dbl.r_c2"].startswith("awaiting approval")
    assert rep["dbl.r_A"].startswith("awaiting approval of k1, k2")   # K2 not even trained yet
    assert "dbl.eval_check_vaes_val" in slurm.names()
    with pytest.raises(SystemExit):
        C.approve(cfg, "k2")                                 # not finished
    C.approve(cfg, "k1", note="checked")
    assert json.loads((C.run_dir(cfg, "k1") / "APPROVED").read_text())["note"] == "checked"
    slurm.queue = []
    _advance(cfg, slurm)
    assert "dbl.r_k2" in slurm.names() and "dbl.r_c2" not in slurm.names()
    _done(cfg, "k2")
    rep = _advance(cfg, slurm)["all"]
    assert rep["dbl.r_A"].startswith("awaiting approval of k2")
    pilot = _cfg(tmp_path / "p", "_pilot")
    nodes = C.nodes(pilot, dict(mode="pilot", seed=None, split="val", max_steps=300))
    assert all(not n.get("gates") for n in nodes.values())
