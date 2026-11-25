# campaign.py
# Anna Preto
# AstroDeep Project
# Self-advancing Slurm campaign: the dependency graph of the deblending stages and evaluations, submitted a few
# jobs at a time instead of all at once. The 2 h dev QoS accepts only a handful of queued jobs per user, so a
# fixed chain of 10 jobs per training (80 jobs for the campaign) cannot be submitted.
#
# How it works:
# - `dp2-ml <cfg> deblend-campaign MODE ...` writes a campaign spec (<ml_root>/deblend/campaign/<name>.json) and
#   advances it once from the login node: it submits the nodes whose dependencies are met, with --after applied to
#   them (e.g. afterok:<d2-merge>).
# - Every job of the campaign runs `dp2-ml <cfg> deblend-advance` when it ends (deblend.slurm, cpu.slurm). That
#   call looks at every active campaign and submits what has become ready: the next stages once their dependencies
#   wrote DONE, and the same stage again when its training stopped on its time budget without finishing.
# - At most deblend.campaign.max_queued jobs of the user are kept in the GPU QoS (all jobs of the user in that QoS
#   count, not only the campaign's). A refused submission is retried at the next advance.
# - A node whose command raised an exception gets a failure marker (<campaign dir>/failed/<job name>) and is not
#   resubmitted until `deblend-advance retry <job name>`. A node is also given up after max_jobs_train /
#   max_jobs_eval submissions (walltime kills, node failures).
# Nothing here imports keras: the campaign is started from a login node.

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

STAGES = ("rubin_vae", "k1", "k2", "A", "B", "c1", "c2", "c3")
DEPENDS = dict(rubin_vae=(), k1=(), k2=("k1",), A=("k1",), B=("k1",), c1=(), c2=("c1",), c3=("c1", "c2"))
STUDENTS = ("A", "B", "c2", "c3")
# Evaluations and the trainings they need; None: every training of the campaign.
EVALS = dict(masking=("k1",), latent=("k1", "c1"), d1=None, d2=None, cost=None)
MODES = dict(pilot=("all", ("masking", "latent", "d1", "d2", "cost")),
             all=("all", ("masking", "latent", "d1", "d2", "cost")),
             seeds=("students", ("d1", "d2", "cost")),
             final=("none", ("d1", "d2", "cost")))
DEFAULTS = dict(max_queued=8, max_jobs_train=30, max_jobs_eval=2, qos="qos_gpu_a100-dev", lock_timeout_s=120)
PREFIX = "dbl."


# ----------------------------------------------------------------------------- names shared with train / evaluate
def deblend_dir(cfg):
    return cfg.root / "deblend"


def run_name(cfg, stage, seed=None):
    # deblend.run_suffix (e.g. _pilot) keeps short pilot runs apart from the real ones.
    base = cfg["deblend"]["runs"][stage] + str(cfg["deblend"].get("run_suffix") or "")
    return base if seed is None else f"{base}_s{int(seed)}"


def run_dir(cfg, stage, seed=None):
    return deblend_dir(cfg) / "runs" / run_name(cfg, stage, seed)


def eval_name(cfg, what, split, seed):
    return what + (f"_{split}" if split else "") + (f"_s{seed}" if seed is not None else "") + \
        str(cfg["deblend"].get("run_suffix") or "")


def eval_dir(cfg, what, split, seed):
    return deblend_dir(cfg) / "evaluations" / eval_name(cfg, what, split, seed)


def campaign_dir(cfg):
    return deblend_dir(cfg) / "campaign"


def settings(cfg):
    s = dict(DEFAULTS)
    s.update(cfg["deblend"].get("campaign", {}) or {})
    return s


def dep_seed(cfg, seed):
    # Seed of the runs a seeded stage depends on (train.dep_dir): shared unless replicate_dependencies.
    return seed if bool(cfg["deblend"].get("replicate_dependencies", False)) else None


# ----------------------------------------------------------------------------- the graph
def nodes(cfg, spec):
    # Ordered dict job name -> node. A node: kind (cpu | train | eval), args (dp2-ml arguments), deps (job names),
    # done, submit (False: must already exist, never submitted by this campaign), max_jobs.
    mode, seed, split = spec["mode"], spec.get("seed"), spec["split"]
    which, evals = MODES[mode]
    st = settings(cfg)
    out = {}
    out[PREFIX + "stats"] = dict(kind="cpu", args=["deblend-stats"], deps=[], submit=True,
                                 done=(deblend_dir(cfg) / "norm_stats.json").exists(), max_jobs=3)
    stats = [PREFIX + "stats"]

    def tname(stage, s):
        return PREFIX + run_name(cfg, stage, s)

    trained = {}       # stage -> job name of the run the evaluations of this campaign use
    if which == "students":
        ds = dep_seed(cfg, seed)
        for stage in STAGES:
            s = seed if stage in STUDENTS else ds
            trained[stage] = (stage, s)
    else:
        trained = {stage: (stage, None) for stage in STAGES}
    needed = set()     # (stage, seed) of every run some node depends on
    for stage in STAGES:
        _, s = trained[stage]
        needed.add((stage, s))
        own = which == "all" or (which == "students" and stage in STUDENTS)
        if own:
            ds = s if s is None else dep_seed(cfg, s)
            needed.update((k, ds) for k in DEPENDS[stage])
            deps = stats + [tname(k, ds) for k in DEPENDS[stage]]
            args = ["deblend-train", stage] + (["--seed", str(s)] if s is not None else []) + \
                (["--max-steps", str(spec["max_steps"])] if spec.get("max_steps") else [])
            out[tname(stage, s)] = dict(kind="train", args=args, deps=deps, submit=True,
                                        done=(run_dir(cfg, stage, s) / "DONE").exists(),
                                        max_jobs=2 if spec.get("max_steps") else int(st["max_jobs_train"]))
    # Runs this campaign relies on without training them (shared stages of `seeds`, everything for `final`).
    for stage, s in sorted(needed, key=lambda t: (STAGES.index(t[0]), -1 if t[1] is None else t[1])):
        name = tname(stage, s)
        if name not in out:
            out[name] = dict(kind="train", args=[], deps=[], submit=False,
                             done=(run_dir(cfg, stage, s) / "DONE").exists(), max_jobs=0)
    eval_seed = seed if which == "students" else None
    for what in evals:
        need = EVALS[what] if EVALS[what] is not None else STAGES
        args = ["deblend-eval", what, "--split", split] + (["--seed", str(eval_seed)] if eval_seed is not None else [])
        out[PREFIX + "eval_" + eval_name(cfg, what, split, eval_seed)] = dict(
            kind="eval", args=args, deps=stats + [tname(*trained[k]) for k in need], submit=True,
            done=(eval_dir(cfg, what, split, eval_seed) / "summary.json").exists(), max_jobs=int(st["max_jobs_eval"]))
    # Dependencies a node needs are always in the graph (external runs included above).
    for n in out.values():
        for d in n["deps"]:
            assert d in out, d
    return out


# ----------------------------------------------------------------------------- Slurm
def run_cmd(cmd, env=None):
    # (return code, stdout, stderr); tests replace it.
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def queue(runner, env):
    # Jobs of the user: list of (job id, name, qos). None if squeue failed (then nothing is submitted).
    user = env.get("USER") or env.get("LOGNAME") or ""
    rc, out, err = runner(["squeue", "-h", "-u", user, "-o", "%i|%j|%q|%T"], None)
    if rc != 0:
        print(f"[campaign] squeue failed ({rc}): {err.strip()}")
        return None
    jobs = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3:
            jobs.append((parts[0].split("_")[0], parts[1], parts[2]))
    return jobs


def _clean_env(env, extra=None, drop=()):
    # Environment for sbatch: without the SLURM_* of the job advancing the campaign (they would leak into the new
    # job), plus the campaign variables.
    e = {k: v for k, v in env.items() if not k.startswith("SLURM_") and k not in drop}
    e.update(extra or {})
    return e


def sbatch_cmd(node, name, spec_path, spec, env, dependency):
    scripts = env.get("DP2_ML_SCRIPTS")
    if not scripts:
        raise RuntimeError("DP2_ML_SCRIPTS is not set (submit_deblend.sh exports it)")
    dep = [f"--dependency={dependency}"] if dependency else []
    sets = " ".join(f"--set {shlex.quote(s)}" for s in spec.get("sets", []))
    jenv = dict(DEBLEND_CAMPAIGN=str(spec_path), DEBLEND_SET=sets)
    if node["kind"] == "cpu":
        cmd = ["sbatch", "--parsable", f"--account={env['DP2_CPU_ACCOUNT']}",
               f"--partition={env.get('DP2_CPU_PARTITION') or 'cpu_p1'}",
               f"--cpus-per-task={env.get('DP2_CPU_CPUS') or 10}", "--time=02:00:00", f"--job-name={name}"] + dep + \
              [f"{scripts}/cpu.slurm"] + node["args"]
        return cmd, _clean_env(env, jenv, drop=("SBATCH_QOS", "SBATCH_TIMELIMIT", "SBATCH_ACCOUNT"))
    extra = shlex.split(env.get("DP2_GPU_EXTRA") or "")
    cmd = ["sbatch", "--parsable", f"--account={env['DP2_GPU_ACCOUNT']}"] + extra + [f"--job-name={name}"] + dep + \
          [f"{scripts}/deblend.slurm"] + node["args"]
    return cmd, _clean_env(env, jenv)


def gpu_qos(env, st):
    for a in shlex.split(env.get("DP2_GPU_EXTRA") or ""):
        if a.startswith("--qos="):
            return a.split("=", 1)[1]
    return st["qos"]


# ----------------------------------------------------------------------------- state and lock
class Lock:
    # Exclusive file created atomically (O_EXCL works on Lustre, flock may not); a lock older than 10 min is stale.
    def __init__(self, path, timeout_s):
        self.path, self.timeout_s, self.held = Path(path), float(timeout_s), False

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {os.environ.get('SLURM_JOB_ID', '')}\n".encode())
                os.close(fd)
                self.held = True
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > 600:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - t0 > self.timeout_s:
                    return self
                time.sleep(2)

    def __exit__(self, *a):
        if self.held:
            self.path.unlink(missing_ok=True)


def _read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path, obj):
    tmp = Path(f"{path}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def failed_marker(cdir, name):
    return Path(cdir) / "failed" / name


def mark_failed(cfg, message):
    # Called by dp2-ml when a deblend command raised inside a campaign job.
    name = os.environ.get("SLURM_JOB_NAME", "")
    if not name.startswith(PREFIX):
        return
    p = failed_marker(campaign_dir(cfg), name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"job {os.environ.get('SLURM_JOB_ID', '?')} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{message}\n")


# ----------------------------------------------------------------------------- advancing
def spec_name(mode, seed, cfg):
    return mode + (f"_s{seed}" if seed is not None else "") + str(cfg["deblend"].get("run_suffix") or "")


def start(cfg, config_path, sets, mode, seed=None, max_steps=None, split="test", after=None,
          runner=run_cmd, env=None):
    if mode not in MODES:
        raise SystemExit(f"campaign mode must be one of {sorted(MODES)}")
    if mode == "seeds" and seed is None:
        raise SystemExit("deblend-campaign seeds needs --seed")
    for s in sets:
        if "," in s:
            raise SystemExit(f"--set values with a comma are not supported in a campaign: {s}")
    cdir = campaign_dir(cfg)
    cdir.mkdir(parents=True, exist_ok=True)
    name = spec_name(mode, seed, cfg)
    spec = dict(name=name, mode=mode, seed=seed, max_steps=max_steps, split=split,
                config=str(Path(config_path).resolve()), sets=list(sets), active=True,
                started=time.strftime("%Y-%m-%d %H:%M:%S"), attempts={}, jobs={})
    path = cdir / f"{name}.json"
    with Lock(cdir / ".lock", settings(cfg)["lock_timeout_s"]):
        # A fresh start retries the nodes of this campaign that had failed.
        for n in nodes(cfg, spec):
            failed_marker(cdir, n).unlink(missing_ok=True)
        _write(path, spec)
    print(f"[campaign] {name}: spec {path}")
    return advance(cfg, cdir, runner=runner, env=env, after={name: after} if after else None,
                   load=lambda p, s: cfg if Path(p).resolve() == Path(config_path).resolve() and s == list(sets)
                   else _load(p, s))


def _load(path, sets):
    from ..mlconfig import MLConfig
    return MLConfig.load(path, sets)


def advance(cfg, cdir=None, runner=run_cmd, env=None, after=None, load=_load, submit=True):
    # Advance every active campaign of <ml_root>/deblend/campaign. Returns {campaign: {job name: state}}.
    env = dict(os.environ if env is None else env)
    cdir = Path(cdir or campaign_dir(cfg))
    st = settings(cfg)
    specs = sorted(cdir.glob("*.json")) if cdir.exists() else []
    report = {}
    with Lock(cdir / ".lock", st["lock_timeout_s"]) as lk:
        if not lk.held:
            print("[campaign] another advance holds the lock: leaving it the work")
            submit = False
        jobs = queue(runner, env)   # also for `status`, so that queued jobs show as queued, not ready
        if jobs is None:
            submit, jobs = False, []
        me = env.get("SLURM_JOB_ID", "").split("_")[0]
        queued = {n for i, n, _ in jobs if i != me}
        qos = gpu_qos(env, st)
        n_gpu = sum(1 for i, _, q in jobs if q == qos and i != me)
        gpu_blocked = False
        for path in specs:
            spec = _read(path, None)
            if not spec or not spec.get("active", True):
                continue
            c = load(spec["config"], spec.get("sets", []))
            graph = nodes(c, spec)
            states = {}
            for name, node in graph.items():
                tries = int(spec["attempts"].get(name, 0))
                if node["done"]:
                    states[name] = "done"
                elif failed_marker(cdir, name).exists():
                    states[name] = "FAILED"
                elif not node["submit"]:
                    states[name] = "missing"
                elif name in queued:
                    states[name] = "queued"
                elif tries >= node["max_jobs"]:
                    states[name] = f"gave up after {tries} jobs"
                else:
                    states[name] = None
            for name, node in graph.items():
                if states[name] is not None:
                    continue
                blockers = [d for d in node["deps"] if states[d] != "done"]
                if any(states[d] in ("FAILED", "missing", "blocked") or str(states[d]).startswith("gave up")
                       for d in blockers):
                    states[name] = "blocked"
                    continue
                if blockers:
                    states[name] = "waiting"
                    continue
                if not submit:
                    states[name] = "ready"
                    continue
                if node["kind"] != "cpu" and (gpu_blocked or n_gpu >= int(st["max_queued"])):
                    states[name] = "deferred (queue full)"
                    continue
                dep = (after or {}).get(spec["name"])
                cmd, jenv = sbatch_cmd(node, name, path, spec, env, dep)
                rc, out, err = runner(cmd, jenv)
                if rc != 0:
                    print(f"[campaign] sbatch {name} refused: {err.strip()}")
                    states[name] = "deferred (sbatch refused)"
                    if node["kind"] != "cpu":
                        gpu_blocked = True
                    continue
                jid = out.strip().split(";")[0]
                spec["attempts"][name] = int(spec["attempts"].get(name, 0)) + 1
                spec["jobs"].setdefault(name, []).append(jid)
                states[name] = f"submitted {jid}"
                if node["kind"] != "cpu":
                    n_gpu += 1
            if submit:
                if all(s == "done" for n, s in states.items() if graph[n]["submit"]):
                    spec["active"] = False
                    spec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _write(path, spec)
            report[spec["name"]] = states
    for camp, states in report.items():
        print(f"[campaign] {camp}")
        for name, s in states.items():
            print(f"    {name:<42s} {s}")
    return report


def retry(cfg, name):
    # Remove the failure marker of a node and give it a fresh job budget; the next advance resubmits it.
    cdir = campaign_dir(cfg)
    failed_marker(cdir, name).unlink(missing_ok=True)
    for path in cdir.glob("*.json"):
        spec = _read(path, None)
        if spec and name in spec.get("attempts", {}):
            spec["attempts"][name] = 0
            spec["active"] = True
            _write(path, spec)


def stop(cfg, campaign):
    # Make a campaign inactive: its queued jobs stay (scancel them), nothing new is submitted for it.
    path = campaign_dir(cfg) / f"{campaign}.json"
    spec = _read(path, None)
    if spec is None:
        raise SystemExit(f"no campaign {campaign} in {campaign_dir(cfg)}")
    spec["active"] = False
    _write(path, spec)
    print(f"[campaign] {campaign} stopped; its queued jobs: {spec.get('jobs')}")
