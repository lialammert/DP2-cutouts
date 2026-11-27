# bench.py
# Anna Preto
# AstroDeep Project
# How fast can each stage go, and what limits it (dp2-ml <cfg> deblend-bench, on one GPU node):
#   GPU     for every stage and batch size: optimisation steps on one batch already in memory (no reading), after
#           a warm-up (compilation): ms per step, examples per second, GPU memory. The batch size from which the
#           examples per second stop growing is where the GPU is saturated; OOM ends the list.
#   loader  for D1 and D2 (training loaders, one batch size): examples per second with the job's workers threads,
#           reading the original arrays, the compact copy memory-mapped, and the compact copy in memory (when it
#           exists and fits).
# A stage trains as fast as min(GPU, loader). Trained stages it depends on are loaded (use the pilot's runs:
# --set deblend.run_suffix=_pilot); a stage whose dependencies are not trained is skipped.
# Writes <ml_root>/deblend/bench/bench_<host>_<time>.json.

import gc
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import keras
import numpy as np

from ..mlconfig import MLConfig
from . import compact, data, train

BATCH_SIZES = (32, 64, 128, 256, 512, 1024, 2048)


def _cfg(cfg, **data_opts):
    raw = json.loads(json.dumps(cfg.raw))
    raw["deblend"]["data"] = dict(raw["deblend"].get("data") or {}, **data_opts)
    return MLConfig(raw=raw, path=cfg.path)


def _gpu_memory():
    try:
        import jax
        st = jax.devices()[0].memory_stats() or {}
        return dict(in_use_gb=st.get("bytes_in_use", 0) / 1e9, peak_gb=st.get("peak_bytes_in_use", 0) / 1e9,
                    limit_gb=st.get("bytes_limit", 0) / 1e9)
    except Exception:  # noqa: BLE001 - CPU backend or no memory stats
        return {}


def _device():
    try:
        import jax
        d = jax.devices()[0]
        return dict(platform=d.platform, kind=getattr(d, "device_kind", str(d)), n=len(jax.devices()))
    except Exception:  # noqa: BLE001
        return dict(platform="?")


def _take(x, b):
    # The first b examples of a batch, tiled when it holds fewer.
    n = len(next(iter(x.values())))
    idx = np.arange(b) % n
    return {k: v[idx] for k, v in x.items()}


def gpu_stage(cfg, stage, sizes, steps):
    deps = {}
    for k in train.DEPENDS[stage]:
        d = train.run_dir(cfg, k)
        if not (d / "DONE").exists():
            return dict(skipped=f"{k} not trained ({d})")
        deps[k] = d
    s = train.settings(cfg, stage)
    s["train"]["batch_size"] = int(max(sizes))
    s["train"]["val_max_examples"] = int(max(sizes))
    s["model"]["stats_batches"] = 1                      # B, C3 latent standardisation: one batch is enough here
    if s["model"].get("lambda_d1") == "auto":
        s["model"]["lambda_d1"] = 1.0
    model, (tr, va), _, subs = train.build(cfg, stage, s, int(cfg["deblend"]["seed"]), deps)
    x_all, _ = va[0]
    model.compile(optimizer=train.optimizer(s["train"]["learning_rate"], s["train"].get("clipnorm")))
    out = dict(n_params={k: int(v.count_params()) for k, v in subs.items()}, sizes={})
    for b in sorted(sizes):
        x, y = _take(x_all, b), np.zeros((b,), np.float32)
        try:
            for _ in range(3):                           # compilation for this shape, then warm
                model.train_on_batch(x, y)
            t0 = time.perf_counter()
            for _ in range(steps):
                model.train_on_batch(x, y)               # returns host floats: the step has finished
            dt = (time.perf_counter() - t0) / steps
        except Exception as e:  # noqa: BLE001 - out of memory ends the list for this stage
            msg = str(e)
            oom = "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()
            out["sizes"][b] = dict(error="OOM" if oom else msg[:300])
            print(f"  {stage} batch {b}: {'out of GPU memory' if oom else msg[:300]}", flush=True)
            break
        r = dict(ms_per_step=1e3 * dt, examples_per_s=b / dt, **_gpu_memory())
        out["sizes"][b] = r
        print(f"  {stage} batch {b:5d}: {r['ms_per_step']:8.1f} ms/step {r['examples_per_s']:9.0f} ex/s"
              f"  GPU memory peak {r.get('peak_gb', float('nan')):.1f} GB", flush=True)
    del model, tr, va, x_all
    keras.backend.clear_session()
    gc.collect()
    return out


def loader_rate(ds, n_batches, workers):
    n = int(n_batches)
    t0 = time.perf_counter()
    ds[0]
    one = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(lambda i: ds[(i + 1) % len(ds)], range(n)):
            pass
    dt = time.perf_counter() - t0
    return dict(examples_per_s=n * ds.batch_size / dt, s_per_batch_one_thread=one, workers=workers,
                batches=n)


def loaders(cfg, b, n_batches, workers):
    out = {}
    have = {w: compact.read_info(cfg, w) is not None for w in ("d1", "d2")}
    modes = [("original", dict(compact=False))]
    modes.append(("compact, memory-mapped", dict(compact=True, require_compact=False, in_memory=[])))
    modes.append(("compact, in memory", dict(compact=True, require_compact=False, in_memory=["d1", "d2"])))
    for which in ("d1", "d2"):
        for label, opts in modes:
            if opts["compact"] and not have[which]:
                out[f"{which} {label}"] = dict(skipped="no compact copy (dp2-ml deblend-compact)")
                continue
            if label == "compact, in memory" and which == "d1":
                continue                                 # 1.4 M rows: never in memory
            c = _cfg(cfg, **opts)
            t0 = time.perf_counter()
            ds = (data.D1Batches if which == "d1" else data.D2Batches)(c, "train", b, True, seed=0,
                                                                         block=int(cfg["deblend"]["train"].get(
                                                                             "read_block", 1)))
            setup = time.perf_counter() - t0
            r = dict(loader_rate(ds, n_batches, workers), setup_s=setup)
            out[f"{which} {label}"] = r
            print(f"  {which} {label:24s}: {r['examples_per_s']:8.0f} ex/s with {workers} threads "
                  f"({r['s_per_batch_one_thread']:.2f} s per batch of {b} on one thread; set-up {setup:.0f} s)",
                  flush=True)
            del ds
            gc.collect()
    return out


def run(cfg, stages=None, sizes=None, loader_batches=30, steps=20):
    stages = list(stages or train.STAGES)
    sizes = [int(b) for b in (sizes or BATCH_SIZES)]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or cfg["deblend"]["train"].get("workers", 8))
    lim = compact.memory_limit_bytes()
    res = dict(host=socket.gethostname(), device=_device(), workers=workers,
               memory_limit_gb=lim / 1e9 if lim else None, stages={}, loaders={})
    print(f"deblend bench on {res['host']}: {res['device']}, {workers} CPU threads, memory limit "
          f"{res['memory_limit_gb'] or float('nan'):.0f} GB", flush=True)
    # The GPU part reads one batch per stage: no copy into memory.
    cg = _cfg(cfg, require_compact=False, in_memory=[])
    for st in stages:
        print(f"GPU, stage {st}:", flush=True)
        res["stages"][st] = gpu_stage(cg, st, sizes, steps)
        if "skipped" in res["stages"][st]:
            print(f"  skipped: {res['stages'][st]['skipped']}", flush=True)
    b = int(cfg["deblend"]["train"]["batch_size"])
    print(f"loaders (training, batch {b}, {loader_batches} batches):", flush=True)
    res["loaders"] = loaders(cfg, b, loader_batches, workers)
    print("\nsummary: examples per second, GPU (best batch size) against the loaders", flush=True)
    for st, r in res["stages"].items():
        ok = {bs: v for bs, v in r.get("sizes", {}).items() if "examples_per_s" in v}
        if not ok:
            continue
        best = max(v["examples_per_s"] for v in ok.values())
        sat = min(bs for bs, v in ok.items() if v["examples_per_s"] >= 0.9 * best)
        r["saturation_batch"] = sat
        print(f"  {st:9s}: GPU up to {best:8.0f} ex/s, 90 % of it from batch {sat}", flush=True)
    out = data.deblend_dir(cfg) / "bench"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"bench_{res['host']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(res, indent=2, default=str))
    print(f"-> {p}")
    return res
