# speed_probe.py
# Anna Preto
# AstroDeep Project
# Can the stages that decode images (C1, the VAEs) go faster on the V100? deblend-bench (2026-11-27) found C1 already
# GPU-bound at batch 32 (~280 examples per second), with XLA's convolution autotuning switched off (level 0: on the
# V100 some cuDNN engines fail while being profiled, env.sh). For each stage, in a fresh process per configuration,
# on one real validation batch: ms per optimisation step and whether the loss stays finite, for
#   autotune level 0 (what env.sh sets), 1, 2, 3 (level 4, XLA's default, fails on the V100)
#   precision float32, and mixed_float16 (V100 tensor cores; Keras scales the loss) at the fastest working level
# The loss of the first step is compared between precisions (same initial weights).
# Usage (on a GPU node): python scripts/inspect/speed_probe.py <ML config> [--stages c1 k1] [--batch 64]
#            [--set deblend.run_suffix=_pilot]

import argparse
import json
import os
import re
import subprocess
import sys
import time

LEVELS = (0, 1, 2, 3)


def child(args):
    os.environ.setdefault("KERAS_BACKEND", "jax")
    out = dict(stage=args.stage, precision=args.precision, level=args.level)
    try:
        import keras
        import numpy as np
        if args.precision == "mixed_float16":
            keras.mixed_precision.set_global_policy("mixed_float16")
        keras.utils.set_random_seed(0)
        from dp2_cutouts.deblend import train
        from dp2_cutouts.mlconfig import MLConfig
        cfg = MLConfig.load(args.config, args.set)
        s = train.settings(cfg, args.stage)
        s["train"].update(batch_size=args.batch, val_max_examples=args.batch)
        s["model"]["stats_batches"] = 1
        if s["model"].get("lambda_d1") == "auto":
            s["model"]["lambda_d1"] = 1.0
        deps = {k: train.dep_dir(cfg, k) for k in train.DEPENDS[args.stage]}
        model, (_, va), _, _ = train.build(cfg, args.stage, s, 0, deps)
        x, _ = va[0]
        y = np.zeros((len(next(iter(x.values()))),), np.float32)
        model.compile(optimizer=train.optimizer(1e-4, 1.0))

        def step():
            logs = model.train_on_batch(x, y, return_dict=True)
            return float(logs.get("total_loss", next(iter(logs.values()))))

        first = step()
        t0 = time.perf_counter()
        step()
        step()
        out["compile_s"] = time.perf_counter() - t0
        losses = []
        t0 = time.perf_counter()
        for _ in range(10):
            losses.append(step())
        dt = (time.perf_counter() - t0) / 10
        out.update(ok=bool(np.isfinite([first] + losses).all()), first_loss=first, last_loss=losses[-1],
                   ms_per_step=1e3 * dt, examples_per_s=args.batch / dt)
    except Exception as e:  # noqa: BLE001 - the point is to report it
        out.update(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")
    print("RESULT " + json.dumps(out), flush=True)


def run_one(args, stage, precision, level):
    flags = re.sub(r"--xla_gpu_autotune_level=\d+", "", os.environ.get("XLA_FLAGS", "")).strip()
    env = dict(os.environ, XLA_FLAGS=f"{flags} --xla_gpu_autotune_level={level}".strip())
    cmd = [sys.executable, __file__, args.config, "--child", "--stage", stage, "--precision", precision,
           "--level", str(level), "--batch", str(args.batch)] + sum([["--set", s] for s in args.set], [])
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    r = json.loads(line[7:]) if line else dict(ok=False, error=f"crashed (code {p.returncode}): "
                                                                 f"{p.stderr.strip()[-400:]}")
    msg = (f"{r['ms_per_step']:8.1f} ms/step {r['examples_per_s']:7.0f} ex/s  first loss {r['first_loss']:.6g}"
           if r.get("ms_per_step") else "")
    print(f"  {stage:9s} {precision:14s} level {level}: {'OK    ' if r.get('ok') else 'FAILED'} {msg} "
          f"{r.get('error', '')}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--stages", nargs="+", default=["c1", "k1"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--stage")
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--level", type=int, default=0)
    args = ap.parse_args()
    if args.child:
        return child(args)
    res = []
    for stage in args.stages:
        print(f"{stage} (batch {args.batch}):", flush=True)
        r32 = [run_one(args, stage, "float32", lv) for lv in LEVELS]
        res += r32
        good = [r for r in r32 if r.get("ok")]
        best = min(good, key=lambda r: r["ms_per_step"])["level"] if good else 0
        r16 = run_one(args, stage, "mixed_float16", best)
        res.append(r16)
        if r16.get("ok") and good:
            ref = next(r for r in r32 if r["level"] == best)
            rel = abs(r16["first_loss"] - ref["first_loss"]) / max(abs(ref["first_loss"]), 1e-12)
            print(f"  -> fastest float32: level {best}; mixed_float16 x{ref['ms_per_step'] / r16['ms_per_step']:.2f}"
                  f" faster, first-step loss differs by {rel:.2e} (relative)", flush=True)
    print("\nRESULTS " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
