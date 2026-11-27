# gpu_probe.py
# Anna Preto
# AstroDeep Project
# Does the dp2-ml virtual environment train on this GPU, and if not, what makes it work? On 2026-11-22 the venv built
# on the A100 failed on the V100 at the first Conv2D (cuDNN "EXECUTION_FAILED / status 5003"). Prints the GPU,
# driver and library versions, then runs, each in a fresh process with its own XLA_FLAGS, a dense step and a
# convolutional training step shaped like the campaign's encoders (Rubin 59 x 59 x 6, VIS 119 x 119 x 2):
#   default                     the venv as is
#   no_autotune                 --xla_gpu_autotune_level=0 (no cuDNN algorithm search)
#   no_triton                   --xla_gpu_enable_triton_gemm=false (Triton GEMM fusions, built for recent GPUs)
#   no_cudnn_fusion             --xla_gpu_enable_cudnn_fmha=false --xla_gpu_cudnn_gemm_fusion_level=0
#   all                         the three together
# Usage (on a GPU node): python scripts/inspect/gpu_probe.py

import json
import os
import subprocess
import sys

CONFIGS = {
    "default": "",
    "no_autotune": "--xla_gpu_autotune_level=0",
    "no_triton": "--xla_gpu_enable_triton_gemm=false",
    "no_cudnn_fusion": "--xla_gpu_enable_cudnn_fmha=false --xla_gpu_cudnn_gemm_fusion_level=0",
    "all": "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cudnn_fmha=false "
           "--xla_gpu_cudnn_gemm_fusion_level=0",
}


def child():
    os.environ.setdefault("KERAS_BACKEND", "jax")
    out = {}
    try:
        import jax
        out["jax"] = jax.__version__
        out["devices"] = [str(d) for d in jax.devices()]
        import numpy as np
        a = jax.numpy.ones((512, 512))
        out["dense"] = float((a @ a).sum())
        import keras
        from keras import layers
        for name, shape in (("rubin", (59, 59, 6)), ("vis", (119, 119, 2))):
            inp = keras.Input(shape)
            h = inp
            for f in (32, 64, 128):
                h = layers.Conv2D(f, 3, padding="same")(h)
                h = layers.PReLU()(h)
                h = layers.Conv2D(f, 3, strides=2, padding="same")(h)
                h = layers.PReLU()(h)
            h = layers.Flatten()(h)
            o = layers.Dense(64)(h)
            m = keras.Model(inp, o)
            m.compile(optimizer="adam", loss="mse")
            x = np.random.default_rng(0).normal(size=(64, *shape)).astype("float32")
            y = np.zeros((64, 64), "float32")
            losses = [float(m.train_on_batch(x, y)) for _ in range(3)]
            out[f"conv_{name}"] = losses[-1]
        out["ok"] = True
    except Exception as e:  # noqa: BLE001 - the point is to report it
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    print("RESULT " + json.dumps(out), flush=True)


def main():
    if "--child" in sys.argv:
        return child()
    for cmd in (["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                [sys.executable, "-m", "pip", "list", "--format=freeze"]):
        try:
            txt = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
        except Exception as e:  # noqa: BLE001
            txt = str(e)
        if "pip" in cmd:
            txt = "\n".join(line for line in txt.splitlines()
                            if any(k in line.lower() for k in ("jax", "keras", "nvidia", "numpy", "ml_dtypes")))
        print(txt, flush=True)
    res = {}
    for name, flags in CONFIGS.items():
        env = dict(os.environ, XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " " + flags).strip())
        p = subprocess.run([sys.executable, __file__, "--child"], env=env, capture_output=True, text=True,
                           timeout=1200)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        r = json.loads(line[7:]) if line else dict(ok=False, error=f"crashed (code {p.returncode}): "
                                                                     f"{p.stderr.strip()[-600:]}")
        res[name] = r
        print(f"{name:16s} {'OK' if r.get('ok') else 'FAILED'}  {r.get('error', '')}", flush=True)
        if name == "default":
            print(f"  jax {r.get('jax')}, devices {r.get('devices')}", flush=True)
    good = [n for n, r in res.items() if r.get("ok")]
    print(f"\nworking: {good or 'none'}", flush=True)


if __name__ == "__main__":
    main()
