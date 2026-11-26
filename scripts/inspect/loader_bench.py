# loader_bench.py
# Anna Preto
# AstroDeep Project
# Reading speed of the deblending batches (deblend/data.py), without a GPU: seconds per training batch of D1 (Rubin
# + Euclid) and D2, with rows drawn one by one (read_block 1, the pilot of 2026-11-26) and in blocks of consecutive
# rows. One thread, batches prepared one after the other: the GPU needs less than the time printed divided by the
# number of workers (deblend.train.workers) to never wait.
# Usage: python scripts/inspect/loader_bench.py <ML config> [--blocks 1 16 64] [--batches 6] [--batch-size 128]

import argparse
import time

from dp2_cutouts.mlconfig import MLConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    from dp2_cutouts.deblend import data
    cfg = MLConfig.load(args.config)
    for name, make in (("D1 rubin+euclid", lambda b: data.D1Batches(cfg, "train", args.batch_size, True, 7, block=b)),
                       ("D2 blend+euclid", lambda b: data.D2Batches(cfg, "train", args.batch_size, True, 7, block=b))):
        for b in args.blocks:
            ds = make(b)
            ds[0]                                       # first touch (metadata, file handles)
            t0 = time.time()
            for i in range(1, args.batches + 1):
                ds[i * 97 % len(ds)]
            dt = (time.time() - t0) / args.batches
            print(f"{name:16s} read_block {b:3d}: {dt:6.2f} s per batch of {args.batch_size}", flush=True)


if __name__ == "__main__":
    main()
