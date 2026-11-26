# mlcli.py
# Anna Preto
# AstroDeep Project
# Command-line entry point of the cross-instrument work: dp2-ml <ml_config.yaml> <command> [options].
#   mmu-mirror [--dry-run]         download the MMU partitions of all patches once (prepost), euclid.mmu_cache
#   probe-euclid --patch P          which Euclid image source works from this node (prepost)
#   phase0 [--run-name N]           reference VAE checks on DP2 stamps (needs --debvader-run); evaluations/N
#   d1 | d2 [--index N|--patches]   build the D1 / D2 shard of one or more patches (prepost: Euclid images)
#   d1-merge | d2-merge             merge the shards into the memory-mapped dataset
#   quality d1|d2                   quality report of a merged dataset
#   repair-meta d1|d2               rewrite meta.parquet of a merged dataset from its shards, arrays untouched
#                                   (datasets merged before the empty-shard fix: identifiers rounded to float64)
#   blends                          controlled blends from D1 (val, test)
#   export-rubin-vae                Rubin stamps of D1 as a debvader-keras real build, to train the reference VAE
#                                   (--debvader-run: template run whose config.json gives the architecture)
#   deblend-stats                   normalisation scales of the deblending campaign (D1 train split)
#   deblend-train STAGE [--seed S]  train one stage of the campaign: rubin_vae, k1, k2, A, B, c1, c2, c3
#                                   (--max-steps N: short pilot / smoke run)
#   deblend-eval WHAT [--seed S]    evaluations of the campaign (see deblend/evaluate.py)
#   deblend-campaign MODE           start a self-advancing Slurm campaign (pilot | all | seeds | final; --seed,
#                                   --max-steps, --split, --after DEPENDENCY), see deblend/campaign.py
#   deblend-advance [status | retry JOB | stop CAMPAIGN]
#                                   submit what has become ready in the active campaigns (run by every campaign job
#                                   when it ends); status: print only
# Legacy (archived architecture, kept runnable for traceability, not part of the active campaign):
#   train-stage1 | eval-stage1      Stage 1 (needs --debvader-run; --ablation picks the loss weights)
#   train-stage2 | eval-stage2      Stage 2 (needs --debvader-run and --stage1-run)
# The DebVAder run folder (paper_baseline) is given with --debvader-run or the DEBVADER_RUN environment variable;
# --debvader-stats points to the stats.json of the dataset it was trained on when the run folder has none.

import argparse
import os
import sys
import traceback

COMMANDS = ("mmu-mirror", "probe-euclid", "phase0", "d1", "d2", "d1-merge", "d2-merge", "quality", "repair-meta", "blends",
            "export-rubin-vae", "train-stage1", "eval-stage1", "train-stage2", "eval-stage2", "deblend-stats",
            "deblend-train", "deblend-eval", "deblend-campaign", "deblend-advance")


def _debvader_run(args):
    run = args.debvader_run or os.environ.get("DEBVADER_RUN")
    if not run:
        sys.exit("give the DebVAder run folder: --debvader-run PATH or DEBVADER_RUN=PATH")
    return run


def _shards(cfg, args, kind):
    from .pairs import dataset1, dataset2, store
    from .pipeline.patches import load_patches
    mod = dataset1 if kind == "d1" else dataset2
    patches = load_patches(cfg.pipeline(), args.patches, args.index)
    ok = True
    for p in patches:
        sd = mod.shard_dir(cfg, p["name"])
        if store.shard_done(sd) and not args.force:
            print(f"[{p['name']}] {kind} shard already built")
            continue
        try:
            print(f"[{p['name']}] {kind}: {mod.build_patch(cfg, p)}")
        except Exception:  # noqa: BLE001 - one patch must not stop the array; the error is logged and returned
            ok = False
            err = traceback.format_exc()
            print(f"[{p['name']}] {kind} FAILED\n{err}", file=sys.stderr)
            sd.parent.mkdir(parents=True, exist_ok=True)
            (sd.parent / f"{p['name']}.error.txt").write_text(err)
    return ok


def _deblend(cfg, args, c):
    if c == "deblend-stats":
        import json
        from .deblend import data
        out = data.deblend_dir(cfg)
        out.mkdir(parents=True, exist_ok=True)
        stats = data.compute_norm_stats(cfg, int(cfg["deblend"].get("stats_max_examples", 20000)))
        (out / "norm_stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))
        return
    if c == "deblend-train":
        from .deblend import train
        if args.what not in train.STAGES:
            sys.exit(f"deblend-train needs a stage: {', '.join(train.STAGES)}")
        budget = args.time_budget_min or float(os.environ.get("DEBLEND_TIME_BUDGET_MIN") or 0) or None
        train.train(cfg, args.what, args.seed, not args.no_resume, args.force, args.max_steps,
                    budget * 60 if budget else None)
        return
    if c == "deblend-eval":
        from .deblend import evaluate as devaluate
        devaluate.run(cfg, args.what, args.seed, args.split)
        return


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dp2-ml")
    ap.add_argument("config")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("what", nargs="?", help="quality: d1 or d2; deblend-train: stage; deblend-eval: evaluation; "
                                            "deblend-campaign: mode; deblend-advance: status | retry | stop")
    ap.add_argument("target", nargs="?", help="deblend-advance retry: job name; stop: campaign name")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a configuration value (repeatable)")
    ap.add_argument("--patches", nargs="+")
    ap.add_argument("--index", type=int)
    ap.add_argument("--patch", help="probe-euclid: patch to take objects from")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="mmu-mirror: list the partitions and their size only")
    ap.add_argument("--debvader-run")
    ap.add_argument("--debvader-stats")
    ap.add_argument("--run-name")
    ap.add_argument("--stage1-run")
    ap.add_argument("--ablation")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3000, help="phase0: number of DP2 stamps")
    ap.add_argument("--reference-npz", help="phase0: predictions saved by the campaign (images, mu)")
    ap.add_argument("--seed", type=int, help="deblend: replicate seed (run folder suffix _s<seed>)")
    ap.add_argument("--max-steps", type=int, help="deblend-train: stop after this many steps (pilot / smoke)")
    ap.add_argument("--time-budget-min", type=float,
                    help="deblend-train: stop after that many minutes without finishing, resumable (default: "
                         "$DEBLEND_TIME_BUDGET_MIN, set by deblend.slurm)")
    ap.add_argument("--after", help="deblend-campaign: Slurm dependency of the first jobs, e.g. afterok:123")
    args = ap.parse_args(argv)

    from .mlconfig import MLConfig
    cfg = MLConfig.load(args.config, args.set)
    c = args.command

    if c in ("deblend-campaign", "deblend-advance"):
        # No keras here: the campaign is started from a login node.
        from .deblend import campaign
        if c == "deblend-campaign":
            campaign.start(cfg, args.config, args.set, args.what, args.seed, args.max_steps, args.split, args.after)
        elif args.what in (None, "status"):
            campaign.advance(cfg, submit=args.what is None)
        elif args.what == "retry" and args.target:
            campaign.retry(cfg, args.target)
            campaign.advance(cfg)
        elif args.what == "stop" and args.target:
            campaign.stop(cfg, args.target)
        else:
            sys.exit("deblend-advance [status | retry JOB_NAME | stop CAMPAIGN]")
        return
    if c in ("d1", "d2"):
        sys.exit(0 if _shards(cfg, args, c) else 1)
    if c == "d1-merge":
        from .pairs import dataset1
        dataset1.merge(cfg)
        return
    if c == "d2-merge":
        from .pairs import dataset2
        dataset2.merge(cfg)
        return
    if c == "repair-meta":
        from .pairs import store
        if args.what not in ("d1", "d2"):
            sys.exit("repair-meta needs d1 or d2")
        root = cfg.dataset_dir(args.what)
        shards = [s for s in sorted((root / "shards").glob("*")) if s.is_dir() and not s.name.endswith(".tmp")]
        fixed = store.repair_meta(shards, root)
        print(f"{args.what}: meta.parquet rewritten from {len(shards)} shards")
        for f in fixed or ["nothing to fix"]:
            print(f"  {f}")
        return
    if c == "quality":
        from .pairs import quality
        from .pairs.store import Store
        if args.what not in ("d1", "d2"):
            sys.exit("quality needs d1 or d2")
        st = Store(cfg.dataset_dir(args.what))
        rep = quality.run(cfg, st, "d1") if args.what == "d1" else quality.run_d2(cfg, st)
        sys.exit(0 if rep["status"] != "fail" else 1)
    if c == "blends":
        from .pairs import blends
        blends.build(cfg)
        return
    if c == "mmu-mirror":
        from .pipeline.patches import load_patches, patch_case
        from .sources import euclid_images
        ec = cfg["euclid"]
        cache = euclid_images.mmu_cache_dir(ec)
        if cache is None:
            sys.exit("euclid.mmu_cache is null: nothing to mirror")
        pcfg = cfg.pipeline()
        boxes = {p["name"]: tuple(patch_case(pcfg, p).sky_box) for p in load_patches(pcfg, args.patches)}
        euclid_images.mirror_mmu(ec["hats_url"], cache, boxes, args.dry_run)
        return
    if c == "export-rubin-vae":
        from .pairs import rubin_vae
        rubin_vae.export(cfg, _debvader_run(args), args.force)
        return

    # Everything below imports keras (backend chosen in dp2_cutouts.ml).
    from . import ml  # noqa: F401
    if c.startswith("deblend-"):
        # Inside a campaign job, an exception leaves a failure marker so that the campaign does not resubmit it.
        try:
            return _deblend(cfg, args, c)
        except BaseException as e:  # noqa: BLE001 (SystemExit from argument checks included)
            if not (isinstance(e, SystemExit) and e.code in (0, None)):
                from .deblend import campaign
                campaign.mark_failed(cfg, "".join(traceback.format_exception(type(e), e, e.__traceback__)))
            raise
    if c == "probe-euclid":
        from .ml import probe
        if not args.patch:
            sys.exit("probe-euclid needs --patch tract_patch")
        probe.run(cfg, args.patch)
        return
    run = _debvader_run(args)
    if c == "phase0":
        from .ml import phase0
        phase0.run(cfg, run, args.debvader_stats, args.n, args.reference_npz, out_name=args.run_name or "phase0",
                   split=args.split)
    elif c == "train-stage1":
        from .ml import stage1
        name = args.run_name or args.ablation or "stage1"
        stage1.train(cfg, run, name, args.ablation, args.debvader_stats, not args.no_resume, args.epochs, args.force)
    elif c == "eval-stage1":
        from .ml import evaluate
        evaluate.evaluate_stage1(cfg, run, args.run_name or args.ablation or "stage1", args.debvader_stats,
                                 args.split)
    elif c == "train-stage2":
        from .ml import stage2
        if not args.stage1_run:
            sys.exit("train-stage2 needs --stage1-run (the Stage 1 run used as teacher)")
        name = args.run_name or args.ablation or "stage2"
        stage2.train(cfg, run, args.stage1_run, name, args.ablation, args.debvader_stats, not args.no_resume,
                     args.epochs, args.force)
    elif c == "eval-stage2":
        from .ml import evaluate
        if not args.stage1_run:
            sys.exit("eval-stage2 needs --stage1-run")
        evaluate.evaluate_stage2(cfg, run, args.stage1_run, args.run_name or args.ablation or "stage2",
                                 args.debvader_stats, args.split)


if __name__ == "__main__":
    main()
