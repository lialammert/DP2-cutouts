# cli.py
# Anna Preto
# AstroDeep Project
# Command-line entry point.
#   Single case (local demo):  dp2-cutouts <config.yaml> <step>, or "all" to run the whole pipeline in order.
#   Patch by patch:            dp2-cutouts <config.yaml> patches | plan | fetch | process | assemble | invariance | run
#                              with --patches 5063_34 ... (default: every planned patch) or --index N (Slurm array).
#   Non-regression:            dp2-cutouts <config.yaml> snapshot | regress

import argparse
import sys

from .config import Config

STEPS = ("rubin_catalog", "euclid_catalog", "copaings", "rubin_images", "catalog_table", "stamps", "package")
TOOLS = ("snapshot", "regress")
PATCH_COMMANDS = ("patches", "plan", "fetch", "process", "assemble", "invariance", "run")


def run_step(cfg, step):
    if step == "rubin_catalog":
        from .sources import rubin_tap
        rubin_tap.run(cfg)
    elif step == "euclid_catalog":
        from .sources import euclid_hf
        euclid_hf.run(cfg)
    elif step == "copaings":
        from .stages import run_copaings
        run_copaings.run(cfg)
    elif step == "rubin_images":
        from .sources import rubin_soda
        rubin_soda.run(cfg)
    elif step == "catalog_table":
        from .stages import catalog_table
        catalog_table.run(cfg)
    elif step == "stamps":
        from .stages import stamps
        stamps.run(cfg)
    elif step == "package":
        from .stages import package
        package.run(cfg)


def run_patch_command(cfg, args):
    from .pipeline import invariance, phases
    from .pipeline.patches import list_patches
    if args.step == "patches":
        list_patches(cfg)
        return True
    if args.step == "assemble":
        phases.assemble(cfg)
        return True
    if args.step == "invariance":
        return invariance.run(cfg, refetch=args.force)
    steps = ("plan", "fetch", "process") if args.step == "run" else (args.step,)
    ok = True
    for phase in steps:
        results = phases.run(cfg, phase, args.patches, args.index, args.force)
        ok &= all(r["status"] != "failed" for r in results)
    if args.step == "run" and args.index is None:
        phases.assemble(cfg)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dp2-cutouts")
    ap.add_argument("config")
    ap.add_argument("step", choices=STEPS + ("all",) + TOOLS + PATCH_COMMANDS)
    ap.add_argument("--patches", nargs="+", help="patches as tract_patch, e.g. 5063_34 (patch commands)")
    ap.add_argument("--index", type=int, help="position in the patch list, e.g. $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--force", action="store_true", help="rerun even if already done")
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    if args.step in TOOLS:
        from . import regression
        ok = getattr(regression, args.step)(cfg)
        sys.exit(0 if ok else 1)
    if args.step in PATCH_COMMANDS:
        sys.exit(0 if run_patch_command(cfg, args) else 1)
    for step in (STEPS if args.step == "all" else (args.step,)):
        print(f"=== {step}")
        run_step(cfg, step)


if __name__ == "__main__":
    main()
