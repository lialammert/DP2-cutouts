# cli.py
# Anna Preto
# AstroDeep Project
# Command-line entry point: dp2-cutouts <config.yaml> <step>, or "all" to run the whole pipeline in order.

import argparse

from .config import Config

STEPS = ("rubin_catalog", 
         "euclid_catalog",
           "copaings", 
           "rubin_images", 
           "empty_sky",
           "catalog_table", 
           "stamps",
           "export_debvader", 
           "package")


def run_step(cfg, step):
    if step == "rubin_catalog":
        # rubin.source: "tap" (default, needs an RSP token) or "butler" (needs the LSST stack).
        if cfg["rubin"].get("source", "tap") == "butler":
            from .sources import rubin_butler
            rubin_butler.run(cfg)
        else:
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
    elif step == "empty_sky":
        from .stages import empty_sky
        empty_sky.run(cfg)
    elif step == "catalog_table":
        from .stages import catalog_table
        catalog_table.run(cfg)
    elif step == "stamps":
        from .stages import stamps
        stamps.run(cfg)
    elif step == "export_debvader":
        from .stages import export_debvader
        export_debvader.run(cfg)
    elif step == "package":
        from .stages import package
        package.run(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dp2-cutouts")
    ap.add_argument("config")
    ap.add_argument("step", choices=STEPS + ("all",))
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    for step in (STEPS if args.step == "all" else (args.step,)):
        print(f"=== {step}")
        run_step(cfg, step)


if __name__ == "__main__":
    main()
