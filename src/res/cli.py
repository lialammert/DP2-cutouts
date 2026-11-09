# cli.py
# Anna Preto
# AstroDeep Project
# Point d'entrée en ligne de commande : res <config.yaml> <étape>, ou « all » pour toute la chaîne dans l'ordre.

import argparse

from .config import Config

STEPS = ("rubin_catalog", "euclid_catalog", "copaings", "rubin_images", "catalog_table", "stamps", "package")


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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="res")
    ap.add_argument("config")
    ap.add_argument("step", choices=STEPS + ("all",))
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    for step in (STEPS if args.step == "all" else (args.step,)):
        print(f"=== {step}")
        run_step(cfg, step)


if __name__ == "__main__":
    main()
