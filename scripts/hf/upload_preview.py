# upload_preview.py
# Anna Preto
# AstroDeep Project
# Puts the preview (export_preview.py) on Hugging Face, everything PRIVATE: the dataset repository
# <user>/<dataset-name> (the parquet files) and a static Space <user>/<space-name> holding the viewer and its own copy
# of the data (static_site.py; static Spaces are free, Gradio ones need a paid subscription). Refuses to upload
# anything to a repository that is not private (Rubin DP2 data rights). Run from a node with internet (login /
# prepost on Jean Zay), after `hf auth login` with a write token. The user name is case-sensitive (hf auth whoami).
# Usage: python scripts/hf/upload_preview.py <export dir> --user <hf user> [--dataset-name dp2-euclid-preview]
#            [--space-name dp2-euclid-viewer] [--no-dataset] [--no-space]

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_site import build_site  # noqa: E402


def private_repo(api, repo_id, repo_type, **kw):
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True, **kw)
    if not api.repo_info(repo_id, repo_type=repo_type).private:
        raise SystemExit(f"{repo_type} {repo_id} exists and is PUBLIC: nothing uploaded. Make it private first.")
    return repo_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--user", required=True)
    ap.add_argument("--dataset-name", default="dp2-euclid-preview")
    ap.add_argument("--space-name", default="dp2-euclid-viewer")
    ap.add_argument("--no-dataset", action="store_true", help="skip the dataset repository (already uploaded)")
    ap.add_argument("--no-space", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    data = Path(args.export_dir)
    for f in ("d1.parquet", "d2.parquet", "info.json", "README.md"):
        if not (data / f).exists():
            raise SystemExit(f"{data / f} missing: run export_preview.py first")
    if not args.no_dataset:
        ds = private_repo(api, f"{args.user}/{args.dataset_name}", "dataset")
        api.upload_folder(folder_path=str(data), repo_id=ds, repo_type="dataset",
                          allow_patterns=["d1.parquet", "d2.parquet", "d1r.parquet", "info.json", "README.md"],
                          commit_message="DP2 x Euclid preview")
        print(f"dataset (private): https://huggingface.co/datasets/{ds}")
    if args.no_space:
        return
    site = build_site(data, data / "site")
    sp = private_repo(api, f"{args.user}/{args.space_name}", "space", space_sdk="static")
    api.upload_folder(folder_path=str(site), repo_id=sp, repo_type="space", commit_message="viewer",
                      delete_patterns=["data/*"])          # chunks of an earlier, larger export
    print(f"space (private): https://huggingface.co/spaces/{sp}")


if __name__ == "__main__":
    main()
