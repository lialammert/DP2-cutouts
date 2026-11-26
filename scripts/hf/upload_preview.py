# upload_preview.py
# Anna Preto
# AstroDeep Project
# Puts the preview (export_preview.py) and the viewer (scripts/hf/space) on Hugging Face, both PRIVATE: a dataset
# repository <user>/<dataset-name> and a Gradio Space <user>/<space-name> reading it. Refuses to upload anything to a
# repository that is not private (Rubin DP2 data rights). Run from a node with internet (login / prepost on Jean
# Zay), after `huggingface-cli login` with a write token (or HF_TOKEN set).
# The Space reads the dataset with its secret HF_TOKEN: export HF_SPACE_READ_TOKEN=<a read token> before running to
# have it set here, or add it later in the Space settings (Variables and secrets).
# Usage: python scripts/hf/upload_preview.py <export dir> --user <hf user or org> [--dataset-name dp2-euclid-preview]
#            [--space-name dp2-euclid-viewer] [--no-space]

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


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
    ap.add_argument("--no-space", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    data = Path(args.export_dir)
    for f in ("d1.parquet", "d2.parquet", "info.json", "README.md"):
        if not (data / f).exists():
            raise SystemExit(f"{data / f} missing: run export_preview.py first")
    ds = private_repo(api, f"{args.user}/{args.dataset_name}", "dataset")
    api.upload_folder(folder_path=str(data), repo_id=ds, repo_type="dataset",
                      allow_patterns=["d1.parquet", "d2.parquet", "info.json", "README.md"],
                      commit_message="DP2 x Euclid preview")
    print(f"dataset (private): https://huggingface.co/datasets/{ds}")
    if args.no_space:
        return
    sp = private_repo(api, f"{args.user}/{args.space_name}", "space", space_sdk="gradio")
    api.upload_folder(folder_path=str(Path(__file__).parent / "space"), repo_id=sp, repo_type="space",
                      allow_patterns=["app.py", "viewer_core.py", "requirements.txt", "README.md"],
                      commit_message="viewer")
    api.add_space_variable(sp, "DATASET_REPO", ds)
    tok = os.environ.get("HF_SPACE_READ_TOKEN")
    if tok:
        api.add_space_secret(sp, "HF_TOKEN", tok)
        print("Space secret HF_TOKEN set from HF_SPACE_READ_TOKEN")
    else:
        print("Space secret HF_TOKEN not set: add a read token in the Space settings (Variables and secrets)")
    print(f"space (private): https://huggingface.co/spaces/{sp}")


if __name__ == "__main__":
    main()
