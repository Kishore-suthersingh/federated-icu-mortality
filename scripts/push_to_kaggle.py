"""Push src/ to Kaggle as a private dataset the notebook can import.

Requires ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token).

First push:   python scripts/push_to_kaggle.py --username YOUR_KAGGLE_USERNAME
Later pushes: python scripts/push_to_kaggle.py --update -m "fixed partitioning"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SLUG = "fedlgbm-src"
ROOT = Path(__file__).resolve().parent.parent


def resolve_username(explicit: str | None) -> str:
    """Find the Kaggle username, or explain how to supply one.

    The CLI accepts three credential forms and only one of them carries the
    username: kaggle.json. With an access_token file or the KAGGLE_KEY
    environment variable the username has to be passed in.
    """
    if explicit:
        return explicit
    if env := os.environ.get("KAGGLE_USERNAME"):
        return env

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if creds.exists():
        name = json.loads(creds.read_text()).get("username")
        if name:
            return name

    token = Path.home() / ".kaggle" / "access_token"
    if token.exists() or os.environ.get("KAGGLE_KEY"):
        sys.exit(
            "Authenticated, but the username is not recoverable from an "
            "access_token/KAGGLE_KEY credential.\n"
            "Re-run with --username YOUR_KAGGLE_USERNAME.\n"
        )

    sys.exit(
        "No Kaggle credentials found. Either:\n"
        "  - https://www.kaggle.com/settings -> API -> Create New Token, saved "
        f"to {creds}, or\n"
        "  - set KAGGLE_USERNAME and KAGGLE_KEY in the environment.\n"
    )


def stage(staging: Path, username: str, update: bool) -> None:
    """Copy the package plus a dataset-metadata file into a staging dir."""
    shutil.copytree(
        ROOT / "src", staging / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "requirements.txt", staging / "requirements.txt")

    metadata = {"title": "fedlgbm-src", "id": f"{username}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}
    (staging / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--username", help="Kaggle username (required on first push)")
    p.add_argument("--update", action="store_true", help="Push a new version")
    p.add_argument("-m", "--message", default="update source")
    args = p.parse_args()

    username = resolve_username(args.username)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        stage(staging, username, args.update)

        cmd = (
            ["kaggle", "datasets", "version", "-m", args.message, "-p", str(staging), "-r", "zip"]
            if args.update
            else ["kaggle", "datasets", "create", "-p", str(staging), "-r", "zip"]
        )
        print("$", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout or "", result.stderr or "")

        if result.returncode != 0:
            if not args.update and "already exists" in (result.stderr or "" ) + (result.stdout or ""):
                sys.exit("Dataset already exists — re-run with --update")
            sys.exit(result.returncode)

    print(f"\nDone. In the notebook: Add Data -> Your Datasets -> {SLUG}")


if __name__ == "__main__":
    main()
