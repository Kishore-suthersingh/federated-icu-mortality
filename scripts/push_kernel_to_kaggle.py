"""Push notebooks/kaggle_multiseed.ipynb to Kaggle as a GPU kernel.

The kernel is created private, with internet disabled (everything it needs is
in the two attached datasets) and the GPU enabled.

    python scripts/push_kernel_to_kaggle.py --username YOUR_KAGGLE_USERNAME

Pushing does not run it. Open the printed URL and use
"Save Version -> Save & Run All (Commit)": an interactive session times out
long before five seeds finish, a committed run does not.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# notebook stem -> (kernel slug, extra competition data sources)
KERNELS = {
    "kaggle_multiseed": ("fedlgbm-multiseed", ()),
    "kaggle_wids": ("fedlgbm-wids", ("widsdatathon2020",)),
    "kaggle_wids_privacy": ("fedlgbm-wids-privacy", ("widsdatathon2020",)),
    "kaggle_wids_multiseed": ("fedlgbm-wids-multiseed", ("widsdatathon2020",)),
}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_to_kaggle import resolve_username  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--username", help="Kaggle username")
    p.add_argument("--notebook", default="kaggle_multiseed",
                   choices=sorted(KERNELS),
                   help="Which notebook under notebooks/ to push")
    args = p.parse_args()

    username = resolve_username(args.username)
    SLUG, competitions = KERNELS[args.notebook]
    NOTEBOOK = ROOT / "notebooks" / f"{args.notebook}.ipynb"
    if not NOTEBOOK.exists():
        sys.exit(f"Notebook not found: {NOTEBOOK}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        shutil.copy2(NOTEBOOK, staging / NOTEBOOK.name)

        metadata = {
            "id": f"{username}/{SLUG}",
            # Derived from SLUG, not written by hand: Kaggle slugs the title to
            # resolve the kernel, so a title belonging to a different kernel
            # collides with it (409) instead of creating this one.
            "title": SLUG.replace("-", " "),
            "code_file": NOTEBOOK.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            # No internet: the run must depend only on the attached datasets,
            # so it stays reproducible and cannot silently pull a new package.
            "enable_internet": False,
            "dataset_sources": [
                f"{username}/fedlgbm-src",
                f"{username}/ciciot2023-fedlgbm-parts",
            ],
            # Competition data is attached in place rather than re-uploaded, so
            # the run reads the official file and nothing can drift.
            "competition_sources": list(competitions),
            "kernel_sources": [],
        }
        (staging / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        cmd = ["kaggle", "kernels", "push", "-p", str(staging)]
        print("$", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout or "", result.stderr or "")
        if result.returncode != 0:
            sys.exit(result.returncode)

    print(
        f"\nhttps://www.kaggle.com/code/{username}/{SLUG}\n"
        "Open it and choose Save Version -> Save & Run All (Commit)."
    )


if __name__ == "__main__":
    main()
