"""Download the CLEF-Flash weights into the Hugging Face cache (respects HF_HOME).

    python download_model.py            # download (resumable)
    python download_model.py --check    # exit 0 if already downloaded, 1 if not (no network)
    python download_model.py --describe # one line: what would be downloaded

Which weights: on an Apple Silicon Mac the 4-bit MLX quantization (mlx-community/clef-flash-4bit, ~6 GB),
elsewhere the full BF16 release (Cloudflare/clef-flash, ~19 GB). CLEF_BACKEND / CLEF_MODEL override that.
Normally you don't call this directly: `./run.sh download` does.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from clef_backend import APPROX_GB, DESCRIPTION, choose_backend, model_repo

BACKEND = choose_backend()
REPO = model_repo(BACKEND)


def is_downloaded(repo: str = REPO) -> bool:
    if Path(repo).expanduser().is_dir():
        return True
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def main() -> int:
    if "--check" in sys.argv:
        return 0 if is_downloaded() else 1
    gb = APPROX_GB[BACKEND]
    if "--describe" in sys.argv:
        print(f"{REPO} ({DESCRIPTION[BACKEND]}, ~{gb} GB)")
        return 0
    from huggingface_hub import snapshot_download

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / 1e9
    partial = cache / "hub" / ("models--" + REPO.replace("/", "--"))
    have_gb = sum(f.stat().st_size for f in partial.rglob("*") if f.is_file()) / 1e9 if partial.is_dir() else 0
    print(f"Downloading {REPO} (~{gb} GB, 9.4B params, {DESCRIPTION[BACKEND]}) into {cache}")
    print(f"Free disk space there: {free_gb:.0f} GB" + (f" ({have_gb:.0f} GB already downloaded)" if have_gb >= 1 else ""))
    if free_gb + have_gb < gb + 1:
        print(f"Warning: that is probably not enough; ~{gb - have_gb:.0f} GB more are needed.", file=sys.stderr)
    print("Interrupted downloads resume where they stopped; just run it again.\n")
    path = snapshot_download(REPO)  # tqdm progress bars per file
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    print(f"\nDone: {size:.1f} GB in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
