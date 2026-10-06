"""Download CLEF-Flash weights into the Hugging Face cache (respects HF_HOME).

    python download_model.py            # download (~19 GB, resumable)
    python download_model.py --check    # exit 0 if already downloaded, 1 if not (no network)

Normally you don't call this directly: `./run.sh download` does.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APPROX_GB = 19
REPO = os.environ.get("CLEF_MODEL", "Cloudflare/clef-flash")


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
    from huggingface_hub import snapshot_download

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / 1e9
    have_gb = sum(f.stat().st_size for f in cache.rglob("*clef-flash*/**/*") if f.is_file()) / 1e9
    print(f"Downloading {REPO} (~{APPROX_GB} GB, 9.4B params in BF16) into {cache}")
    print(f"Free disk space there: {free_gb:.0f} GB" + (f" ({have_gb:.0f} GB already downloaded)" if have_gb >= 1 else ""))
    if free_gb + have_gb < APPROX_GB + 1:
        print(f"Warning: that is probably not enough; ~{APPROX_GB - have_gb:.0f} GB more are needed.", file=sys.stderr)
    print("Interrupted downloads resume where they stopped; just run it again.\n")
    path = snapshot_download(REPO)  # tqdm progress bars per file
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    print(f"\nDone: {size:.1f} GB in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
