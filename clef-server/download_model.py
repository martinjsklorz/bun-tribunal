"""Download the CLEF-Flash weights into the Hugging Face cache (respects HF_HOME).

    python download_model.py             # download (resumable)
    python download_model.py --check     # exit 0 if already downloaded, 1 if not (no network)
    python download_model.py --describe  # one line: what would be downloaded

Which weights: on an Apple Silicon Mac the 4-bit MLX quantization (mlx-community/clef-flash-4bit, ~6 GB),
elsewhere the full BF16 release (Cloudflare/clef-flash, ~19 GB). CLEF_BACKEND / CLEF_MODEL override that.
Normally you don't call this directly: `./run.sh download` does.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from clef_backend import APPROX_GB, DESCRIPTION, choose_backend, find_local_weights, model_repo


def dir_gb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def download(repo: str, backend: str) -> None:
    from huggingface_hub import snapshot_download

    gb = APPROX_GB[backend]
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(cache).free / 1e9
    have_gb = dir_gb(cache / "hub" / ("models--" + repo.replace("/", "--")))  # from an interrupted download

    print(f"Downloading {repo} (~{gb} GB, 9.4B params, {DESCRIPTION[backend]}) into {cache}")
    resumed = f" ({have_gb:.0f} GB already downloaded)" if have_gb >= 1 else ""
    print(f"Free disk space there: {free_gb:.0f} GB{resumed}")
    if free_gb + have_gb < gb + 1:
        print(f"Warning: that is probably not enough; ~{gb - have_gb:.0f} GB more are needed.", file=sys.stderr)
    print("Interrupted downloads resume where they stopped; just run it again.\n")

    path = snapshot_download(repo)  # tqdm progress bars per file
    print(f"\nDone: {dir_gb(Path(path)):.1f} GB in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the CLEF-Flash weights (see ./run.sh download).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 0 if already downloaded, 1 if not (no network)")
    mode.add_argument("--describe", action="store_true", help="print what would be downloaded")
    args = parser.parse_args()

    backend = choose_backend()
    repo = model_repo(backend)
    if args.check:
        return 0 if find_local_weights(repo) else 1
    if args.describe:
        print(f"{repo} ({DESCRIPTION[backend]}, ~{APPROX_GB[backend]} GB)")
        return 0
    download(repo, backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
