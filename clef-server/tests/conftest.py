import os
import sys
import tempfile
from pathlib import Path

# Tests never touch real weights: an empty Hugging Face cache, no network downloads.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="hf-empty-")
os.environ.pop("CLEF_AUTO_DOWNLOAD", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
