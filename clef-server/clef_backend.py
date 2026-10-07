"""CLEF-Flash backends.

Two ways to run the same model, picked by `choose_backend()`:
  mlx    mlx-community/clef-flash-4bit: 4-bit MLX quantization, ~6 GB, Apple Silicon (default on a Mac)
  torch  Cloudflare/clef-flash: the full BF16 release, ~19 GB, NVIDIA GPU / CPU (default elsewhere)

Heavy libraries (torch, mlx) are imported lazily so this module imports without them.
"""
from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prompt import ClefPrompt

log = logging.getLogger("clef")

REPOS = {"mlx": "mlx-community/clef-flash-4bit", "torch": "Cloudflare/clef-flash"}
APPROX_GB = {"mlx": 6, "torch": 19}
DESCRIPTION = {"mlx": "4-bit MLX quantization", "torch": "full BF16 weights"}

_DTYPES = {"bf16": "bfloat16", "bfloat16": "bfloat16",
           "fp16": "float16", "float16": "float16", "half": "float16",
           "fp32": "float32", "float32": "float32"}


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def choose_backend() -> str:
    """CLEF_BACKEND=mlx|torch overrides; otherwise MLX on Apple Silicon, torch everywhere else."""
    env = os.environ.get("CLEF_BACKEND", "auto").strip().lower()
    if env in REPOS:
        return env
    if env not in ("", "auto"):
        raise ValueError(f"CLEF_BACKEND must be auto, mlx or torch, not {env!r}")
    return "mlx" if is_apple_silicon() else "torch"


def model_repo(backend: str) -> str:
    """CLEF_MODEL (Hub repo id or local dir) overrides the backend's default repo."""
    return os.environ.get("CLEF_MODEL") or REPOS[backend]


def pick_device(torch) -> str:
    env = os.environ.get("DEVICE", "").strip().lower()
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class WeightsMissing(RuntimeError):
    """The weights are not in the local Hugging Face cache (and auto-download is off)."""


def resolve_model_path(model: str, approx_gb: int = 19) -> str:
    """Local dir -> as-is; otherwise the local HF cache (respects HF_HOME).

    Never starts a multi-GB download implicitly: `./run.sh download` does that on purpose.
    Set CLEF_AUTO_DOWNLOAD=1 to allow downloading on server start anyway.
    """
    if Path(model).expanduser().is_dir():
        return str(Path(model).expanduser().resolve())
    from huggingface_hub import snapshot_download
    if os.environ.get("CLEF_AUTO_DOWNLOAD", "0") == "1":
        log.info("downloading / resolving %s from the Hub (~%s GB on first run)", model, approx_gb)
        return snapshot_download(model)
    try:
        return snapshot_download(model, local_files_only=True)
    except Exception as e:
        raise WeightsMissing(
            f"CLEF-Flash weights are not downloaded yet — run ./run.sh download (~{approx_gb} GB)") from e


def make_backend(prompt: ClefPrompt):
    kind = choose_backend()
    cls = MlxBackend if kind == "mlx" else ClefBackend
    return cls(model_repo(kind), prompt)


def _to_labels(prompt: ClefPrompt, option_probs) -> dict[str, float]:
    """[(option_id, p), ...] -> {"hotdog": p, "not_hotdog": p} via the schema's label_map."""
    result = {"hotdog": 0.0, "not_hotdog": 0.0}
    for opt, p in option_probs:
        label = prompt.label_map.get(str(opt))
        if label is None:
            raise RuntimeError(f"unexpected option id from model: {opt!r}")
        result[label] += float(p)
    return result


class _Loader:
    """Loads the model in the background; /health reports `ready` / `error` meanwhile."""
    kind = ""

    def __init__(self, model_id: str, prompt: ClefPrompt):
        self.model_id = model_id
        self.prompt = prompt
        self.ready = False
        self.error: str | None = None

    @property
    def weights(self) -> str:
        return DESCRIPTION[self.kind]

    def _load_safe(self) -> None:
        try:
            self._load()
        except WeightsMissing as e:  # expected before `./run.sh download`; no traceback spam
            log.warning("%s", e)
            self.error = str(e)
        except Exception as e:  # surfaced via /health
            log.exception("model load failed")
            self.error = f"{type(e).__name__}: {e}"

    def _load(self) -> None:
        raise NotImplementedError


class ClefBackend(_Loader):
    """Full-size PyTorch release (Cloudflare/clef-flash): CUDA, MPS or CPU."""
    kind = "torch"

    def __init__(self, model_id: str, prompt: ClefPrompt):
        super().__init__(model_id, prompt)
        self.device = os.environ.get("DEVICE", "") or "pending"
        self.dtype_name = _DTYPES.get(os.environ.get("DTYPE", "bf16").lower(), "bfloat16")
        self._lock = threading.Lock()
        self._torch = None
        self._model = None
        self._processor = None
        self._encode = None
        self._collate = None

    # ---------------------------------------------------------------- loading
    def start(self) -> None:
        threading.Thread(target=self._load_safe, name="clef-loader", daemon=True).start()

    def _load(self) -> None:
        import torch
        self._torch = torch
        self.device = pick_device(torch)
        path = resolve_model_path(self.model_id, APPROX_GB[self.kind])
        if path not in sys.path:
            sys.path.insert(0, path)
        from joint_schema_model import collate_records, encode_record, load_release_model
        self._encode, self._collate = encode_record, collate_records

        dtype = getattr(torch, self.dtype_name)
        try:
            self._load_and_warmup(load_release_model, path, dtype)
        except Exception as e:
            # MPS bf16 support is patchy (older macOS / some kernels) -> retry fp16.
            if self.device == "mps" and dtype == torch.bfloat16:
                log.warning("bf16 on mps failed (%s); retrying in fp16", e)
                self._model = None
                torch.mps.empty_cache()
                self.dtype_name = "float16"
                self._load_and_warmup(load_release_model, path, torch.float16)
            else:
                raise
        self.ready = True
        log.info("CLEF-Flash ready on %s (%s)", self.device, self.dtype_name)

    def _load_and_warmup(self, load_release_model, path: str, dtype) -> None:
        t = time.perf_counter()
        self._model, self._processor = load_release_model(path, device=self.device, dtype=dtype)
        self._model.eval()
        log.info("weights loaded in %.1fs", time.perf_counter() - t)
        from PIL import Image
        self._infer(Image.new("RGB", (64, 64), (128, 128, 128)))  # warmup / dtype check

    # -------------------------------------------------------------- inference
    def _infer(self, image) -> tuple[dict[str, float], float]:
        torch = self._torch
        tok = self._processor.tokenizer
        record = self.prompt.record([image])
        encoded = self._encode(tok, record, processor=self._processor)
        batch = self._collate([encoded], tok.pad_token_id, torch.device(self.device))

        qid = self.prompt.question_id
        q_index, option_ids = _find_question(encoded, qid)

        with torch.inference_mode():
            _sync(torch, self.device)
            t0 = time.perf_counter()
            out = self._model(batch)
            _sync(torch, self.device)
            latency_ms = (time.perf_counter() - t0) * 1000.0

        per_question = out[0]                 # list over questions (batch of 1)
        logits = per_question[q_index].float().reshape(-1)
        if logits.numel() != len(option_ids):
            raise RuntimeError(f"logit/option mismatch: {logits.numel()} vs {option_ids}")
        probs = logits.softmax(-1).cpu().tolist()

        return _to_labels(self.prompt, zip(option_ids, probs)), latency_ms

    def classify(self, image) -> tuple[dict[str, float], float]:
        if not self.ready:
            raise RuntimeError("model not ready")
        with self._lock:  # single GPU: serialize
            return self._infer(image)


def _find_question(encoded, qid: str) -> tuple[int, list]:
    """Return (index, option_ids) of our question in the encoded record.
    Defensive about the exact shape of encoded.questions."""
    questions = getattr(encoded, "questions", None)
    if questions is None and isinstance(encoded, dict):
        questions = encoded["questions"]
    if isinstance(questions, dict):
        questions = list(questions.values())
    for i, q in enumerate(questions):
        get = (lambda k: q.get(k)) if isinstance(q, dict) else (lambda k: getattr(q, k, None))
        name = get("question_id") or get("name") or get("id") or get("key")
        if name is None or name == qid:
            opts = get("option_ids")
            if opts is None:
                raise RuntimeError("encoded question has no option_ids")
            return i, list(opts)
    raise RuntimeError(f"question {qid!r} not found in encoded record")


class MlxBackend(_Loader):
    """4-bit MLX quantization (mlx-community/clef-flash-4bit) for Apple Silicon.

    The repo ships its own loader, clef_mlx.py: `clef_mlx.load(path).predict(record)` returns
    {question_id: {option_id: probability}} from a single forward pass.
    All MLX work runs on one dedicated thread: MLX streams belong to the thread that created them.
    """
    kind = "mlx"

    def __init__(self, model_id: str, prompt: ClefPrompt):
        super().__init__(model_id, prompt)
        self.device = "mlx · 4-bit"
        self._model = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clef-mlx")

    def start(self) -> None:
        self._worker.submit(self._load_safe)

    def _load(self) -> None:
        path = resolve_model_path(self.model_id, APPROX_GB[self.kind])  # first: "not set up" works anywhere
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            import clef_mlx
        except ImportError as e:
            hint = ("run ./run.sh setup to install mlx-vlm" if is_apple_silicon()
                    else "MLX needs an Apple Silicon Mac; elsewhere use CLEF_BACKEND=torch")
            raise RuntimeError(f"cannot load the MLX model ({e}); {hint}") from e
        t = time.perf_counter()
        self._model = clef_mlx.load(path)
        log.info("weights loaded in %.1fs", time.perf_counter() - t)
        from PIL import Image
        self._infer(Image.new("RGB", (64, 64), (128, 128, 128)))  # warmup: compiles the kernels
        self.ready = True
        log.info("CLEF-Flash ready on MLX (%s)", self.weights)

    def _infer(self, image) -> tuple[dict[str, float], float]:
        record = self.prompt.record([image])
        t0 = time.perf_counter()
        out = self._model.predict(record)  # returns Python floats, so the GPU work is done here
        latency_ms = (time.perf_counter() - t0) * 1000.0
        qid = self.prompt.question_id
        if qid not in out:
            raise RuntimeError(f"question {qid!r} missing from model output {list(out)}")
        return _to_labels(self.prompt, out[qid].items()), latency_ms

    def classify(self, image) -> tuple[dict[str, float], float]:
        if not self.ready:
            raise RuntimeError("model not ready")
        return self._worker.submit(self._infer, image).result()  # one at a time, on the MLX thread
