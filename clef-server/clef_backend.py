"""CLEF-Flash backends: two ways to run the same 9.4B model, picked by `choose_backend()`.

  mlx    mlx-community/clef-flash-4bit: 4-bit MLX quantization, ~6 GB, Apple Silicon (default on a Mac)
  torch  Cloudflare/clef-flash: the full BF16 release, ~19 GB, NVIDIA GPU / CPU (default elsewhere)

torch and mlx are imported lazily in the loader thread, so this module imports without them
(run.sh and download_model.py rely on that).
"""

import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from prompt import ClefPrompt

log = logging.getLogger("clef")

REPOS = {"mlx": "mlx-community/clef-flash-4bit", "torch": "Cloudflare/clef-flash"}
APPROX_GB = {"mlx": 6, "torch": 19}
DESCRIPTION = {"mlx": "4-bit MLX quantization", "torch": "full BF16 weights"}

_TORCH_DTYPES = {
    **dict.fromkeys(("bf16", "bfloat16"), "bfloat16"),
    **dict.fromkeys(("fp16", "float16", "half"), "float16"),
    **dict.fromkeys(("fp32", "float32"), "float32"),
}

Probs = dict[str, float]  # {"hotdog": p, "not_hotdog": p}


# ------------------------------------------------------------------------------------------ configuration
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


def make_backend(prompt: ClefPrompt) -> "Backend":
    kind = choose_backend()
    backend_cls = MlxBackend if kind == "mlx" else TorchBackend
    return backend_cls(model_repo(kind), prompt)


# ------------------------------------------------------------------------------------------------ weights
class WeightsMissing(RuntimeError):
    """The weights are not in the local Hugging Face cache (and auto-download is off)."""


def find_local_weights(model: str) -> str | None:
    """A local dir, or the model's snapshot in the local HF cache (respects HF_HOME); None if neither. No network."""
    local = Path(model).expanduser()
    if local.is_dir():
        return str(local.resolve())
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model, local_files_only=True)
    except Exception:  # LocalEntryNotFoundError & co.: simply not downloaded (yet)
        return None


def resolve_model_path(model: str, approx_gb: int = APPROX_GB["torch"]) -> str:
    """Path to the weights. Never starts a multi-GB download implicitly: `./run.sh download` does that on
    purpose. CLEF_AUTO_DOWNLOAD=1 allows downloading on server start anyway."""
    if os.environ.get("CLEF_AUTO_DOWNLOAD") == "1" and not Path(model).expanduser().is_dir():
        from huggingface_hub import snapshot_download

        log.info("downloading / resolving %s from the Hub (~%s GB on first run)", model, approx_gb)
        return snapshot_download(model)
    path = find_local_weights(model)
    if path is None:
        raise WeightsMissing(f"CLEF-Flash weights are not downloaded yet — run ./run.sh download (~{approx_gb} GB)")
    return path


def _import_from(path: str) -> None:
    """Both releases ship their own Python loader next to the weights."""
    if path not in sys.path:
        sys.path.insert(0, path)


def _to_labels(prompt: ClefPrompt, option_probs: Iterable[tuple[Any, float]]) -> Probs:
    """[(option_id, p), ...] -> {"hotdog": p, "not_hotdog": p} via the schema's label_map."""
    probs = {"hotdog": 0.0, "not_hotdog": 0.0}
    for option, p in option_probs:
        label = prompt.label_map.get(str(option))
        if label is None:
            raise RuntimeError(f"unexpected option id from model: {option!r}")
        probs[label] += float(p)
    return probs


def pick_device(torch: Any) -> str:
    """DEVICE overrides; otherwise cuda → mps → cpu."""
    if env := os.environ.get("DEVICE", "").strip().lower():
        return env
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ----------------------------------------------------------------------------------------------- backends
class Backend:
    """Loads the model in the background so /health can answer meanwhile.

    `ready` turns True once the model is loaded and warmed up; `error` explains a failed load
    (it is the "not set up" detail that /health and /classify report).
    """

    kind: str
    device: str

    def __init__(self, model_id: str, prompt: ClefPrompt):
        self.model_id = model_id
        self.prompt = prompt
        self.ready = False
        self.error: str | None = None

    @property
    def weights(self) -> str:
        return DESCRIPTION[self.kind]

    def start(self) -> None:
        """Begin loading in the background; returns immediately."""
        raise NotImplementedError

    def classify(self, image: Image.Image) -> tuple[Probs, float]:
        """-> (label probabilities, forward-pass latency in ms). Safe to call from any thread."""
        if not self.ready:
            raise RuntimeError("model not ready")
        return self._infer_exclusively(image)

    def _infer_exclusively(self, image: Image.Image) -> tuple[Probs, float]:
        """`_infer`, one request at a time: there is one model on one accelerator."""
        raise NotImplementedError

    def _load(self) -> None:
        raise NotImplementedError

    def _infer(self, image: Image.Image) -> tuple[Probs, float]:
        raise NotImplementedError

    def _load_safe(self) -> None:
        try:
            self._load()
        except WeightsMissing as e:  # expected before `./run.sh download`; no traceback spam
            log.warning("%s", e)
            self.error = str(e)
        except Exception as e:
            log.exception("model load failed")
            self.error = f"{type(e).__name__}: {e}"

    def _warmup(self) -> None:
        """A tiny forward pass: compiles kernels and surfaces dtype/device problems before the first request."""
        self._infer(Image.new("RGB", (64, 64), (128, 128, 128)))


class TorchBackend(Backend):
    """The full-size PyTorch release (Cloudflare/clef-flash) on CUDA, MPS or CPU."""

    kind = "torch"

    def __init__(self, model_id: str, prompt: ClefPrompt):
        super().__init__(model_id, prompt)
        self.device = os.environ.get("DEVICE") or "pending"
        self.dtype_name = _TORCH_DTYPES.get(os.environ.get("DTYPE", "bf16").lower(), "bfloat16")
        self._lock = threading.Lock()
        self._torch: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._encode: Any = None
        self._collate: Any = None

    def start(self) -> None:
        threading.Thread(target=self._load_safe, name="clef-loader", daemon=True).start()

    def _infer_exclusively(self, image: Image.Image) -> tuple[Probs, float]:
        with self._lock:
            return self._infer(image)

    def _load(self) -> None:
        import torch

        self._torch = torch
        self.device = pick_device(torch)
        path = resolve_model_path(self.model_id, APPROX_GB[self.kind])
        _import_from(path)
        from joint_schema_model import collate_records, encode_record, load_release_model

        self._encode, self._collate = encode_record, collate_records
        dtype = getattr(torch, self.dtype_name)
        try:
            self._load_weights(load_release_model, path, dtype)
        except Exception as e:
            if self.device != "mps" or dtype != torch.bfloat16:
                raise
            # MPS bf16 support is patchy (older macOS, some kernels): retry in fp16.
            log.warning("bf16 on mps failed (%s); retrying in fp16", e)
            self._model = None
            torch.mps.empty_cache()
            self.dtype_name = "float16"
            self._load_weights(load_release_model, path, torch.float16)
        self.ready = True
        log.info("CLEF-Flash ready on %s (%s)", self.device, self.dtype_name)

    def _load_weights(self, load_release_model: Callable, path: str, dtype: Any) -> None:
        started = time.perf_counter()
        self._model, self._processor = load_release_model(path, device=self.device, dtype=dtype)
        self._model.eval()
        log.info("weights loaded in %.1fs", time.perf_counter() - started)
        self._warmup()  # part of the load so a failing dtype triggers the fp16 retry

    def _infer(self, image: Image.Image) -> tuple[Probs, float]:
        torch = self._torch
        tokenizer = self._processor.tokenizer
        encoded = self._encode(tokenizer, self.prompt.record([image]), processor=self._processor)
        batch = self._collate([encoded], tokenizer.pad_token_id, torch.device(self.device))
        q_index, option_ids = _find_question(encoded, self.prompt.question_id)

        with torch.inference_mode():
            _sync(torch, self.device)  # time the forward pass only, not queued async work
            started = time.perf_counter()
            out = self._model(batch)
            _sync(torch, self.device)
            latency_ms = (time.perf_counter() - started) * 1000.0

        logits = out[0][q_index].float().reshape(-1)  # out: per sample (batch of 1) -> per question
        if logits.numel() != len(option_ids):
            raise RuntimeError(f"logit/option mismatch: {logits.numel()} vs {option_ids}")
        probs = logits.softmax(-1).cpu().tolist()
        return _to_labels(self.prompt, zip(option_ids, probs, strict=True)), latency_ms


_QUESTION_NAME_KEYS = ("question_id", "name", "id", "key")


def _find_question(encoded: Any, question_id: str) -> tuple[int, list]:
    """(index, option_ids) of our question in the encoded record.

    Defensive about its exact shape: the record and its questions may be dicts or objects.
    """
    questions = _field(encoded, "questions")
    if questions is None:
        raise RuntimeError("encoded record has no questions")
    if isinstance(questions, dict):
        questions = questions.values()
    for index, question in enumerate(questions):
        name = next(filter(None, (_field(question, key) for key in _QUESTION_NAME_KEYS)), None)
        if name is None or name == question_id:
            option_ids = _field(question, "option_ids")
            if option_ids is None:
                raise RuntimeError("encoded question has no option_ids")
            return index, list(option_ids)
    raise RuntimeError(f"question {question_id!r} not found in encoded record")


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


class MlxBackend(Backend):
    """The 4-bit MLX quantization (mlx-community/clef-flash-4bit) for Apple Silicon.

    The repo ships its own loader, clef_mlx.py: `clef_mlx.load(path).predict(record)` returns
    {question_id: {option_id: probability}} from a single forward pass.
    All MLX work runs on one dedicated thread because MLX streams belong to the thread that created them.
    """

    kind = "mlx"

    def __init__(self, model_id: str, prompt: ClefPrompt):
        super().__init__(model_id, prompt)
        self.device = "mlx · 4-bit"
        self._model: Any = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clef-mlx")

    def start(self) -> None:
        self._worker.submit(self._load_safe)

    def _infer_exclusively(self, image: Image.Image) -> tuple[Probs, float]:
        return self._worker.submit(self._infer, image).result()  # the single MLX thread also serializes

    def _load(self) -> None:
        # Resolve first, so "not set up" is reported the same way on any machine.
        path = resolve_model_path(self.model_id, APPROX_GB[self.kind])
        _import_from(path)
        try:
            import clef_mlx
        except ImportError as e:
            hint = (
                "run ./run.sh setup to install mlx-vlm"
                if is_apple_silicon()
                else "MLX needs an Apple Silicon Mac; elsewhere use CLEF_BACKEND=torch"
            )
            raise RuntimeError(f"cannot load the MLX model ({e}); {hint}") from e
        started = time.perf_counter()
        self._model = clef_mlx.load(path)
        log.info("weights loaded in %.1fs", time.perf_counter() - started)
        self._warmup()
        self.ready = True
        log.info("CLEF-Flash ready on MLX (%s)", self.weights)

    def _infer(self, image: Image.Image) -> tuple[Probs, float]:
        record = self.prompt.record([image])
        started = time.perf_counter()
        out = self._model.predict(record)  # returns Python floats, so the GPU work is finished here
        latency_ms = (time.perf_counter() - started) * 1000.0
        question_id = self.prompt.question_id
        if question_id not in out:
            raise RuntimeError(f"question {question_id!r} missing from model output {list(out)}")
        return _to_labels(self.prompt, out[question_id].items()), latency_ms
