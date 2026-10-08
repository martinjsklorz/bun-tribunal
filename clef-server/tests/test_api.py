"""CLEF-Flash server tests.

They run without real weights, so the server is in its "not set up" state; image handling, the prompt
and the download policy are tested directly, and the MLX backend runs against a fake clef_mlx module.
"""

import io
import sys
import time

import huggingface_hub
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
import clef_backend
from imaging import BadImage, load_image
from prompt import load_prompt

HOT_RED = (200, 80, 40)  # the fake model calls red pictures hot dogs
COLD_BLUE = (20, 120, 200)


# ------------------------------------------------------------------------------------------------ helpers
def image_bytes(size=(320, 200), color=HOT_RED, fmt="PNG", mode="RGB", **save_kw) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt, **save_kw)
    return buf.getvalue()


def upload(client: TestClient, data: bytes):
    return client.post("/classify", files={"file": ("x.png", data, "image/png")})


def wait_until_settled(backend, timeout: float = 120) -> None:
    """Wait for the background load to succeed or fail (without weights the torch backend
    fails fast, after importing torch, which can take a while)."""
    deadline = time.time() + timeout
    while not backend.ready and backend.error is None and time.time() < deadline:
        time.sleep(0.02)


@pytest.fixture()
def client():
    with TestClient(app_module.create_app()) as c:
        wait_until_settled(c.app.state.backend)
        yield c


# ---------------------------------------------------------------------------------- HTTP, not set up yet
def test_health_without_weights_says_how_to_fix(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model"] == "clef-flash" and health["ready"] is False
    assert "./run.sh download" in health["detail"]


def test_classify_without_weights_is_503_with_the_same_hint(client):
    r = upload(client, image_bytes())
    assert r.status_code == 503 and "./run.sh download" in r.json()["detail"]


def test_cors_only_for_the_ui(client):
    r = client.get("/health", headers={"Origin": "http://localhost:8080"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8080"
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_weights_missing_never_downloads(monkeypatch):
    """Without the weights in the local cache the server must not start a 19 GB download."""
    calls = []

    def fake_snapshot(repo, local_files_only=False, **kw):
        calls.append(local_files_only)
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    with pytest.raises(clef_backend.WeightsMissing, match=r"\./run\.sh download \(~19 GB\)"):
        clef_backend.resolve_model_path("Cloudflare/clef-flash")
    assert calls == [True]  # only ever asked the local cache


# --------------------------------------------------------------------------------------- image handling
def test_resize_and_exif():
    img = Image.new("RGB", (4000, 1000))
    exif = img.getexif()
    exif[0x0112] = 6  # orientation: rotate 90° clockwise
    out = load_image(image_bytes(img.size, fmt="JPEG", exif=exif), max_side=1024)
    assert out.mode == "RGB"
    assert out.size == (256, 1024)  # rotated + downscaled


@pytest.mark.parametrize(("fmt", "mode"), [("PNG", "RGBA"), ("JPEG", "RGB"), ("WEBP", "RGB")])
def test_load_image_formats(fmt, mode):
    color = (10, 20, 30, 128)[: len(mode)]
    out = load_image(image_bytes((3000, 1500), color, fmt=fmt, mode=mode), max_side=1024)
    assert out.mode == "RGB" and max(out.size) == 1024


def test_large_jpeg_is_decoded_downscaled_to_exact_size():
    # 4096x3072 lets libjpeg decode at 1/2 scale; the result must still be exactly max_side wide.
    out = load_image(image_bytes((4096, 3072), fmt="JPEG"), max_side=1024)
    assert out.mode == "RGB" and out.size == (1024, 768)


def test_decompression_bomb_rejected(monkeypatch):
    import imaging

    monkeypatch.setattr(imaging, "MAX_PIXELS", 100 * 100)
    with pytest.raises(BadImage, match="too large"):
        load_image(image_bytes((200, 200)), max_side=1024)


@pytest.mark.parametrize("data", [b"", b"definitely not an image"])
def test_load_image_rejects_garbage(data):
    with pytest.raises(BadImage):
        load_image(data, max_side=1024)


def test_prompt_schema_loads():
    prompt = load_prompt()
    question = prompt.record(["img"])["questions"][prompt.question_id]
    assert question["type"] == "choice" and set(question["criteria"]) == {"hotdog", "not_hotdog"}


# ------------------------------------------------------------------------- backend choice + MLX backend
@pytest.mark.parametrize(
    ("env", "mac", "expected"),
    [
        ("auto", True, "mlx"),
        ("auto", False, "torch"),
        ("", True, "mlx"),
        ("torch", True, "torch"),
        ("mlx", False, "mlx"),
    ],
)
def test_backend_choice(monkeypatch, env, mac, expected):
    monkeypatch.setenv("CLEF_BACKEND", env)
    monkeypatch.setattr(clef_backend, "is_apple_silicon", lambda: mac)
    monkeypatch.delenv("CLEF_MODEL", raising=False)
    kind = clef_backend.choose_backend()
    assert kind == expected
    assert (
        clef_backend.model_repo(kind)
        == {"mlx": "mlx-community/clef-flash-4bit", "torch": "Cloudflare/clef-flash"}[kind]
    )


def test_bad_backend_name_rejected(monkeypatch):
    monkeypatch.setenv("CLEF_BACKEND", "tpu")
    with pytest.raises(ValueError):
        clef_backend.choose_backend()


def test_mlx_without_weights_names_the_6gb_download(monkeypatch):
    monkeypatch.setenv("CLEF_BACKEND", "mlx")
    monkeypatch.delenv("CLEF_MODEL", raising=False)
    with TestClient(app_module.create_app()) as c:
        wait_until_settled(c.app.state.backend)
        health = c.get("/health").json()
    assert health["ready"] is False and health["backend"] == "mlx"
    assert health["model_id"] == "mlx-community/clef-flash-4bit"
    assert "./run.sh download (~6 GB)" in health["detail"]


FAKE_CLEF_MLX = """
import threading
CALLS = []

class _Model:
    def __init__(self):
        self.thread = threading.get_ident()

    def predict(self, record):
        assert threading.get_ident() == self.thread, "MLX must stay on the thread that loaded it"
        (qid, q), = record["questions"].items()
        assert q["type"] == "choice" and len(record["images"]) == 1
        image = record["images"][0]
        CALLS.append(image.size)
        p = 0.9 if image.getpixel((0, 0))[0] > 150 else 0.2  # red pictures are hot dogs
        return {qid: {"not_hotdog": 1 - p, "hotdog": p}}  # keyed by option id, any order

def load(path, **kw):
    return _Model()
"""


@pytest.fixture()
def fake_mlx_model_dir(monkeypatch, tmp_path):
    """A model dir whose clef_mlx.py has the real load()/predict() interface but needs no Apple GPU."""
    model_dir = tmp_path / "clef-flash-4bit"
    model_dir.mkdir()
    (model_dir / "clef_mlx.py").write_text(FAKE_CLEF_MLX)
    monkeypatch.delitem(sys.modules, "clef_mlx", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))  # undo the backend's sys.path insert afterwards
    return model_dir


def test_mlx_backend_classifies_via_clef_mlx(monkeypatch, fake_mlx_model_dir):
    monkeypatch.setenv("CLEF_BACKEND", "mlx")
    monkeypatch.setenv("CLEF_MODEL", str(fake_mlx_model_dir))

    with TestClient(app_module.create_app()) as c:
        wait_until_settled(c.app.state.backend)
        health = c.get("/health").json()
        assert health["ready"] is True and health["device"] == "mlx · 4-bit"
        assert health["weights"] == "4-bit MLX quantization" and "detail" not in health
        hot = upload(c, image_bytes(color=HOT_RED)).json()
        cold = upload(c, image_bytes((100, 100), COLD_BLUE)).json()

    assert hot["model"] == "clef-flash" and hot["label"] == "hotdog" and hot["is_hotdog"] is True
    assert hot["confidence"] == pytest.approx(0.9)
    assert hot["probabilities"] == pytest.approx({"hotdog": 0.9, "not_hotdog": 0.1})
    assert cold["label"] == "not_hotdog" and cold["is_hotdog"] is False
    assert cold["confidence"] == pytest.approx(0.8)
    assert hot["latency_ms"] >= 0 and hot["total_ms"] >= hot["latency_ms"]
    assert sys.modules["clef_mlx"].CALLS[0] == (64, 64)  # the warmup ran before the first request


def test_mlx_backend_rejects_bad_uploads(monkeypatch, fake_mlx_model_dir):
    monkeypatch.setenv("CLEF_BACKEND", "mlx")
    monkeypatch.setenv("CLEF_MODEL", str(fake_mlx_model_dir))
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")  # ~1 KB

    with TestClient(app_module.create_app()) as c:
        wait_until_settled(c.app.state.backend)
        assert upload(c, b"definitely not an image").status_code == 400
        assert upload(c, b"x" * 2000).status_code == 413


def test_mlx_needs_apple_silicon_message(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "clef_mlx", raising=False)
    monkeypatch.setattr(clef_backend, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    backend = clef_backend.MlxBackend(str(tmp_path), load_prompt())
    backend.start()
    wait_until_settled(backend)
    assert not backend.ready and "CLEF_BACKEND=torch" in backend.error
