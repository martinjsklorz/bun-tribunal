"""CLEF-Flash server tests. They run without any weights, so the server is in its
"not set up" state; image handling, the prompt and the download policy are tested directly."""
import io
import sys
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module


@pytest.fixture()
def client():
    with TestClient(app_module.create_app()) as c:
        backend = c.app.state.backend
        deadline = time.time() + 120  # fails fast without weights (after importing torch, which can take a while)
        while backend.error is None and time.time() < deadline:
            time.sleep(0.05)
        yield c


def _png(size=(320, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 80, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_without_weights_says_how_to_fix(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "clef-flash" and h["ready"] is False
    assert "./run.sh download" in h["detail"]


def test_classify_without_weights_is_503_with_the_same_hint(client):
    r = client.post("/classify", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 503 and "./run.sh download" in r.json()["detail"]


def test_cors_only_for_the_ui(client):
    r = client.get("/health", headers={"Origin": "http://localhost:8080"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8080"
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_weights_missing_never_downloads(monkeypatch):
    """Without the weights in the local cache the server must not start a 19 GB download."""
    import huggingface_hub
    import clef_backend

    calls = []

    def fake_snapshot(repo, local_files_only=False, **kw):
        calls.append(local_files_only)
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    with pytest.raises(clef_backend.WeightsMissing, match=r"\./run\.sh download"):
        clef_backend.resolve_model_path("Cloudflare/clef-flash")
    assert calls == [True]  # only ever asked the local cache


def test_resize_and_exif():
    from imaging import load_image
    img = Image.new("RGB", (4000, 1000))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 CW
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    out = load_image(buf.getvalue(), max_side=1024)
    assert out.mode == "RGB"
    assert out.size == (256, 1024)  # rotated + downscaled


@pytest.mark.parametrize("fmt,mode", [("PNG", "RGBA"), ("JPEG", "RGB"), ("WEBP", "RGB")])
def test_load_image_formats(fmt, mode):
    from imaging import load_image
    buf = io.BytesIO()
    Image.new(mode, (3000, 1500), (10, 20, 30, 128)[: len(mode)]).save(buf, format=fmt)
    out = load_image(buf.getvalue(), max_side=1024)
    assert out.mode == "RGB" and max(out.size) == 1024


@pytest.mark.parametrize("data", [b"", b"definitely not an image"])
def test_load_image_rejects_garbage(data):
    from imaging import BadImage, load_image
    with pytest.raises(BadImage):
        load_image(data, max_side=1024)


def test_prompt_schema_loads():
    from prompt import load_prompt
    p = load_prompt()
    rec = p.record(["img"])
    q = rec["questions"][p.question_id]
    assert q["type"] == "choice" and set(q["criteria"]) == {"hotdog", "not_hotdog"}


# ---------------------------------------------------------------- backend choice + MLX (4-bit) backend
@pytest.mark.parametrize("env,mac,expected", [
    ("auto", True, "mlx"), ("auto", False, "torch"), ("", True, "mlx"),
    ("torch", True, "torch"), ("mlx", False, "mlx"),
])
def test_backend_choice(monkeypatch, env, mac, expected):
    import clef_backend
    monkeypatch.setenv("CLEF_BACKEND", env)
    monkeypatch.setattr(clef_backend, "is_apple_silicon", lambda: mac)
    monkeypatch.delenv("CLEF_MODEL", raising=False)
    kind = clef_backend.choose_backend()
    assert kind == expected
    assert clef_backend.model_repo(kind) == {"mlx": "mlx-community/clef-flash-4bit",
                                             "torch": "Cloudflare/clef-flash"}[expected]


def test_bad_backend_name_rejected(monkeypatch):
    import clef_backend
    monkeypatch.setenv("CLEF_BACKEND", "tpu")
    with pytest.raises(ValueError):
        clef_backend.choose_backend()


def _wait(backend, timeout=30):
    deadline = time.time() + timeout
    while not backend.ready and backend.error is None and time.time() < deadline:
        time.sleep(0.02)


def test_mlx_without_weights_names_the_6gb_download(monkeypatch):
    monkeypatch.setenv("CLEF_BACKEND", "mlx")
    monkeypatch.delenv("CLEF_MODEL", raising=False)
    with TestClient(app_module.create_app()) as c:
        _wait(c.app.state.backend)
        h = c.get("/health").json()
    assert h["ready"] is False and h["backend"] == "mlx" and h["model_id"] == "mlx-community/clef-flash-4bit"
    assert "./run.sh download (~6 GB)" in h["detail"]


FAKE_CLEF_MLX = '''
import threading
CALLS = []

class _Model:
    def __init__(self):
        self.thread = threading.get_ident()
    def predict(self, record):
        assert threading.get_ident() == self.thread, "MLX must stay on the thread that loaded it"
        (qid, q), = record["questions"].items()
        assert q["type"] == "choice" and len(record["images"]) == 1
        CALLS.append(record["images"][0].size)
        red = record["images"][0].getpixel((0, 0))[0] > 150      # "red pictures are hot dogs"
        p = 0.9 if red else 0.2
        return {qid: {"not_hotdog": 1 - p, "hotdog": p}}       # keyed by option id, any order

def load(path, **kw):
    return _Model()
'''


def test_mlx_backend_classifies_via_clef_mlx(monkeypatch, tmp_path):
    """A stand-in for the repo's clef_mlx.py: same load()/predict() interface, no Apple GPU needed."""
    model_dir = tmp_path / "clef-flash-4bit"
    model_dir.mkdir()
    (model_dir / "clef_mlx.py").write_text(FAKE_CLEF_MLX)
    monkeypatch.setenv("CLEF_BACKEND", "mlx")
    monkeypatch.setenv("CLEF_MODEL", str(model_dir))
    monkeypatch.delitem(sys.modules, "clef_mlx", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    with TestClient(app_module.create_app()) as c:
        _wait(c.app.state.backend)
        h = c.get("/health").json()
        assert h["ready"] is True and h["device"] == "mlx · 4-bit" and h["weights"] == "4-bit MLX quantization"
        hot = c.post("/classify", files={"file": ("x.png", _png(), "image/png")}).json()
        buf = io.BytesIO()
        Image.new("RGB", (100, 100), (20, 120, 200)).save(buf, format="PNG")
        cold = c.post("/classify", files={"file": ("y.png", buf.getvalue(), "image/png")}).json()

    assert hot["label"] == "hotdog" and hot["confidence"] == pytest.approx(0.9)
    assert hot["probabilities"] == pytest.approx({"hotdog": 0.9, "not_hotdog": 0.1})
    assert cold["label"] == "not_hotdog" and cold["confidence"] == pytest.approx(0.8)
    assert hot["latency_ms"] >= 0 and hot["total_ms"] >= hot["latency_ms"]
    assert sys.modules["clef_mlx"].CALLS[0] == (64, 64)  # the warmup ran before the first request


def test_mlx_needs_apple_silicon_message(monkeypatch, tmp_path):
    import clef_backend
    from prompt import load_prompt
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    monkeypatch.delitem(sys.modules, "clef_mlx", raising=False)
    monkeypatch.setattr(clef_backend, "is_apple_silicon", lambda: False)
    b = clef_backend.MlxBackend(str(model_dir), load_prompt())
    b.start()
    _wait(b)
    assert not b.ready and "CLEF_BACKEND=torch" in b.error
