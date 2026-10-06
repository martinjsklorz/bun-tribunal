"""CLEF-Flash server tests. They run without the 19 GB weights, so the server is in its
"not set up" state; image handling, the prompt and the download policy are tested directly."""
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module


@pytest.fixture()
def client():
    with TestClient(app_module.create_app()) as c:
        backend = c.app.state.backend
        deadline = time.time() + 30  # the background loader fails fast without weights
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
