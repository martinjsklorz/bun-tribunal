"""Server tests: missing artifacts, and a real tiny model trained by the notebook in SMOKE mode.

Run from cnn/:  pytest -q tests
The real-model tests execute train_hotdog_cnn.ipynb with SMOKE=1 ARCH=mobilenet_v3_large into a temp dir
(about a minute on CPU).
Set CNN_TEST_ARTIFACTS=/path/to/artifacts to reuse existing artifacts instead.
"""
from __future__ import annotations

import importlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

CNN_DIR = Path(__file__).resolve().parents[1]
SERVER_DIR = CNN_DIR / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(CNN_DIR))


def make_client(monkeypatch, **env) -> TestClient:
    for k in ("CNN_ARTIFACTS", "DEVICE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import app as app_module

    app_module = importlib.reload(app_module)  # model is loaded at import time
    return TestClient(app_module.app)


def png_bytes(color=(200, 60, 40), size=(320, 240), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def check_classify_body(body: dict):
    assert body["model"] == "cnn"
    assert body["label"] in ("hotdog", "not_hotdog")
    assert body["is_hotdog"] == (body["label"] == "hotdog")
    assert set(body["probabilities"]) == {"hotdog", "not_hotdog"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["confidence"] == pytest.approx(body["probabilities"][body["label"]], abs=1e-4)
    assert body["confidence"] >= 0.5
    assert 0 <= body["latency_ms"] <= body["total_ms"]


# ---------------- missing artifacts ----------------
def test_missing_artifacts(monkeypatch, tmp_path):
    c = make_client(monkeypatch, CNN_ARTIFACTS=tmp_path / "nope")
    b = c.get("/health").json()
    assert b["ready"] is False
    assert b["detail"] == "CNN not trained yet — run ./run.sh train (or open cnn/train_hotdog_cnn.ipynb)"
    assert "./run.sh train" in b["detail"]
    r = c.post("/classify", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 503 and r.json()["detail"] == b["detail"]


def test_cors_only_for_the_ui(monkeypatch, tmp_path):
    c = make_client(monkeypatch, CNN_ARTIFACTS=tmp_path / "nope")
    pre = {"Access-Control-Request-Method": "POST"}
    ok = c.options("/classify", headers={"Origin": "http://localhost:8080", **pre})
    bad = c.options("/classify", headers={"Origin": "https://evil.example", **pre})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8080"
    assert "access-control-allow-origin" not in bad.headers


def test_meta_without_weights_counts_as_untrained(monkeypatch, tmp_path):
    (tmp_path / "model_meta.json").write_text(json.dumps({"arch": "convnext_tiny", "classes": ["hotdog", "not_hotdog"]}))
    b = make_client(monkeypatch, CNN_ARTIFACTS=tmp_path).get("/health").json()
    assert b["ready"] is False and "./run.sh train" in b["detail"]


# ---------------- real SMOKE-trained model ----------------
@pytest.fixture(scope="session")
def smoke_artifacts(tmp_path_factory) -> Path:
    existing = os.environ.get("CNN_TEST_ARTIFACTS")
    if existing:
        return Path(existing)
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    out = tmp_path_factory.mktemp("artifacts")
    keys = ("SMOKE", "ARCH", "CNN_ARTIFACTS", "MPLBACKEND", "DEVICE")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(SMOKE="1", ARCH="mobilenet_v3_large", CNN_ARTIFACTS=str(out), MPLBACKEND="Agg", DEVICE="cpu")
    try:
        nb = nbformat.read(CNN_DIR / "train_hotdog_cnn.ipynb", as_version=4)
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(CNN_DIR)}}).execute()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert (out / "model_meta.json").exists() and (out / "hotdog_cnn.pt").exists()
    return out


def test_real_health(monkeypatch, smoke_artifacts):
    c = make_client(monkeypatch, CNN_ARTIFACTS=smoke_artifacts, DEVICE="cpu")
    b = c.get("/health").json()
    assert b == {**b, "status": "ok", "model": "cnn", "device": "cpu", "ready": True}
    assert b["weights"] == "state_dict"
    if not os.environ.get("CNN_TEST_ARTIFACTS"):  # our own SMOKE model; a user's model may be any arch
        assert b["arch"] == "mobilenet_v3_large"
    assert b["temperature"] > 0


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_real_classify(monkeypatch, smoke_artifacts, fmt, mime):
    c = make_client(monkeypatch, CNN_ARTIFACTS=smoke_artifacts, DEVICE="cpu")
    r = c.post("/classify", files={"file": (f"a.{fmt.lower()}", png_bytes(size=(1024, 640), fmt=fmt), mime)})
    assert r.status_code == 200, r.text
    check_classify_body(r.json())


def test_real_bad_image(monkeypatch, smoke_artifacts):
    c = make_client(monkeypatch, CNN_ARTIFACTS=smoke_artifacts, DEVICE="cpu")
    r = c.post("/classify", files={"file": ("x.jpg", b"\xff\xd8\xff garbage", "image/jpeg")})
    assert r.status_code == 400
    assert c.post("/classify", files={"file": ("e.png", b"", "image/png")}).status_code == 400


def test_real_torchscript_fallback(monkeypatch, smoke_artifacts, tmp_path):
    ts_only = tmp_path / "ts_only"
    ts_only.mkdir()
    for f in ("model_meta.json", "hotdog_cnn.ts.pt"):
        shutil.copy(smoke_artifacts / f, ts_only / f)
    c = make_client(monkeypatch, CNN_ARTIFACTS=ts_only, DEVICE="cpu")
    assert c.get("/health").json()["weights"] == "torchscript"
    r = c.post("/classify", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    check_classify_body(r.json())


def test_unsupported_arch_falls_back_to_torchscript(monkeypatch, smoke_artifacts, tmp_path):
    old = tmp_path / "old"
    shutil.copytree(smoke_artifacts, old)
    meta = json.loads((old / "model_meta.json").read_text())
    (old / "model_meta.json").write_text(json.dumps({**meta, "arch": "scratch"}))
    c = make_client(monkeypatch, CNN_ARTIFACTS=old, DEVICE="cpu")
    b = c.get("/health").json()
    assert b["ready"] is True and b["weights"] == "torchscript"


def test_unsupported_arch_without_torchscript(monkeypatch, smoke_artifacts, tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    shutil.copy(smoke_artifacts / "hotdog_cnn.pt", old / "hotdog_cnn.pt")
    meta = json.loads((smoke_artifacts / "model_meta.json").read_text())
    (old / "model_meta.json").write_text(json.dumps({**meta, "arch": "scratch"}))
    b = make_client(monkeypatch, CNN_ARTIFACTS=old, DEVICE="cpu").get("/health").json()
    assert b["ready"] is False
    assert b["detail"] == "artifacts are from an unsupported architecture 'scratch' — retrain with ./run.sh train"
