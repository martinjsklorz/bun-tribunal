"""LLM server tests: parsing, logprob extraction, fallbacks, and the HTTP contract (no real API calls)."""
import importlib
import io
import json
import math
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import llm_client  # noqa: E402
from llm_client import LLMClient, LLMConfig, label_prob_from_logprobs, parse_answer  # noqa: E402


def img_bytes(fmt="PNG", size=(320, 240), color=(200, 90, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def tok(t, lp=0.0, top=None):
    return {"token": t, "logprob": lp, "top_logprobs": top or []}


def logprob_content(p_hot: float):
    """Token stream for {"label": "hotdog", ...} with alternatives at the value position."""
    lp_hot, lp_not = math.log(p_hot), math.log(1 - p_hot)
    first = "hot" if p_hot >= 0.5 else "not"
    return {"content": [
        tok('{"'), tok("label"), tok('":'), tok(' "'),
        tok(first, max(lp_hot, lp_not), [{"token": "hot", "logprob": lp_hot}, {"token": "not", "logprob": lp_not},
                                         {"token": "maybe", "logprob": -12.0}]),
        tok("dog"), tok('",'),
    ]}


def completion(content, logprobs=None):
    choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"id": "x", "object": "chat.completion", "choices": [choice]}


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("content,label,conf", [
    ('{"label": "hotdog", "confidence": 0.93, "reason": "bun + sausage"}', "hotdog", 0.93),
    ('```json\n{"label":"not_hotdog","confidence":88,"reason":"pizza"}\n```', "not_hotdog", 0.88),
    ('Sure! {"label": "Hot Dog", "confidence": "0.7"}', "hotdog", 0.7),
    ('{"label": "not hotdog"}', "not_hotdog", None),
    ("This is not a hot dog, it is a taco.", "not_hotdog", None),
    ("Definitely a hot dog.", "hotdog", None),
])
def test_parse_answer(content, label, conf):
    a = parse_answer(content)
    assert a["label"] == label
    assert (a["confidence"] is None) if conf is None else a["confidence"] == pytest.approx(conf)


def test_parse_answer_garbage():
    with pytest.raises(llm_client.UpstreamError):
        parse_answer("I like turtles.")


def test_logprobs_extraction():
    assert label_prob_from_logprobs(logprob_content(0.9)) == pytest.approx(0.9, abs=1e-6)
    assert label_prob_from_logprobs(logprob_content(0.2)) == pytest.approx(0.2, abs=1e-6)
    assert label_prob_from_logprobs(None) is None
    assert label_prob_from_logprobs({"content": [tok("hello")]}) is None


# ---------------------------------------------------------------- client against a fake API
def make_client(handler, **cfg):
    cfg.setdefault("backoff_s", 0.0)  # no real sleeping in tests
    c = LLMConfig(base_url="http://fake/v1", api_key="k", model=cfg.pop("model", "fake-vision"), **cfg)
    return LLMClient(c, transport=httpx.MockTransport(handler))


async def _classify(client):
    try:
        return await client.classify(Image.new("RGB", (1600, 900), (10, 200, 30)))
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_client_uses_logprobs_and_sends_image():
    seen = {}

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.update(body)
        assert req.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json=completion('{"label":"hotdog","confidence":0.6,"reason":"bun"}', logprob_content(0.97)))

    r = await _classify(make_client(handler))
    assert r["confidence_source"] == "logprobs"
    assert r["probabilities"]["hotdog"] == pytest.approx(0.97, abs=1e-6)
    assert r["reason"] == "bun"
    img_part = seen["messages"][1]["content"][1]
    assert img_part["type"] == "image_url" and img_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert seen["logprobs"] is True and seen["response_format"] == {"type": "json_object"} and seen["temperature"] == 0
    # image was downscaled before upload
    import base64
    raw = base64.b64decode(img_part["image_url"]["url"].split(",", 1)[1])
    assert max(Image.open(io.BytesIO(raw)).size) <= 768


@pytest.mark.anyio
async def test_client_falls_back_when_server_rejects_logprobs():
    calls = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        calls.append(body)
        if "logprobs" in body:
            return httpx.Response(400, json={"error": {"message": "logprobs is not supported for this model"}})
        return httpx.Response(200, json=completion('{"label":"not_hotdog","confidence":0.8,"reason":"a shoe"}'))

    client = make_client(handler)
    r = await client.classify(Image.new("RGB", (64, 64)))
    assert len(calls) == 2 and "logprobs" not in calls[1]
    assert r["confidence_source"] == "self-reported"
    assert r["probabilities"]["not_hotdog"] == pytest.approx(0.8)
    # remembered: next call doesn't try logprobs again
    await client.classify(Image.new("RGB", (64, 64)))
    assert len(calls) == 3 and "logprobs" not in calls[2]
    await client.aclose()


@pytest.mark.anyio
async def test_client_upstream_errors():
    with pytest.raises(llm_client.UpstreamError, match=r"fake: boom \(failed 3× in a row\)"):
        await _classify(make_client(lambda req: httpx.Response(500, json={"error": {"message": "boom"}})))
    with pytest.raises(llm_client.UpstreamError, match="could not parse"):
        await _classify(make_client(lambda req: httpx.Response(200, json=completion("I like turtles"))))

    def slow(req):
        raise httpx.ReadTimeout("slow", request=req)
    with pytest.raises(llm_client.UpstreamTimeout):
        await _classify(make_client(slow))


# The exact payload OpenRouter sent when the free Nvidia endpoint was saturated: HTTP 200, no choices.
OPENROUTER_EXHAUSTED = {"id": "gen-1791284586-3qhqWSber1qcZJKrEPSU", "error": {
    "message": "Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (377/16)",
    "code": 502, "metadata": {"error_type": "provider_error"}}}


@pytest.mark.anyio
async def test_error_in_200_body_is_retried_then_succeeds():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json=OPENROUTER_EXHAUSTED)
        return httpx.Response(200, json=completion('{"label":"hotdog","confidence":0.9,"reason":"bun"}'))

    r = await _classify(make_client(handler))
    assert len(calls) == 2 and r["attempts"] == 2 and r["probabilities"]["hotdog"] == pytest.approx(0.9)


@pytest.mark.anyio
async def test_persistent_provider_error_gives_readable_message():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, json=OPENROUTER_EXHAUSTED)

    with pytest.raises(llm_client.UpstreamError) as ei:
        await _classify(make_client(handler, model="nvidia/some-model:free"))
    msg = str(ei.value)
    assert len(calls) == 3  # 1 + LLM_RETRIES (2)
    assert "Nvidia: ResourceExhausted" in msg and "failed 3×" in msg and ":free" in msg
    assert "unexpected response shape" not in msg


@pytest.mark.anyio
async def test_non_transient_error_is_not_retried():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(404, json={"error": {"message": "No endpoints found for model foo"}})

    with pytest.raises(llm_client.UpstreamError, match="No endpoints found") as ei:
        await _classify(make_client(handler))
    assert len(calls) == 1 and "failed" not in str(ei.value)


@pytest.mark.anyio
async def test_retry_after_header_and_connection_errors():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=req)
        if len(calls) == 2:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=completion('{"label":"not_hotdog","confidence":0.7}'))

    r = await _classify(make_client(handler))
    assert len(calls) == 3 and r["attempts"] == 3 and r["probabilities"]["not_hotdog"] == pytest.approx(0.7)


@pytest.mark.anyio
async def test_reasoning_model_out_of_tokens_hint():
    resp = completion(None)
    resp["choices"][0]["finish_reason"] = "length"
    with pytest.raises(llm_client.UpstreamError, match="LLM_MAX_TOKENS=2000"):
        await _classify(make_client(lambda req: httpx.Response(200, json=resp)))


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------- HTTP contract
def load_app(monkeypatch, **env):
    for k in ("LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(HERE)
    import app as app_module
    return importlib.reload(app_module)


def check_body(b, model_id):
    assert b["model"] == "llm" and b["label"] in ("hotdog", "not_hotdog")
    assert b["is_hotdog"] == (b["label"] == "hotdog")
    assert abs(sum(b["probabilities"].values()) - 1) < 1e-6
    assert b["confidence"] == b["probabilities"][b["label"]] >= 0.5
    assert b["latency_ms"] >= 0 and b["total_ms"] >= b["latency_ms"]
    assert b["confidence_source"] in ("logprobs", "self-reported") and b["model_id"] == model_id


def test_missing_key_not_ready(monkeypatch):
    m = load_app(monkeypatch, LLM_BASE_URL="https://api.openai.com/v1")
    monkeypatch.setattr(m, "load_dotenv", lambda *_: None)
    with TestClient(m.app) as c:
        h = c.get("/health").json()
        assert h["ready"] is False and "LLM_API_KEY" in h["detail"] and h["base_url_host"] == "api.openai.com"
        r = c.post("/classify", files={"file": ("a.png", img_bytes(), "image/png")})
        assert r.status_code == 503 and "LLM_API_KEY" in r.json()["detail"]


@pytest.mark.parametrize("handler,status,body_check", [
    (lambda req: httpx.Response(200, json={"data": []}) if req.url.path.endswith("/models")
     else httpx.Response(200, json=completion('{"label":"hotdog","confidence":0.9,"reason":"yum"}', logprob_content(0.95))),
     200, lambda b: b["confidence_source"] == "logprobs" and b["probabilities"]["hotdog"] == pytest.approx(0.95, abs=1e-5)),
    (lambda req: httpx.Response(200, json={"data": []}) if req.url.path.endswith("/models")
     else httpx.Response(429, json={"error": {"message": "rate limited"}}),
     502, lambda b: "rate limited" in b["detail"] and "failed 3×" in b["detail"]),
])
def test_real_mode_with_fake_api(monkeypatch, handler, status, body_check):
    m = load_app(monkeypatch, LLM_BASE_URL="http://fake/v1", LLM_API_KEY="k", LLM_MODEL="fake-vision", LLM_RETRY_BACKOFF="0")
    real_init = m.LLMClient.__init__
    monkeypatch.setattr(m.LLMClient, "__init__",
                        lambda self, cfg, transport=None: real_init(self, cfg, transport=httpx.MockTransport(handler)))
    with TestClient(m.app) as c:
        h = c.get("/health").json()
        assert h["ready"] is True and h["model_id"] == "fake-vision"
        r = c.post("/classify", files={"file": ("a.jpg", img_bytes("JPEG"), "image/jpeg")})
        assert r.status_code == status, r.text
        if status == 200:
            check_body(r.json(), "fake-vision")
        assert body_check(r.json())


def test_dotenv_inline_comments_and_last_wins(tmp_path, monkeypatch):
    from llm_client import load_dotenv
    for k in ("HD_A", "HD_B", "HD_C", "HD_D"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HD_D", "from-env")
    f = tmp_path / ".env"
    f.write_text('# comment\nHD_A=60   # seconds\nHD_B=first\nHD_B=second\nHD_C="has # inside"\nHD_D=from-file\n')
    load_dotenv(f)
    import os
    assert os.environ["HD_A"] == "60" and os.environ["HD_B"] == "second"
    assert os.environ["HD_C"] == "has # inside" and os.environ["HD_D"] == "from-env"


def test_cors_only_for_the_ui(monkeypatch):
    m = load_app(monkeypatch, LLM_BASE_URL="https://api.openai.com/v1")  # not configured: no network needed
    with TestClient(m.app) as c:
        pre = {"Access-Control-Request-Method": "POST"}
        ok = c.options("/classify", headers={"Origin": "http://192.168.1.20:8080", **pre})
        bad = c.options("/classify", headers={"Origin": "https://evil.example", **pre})
        assert ok.headers.get("access-control-allow-origin") == "http://192.168.1.20:8080"
        assert "access-control-allow-origin" not in bad.headers


def test_bad_or_empty_image_is_400(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json={"data": []})

    m = load_app(monkeypatch, LLM_BASE_URL="http://fake/v1", LLM_API_KEY="k", LLM_MODEL="fake-vision")
    real_init = m.LLMClient.__init__
    monkeypatch.setattr(m.LLMClient, "__init__",
                        lambda self, cfg, transport=None: real_init(self, cfg, transport=httpx.MockTransport(handler)))
    with TestClient(m.app) as c:
        assert c.post("/classify", files={"file": ("x.jpg", b"garbage", "image/jpeg")}).status_code == 400
        assert c.post("/classify", files={"file": ("e.png", b"", "image/png")}).status_code == 400
    assert not any(p.endswith("/chat/completions") for p in calls)  # never sent to the API
