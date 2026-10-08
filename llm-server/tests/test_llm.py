"""LLM server tests: parsing, logprob extraction, image prep, retries and the HTTP contract (no real API calls)."""

import asyncio
import base64
import importlib
import io
import json
import math
import os
import struct
import sys
import time
import zlib
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import llm_client  # noqa: E402  # needs the sys.path entry above
from llm_client import (  # noqa: E402
    Answer,
    LLMClient,
    LLMConfig,
    image_to_data_url,
    label_prob_from_logprobs,
    parse_answer,
    retry_after_s,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def img_bytes(fmt="PNG", size=(320, 240), color=(200, 90, 40), **save):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt, **save)
    return buf.getvalue()


def broken_png() -> bytes:
    """A PNG whose second image-data chunk has a garbage type: Pillow raises SyntaxError("broken PNG file")."""
    good = img_bytes("PNG", size=(32, 32))
    start = good.find(b"IDAT") - 4
    length = struct.unpack(">I", good[start : start + 4])[0]
    data = good[start + 8 : start + 8 + length]

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    return good[:start] + chunk(b"IDAT", data[:10]) + chunk(b"\0\0\0\0", data[10:]) + good[start + 12 + length :]


def decode_data_url(url: str) -> Image.Image:
    assert url.startswith("data:image/jpeg;base64,")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def tok(t, lp=0.0, top=None):
    return {"token": t, "logprob": lp, "top_logprobs": top or []}


def logprob_content(p_hot: float):
    """Token stream for {"label": "hotdog", ...} with alternatives at the value position."""
    lp_hot, lp_not = math.log(p_hot), math.log(1 - p_hot)
    first = "hot" if p_hot >= 0.5 else "not"
    alternatives = [
        {"token": "hot", "logprob": lp_hot},
        {"token": "not", "logprob": lp_not},
        {"token": "maybe", "logprob": -12.0},
    ]
    return {
        "content": [
            tok('{"'),
            tok("label"),
            tok('":'),
            tok(' "'),
            tok(first, max(lp_hot, lp_not), alternatives),
            tok("dog"),
            tok('",'),
        ]
    }


def completion(content, logprobs=None):
    choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"id": "x", "object": "chat.completion", "choices": [choice]}


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    "content,label,conf",
    [
        ('{"label": "hotdog", "confidence": 0.93, "reason": "bun + sausage"}', "hotdog", 0.93),
        ('```json\n{"label":"not_hotdog","confidence":88,"reason":"pizza"}\n```', "not_hotdog", 0.88),
        ('Sure! {"label": "Hot Dog", "confidence": "0.7"}', "hotdog", 0.7),
        ('{"label": "not hotdog"}', "not_hotdog", None),
        ('{"label": "hotdog", "confidence": "NaN"}', "hotdog", None),
        ("This is not a hot dog, it is a taco.", "not_hotdog", None),
        ("Definitely a hot dog.", "hotdog", None),
    ],
)
def test_parse_answer(content, label, conf):
    a = parse_answer(content)
    assert a.label == label
    assert (a.confidence is None) if conf is None else a.confidence == pytest.approx(conf)


@pytest.mark.parametrize("text", ["I like turtles.", '{"label": "maybe"}'])
def test_parse_answer_garbage(text):
    with pytest.raises(llm_client.UpstreamError):
        parse_answer(text)


@pytest.mark.parametrize(
    "answer,p_hot",
    [
        (Answer("hotdog", 0.9), 0.9),
        (Answer("not_hotdog", 0.8), 0.2),
        (Answer("hotdog", 0.1), 0.5),  # a label below 50% is still the model's pick
        (Answer("not_hotdog"), 0.25),  # no number: default confidence
    ],
)
def test_self_reported_probability(answer, p_hot):
    assert answer.p_hotdog() == pytest.approx(p_hot)


def test_logprobs_extraction():
    assert label_prob_from_logprobs(logprob_content(0.9)) == pytest.approx(0.9, abs=1e-6)
    assert label_prob_from_logprobs(logprob_content(0.2)) == pytest.approx(0.2, abs=1e-6)
    assert label_prob_from_logprobs(None) is None
    assert label_prob_from_logprobs({"content": [tok("hello")]}) is None


def test_retry_after_parsing():
    assert retry_after_s(httpx.Headers({"retry-after": "3"})) == 3.0
    assert retry_after_s(httpx.Headers({"retry-after": "soon"})) is None
    assert retry_after_s(httpx.Headers()) is None
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert 25 < retry_after_s(httpx.Headers({"retry-after": later})) <= 30


# ---------------------------------------------------------------- image preparation
def test_image_is_bounded_jpeg_and_caller_image_untouched():
    src = Image.new("RGB", (1600, 900), (10, 200, 30))
    out = decode_data_url(image_to_data_url(src, 768))
    assert out.format == "JPEG" and out.size == (768, 432)
    assert src.size == (1600, 900)
    small = decode_data_url(image_to_data_url(Image.new("RGB", (100, 50)), 768))
    assert small.size == (100, 50)  # never upscaled


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_transparency_becomes_white(mode):
    src = Image.new("RGBA", (40, 40), (0, 0, 0, 0)).convert(mode)
    if mode == "P":
        src.info["transparency"] = src.getpixel((0, 0))
    out = decode_data_url(image_to_data_url(src, 768)).convert("RGB")
    assert all(c > 240 for c in out.getpixel((20, 20)))


def test_decode_image_is_upright_and_drafts_big_jpegs(monkeypatch):
    m = load_app(monkeypatch, LLM_BASE_URL="https://api.openai.com/v1")
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° CW to display
    upright = m.decode_image(img_bytes("JPEG", size=(200, 100), exif=exif.tobytes()), 768)
    assert upright.size == (100, 200)
    big = m.decode_image(img_bytes("JPEG", size=(4000, 3000)), 768)
    assert min(big.size) >= 768 and max(big.size) <= 2000  # decoded at reduced DCT scale
    assert m.decode_image(img_bytes("PNG", size=(4000, 3000)), 768).size == (4000, 3000)  # no draft for PNG
    for bad in (b"", b"garbage", broken_png()):
        with pytest.raises(ValueError):
            m.decode_image(bad, 768)


# ---------------------------------------------------------------- client against a fake API
def make_client(handler, **cfg):
    cfg.setdefault("backoff_s", 0.0)  # no real sleeping in tests
    c = LLMConfig(base_url="http://fake/v1", api_key="k", model=cfg.pop("model", "fake-vision"), **cfg)
    return LLMClient(c, transport=httpx.MockTransport(handler))


async def _classify(client):
    async with client:
        return await client.classify(Image.new("RGB", (1600, 900), (10, 200, 30)))


@pytest.mark.anyio
async def test_client_uses_logprobs_and_sends_image():
    seen = {}

    def handler(req: httpx.Request):
        seen.update(json.loads(req.content))
        assert req.headers["authorization"] == "Bearer k"
        assert req.headers["content-type"] == "application/json"
        content = completion('{"label":"hotdog","confidence":0.6,"reason":"bun"}', logprob_content(0.97))
        return httpx.Response(200, json=content)

    r = await _classify(make_client(handler))
    assert r.confidence_source == "logprobs" and r.label == "hotdog" and r.attempts == 1
    assert r.probabilities["hotdog"] == pytest.approx(0.97, abs=1e-6)
    assert r.reason == "bun"
    img_part = seen["messages"][1]["content"][1]
    assert img_part["type"] == "image_url" and img_part["image_url"]["detail"] == "low"
    assert max(decode_data_url(img_part["image_url"]["url"]).size) <= 768  # downscaled before upload
    assert seen["logprobs"] is True and seen["response_format"] == {"type": "json_object"} and seen["temperature"] == 0


@pytest.mark.anyio
async def test_optional_features_can_be_switched_off():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=completion('{"label":"hotdog","confidence":0.9}'))

    await _classify(make_client(handler, use_logprobs=False, json_mode=False, max_tokens=2000))
    assert "logprobs" not in seen and "response_format" not in seen and seen["max_tokens"] == 2000


@pytest.mark.anyio
async def test_client_falls_back_when_server_rejects_logprobs():
    calls = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        calls.append(body)
        if "logprobs" in body:
            return httpx.Response(400, json={"error": {"message": "logprobs is not supported for this model"}})
        return httpx.Response(200, json=completion('{"label":"not_hotdog","confidence":0.8,"reason":"a shoe"}'))

    async with make_client(handler) as client:
        r = await client.classify(Image.new("RGB", (64, 64)))
        assert len(calls) == 2 and "logprobs" not in calls[1] and "response_format" in calls[1]
        assert r.confidence_source == "self-reported"
        assert r.probabilities["not_hotdog"] == pytest.approx(0.8)
        # remembered: the next call doesn't try logprobs again
        await client.classify(Image.new("RGB", (64, 64)))
        assert len(calls) == 3 and "logprobs" not in calls[2]


@pytest.mark.anyio
async def test_unclear_400_drops_both_optional_features():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"message": "invalid request"}})
        return httpx.Response(200, json=completion('{"label":"hotdog","confidence":0.9}'))

    await _classify(make_client(handler))
    assert "logprobs" not in calls[1] and "response_format" not in calls[1]


@pytest.mark.anyio
async def test_client_upstream_errors():
    with pytest.raises(llm_client.UpstreamError, match=r"fake: boom \(failed 3× in a row\)"):
        await _classify(make_client(lambda req: httpx.Response(500, json={"error": {"message": "boom"}})))
    with pytest.raises(llm_client.UpstreamError, match="could not parse"):
        await _classify(make_client(lambda req: httpx.Response(200, json=completion("I like turtles"))))
    with pytest.raises(llm_client.UpstreamError, match="unexpected response shape"):
        await _classify(make_client(lambda req: httpx.Response(200, text="<html>proxy page</html>")))
    with pytest.raises(llm_client.UpstreamError, match="unexpected response shape"):
        await _classify(make_client(lambda req: httpx.Response(200, json={"choices": []})))

    calls = []

    def slow(req):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(llm_client.UpstreamTimeout):
        await _classify(make_client(slow))
    assert len(calls) == 1  # a timeout is not retried


# The exact payload OpenRouter sent when the free Nvidia endpoint was saturated: HTTP 200, no choices.
OPENROUTER_EXHAUSTED = {
    "id": "gen-1791284586-3qhqWSber1qcZJKrEPSU",
    "error": {
        "message": "Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (377/16)",
        "code": 502,
        "metadata": {"error_type": "provider_error"},
    },
}


@pytest.mark.anyio
async def test_error_in_200_body_is_retried_then_succeeds():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json=OPENROUTER_EXHAUSTED)
        return httpx.Response(200, json=completion('{"label":"hotdog","confidence":0.9,"reason":"bun"}'))

    r = await _classify(make_client(handler))
    assert len(calls) == 2 and r.attempts == 2 and r.probabilities["hotdog"] == pytest.approx(0.9)


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
async def test_retry_after_header_and_connection_errors(monkeypatch):
    calls, sleeps = [], []
    real_sleep = asyncio.sleep

    async def fake_sleep(s):
        sleeps.append(s)
        await real_sleep(0)

    monkeypatch.setattr(llm_client.asyncio, "sleep", fake_sleep)

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=req)
        if len(calls) == 2:
            return httpx.Response(429, headers={"retry-after": "60"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=completion('{"label":"not_hotdog","confidence":0.7}'))

    r = await _classify(make_client(handler, backoff_s=0.5))
    assert len(calls) == 3 and r.attempts == 3 and r.probabilities["not_hotdog"] == pytest.approx(0.7)
    assert sleeps == [0.5, llm_client.MAX_RETRY_DELAY_S]  # backoff, then Retry-After (capped)


@pytest.mark.anyio
async def test_reasoning_model_out_of_tokens_hint():
    resp = completion(None)
    resp["choices"][0]["finish_reason"] = "length"
    with pytest.raises(llm_client.UpstreamError, match="LLM_MAX_TOKENS=2000"):
        await _classify(make_client(lambda req: httpx.Response(200, json=resp)))


@pytest.mark.anyio
@pytest.mark.parametrize(
    "base_url,handler,ok,hint",
    [
        ("http://fake/v1", lambda req: httpx.Response(404), True, None),  # no /models endpoint is fine
        ("http://fake/v1", lambda req: httpx.Response(401), False, "rejected the API key"),
        ("http://localhost:11434/v1", None, False, "is Ollama / LM Studio running?"),
    ],
)
async def test_ping(base_url, handler, ok, hint):
    def refuse(req):
        raise httpx.ConnectError("refused", request=req)

    cfg = LLMConfig(base_url=base_url, api_key="k", model="m")
    async with LLMClient(cfg, transport=httpx.MockTransport(handler or refuse)) as client:
        ready, detail = await client.ping()
    assert ready is ok and (detail is None if hint is None else hint in detail)


# ---------------------------------------------------------------- HTTP contract
def load_app(monkeypatch, **env):
    for k in ("LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(llm_client, "load_dotenv", lambda *_: None)  # ignore the user's real .env
    import app as app_module

    return importlib.reload(app_module)


def fake_api(monkeypatch, m, handler):
    """Route the app's LLMClient through `handler` instead of the network."""
    real_init = m.LLMClient.__init__
    monkeypatch.setattr(
        m.LLMClient, "__init__", lambda self, cfg, transport=None: real_init(self, cfg, httpx.MockTransport(handler))
    )


def models_or(chat_response):
    return lambda req: (
        httpx.Response(200, json={"data": []}) if req.url.path.endswith("/models") else chat_response(req)
    )


def check_body(b, model_id):
    assert b["model"] == "llm" and b["label"] in ("hotdog", "not_hotdog")
    assert b["is_hotdog"] == (b["label"] == "hotdog")
    assert abs(sum(b["probabilities"].values()) - 1) < 1e-6
    assert b["confidence"] == b["probabilities"][b["label"]] >= 0.5
    assert b["latency_ms"] >= 0 and b["total_ms"] >= b["latency_ms"]
    assert b["confidence_source"] in ("logprobs", "self-reported") and b["model_id"] == model_id
    assert b["attempts"] >= 1


def test_missing_key_not_ready(monkeypatch):
    m = load_app(monkeypatch, LLM_BASE_URL="https://api.openai.com/v1")
    with TestClient(m.app) as c:
        h = c.get("/health").json()
        assert h["ready"] is False and "LLM_API_KEY" in h["detail"] and "./run.sh llm" in h["detail"]
        assert h["base_url_host"] == "api.openai.com" and h["device"] == "api" and h["model"] == "llm"
        r = c.post("/classify", files={"file": ("a.png", img_bytes(), "image/png")})
        assert r.status_code == 503 and "LLM_API_KEY" in r.json()["detail"]


def _timeout(req):
    raise httpx.ReadTimeout("slow", request=req)


@pytest.mark.parametrize(
    "chat,status,body_check",
    [
        (
            lambda req: httpx.Response(
                200, json=completion('{"label":"hotdog","confidence":0.9,"reason":"yum"}', logprob_content(0.95))
            ),
            200,
            lambda b: (
                b["confidence_source"] == "logprobs"
                and b["probabilities"]["hotdog"] == pytest.approx(0.95, abs=1e-5)
                and b["reason"] == "yum"
                and b["attempts"] == 1
            ),
        ),
        (
            lambda req: httpx.Response(429, json={"error": {"message": "rate limited"}}),
            502,
            lambda b: "rate limited" in b["detail"] and "failed 3×" in b["detail"],
        ),
        (_timeout, 504, lambda b: "did not answer" in b["detail"]),
    ],
)
def test_classify_with_fake_api(monkeypatch, chat, status, body_check):
    m = load_app(
        monkeypatch, LLM_BASE_URL="http://fake/v1", LLM_API_KEY="k", LLM_MODEL="fake-vision", LLM_RETRY_BACKOFF="0"
    )
    fake_api(monkeypatch, m, models_or(chat))
    with TestClient(m.app) as c:
        h = c.get("/health").json()
        assert h["ready"] is True and h["model_id"] == "fake-vision" and "detail" not in h
        r = c.post("/classify", files={"file": ("a.jpg", img_bytes("JPEG"), "image/jpeg")})
        assert r.status_code == status, r.text
        if status == 200:
            check_body(r.json(), "fake-vision")
        assert body_check(r.json())


def test_health_is_cached_and_concurrent_checks_share_one_ping(monkeypatch):
    pings = []

    async def handler(req):
        pings.append(req.url.path)
        await asyncio.sleep(0)  # let the other callers pile up
        return httpx.Response(200, json={"data": []})

    async def scenario(m):
        async with LLMClient(LLMConfig("http://fake/v1", "k", "m"), httpx.MockTransport(handler)) as client:
            backend = m.Backend(client)
            await asyncio.gather(*(backend.refresh() for _ in range(5)))
            await backend.refresh()
            assert backend.ready is True
            await backend.refresh(force=True)

    asyncio.run(scenario(load_app(monkeypatch, LLM_BASE_URL="http://fake/v1")))
    assert pings == ["/v1/models", "/v1/models"]


def test_concurrent_not_ready_classify_calls_share_one_ping(monkeypatch):
    ping_s, pings = 0.3, []

    async def handler(req):
        pings.append(req.url.path)
        await asyncio.sleep(ping_s)
        return httpx.Response(401)  # bad key: stays not ready

    async def scenario(m):
        async with LLMClient(m.CFG, httpx.MockTransport(handler)) as client:
            m.backend = m.Backend(client)
            transport = httpx.ASGITransport(app=m.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                files = {"file": ("a.png", img_bytes(), "image/png")}
                t0 = time.monotonic()
                rs = await asyncio.gather(*(http.post("/classify", files=files) for _ in range(5)))
                elapsed = time.monotonic() - t0
        assert [r.status_code for r in rs] == [503] * 5
        assert all("rejected the API key" in r.json()["detail"] for r in rs)
        return elapsed

    m = load_app(monkeypatch, LLM_BASE_URL="http://fake/v1", LLM_API_KEY="k")
    elapsed = asyncio.run(scenario(m))
    assert pings == ["/v1/models"]
    assert elapsed < 2 * ping_s  # one shared ping, not five in a row


def test_dotenv_inline_comments_and_last_wins(tmp_path, monkeypatch):
    keys = ("HD_A", "HD_B", "HD_C", "HD_D", "HD_E", "HD_F")
    for k in keys:  # set-then-delete so monkeypatch also removes what load_dotenv adds
        monkeypatch.setenv(k, "")
        monkeypatch.delenv(k)
    monkeypatch.setenv("HD_D", "from-env")
    f = tmp_path / ".env"
    f.write_text(
        "# comment\nHD_A=60   # seconds\nHD_B=first\nHD_B=second\n"
        "HD_C=\"has # inside\"\nHD_D=from-file\nHD_E='single'  # c\nHD_F=\nnot a setting\n"
    )
    llm_client.load_dotenv(f)
    assert os.environ["HD_A"] == "60" and os.environ["HD_B"] == "second"
    assert os.environ["HD_C"] == "has # inside" and os.environ["HD_D"] == "from-env"
    assert os.environ["HD_E"] == "single" and os.environ["HD_F"] == ""
    llm_client.load_dotenv(tmp_path / "missing.env")  # no file: no error


def test_cors_only_for_the_ui(monkeypatch):
    m = load_app(monkeypatch, LLM_BASE_URL="https://api.openai.com/v1")  # not configured: no network needed
    with TestClient(m.app) as c:
        pre = {"Access-Control-Request-Method": "POST"}
        ok = c.options("/classify", headers={"Origin": "http://192.168.1.20:8080", **pre})
        bad = c.options("/classify", headers={"Origin": "https://evil.example", **pre})
        other_port = c.options("/classify", headers={"Origin": "http://localhost:3000", **pre})
        assert ok.headers.get("access-control-allow-origin") == "http://192.168.1.20:8080"
        assert "access-control-allow-origin" not in bad.headers
        assert "access-control-allow-origin" not in other_port.headers


def test_bad_empty_or_huge_upload_never_reaches_the_api(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json={"data": []})

    m = load_app(monkeypatch, LLM_BASE_URL="http://fake/v1", LLM_API_KEY="k", LLM_MODEL="fake-vision")
    fake_api(monkeypatch, m, handler)
    with TestClient(m.app) as c:
        assert c.post("/classify", files={"file": ("x.jpg", b"garbage", "image/jpeg")}).status_code == 400
        assert c.post("/classify", files={"file": ("e.png", b"", "image/png")}).status_code == 400
        r = c.post("/classify", files={"file": ("b.png", broken_png(), "image/png")})
        assert r.status_code == 400 and "broken PNG file" in r.json()["detail"]
        huge = b"\0" * (m.MAX_UPLOAD_BYTES + 1)
        r = c.post("/classify", files={"file": ("h.jpg", huge, "image/jpeg")})
        assert r.status_code == 413 and "20 MB" in r.json()["detail"]
    assert not any(p.endswith("/chat/completions") for p in calls)
