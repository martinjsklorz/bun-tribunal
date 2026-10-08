"""Talk to any OpenAI-compatible chat-completions endpoint that accepts images.

Works with OpenAI, Ollama (`/v1`), LM Studio, vLLM, llama.cpp server, OpenRouter, Together, Groq, …
Only needs `httpx`, no vendor SDK.

Probability strategy:
1. Ask for a tiny JSON answer: {"label": "hotdog"|"not_hotdog", "confidence": 0..1, "reason": "..."}
2. Request token logprobs. If the API returns them, read the probability mass of the *first token of
   the label value* ("hot…" vs "not…"): a real model probability ("logprobs").
3. Otherwise fall back to the model's own `confidence` number ("self-reported").
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

SYSTEM_PROMPT = (
    "You are a strict food-image classifier. Decide whether the photo shows a HOT DOG: a sausage "
    "(frankfurter, wiener, bratwurst, etc.) served in a sliced bun or roll, with or without toppings. "
    "A sausage without a bun, a burger, a sandwich, a burrito, a taco or a corn dog is NOT a hot dog. "
    "A dog (the animal) is NOT a hot dog. "
    'Reply with ONLY a JSON object, no markdown: {"label": "hotdog" or "not_hotdog", '
    '"confidence": number between 0 and 1, "reason": "one short sentence"}. '
    'Write the "label" key first.'
)
USER_PROMPT = "Hot dog or not hot dog?"

JPEG_QUALITY = 88
PING_TIMEOUT_S = 5.0
MAX_RETRY_DELAY_S = 10.0  # never stall a request for long, whatever Retry-After says
SELF_REPORTED_DEFAULT = 0.75  # used when the model names a label but no usable confidence

Label = Literal["hotdog", "not_hotdog"]


# ---------------------------------------------------------------- configuration
def load_dotenv(path: str | os.PathLike[str]) -> None:
    """Minimal .env loader, so the server needs no python-dotenv.

    KEY=VALUE lines; `# comments` (also after an unquoted value); later lines win;
    real environment variables always win over the file.
    """
    path = Path(path)
    if not path.exists():
        return
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _dotenv_value(value.strip())
    for key, value in values.items():
        os.environ.setdefault(key, value)


def _dotenv_value(value: str) -> str:
    quote = value[:1]
    if quote in ("'", '"') and quote in value[1:]:
        return value[1 : value.index(quote, 1)]  # quoted: a '#' inside is part of the value
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0
    max_image_side: int = 768
    use_logprobs: bool = True
    json_mode: bool = True
    detail: str = "low"  # OpenAI image detail; ignored by most other servers
    max_tokens: int = 512  # reasoning models "think" first: give them 2000+
    retries: int = 2  # automatic retries for transient errors (rate limits, overloaded providers)
    backoff_s: float = 1.0  # first retry delay; doubles each time (unless the server sends Retry-After)

    @classmethod
    def from_env(cls) -> LLMConfig:
        env = os.environ.get
        return cls(
            base_url=env("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=env("LLM_API_KEY") or env("OPENAI_API_KEY") or "",
            model=env("LLM_MODEL", "gpt-4o-mini"),
            timeout_s=float(env("LLM_TIMEOUT", "60")),
            max_image_side=int(env("LLM_MAX_IMAGE_SIDE", "768")),
            use_logprobs=env("LLM_LOGPROBS", "1") != "0",
            json_mode=env("LLM_JSON_MODE", "1") != "0",
            detail=env("LLM_IMAGE_DETAIL", "low"),
            max_tokens=int(env("LLM_MAX_TOKENS", "512")),
            retries=max(0, int(env("LLM_RETRIES", "2"))),
            backoff_s=float(env("LLM_RETRY_BACKOFF", "1.0")),
        )

    @property
    def host(self) -> str:
        return urlparse(self.base_url).netloc or self.base_url

    @property
    def needs_key(self) -> bool:
        return "api.openai.com" in self.base_url or "openrouter.ai" in self.base_url

    @property
    def is_local(self) -> bool:
        return "localhost" in self.base_url or "127.0.0.1" in self.base_url


class UpstreamError(RuntimeError):
    """The API call failed or returned something we could not use (-> HTTP 502)."""


class UpstreamTimeout(UpstreamError):
    """The API did not answer in time (-> HTTP 504)."""


# ---------------------------------------------------------------- image -> request
def image_to_data_url(img: Image.Image, max_side: int) -> str:
    """Bounded, alpha-flattened JPEG as a data URL: small uploads keep latency and token cost down.

    CPU-bound; call it off the event loop. Never modifies `img`.
    """
    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    work_mode = "RGBA" if has_alpha else "RGB"
    if img.mode != work_mode:
        img = img.convert(work_mode)  # before resizing: palette images would resize with NEAREST
    if max(img.size) > max_side:
        img = ImageOps.contain(img, (max_side, max_side), Image.Resampling.LANCZOS)
    if has_alpha:  # transparent areas become white, not black
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        img = background
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getbuffer()).decode("ascii")


# ---------------------------------------------------------------- response parsing
_LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]+)"', re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_LABEL_KEY_RE = re.compile(r'"label"\s*:\s*"?$')


@dataclass(frozen=True, slots=True)
class Answer:
    """What the model wrote: its label, its own confidence (if any) and a one-line reason."""

    label: Label
    confidence: float | None = None
    reason: str | None = None

    def p_hotdog(self) -> float:
        """P(hotdog) from the self-reported confidence; the stated label is the argmax by definition."""
        conf = max(self.confidence if self.confidence is not None else SELF_REPORTED_DEFAULT, 0.5)
        return conf if self.label == "hotdog" else 1.0 - conf


def normalize_label(text: object) -> Label | None:
    s = re.sub(r"[^a-z]", "", str(text).lower())
    if s in {"hotdog", "hotdogs", "yes", "true"}:
        return "hotdog"
    if s.startswith(("not", "no")) or s in {"false", "other"}:
        return "not_hotdog"
    return None


def parse_answer(content: str) -> Answer:
    """Extract the answer from the model's text, tolerating fences and chatter around the JSON."""
    text = _FENCE_RE.sub("", content.strip())
    obj = _json_object(text)
    if obj is None:
        return Answer(_label_from_prose(text))
    label = normalize_label(obj.get("label", ""))
    if label is None:
        raise UpstreamError(f"model returned an unknown label: {obj.get('label')!r}")
    reason = obj.get("reason")
    return Answer(label, _confidence(obj.get("confidence")), str(reason).strip()[:300] if reason else None)


def _json_object(text: str) -> dict[str, Any] | None:
    m = _OBJECT_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _label_from_prose(text: str) -> Label:
    m = _LABEL_RE.search(text)
    label = normalize_label(m.group(1)) if m else None
    if label is None:  # last resort: a plain-text answer
        low = text.lower()
        if re.search(r"\bnot[ _-]?(a )?hot ?dog\b", low):
            label = "not_hotdog"
        elif re.search(r"\bhot ?dog\b", low):
            label = "hotdog"
    if label is None:
        raise UpstreamError(f"could not parse a label from the model output: {text[:200]!r}")
    return label


def _confidence(value: Any) -> float | None:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(conf):
        return None
    if 1.0 < conf <= 100.0:  # some models answer in percent
        conf /= 100.0
    return min(max(conf, 0.0), 1.0)


def label_prob_from_logprobs(logprobs: dict[str, Any] | None) -> float | None:
    """P(hotdog) from the first token of the label value, using the top alternatives at that position.

    The JSON looks like {"label": "hotdog", ...}: find the first token after the `label` key that
    starts the value, then compare the probability mass of alternatives starting with "hot" vs "not".
    None when the API gave no usable logprobs.
    """
    if not logprobs or not logprobs.get("content"):
        return None
    text_so_far = ""
    seen_label_key = False
    for t in logprobs["content"]:
        tok = t.get("token", "")
        text_so_far += tok
        if not seen_label_key:
            seen_label_key = bool(_LABEL_KEY_RE.search(text_so_far))
            continue
        if not _token_core(tok):  # the opening quote of the value can be its own token
            continue
        alternatives = t.get("top_logprobs") or [{"token": tok, "logprob": t.get("logprob", 0.0)}]
        p_hot = p_not = 0.0
        for alt in alternatives:
            core, p = _token_core(alt.get("token", "")), math.exp(alt.get("logprob", -1e9))
            if core.startswith(("hot", "hd")):
                p_hot += p
            elif core.startswith(("not", "no")):
                p_not += p
        return p_hot / (p_hot + p_not) if p_hot + p_not > 0 else None
    return None


def _token_core(token: str) -> str:
    return token.strip().lstrip('"').lower()


# ---------------------------------------------------------------- API failures and retries
# Worth retrying automatically: rate limits, overloaded providers, flaky gateways.
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529})
TRANSIENT_HINTS = (
    "rate limit",
    "ratelimit",
    "rate-limit",
    "resourceexhausted",
    "resource exhausted",
    "overloaded",
    "temporarily",
    "try again",
    "capacity",
    "busy",
    "limit reached",
    "unavailable",
    "timed out",
)


def is_transient(status: int, message: str) -> bool:
    m = message.lower()
    return status in TRANSIENT_STATUS or any(h in m for h in TRANSIENT_HINTS)


@dataclass(frozen=True, slots=True)
class ApiFailure:
    status: int
    message: str

    @property
    def transient(self) -> bool:
        return is_transient(self.status, self.message)


def find_failure(status_code: int, data: object, text: str) -> ApiFailure | None:
    """The API's error, if the response is one.

    Covers real HTTP errors and the OpenRouter-style "HTTP 200 with an `error` object and no
    `choices`" (provider failures are often reported that way).
    """
    body = data if isinstance(data, dict) else {}
    err = body.get("error")
    if status_code >= 400:
        message = text[:300]
        if err:
            message = err.get("message", message) if isinstance(err, dict) else str(err)
        return ApiFailure(status_code, message)
    if err and not body.get("choices"):
        if not isinstance(err, dict):
            return ApiFailure(502, str(err))
        code = err.get("code")
        status = int(code) if isinstance(code, int) or (isinstance(code, str) and code.isdigit()) else 502
        return ApiFailure(status, err.get("message", "unknown error"))
    return None


def retry_after_s(headers: httpx.Headers) -> float | None:
    """Seconds the server asked us to wait (Retry-After as seconds or an HTTP date), if it said."""
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _json_or_none(r: httpx.Response) -> Any:
    try:
        return r.json()
    except ValueError:
        return None


# ---------------------------------------------------------------- client
@dataclass(frozen=True, slots=True)
class Verdict:
    p_hotdog: float
    reason: str | None
    confidence_source: Literal["logprobs", "self-reported"]
    latency_ms: float  # API round trip of the successful attempt
    attempts: int

    @property
    def probabilities(self) -> dict[str, float]:
        return {"hotdog": self.p_hotdog, "not_hotdog": 1.0 - self.p_hotdog}

    @property
    def label(self) -> Label:
        return "hotdog" if self.p_hotdog >= 0.5 else "not_hotdog"


class LLMClient:
    """One pooled HTTP client for the app's lifetime; also remembers which optional features the server rejects."""

    def __init__(self, cfg: LLMConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        self._http = httpx.AsyncClient(
            base_url=cfg.base_url, headers=headers, timeout=cfg.timeout_s, transport=transport
        )
        self._logprobs_ok = cfg.use_logprobs
        self._json_mode_ok = cfg.json_mode

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def ping(self) -> tuple[bool, str | None]:
        """Cheap reachability and key check via GET /models (spends no tokens)."""
        cfg = self.cfg
        if cfg.needs_key and not cfg.api_key:
            return False, f"LLM not configured: no API key for {cfg.host} (LLM_API_KEY) — run ./run.sh llm"
        try:
            r = await self._http.get("/models", timeout=PING_TIMEOUT_S)
        except httpx.HTTPError as e:
            hint = "is Ollama / LM Studio running?" if cfg.is_local else "check your network, or run ./run.sh llm"
            return False, f"cannot reach {cfg.base_url}: {type(e).__name__} — {hint}"
        if r.status_code in (401, 403):
            return False, f"{cfg.host} rejected the API key (HTTP {r.status_code}) — run ./run.sh llm"
        # Some servers don't implement /models (404/405) but chat still works: treat as reachable.
        return True, None

    async def classify(self, img: Image.Image) -> Verdict:
        data_url = await asyncio.to_thread(image_to_data_url, img, self.cfg.max_image_side)
        data, latency_ms, attempts = await self._complete(data_url)
        choice, content = self._first_choice(data)
        answer = parse_answer(content)
        p_hot = label_prob_from_logprobs(choice.get("logprobs"))
        source: Literal["logprobs", "self-reported"] = "logprobs"
        if p_hot is None:
            p_hot, source = answer.p_hotdog(), "self-reported"
        return Verdict(min(max(float(p_hot), 0.0), 1.0), answer.reason, source, latency_ms, attempts)

    # -- request building
    def _payload(self, data_url: str) -> dict[str, Any]:
        user_content = [
            {"type": "text", "text": USER_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url, "detail": self.cfg.detail}},
        ]
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "temperature": 0,
            "max_tokens": self.cfg.max_tokens,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
        }
        if self._json_mode_ok:
            body["response_format"] = {"type": "json_object"}
        if self._logprobs_ok:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        return body

    # -- transport
    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return await self._http.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise UpstreamTimeout(f"{self.cfg.host} did not answer within {self.cfg.timeout_s:.0f}s") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"cannot reach {self.cfg.base_url}: {type(e).__name__}: {e}") from e

    async def _send(self, data_url: str) -> httpx.Response:
        """POST once; on a 400 that blames an optional feature, drop it for good and resend."""
        r = await self._post(self._payload(data_url))
        if r.status_code == 400 and (self._logprobs_ok or self._json_mode_ok):
            self._disable_rejected_features(r.text.lower())
            r = await self._post(self._payload(data_url))
        return r

    def _disable_rejected_features(self, message: str) -> None:
        blames_logprobs = "logprob" in message
        blames_json = "response_format" in message or "json" in message
        if not (blames_logprobs or blames_json):  # unclear which one: drop both
            blames_logprobs = blames_json = True
        if blames_logprobs:
            self._logprobs_ok = False
        if blames_json:
            self._json_mode_ok = False

    # -- retries
    async def _complete(self, data_url: str) -> tuple[dict[str, Any], float, int]:
        """(response JSON, latency of the successful attempt in ms, attempts), retrying transient failures."""
        max_attempts = self.cfg.retries + 1
        for attempt in range(1, max_attempts + 1):
            last = attempt == max_attempts
            t0 = time.perf_counter()
            try:
                r = await self._send(data_url)
            except UpstreamTimeout:
                raise  # already waited LLM_TIMEOUT; retrying would multiply that
            except UpstreamError as e:  # connection refused / reset: often transient
                if last:
                    raise UpstreamError(self._with_attempts(str(e), attempt)) from e
                await asyncio.sleep(self._retry_delay(attempt))
                continue
            latency_ms = (time.perf_counter() - t0) * 1000.0
            data = _json_or_none(r)
            failure = find_failure(r.status_code, data, r.text)
            if failure is None:
                if not isinstance(data, dict):
                    raise UpstreamError(f"unexpected response shape from {self.cfg.host}: {r.text[:200]}")
                return data, latency_ms, attempt
            if last or not failure.transient:
                raise UpstreamError(self._explain(failure, attempt))
            await asyncio.sleep(self._retry_delay(attempt, retry_after_s(r.headers)))
        raise AssertionError("unreachable: the loop always returns or raises")

    def _retry_delay(self, attempt: int, retry_after: float | None = None) -> float:
        delay = retry_after if retry_after is not None else self.cfg.backoff_s * 2 ** (attempt - 1)
        return min(delay, MAX_RETRY_DELAY_S)

    # -- response reading
    def _first_choice(self, data: dict[str, Any]) -> tuple[dict[str, Any], str]:
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise UpstreamError(f"unexpected response shape from {self.cfg.host}: {json.dumps(data)[:200]}") from e
        if not content.strip():
            if choice.get("finish_reason") == "length":
                raise UpstreamError(
                    f"{self.cfg.model} ran out of tokens before answering (max_tokens={self.cfg.max_tokens}). "
                    "Reasoning models think first: set LLM_MAX_TOKENS=2000 or more."
                )
            raise UpstreamError(f"{self.cfg.model} returned an empty answer")
        return choice, content

    # -- error messages
    @staticmethod
    def _with_attempts(message: str, attempts: int) -> str:
        return f"{message} (failed {attempts}× in a row)" if attempts > 1 else message

    def _explain(self, failure: ApiFailure, attempts: int) -> str:
        text = self._with_attempts(f"{self.cfg.host}: {failure.message.strip()}", attempts)
        if failure.transient:
            text += ". The provider is overloaded or rate-limiting; retry in a moment"
            if self.cfg.model.endswith(":free"):
                text += ', or pick a paid / different LLM_MODEL (":free" models are heavily rate-limited).'
            else:
                text += " or switch LLM_MODEL."
        return text
