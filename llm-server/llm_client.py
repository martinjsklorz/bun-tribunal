"""Talk to any OpenAI-compatible chat-completions endpoint that accepts images.

Works with OpenAI, Ollama (`/v1`), LM Studio, vLLM, llama.cpp server, OpenRouter, Together, Groq, …
Only needs `httpx` — no vendor SDK.

Probability strategy:
1. Ask for a tiny JSON answer: {"label": "hotdog"|"not_hotdog", "confidence": 0..1, "reason": "..."}
2. Request token logprobs. If the API returns them, read the probability mass of the *first token of
   the label value* ("hot…" vs "not…") → a real model probability ("logprobs").
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
from urllib.parse import urlparse

import httpx
from PIL import Image

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


def load_dotenv(path) -> None:
    """Minimal .env loader. KEY=VALUE lines; `# comments` (also after a value); later lines win;
    real environment variables always win over the file."""
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v[:1] in ("'", '"') and v[:1] in v[1:]:
            v = v[1:v.index(v[0], 1)]          # quoted value: keep '#' inside quotes
        else:
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
        values[k.strip()] = v
    for k, v in values.items():
        os.environ.setdefault(k, v)


class UpstreamError(RuntimeError):
    """The API call failed or returned something we could not use (-> HTTP 502)."""


class UpstreamTimeout(UpstreamError):
    """The API did not answer in time (-> HTTP 504)."""


# Errors worth retrying automatically: rate limits, overloaded providers, flaky gateways.
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
TRANSIENT_HINTS = ("rate limit", "ratelimit", "rate-limit", "resourceexhausted", "resource exhausted", "overloaded",
                   "temporarily", "try again", "capacity", "busy", "limit reached", "unavailable", "timed out")


def is_transient(status: int, message: str) -> bool:
    m = message.lower()
    return status in TRANSIENT_STATUS or any(h in m for h in TRANSIENT_HINTS)


def error_in_response(r: httpx.Response) -> tuple[int, str] | None:
    """(status, message) if the response is an error, else None.

    Handles both real HTTP errors and the OpenRouter-style "HTTP 200 with an `error` object and no
    `choices`" (provider failures are often reported that way).
    """
    try:
        data = r.json()
    except Exception:
        data = None
    if r.status_code >= 400:
        msg = r.text[:300]
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            msg = err.get("message", msg) if isinstance(err, dict) else str(err)
        return r.status_code, msg
    if isinstance(data, dict) and data.get("error") and not data.get("choices"):
        err = data["error"]
        msg = err.get("message", "unknown error") if isinstance(err, dict) else str(err)
        code = err.get("code") if isinstance(err, dict) else None
        status = int(code) if isinstance(code, int) or (isinstance(code, str) and code.isdigit()) else 502
        return status, msg
    return None


def retry_after_s(r: httpx.Response) -> float | None:
    v = r.headers.get("retry-after")
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


@dataclass
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
    backoff_s: float = 1.0  # first retry delay; doubles each time (or the server's Retry-After)

    @classmethod
    def from_env(cls) -> "LLMConfig":
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


def image_to_data_url(img: Image.Image, max_side: int) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- response parsing
_LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]+)"', re.I)


def normalize_label(text: str) -> str | None:
    s = re.sub(r"[^a-z]", "", str(text).lower())
    if s in {"hotdog", "hotdogs", "yes", "true"}:
        return "hotdog"
    if s.startswith("not") or s.startswith("no") or s in {"false", "other"}:
        return "not_hotdog"
    return None


def parse_answer(content: str) -> dict:
    """Extract {label, confidence, reason} from the model's text, tolerating fences and chatter."""
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    obj = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        lm = _LABEL_RE.search(text)
        label = normalize_label(lm.group(1)) if lm else None
        if label is None:  # last resort: plain-text answer
            low = text.lower()
            if re.search(r"\bnot[ _-]?(a )?hot ?dog\b", low):
                label = "not_hotdog"
            elif re.search(r"\bhot ?dog\b", low):
                label = "hotdog"
        if label is None:
            raise UpstreamError(f"could not parse a label from the model output: {text[:200]!r}")
        return {"label": label, "confidence": None, "reason": None}

    label = normalize_label(obj.get("label", ""))
    if label is None:
        raise UpstreamError(f"model returned an unknown label: {obj.get('label')!r}")
    conf = obj.get("confidence")
    try:
        conf = float(conf)
        if conf > 1.0 and conf <= 100.0:  # some models answer in percent
            conf /= 100.0
        conf = min(max(conf, 0.0), 1.0)
    except (TypeError, ValueError):
        conf = None
    reason = obj.get("reason")
    reason = str(reason).strip()[:300] if reason else None
    return {"label": label, "confidence": conf, "reason": reason}


def label_prob_from_logprobs(logprobs: dict | None) -> float | None:
    """P(hotdog) from the first token of the label value, using the top alternatives at that position.

    The JSON looks like {"label": "hotdog", ...}; we walk the tokens, find the first one after the
    `label` key whose text starts the value, then sum probability over alternatives starting with
    "hot" vs "not". Returns None when the API gave no usable logprobs.
    """
    if not logprobs or not logprobs.get("content"):
        return None
    toks = logprobs["content"]
    seen_label_key = False
    text_so_far = ""
    for t in toks:
        tok = t.get("token", "")
        text_so_far += tok
        if not seen_label_key:
            if re.search(r'"label"\s*:\s*"?$', text_so_far):
                seen_label_key = True
            continue
        core = tok.strip().lstrip('"').lower()
        if not core:  # the opening quote of the value can be its own token
            continue
        alts = t.get("top_logprobs") or [{"token": tok, "logprob": t.get("logprob", 0.0)}]
        p_hot = p_not = 0.0
        for a in alts:
            c = a.get("token", "").strip().lstrip('"').lower()
            p = math.exp(a.get("logprob", -1e9))
            if c.startswith(("hot", "hd")):
                p_hot += p
            elif c.startswith(("not", "no")):
                p_not += p
        if p_hot + p_not == 0:
            return None
        return p_hot / (p_hot + p_not)
    return None


# ---------------------------------------------------------------- client
class LLMClient:
    def __init__(self, cfg: LLMConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self._http = httpx.AsyncClient(base_url=cfg.base_url, headers=headers,
                                       timeout=cfg.timeout_s, transport=transport)
        # Some servers reject logprobs / response_format; remember and stop sending them.
        self._logprobs_ok = cfg.use_logprobs
        self._json_mode_ok = cfg.json_mode

    async def aclose(self) -> None:
        await self._http.aclose()

    async def ping(self) -> tuple[bool, str | None]:
        """Cheap reachability check: GET /models (doesn't spend tokens)."""
        if self.cfg.needs_key and not self.cfg.api_key:
            return False, f"LLM not configured: no API key for {self.cfg.host} (LLM_API_KEY) — run ./run.sh llm"
        try:
            r = await self._http.get("/models", timeout=5.0)
        except httpx.HTTPError as e:
            hint = " — is Ollama / LM Studio running?" if any(h in self.cfg.base_url for h in ("localhost", "127.0.0.1")) \
                else " — check your network, or run ./run.sh llm"
            return False, f"cannot reach {self.cfg.base_url}: {type(e).__name__}{hint}"
        if r.status_code in (401, 403):
            return False, f"{self.cfg.host} rejected the API key (HTTP {r.status_code}) — run ./run.sh llm"
        # Some servers don't implement /models (404/405) but chat still works -> treat as reachable.
        return True, None

    def _payload(self, data_url: str) -> dict:
        body: dict = {
            "model": self.cfg.model,
            "temperature": 0,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": self.cfg.detail}},
                ]},
            ],
        }
        if self._json_mode_ok:
            body["response_format"] = {"type": "json_object"}
        if self._logprobs_ok:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        return body

    async def _post(self, body: dict) -> httpx.Response:
        try:
            return await self._http.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise UpstreamTimeout(f"{self.cfg.host} did not answer within {self.cfg.timeout_s:.0f}s") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"cannot reach {self.cfg.base_url}: {type(e).__name__}: {e}") from e

    async def _request_once(self, data_url: str) -> httpx.Response:
        r = await self._post(self._payload(data_url))
        # Graceful degradation: retry once without optional features the server doesn't support.
        if r.status_code == 400 and (self._logprobs_ok or self._json_mode_ok):
            msg = r.text.lower()
            if "logprob" in msg:
                self._logprobs_ok = False
            if "response_format" in msg or "json" in msg:
                self._json_mode_ok = False
            if "logprob" not in msg and "response_format" not in msg and "json" not in msg:
                self._logprobs_ok = self._json_mode_ok = False
            r = await self._post(self._payload(data_url))
        return r

    def _friendly(self, status: int, message: str, attempts: int) -> str:
        text = f"{self.cfg.host}: {message.strip()}"
        if attempts > 1:
            text += f" (failed {attempts}× in a row)"
        if is_transient(status, message):
            text += ". The provider is overloaded or rate-limiting; retry in a moment"
            text += ", or pick a paid / different LLM_MODEL (\":free\" models are heavily rate-limited)." \
                if self.cfg.model.endswith(":free") else " or switch LLM_MODEL."
        return text

    async def classify(self, img: Image.Image) -> dict:
        data_url = image_to_data_url(img, self.cfg.max_image_side)
        attempts = self.cfg.retries + 1
        for attempt in range(1, attempts + 1):
            t0 = time.perf_counter()
            try:
                r = await self._request_once(data_url)
            except UpstreamTimeout:
                raise  # already waited LLM_TIMEOUT; retrying would multiply that
            except UpstreamError as e:  # connection refused / reset: often transient
                if attempt < attempts:
                    await asyncio.sleep(self.cfg.backoff_s * 2 ** (attempt - 1))
                    continue
                raise UpstreamError(f"{e} (failed {attempts}× in a row)" if attempts > 1 else str(e)) from e
            latency_ms = (time.perf_counter() - t0) * 1000.0
            err = error_in_response(r)
            if err is None:
                break
            status, message = err
            if is_transient(status, message) and attempt < attempts:
                delay = retry_after_s(r) or self.cfg.backoff_s * 2 ** (attempt - 1)
                await asyncio.sleep(min(delay, 10.0))
                continue
            raise UpstreamError(self._friendly(status, message, attempt))

        try:
            choice = r.json()["choices"][0]
            content = choice["message"].get("content") or ""
        except Exception as e:
            raise UpstreamError(f"unexpected response shape from {self.cfg.host}: {r.text[:200]}") from e
        if not content.strip():
            if choice.get("finish_reason") == "length":
                raise UpstreamError(f"{self.cfg.model} ran out of tokens before answering (max_tokens="
                                    f"{self.cfg.max_tokens}). Reasoning models think first: set LLM_MAX_TOKENS=2000 or more.")
            raise UpstreamError(f"{self.cfg.model} returned an empty answer")

        ans = parse_answer(content)
        p_hot = label_prob_from_logprobs(choice.get("logprobs"))
        if p_hot is not None:
            source = "logprobs"
        else:
            source = "self-reported"
            conf = ans["confidence"] if ans["confidence"] is not None else 0.75
            conf = max(conf, 0.5)  # the stated label is the argmax by definition
            p_hot = conf if ans["label"] == "hotdog" else 1.0 - conf
        p_hot = min(max(float(p_hot), 0.0), 1.0)
        return {
            "probabilities": {"hotdog": p_hot, "not_hotdog": 1.0 - p_hot},
            "reason": ans["reason"],
            "confidence_source": source,
            "latency_ms": latency_ms,
            "raw_label": ans["label"],
            "attempts": attempt,
        }
