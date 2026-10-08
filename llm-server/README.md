# Bun Tribunal: LLM contender

A general-purpose **vision LLM** behind any **OpenAI-compatible** chat API, wrapped in the same `/health` +
`/classify` API as the CLEF-Flash and CNN servers ([`../CONTRACT.md`](../CONTRACT.md)). Port **8003**, model name
**`llm`**. No weights, no torch: a thin FastAPI wrapper that calls the API with `httpx`.

## Configure

```bash
./run.sh llm      # from the project root: pick a provider, enter a key, test it with one picture
```

This writes `llm-server/.env` (readable only by you; the previous one is kept as `.env.bak`). You can also edit it
by hand; [`.env.example`](.env.example) lists every option with its default. Real environment variables override
`.env`, and `OPENAI_API_KEY` is used when `LLM_API_KEY` is not set. Restart `./run.sh` after a change.

| Provider | `LLM_BASE_URL` | `LLM_MODEL` (examples) | Key |
|---|---|---|---|
| OpenAI (default) | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4.1-mini` | required |
| OpenRouter | `https://openrouter.ai/api/v1` | `google/gemini-2.5-flash`, … | required |
| Ollama (local) | `http://localhost:11434/v1` | `qwen2.5vl:7b`, `llama3.2-vision`, `gemma3:12b` | any string |
| LM Studio (local) | `http://localhost:1234/v1` | the id LM Studio shows | any string |

The model **must accept images** (`image_url` content parts).

## How it classifies

1. The image is downscaled (longest side `LLM_MAX_IMAGE_SIDE`, 768 px), JPEG-encoded and sent as a `data:` URL with a
   strict system prompt. The prompt defines a hot dog as a sausage in a bun, which excludes corn dogs, burgers and
   the animal.
2. The model replies with JSON: `{"label": "hotdog" | "not_hotdog", "confidence": 0..1, "reason": "…"}`. The request
   uses `temperature: 0` and `response_format: json_object`.
3. **Probability** (`confidence_source` in the response):
   * **`logprobs`**: if the API returns token logprobs (OpenAI does), `P(hotdog)` is read from the model's own
     token distribution at the first token of the label value (`hot…` vs `not…`). This is a real probability.
   * **`self-reported`**: otherwise the model's stated `confidence` is used. LLMs are notoriously overconfident about
     these numbers, so the UI flags them.
4. If a server rejects `logprobs` or `response_format` (some local servers do), the client retries once without
   them and remembers that for later requests.

The response has the standard fields plus `reason`, `confidence_source`, `model_id` and `attempts`. `latency_ms` is
the API round trip of the successful attempt (network + generation).

| HTTP | When |
|---|---|
| 503 | not configured (e.g. no key for OpenAI / OpenRouter), the key was rejected, or the API can't be reached. `/health` explains why in `detail` |
| 502 | the API returned an error (rate limit, bad model id, …) or output that couldn't be parsed |
| 504 | the API took longer than `LLM_TIMEOUT` (60 s); not retried |

`/health` checks the API with a cheap `GET /models` (no tokens spent) and caches the result for 30 s.

**Automatic retries:** rate limits and overloaded providers (HTTP 429/5xx, or OpenRouter-style errors inside an
HTTP 200 body such as `ResourceExhausted`) are retried up to `LLM_RETRIES` (2) times, waiting `LLM_RETRY_BACKOFF`
(1 s) and doubling each time. A `Retry-After` header (seconds or an HTTP date) takes precedence; any wait is capped at
10 s. Permanent errors such as an unknown model id fail immediately. If all attempts fail, the 502 message quotes the
provider's own error. The UI shows "N attempts" on the card when retries were needed, and its **Retry** button sends
the image again.

**Reasoning models** (o-series, R1, `gpt-5`, `*-reasoning` …) think before they answer. If they run out of tokens the
reply is empty and the server says so; set `LLM_MAX_TOKENS=2000` or more (`./run.sh llm` does this when the model
name looks like a reasoning model).

## Run and test

`./run.sh` starts it with everything else. To run it on its own:

```bash
cd llm-server && ../.venv/bin/python -m uvicorn app:app --port 8003
curl -s localhost:8003/health
curl -s -F "file=@hotdog.jpg" localhost:8003/classify
```

The tests (`./run.sh test`, or `../.venv/bin/python -m pytest` here) cover parsing, logprob extraction, retries and
the HTTP API against a fake provider, so no key is needed. `llm_client.py` can also be used on its own:
`async with LLMClient(LLMConfig.from_env()) as client: verdict = await client.classify(pil_image)` returns a
`Verdict` (`label`, `probabilities`, `reason`, `confidence_source`, `latency_ms`, `attempts`).

## Cost note

With `gpt-4o-mini` and `LLM_IMAGE_DETAIL=low` (the default), each image costs a few hundred input tokens, a fraction
of a cent. Local Ollama or LM Studio costs nothing, but expect a few seconds per image on a laptop.
