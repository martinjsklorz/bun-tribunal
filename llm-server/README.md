# Bun Tribunal: LLM contender

A general-purpose **vision LLM** behind any **OpenAI-compatible** chat API, wrapped in the same `/health` +
`/classify` contract as the CLEF-Flash and CNN servers (`../CONTRACT.md`). Port **8003**, model name **`llm`**.
It has no weights or torch dependency: it is a thin FastAPI wrapper that uses `httpx` to call the API.

## Configure

```bash
./run.sh llm      # from the project root: pick a provider, enter a key, test it with one picture
```

This writes `llm-server/.env`. You can also edit that file by hand; `.env.example` lists every option.

| Provider | `LLM_BASE_URL` | `LLM_MODEL` (examples) | Key |
|---|---|---|---|
| OpenAI (default) | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4.1-mini` | required |
| Ollama (local, Mac) | `http://localhost:11434/v1` | `qwen2.5vl:7b`, `llama3.2-vision`, `gemma3:12b` | any string |
| LM Studio (local) | `http://localhost:1234/v1` | the id LM Studio shows | any string |
| OpenRouter | `https://openrouter.ai/api/v1` | `google/gemini-2.5-flash`, … | required |

The model **must accept images** (`image_url` content parts). Real environment variables override `.env`.

## How it classifies

1. The image is downscaled (longest side 768 px), JPEG-encoded and sent as a `data:` URL with a strict system prompt.
   The prompt defines what counts as a hot dog: a sausage in a bun, which excludes corn dogs, burgers and the animal.
2. The model replies with JSON: `{"label": "hotdog" | "not_hotdog", "confidence": 0..1, "reason": "…"}`. The request
   uses `temperature: 0` and `response_format: json_object`.
3. **Probability:**
   * **`logprobs`**: if the API returns token logprobs (OpenAI does), `P(hotdog)` is read from the model's own
     token distribution at the first token of the label value (`hot…` vs `not…`). This is a real probability.
   * **`self-reported`**: otherwise the model's stated `confidence` is used. LLMs are notoriously overconfident about
     these numbers, so the UI flags them.
4. If a server rejects `logprobs` or `response_format` (some local servers do), the client retries once without
   them and remembers that for later requests.

The response contains the standard fields plus `reason`, `confidence_source` and `model_id`. `latency_ms` is the API
round trip, which includes network and generation time.

| HTTP | When |
|---|---|
| 503 | not configured (for example, no key for OpenAI) or the API can't be reached. `/health` explains why in `detail` |
| 502 | the API returned an error (rate limit, bad model id, …) or output that couldn't be parsed |
| 504 | the API took longer than `LLM_TIMEOUT` (60 s) |

**Automatic retries:** rate limits and overloaded providers (HTTP 429/5xx, or OpenRouter-style errors delivered inside an
HTTP 200 body such as `ResourceExhausted`) are retried up to `LLM_RETRIES` (default 2) times, with a backoff that
starts at 1 s and doubles. A `Retry-After` header takes precedence. Permanent errors such as an unknown model id fail
immediately. If all attempts fail, the 502 message quotes the provider's own error. The UI's **Retry** button sends
the image again.

**Reasoning models** (`*-reasoning`, o-series, R1 …) think before they answer. If they run out of tokens the reply
is empty and the server says so; set `LLM_MAX_TOKENS=2000` or more.

## Run and test

`./run.sh` starts it with everything else. Its 22 tests (parsing, logprob extraction, retries, the HTTP contract)
run against a fake API, so no key is needed.

```bash
curl -s localhost:8003/health
curl -s -F "file=@hotdog.jpg" localhost:8003/classify
```

## Cost note

With `gpt-4o-mini` and `LLM_IMAGE_DETAIL=low`, each image costs a few hundred input tokens, which is a fraction of a cent.
Local Ollama or LM Studio costs nothing, but expect a few seconds per image on a laptop.
