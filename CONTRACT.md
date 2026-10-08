# Bun Tribunal: the shared API

Three model servers speak the same small JSON API; the UI talks only to that. `./run.sh` starts all of them.

| Component | Dir | Port | Tech |
|---|---|---|---|
| LLM | `llm-server/` | 8003 | FastAPI + any OpenAI-compatible vision chat API (OpenAI, OpenRouter, Ollama, LM Studio …) |
| CLEF-Flash | `clef-server/` | 8001 | FastAPI + MLX on Apple Silicon (`mlx-community/clef-flash-4bit`, 4-bit) or PyTorch elsewhere (`Cloudflare/clef-flash`, BF16). 9.4B: Qwen3.5-9B backbone + joint schema head |
| CNN | `cnn/` (notebook + `cnn/server/`) | 8002 | FastAPI + PyTorch, ConvNeXt-Tiny fine-tuned by the notebook |
| UI | `ui/` | 8080 (`PORT_UI`) | static HTML/CSS/JS, no build step |

## Endpoints (every classifier server implements both)

### `GET /health`

```json
{"status": "ok", "model": "cnn", "device": "mps", "ready": true}
```

`model` is `llm`, `clef-flash` or `cnn`; `device` is e.g. `cuda`, `mps`, `cpu`, `mlx · 4-bit` or `api` (LLM).
`ready` is false in two cases, and the UI tells them apart:

- **loading** (no `detail`): e.g. CLEF-Flash is still loading its weights.
- **not set up** (`detail` explains why and names the fix: `./run.sh llm`, `./run.sh download` or `./run.sh train`).

### `POST /classify`

Request: `multipart/form-data`, field **`file`** (JPEG, PNG or WebP). The server decodes, EXIF-rotates and resizes
as needed; the UI already downscales big photos to 1280 px before uploading.

Response 200:

```jsonc
{
  "model": "cnn",
  "label": "hotdog",                // "hotdog" | "not_hotdog"
  "is_hotdog": true,
  "confidence": 0.973,              // probability of the returned label
  "probabilities": {"hotdog": 0.973, "not_hotdog": 0.027},
  "latency_ms": 41.2,               // model forward pass only
  "total_ms": 55.8                  // incl. decoding and preprocessing
}
```

| Status | When |
|---|---|
| 400 | not a decodable image (CLEF-Flash also: more than `MAX_IMAGE_PIXELS` pixels) |
| 413 | upload too large: 25 MB (CNN and CLEF-Flash, `MAX_UPLOAD_MB`), 20 MB (LLM) |
| 503 | loading or not set up (same `detail` as `/health`) |

Errors are FastAPI's `{"detail": "..."}`.

## Server-specific additions

**CLEF-Flash** `/health` adds `backend` (`mlx` | `torch`), `weights` (`4-bit MLX quantization` | `full BF16 weights`)
and `model_id`. When not ready it reports the reason as `detail` and, for older UIs, also as `error`.

**CNN** `/health` adds, once ready, `arch`, `weights` (`state_dict` | `torchscript`) and `temperature` (calibration).

**LLM** (`"model": "llm"`, `"device": "api"`): `/health` adds `model_id` (e.g. `gpt-4o-mini`) and `base_url_host`
(e.g. `api.openai.com`). `/classify` adds:

```jsonc
{
  "reason": "A grilled sausage in a split bun with mustard.",  // one sentence from the LLM
  "confidence_source": "logprobs",  // "logprobs" (token probabilities from the API) | "self-reported" (the LLM's own number)
  "model_id": "gpt-4o-mini",
  "attempts": 1                     // >1 when rate limits / overloaded providers were retried automatically
}
```

Its `latency_ms` is the API round trip (network + generation); there is no local forward pass. Extra errors:
`502` when the upstream API fails or returns something unparsable, `504` when it doesn't answer within `LLM_TIMEOUT`.

## Cross-cutting rules

- **CORS:** only pages served on the UI port (`PORT_UI`, default 8080) may call the servers, so other websites can't.
- **Device** (PyTorch servers): `DEVICE` overrides; else cuda → mps → cpu. CLEF-Flash on Apple Silicon uses MLX
  instead (`CLEF_BACKEND=torch` to opt out).
- **Binding:** servers listen on 127.0.0.1; `LAN=1 ./run.sh` binds 0.0.0.0 so a phone on the same Wi-Fi can use the UI.
- **One at a time:** each local model runs one forward pass at a time; decoding and preprocessing still run in parallel.
