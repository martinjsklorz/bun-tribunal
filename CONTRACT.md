# Bun Tribunal: the shared API

Three model servers speak the same small JSON API; the UI talks only to that. `./run.sh` starts all of them.

| Component | Dir | Port | Tech |
|---|---|---|---|
| LLM | `llm-server/` | 8003 | FastAPI + any OpenAI-compatible vision chat API (OpenAI, OpenRouter, Ollama, LM Studio …) |
| CLEF-Flash | `clef-server/` | 8001 | FastAPI + PyTorch, `Cloudflare/clef-flash` (9.4B, Qwen3.5-9B backbone + joint schema head, BF16) |
| CNN | `cnn/` (training notebook + `cnn/server/`) | 8002 | FastAPI + PyTorch, ConvNeXt-Tiny fine-tuned by the notebook |
| UI | `ui/` | 8080 | static HTML/CSS/JS, no build step |

## Endpoints (every classifier server implements exactly this)

### `GET /health`
```json
{"status": "ok", "model": "clef-flash", "device": "mps", "ready": true}
```
`ready` is false in two cases, and the UI tells them apart:
- **loading** (no `detail`): e.g. CLEF-Flash is still loading its weights.
- **not set up** (`detail` explains why and names the fix: `./run.sh llm`, `./run.sh download` or `./run.sh train`).

### `POST /classify`
Request: `multipart/form-data`, field name **`file`** (jpeg/png/webp; any size — server resizes as needed).

Response 200:
```json
{
  "model": "clef-flash",            // or "cnn"
  "label": "hotdog",                // "hotdog" | "not_hotdog"
  "is_hotdog": true,
  "confidence": 0.973,              // probability of the returned label
  "probabilities": {"hotdog": 0.973, "not_hotdog": 0.027},
  "latency_ms": 41.2,               // model forward time only
  "total_ms": 55.8                  // incl. decode/preprocess
}
```
Errors: `400` bad image, `413` upload too large (25 MB CNN/CLEF, 20 MB LLM), `503` loading or not set up (same `detail` as `/health`).

The CNN's `/health` also reports `arch`, `weights` (`state_dict` | `torchscript`) and `temperature` (calibration).

## Cross-cutting rules
- CORS: only pages served on the UI port (`PORT_UI`, default 8080) may call the servers, so other websites can't.
- Device auto-select: `DEVICE` overrides; else cuda → mps → cpu.
- Servers bind to 127.0.0.1; `LAN=1 ./run.sh` binds 0.0.0.0 so a phone on the same Wi-Fi can use the UI.

## LLM-specific additions (port 8003, `"model": "llm"`)
Same contract, plus optional fields the UI shows when present:

`GET /health` adds `"model_id": "gpt-4o-mini"`, `"base_url_host": "api.openai.com"` and, when not ready, `"detail"`.

`POST /classify` adds:
```json
{
  "reason": "A grilled sausage in a split bun with mustard.",   // one sentence from the LLM
  "confidence_source": "logprobs",   // "logprobs" (token probabilities from the API) | "self-reported" (the LLM's own number)
  "model_id": "gpt-4o-mini",
  "attempts": 1                       // >1 when rate limits / overloaded providers were retried automatically
}
```
`latency_ms` = the API round trip (network + generation); there is no local forward pass.
Errors: `502 {"detail": "..."}` when the upstream API fails or returns something unparsable, `504` on timeout.
