# CLEF-Flash contender (port 8001)

A FastAPI server for [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) (Apache-2.0). It's a
9.4B-parameter model: a Qwen3.5-9B backbone with a vision encoder and a "joint schema head". Instead of writing text,
it returns a probability for each allowed answer. We ask it one `choice` question with the options `hotdog` and
`not_hotdog`. The API is described in `../CONTRACT.md`.

## Setup

```bash
./run.sh download     # from the project root: ~19 GB into the Hugging Face cache, resumable
./run.sh              # starts it with everything else; the weights load in the background
```

The server **never downloads on its own**. Without the weights, `/health` reports "not set up" and names the command
above. Set `CLEF_AUTO_DOWNLOAD=1` to allow downloading on start anyway.

| Hardware | Notes |
|---|---|
| Apple Silicon, ≥ 32 GB unified memory | Runs on MPS. bf16 is tried first; the server falls back to fp16 if bf16 fails |
| NVIDIA GPU, ≥ 24 GB VRAM | bf16 weights need about 19–20 GB |
| CPU | Works but slowly (seconds per image). Needs about 20 GB RAM |

Loading takes a minute or two. Until it finishes, the UI shows CLEF-Flash as "loading", and `/classify` returns 503.

## How it classifies

The prompt lives in `clef_schema.json`: the state, the question, and a description of each option. Edit it and
restart to see how accuracy moves. Probabilities are matched to labels by the model's `option_ids`, never by
position. Requests run one at a time (one GPU). `latency_ms` is the forward pass only, with a device sync on both
sides; `total_ms` also includes decoding and resizing the upload (longest side 1024 px, EXIF-rotated).

## Configuration (environment variables)

| Variable | Default | |
|---|---|---|
| `DEVICE` | auto: cuda → mps → cpu | e.g. `cuda:1` |
| `DTYPE` | `bf16` | `bf16` / `fp16` / `fp32` |
| `CLEF_MODEL` | `Cloudflare/clef-flash` | Hub repo id or a local snapshot folder |
| `CLEF_SCHEMA` | `clef_schema.json` | the prompt |
| `MAX_IMAGE_SIDE` / `MAX_UPLOAD_MB` | `1024` / `25` | upload handling |

## Files

`app.py` (FastAPI) · `clef_backend.py` (the model; torch is imported lazily) · `imaging.py` ·
`prompt.py` + `clef_schema.json` · `download_model.py` (used by `./run.sh download`) · `tests/` (`./run.sh test`)
