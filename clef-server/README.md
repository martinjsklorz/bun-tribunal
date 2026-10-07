# CLEF-Flash contender (port 8001)

A FastAPI server for Cloudflare's CLEF-Flash (Apache-2.0). It's a 9.4B-parameter model: a Qwen3.5-9B backbone with
a vision encoder and a "joint schema head". Instead of writing text, it returns a probability for each allowed
answer. We ask it one `choice` question with the options `hotdog` and `not_hotdog`. The API is described in
`../CONTRACT.md`.

## Which weights

| Backend | Weights | Size | Runs on |
|---|---|---|---|
| **`mlx`** (default on Apple Silicon) | [`mlx-community/clef-flash-4bit`](https://huggingface.co/mlx-community/clef-flash-4bit): a **4-bit MLX quantization** of the model | ~6 GB on disk, 7–9 GB memory while running | Apple Silicon Mac, 16 GB+ |
| `torch` (default elsewhere) | [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash): the full BF16 release | ~19 GB | NVIDIA GPU with 24 GB+ VRAM, or CPU (slow) |

The 4-bit version is about a third of the size and loads much faster, at the cost of a little precision in the
probabilities. `CLEF_BACKEND=torch ./run.sh` forces the full model on a Mac (MPS, needs 32 GB+). `/health` reports
which one is running (`backend`, `weights`, `model_id`), and the UI shows the device as `mlx · 4-bit`.

## Setup

```bash
./run.sh download     # from the project root: the right weights for this machine, resumable
./run.sh              # starts it with everything else; the weights load in the background
```

The server **never downloads on its own**. Without the weights, `/health` reports "not set up" and names the command
above. Set `CLEF_AUTO_DOWNLOAD=1` to allow downloading on start anyway.

Loading takes a few seconds (MLX) to a minute or two (full model). Until it finishes, the UI shows CLEF-Flash as
"loading", and `/classify` returns 503.

## How it classifies

The prompt lives in `clef_schema.json`: the state, the question, and a description of each option. Edit it and
restart to see how accuracy moves. Probabilities are matched to labels by option id, never by position. Requests run
one at a time. `latency_ms` is the model call only; `total_ms` also includes decoding and resizing the upload
(longest side 1024 px, EXIF-rotated).

With MLX, the model repo brings its own loader (`clef_mlx.py`, from `mlx-vlm`); the server calls
`clef_mlx.load(path).predict(record)`, all on one dedicated thread because MLX streams belong to the thread that
created them.

## Configuration (environment variables)

| Variable | Default | |
|---|---|---|
| `CLEF_BACKEND` | `auto` | `mlx` / `torch`; auto = MLX on Apple Silicon, torch elsewhere |
| `CLEF_MODEL` | depends on the backend (table above) | Hub repo id or a local snapshot folder |
| `DEVICE` | auto: cuda → mps → cpu | torch only, e.g. `cuda:1` |
| `DTYPE` | `bf16` | torch only: `bf16` / `fp16` / `fp32` |
| `CLEF_SCHEMA` | `clef_schema.json` | the prompt |
| `MAX_IMAGE_SIDE` / `MAX_UPLOAD_MB` | `1024` / `25` | upload handling |

## Files

`app.py` (FastAPI) · `clef_backend.py` (both backends; torch / mlx are imported lazily) · `imaging.py` ·
`prompt.py` + `clef_schema.json` · `download_model.py` (used by `./run.sh download`) · `tests/` (`./run.sh test`)
