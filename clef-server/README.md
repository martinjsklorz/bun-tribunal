# CLEF-Flash contender (port 8001)

A FastAPI server for Cloudflare's CLEF-Flash (Apache-2.0): a 9.4B-parameter model with a Qwen3.5-9B backbone, a
vision encoder and a "joint schema head". Instead of writing text, it returns a probability for each allowed answer.
We ask it one `choice` question with the options `hotdog` and `not_hotdog`. API: [`../CONTRACT.md`](../CONTRACT.md).

## Which weights

| Backend | Weights | Size | Runs on |
|---|---|---|---|
| **`mlx`** (default on Apple Silicon) | [`mlx-community/clef-flash-4bit`](https://huggingface.co/mlx-community/clef-flash-4bit): a **4-bit MLX quantization** | ~6 GB on disk, 7–9 GB memory while running | Apple Silicon Mac, 16 GB+ |
| `torch` (default elsewhere) | [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash): the full BF16 release | ~19 GB | NVIDIA GPU with 24 GB+ VRAM, or CPU (slow) |

The 4-bit version is about a third of the size and loads much faster, at the cost of a little precision in the
probabilities. To run the full model on a Mac (MPS, 32 GB+), use `CLEF_BACKEND=torch` for both
`./run.sh download` and `./run.sh`. `/health` reports which one is running (`backend`, `weights`, `model_id`), and
the UI shows the device as `mlx · 4-bit`.

## Setup

```bash
./run.sh download     # from the project root: the right weights for this machine, resumable
./run.sh              # starts it with everything else; the weights load in the background
```

The weights go to the Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME`). `download_model.py` does the work
behind `./run.sh download`; run it from `clef-server/` with `--check` (exit 0 if downloaded, no network),
`--describe` (what would be downloaded) or `--help`.

The server **never downloads on its own**: without the weights, `/health` reports "not set up" and names
`./run.sh download`. Set `CLEF_AUTO_DOWNLOAD=1` to allow downloading on start anyway.

Loading takes a few seconds (MLX) to a minute or two (full model). Until it finishes, the UI shows CLEF-Flash as
"loading" and `/classify` returns 503.

## How it classifies

- **The prompt** lives in `clef_schema.json`: the state, the instructions, and a description of each option. Edit it
  and restart to see how accuracy moves. Probabilities are matched to labels by option id (`label_map`), never by
  position.
- **The image** is EXIF-rotated, flattened onto white if transparent, and shrunk to a longest side of 1024 px. Big
  JPEGs are decoded directly at a reduced scale, which is much faster than decoding the full photo. Images with
  more than `MAX_IMAGE_PIXELS` pixels are rejected with 400.
- **One request at a time** runs on the model. `latency_ms` is the model call only; `total_ms` also includes
  decoding and resizing.
- **MLX:** the model repo brings its own loader (`clef_mlx.py`, built on `mlx-vlm`); the server calls
  `clef_mlx.load(path).predict(record)`, all on one dedicated thread because MLX streams belong to the thread that
  created them.
- **torch:** if bf16 fails on MPS, the server retries in fp16.

## Configuration (environment variables)

| Variable | Default | |
|---|---|---|
| `CLEF_BACKEND` | `auto` | `mlx` / `torch`; auto = MLX on Apple Silicon, torch elsewhere |
| `CLEF_MODEL` | depends on the backend (table above) | Hub repo id or a local snapshot folder |
| `CLEF_AUTO_DOWNLOAD` | `0` | `1`: download missing weights on start |
| `DEVICE` | auto: cuda → mps → cpu | torch only, e.g. `cuda:1` |
| `DTYPE` | `bf16` | torch only: `bf16` / `fp16` / `fp32` |
| `CLEF_SCHEMA` | `clef_schema.json` | the prompt (same shape) |
| `MAX_IMAGE_SIDE` | `1024` | longest side the model sees |
| `MAX_IMAGE_PIXELS` | `268435456` (16384²) | larger images → 400 |
| `MAX_UPLOAD_MB` | `25` | larger uploads → 413 |
| `PORT_UI` | `8080` | the UI port allowed by CORS |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `HF_HOME` | `~/.cache/huggingface` | where the weights are cached |

## Files

`app.py` (FastAPI) · `clef_backend.py` (both backends; torch / mlx are imported lazily) · `imaging.py` (upload
decoding) · `prompt.py` + `clef_schema.json` · `download_model.py` · `tests/` (`./run.sh test`)
