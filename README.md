# Bun Tribunal 🌭

**Three very different AI models answer the most important question in computer vision: _hot dog or not hot dog?_**

![Bun Tribunal: LLM, CLEF-Flash and CNN judging the same photo side by side](docs/screenshot.png)

| | Contender | What it is | Runs |
|---|---|---|---|
| 🟣 | **LLM** | A general-purpose vision chatbot asked for a verdict, a confidence and a one-line reason | via any OpenAI-compatible API: OpenAI, OpenRouter, or locally with Ollama / LM Studio |
| 🟠 | **CLEF-Flash** | Cloudflare's 9.4B decision model. It doesn't write text; it returns a probability per allowed answer | locally (~19 GB of weights) |
| 🔵 | **CNN** | ConvNeXt-Tiny (28M params), fine-tuned on your laptop in ~20 min | locally |

Upload, drop, paste or snap a photo. All three judge it in parallel, and you get their verdicts, confidences and
speeds side by side, plus a session scoreboard once you tell it the truth.

## Quick start

```bash
./run.sh
```

That's it. The first run creates a Python environment, installs everything and walks you through setting up each
model (you can skip any of them), then starts the app and opens **http://localhost:8080**. Ctrl+C stops everything.

**Needs:** macOS or Linux, Python 3.11+ (`brew install python@3.12`), and for the real models an Apple Silicon Mac
with 32 GB+ memory (CLEF-Flash) or an NVIDIA GPU. The LLM and the CNN also run fine on smaller machines.

## Everything `run.sh` does

| Command | |
|---|---|
| `./run.sh` | Set up what's missing (it asks first), start everything, open the browser |
| `./run.sh llm` | Pick the LLM provider (OpenAI, OpenRouter, Ollama, LM Studio, other), enter a key, test it with one picture |
| `./run.sh train` | Train the CNN: downloads the data, fine-tunes, evaluates, exports. Shows live progress and writes an HTML report |
| `./run.sh download` | Download the CLEF-Flash weights (~19 GB, resumable) |
| `./run.sh status` | What's set up, what's running |
| `./run.sh test` | All test suites (no keys or downloads needed) |
| `LAN=1 ./run.sh` | Also serve your Wi-Fi, to try it on your phone (the address is printed). Anyone on that Wi-Fi can then use your LLM key, so use it on networks you trust |

A model that isn't set up yet doesn't break anything. Its card says **not set up** and shows the exact command that
fixes it, and the other models keep working. If a provider has a hiccup, every card has a **Retry** button.

## The three contenders in a bit more detail

**LLM** (`llm-server/`). It sends the picture with a strict definition: a sausage in a bun, so no corn dogs and no
dachshunds. If the API returns token probabilities (OpenAI does), the confidence comes from the model's actual token
distribution; otherwise it's the number the model states itself, and the card marks it as "self-reported".
Rate limits and overloaded providers are retried automatically. Reasoning models get extra room to think.
→ [llm-server/README.md](llm-server/README.md)

**CLEF-Flash** (`clef-server/`). This is [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash),
asked one typed question with the options `hotdog` / `not_hotdog`, each with a short description. Edit
`clef-server/clef_schema.json` to change the question. → [clef-server/README.md](clef-server/README.md)

**CNN** (`cnn/`). An ImageNet-pretrained ConvNeXt-Tiny, fine-tuned on Kaggle's SeeFood hot dog set plus extra hot
dogs and look-alike dishes (fries, tacos, prime rib …) from Food-101. Food-101 images that duplicate any Kaggle image are
removed first, so the test score isn't inflated by leaked test images. A typical run scores **94 % accuracy on the 500 held-out test images** (precision
0.99, recall 0.89, ROC-AUC 0.99), with calibrated confidences, in about 20 minutes on an Apple Silicon Mac.
The notebook `cnn/train_hotdog_cnn.ipynb` explains every step and shows Grad-CAM heatmaps of what the model looks
at. → [cnn/README.md](cnn/README.md)

## Project layout

```
run.sh            the one script
ui/               the web app (static HTML/CSS/JS, no build step)
llm-server/       LLM contender, port 8003
clef-server/      CLEF-Flash contender, port 8001
cnn/              training notebook + CNN server, port 8002
scripts/          small helpers used by run.sh
CONTRACT.md       the JSON API all three servers share
LICENSE           MIT
docs/             screenshot
```

Local state that is never committed: `.venv/` (Python packages), `llm-server/.env` (your LLM settings and key),
`cnn/artifacts/` (your trained model and training report), `cnn/data/` (datasets), `logs/` (server logs).

## Troubleshooting

- **"Port 8080 is already in use"**: Bun Tribunal is probably still running in another terminal. If another app
  owns 8080, use `PORT_UI=9090 ./run.sh`.
- **Changed the LLM settings?** Restart `./run.sh` so the LLM server picks them up.
- **A card says "not set up"**: run the command it shows, then restart `./run.sh`.
- **Something crashed**: `run.sh` prints the last lines of the log; full logs are in `logs/`.
- **Start over with a model's setup questions**: `./run.sh setup` asks again for everything that's missing.

## License

The code in this repository is released under the [MIT License](LICENSE).

It does not cover the models and data the project downloads, which keep their own terms:
[CLEF-Flash](https://huggingface.co/Cloudflare/clef-flash) (Apache-2.0), the ImageNet-pretrained torchvision weights,
the Kaggle [SeeFood dataset](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog),
[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), and the LLM provider you connect.
The UI is an unofficial parody; all of its artwork is original.
