# Bun Tribunal: CNN contender

A convolutional network for **hotdog / not hotdog**: an ImageNet-pretrained ConvNeXt-Tiny, fine-tuned on the Kaggle
SeeFood set plus Food-101 hard negatives, calibrated, and served with the same API as the 9B CLEF-Flash VLM
(see `../CONTRACT.md`). Port **8002**, model name **`cnn`**.

```
cnn/
├── train_hotdog_cnn.ipynb   # data -> augment -> train -> calibrate -> evaluate -> export
├── hotdog_model.py          # architectures (shared by notebook and server)
├── hotdog_data.py           # data sources, Food-101 sampling, test-leak de-duplication
├── artifacts/               # written by the notebook (git-ignored)
├── server/app.py            # FastAPI inference server
├── tests/                   # server + data/arch tests
└── requirements.txt         # everything: training, server, tests (installed by ./run.sh)
```

## 1. Train

From the project root:

```bash
./run.sh train
```

This runs `train_hotdog_cnn.ipynb` headless, top to bottom, prints progress as it goes, saves an executed copy of the
notebook (with all plots and metrics) and writes the model to `cnn/artifacts/`.

For the interactive version, open the notebook in Jupyter and run all cells:

```bash
source .venv/bin/activate
cd cnn && jupyter lab train_hotdog_cnn.ipynb
```

### Configuration

Set these as environment variables (e.g. `ARCH=efficientnet_b0 ./run.sh train`) or edit the **Configuration** cell:

| Setting | Default | Options |
|---|---|---|
| `DATA_SOURCE` | `kaggle`: [`dansbecker/hot-dog-not-hot-dog`](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog) | `hf`: a Hugging Face dataset (check the id and columns in the notebook) · `folder`: `cnn/data/{train,test}/{hot_dog,not_hot_dog}/` |
| `ADD_FOOD101` | `1`: adds ~1 000 Food-101 hotdogs + ~2 200 negatives (look-alikes such as fries, prime rib and tacos oversampled) to **training only** | `0`: source data only |
| `ARCH` | `convnext_tiny` (27.8M params, most accurate) | `efficientnet_b0` (4.0M, about 2× faster) · `mobilenet_v3_large` (4.2M, fastest) |
| `DEVICE` | auto: CUDA, then Apple MPS, then CPU | force one, e.g. `cpu` |

**Kaggle credentials** are required for the default source: `~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME` and
`KAGGLE_KEY` in the environment. **Food-101** is a one-time ~5 GB download into `cnn/data/`. Food-101 images that
duplicate any Kaggle image are removed before training, so the test set stays unseen.

### Expected result

On an Apple Silicon Mac with the defaults, training takes roughly 15–25 min. On the 500 Kaggle test images:

| Accuracy | Precision | Recall | ROC-AUC | ECE |
|---|---|---|---|---|
| 94.0 % | 0.99 | 0.89 | 0.989 | 0.017 |

(Positive class = hotdog. ECE is the calibration error after temperature scaling.)

The notebook writes:

* `artifacts/hotdog_cnn.pt`: state_dict (primary)
* `artifacts/hotdog_cnn.ts.pt`: TorchScript (fallback)
* `artifacts/model_meta.json`: arch, classes, preprocessing, calibration `temperature`, metrics, data composition, latency

`SMOKE=1` trains 2 epochs on synthetic images with no downloads at all, to check the pipeline end to end. It
overwrites `artifacts/` with a useless model, so retrain afterwards.

## 2. Serve

`./run.sh` starts the server on port 8002 together with the rest of the arena. To run it on its own, from the project root:

```bash
uvicorn --app-dir cnn/server app:app --port 8002
```

| Env | Default | Meaning |
|---|---|---|
| `CNN_ARTIFACTS` | `cnn/artifacts` | directory with `model_meta.json` and weights |
| `DEVICE` | auto (cuda, then mps, then cpu) | force a device |

Without trained artifacts, `/health` reports `"ready": false` with
`"detail": "CNN not trained yet — run ./run.sh train (or open cnn/train_hotdog_cnn.ipynb)"`, and `/classify` returns 503.
Preprocessing matches evaluation: EXIF transpose, RGB, Resize(256), CenterCrop(224), ImageNet normalize.
Probabilities are `softmax(logits / temperature)`. `latency_ms` is the forward pass only (device-synchronized);
`total_ms` includes decoding and preprocessing.

```bash
curl -s localhost:8002/health
# {"status":"ok","model":"cnn","device":"mps","ready":true,"arch":"convnext_tiny","weights":"state_dict","temperature":1.31}

curl -s -F "file=@hotdog.jpg" localhost:8002/classify
# {"model":"cnn","label":"hotdog","is_hotdog":true,"confidence":0.9412,
#  "probabilities":{"hotdog":0.9412,"not_hotdog":0.0588},"latency_ms":3.1,"total_ms":14.7}
```

## 3. Test

```bash
cd cnn
../.venv/bin/python -m pytest -q
```

The tests cover missing or outdated artifacts, CORS, the architectures, test-leak de-duplication, and a real model:
the notebook runs in SMOKE mode (`ARCH=mobilenet_v3_large`) into a temporary directory, so your `artifacts/` stays
untouched. Set `CNN_TEST_ARTIFACTS=artifacts` to test against your trained model instead.
