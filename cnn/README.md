# Bun Tribunal: CNN contender

A convolutional network for **hotdog / not hotdog**: an ImageNet-pretrained ConvNeXt-Tiny, fine-tuned on Food-101
(hot dogs plus look-alike dishes as hard negatives), calibrated, and served with the same API as the other two
contenders ([`../CONTRACT.md`](../CONTRACT.md)). Port **8002**, model name **`cnn`**.

```
cnn/
├── train_hotdog_cnn.ipynb   # data -> augment -> train -> calibrate -> evaluate -> export
├── run_training.py          # runs the notebook headless (used by ./run.sh train)
├── hotdog_model.py          # architectures (shared by notebook and server)
├── hotdog_data.py           # data sources, Food-101 sampling, test-leak de-duplication
├── server/app.py            # FastAPI inference server
├── tests/                   # server + data/arch tests
├── artifacts/               # the trained model and training report (git-ignored)
├── data/                    # downloaded datasets (git-ignored)
└── requirements.txt         # training, server and tests (installed by ./run.sh)
```

## 1. Train

From the project root:

```bash
./run.sh train
```

This runs `train_hotdog_cnn.ipynb` headless, top to bottom, prints progress as it goes, and writes the model plus an
executed copy of the notebook with all plots and metrics (`training_report.ipynb` / `.html`) to `cnn/artifacts/`.

For the interactive version, open the notebook in Jupyter and run all cells:

```bash
source .venv/bin/activate
cd cnn && jupyter lab train_hotdog_cnn.ipynb
```

### Configuration

Set these as environment variables (e.g. `ARCH=efficientnet_b0 ./run.sh train`; an empty value means the default)
or edit the notebook's **Configuration** cell:

| Setting | Default | Options |
|---|---|---|
| `DATA_SOURCE` | `food101`: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), free, no account | `kaggle`: the [SeeFood set](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog) (needs a Kaggle account) · `hf`: a Hugging Face dataset (check the id and columns in the notebook) · `folder`: `cnn/data/{train,test}/{hot_dog,not_hot_dog}/` |
| `ADD_FOOD101` | `1` (only for `kaggle` / `hf` / `folder`): adds ~1 000 Food-101 hot dogs + ~2 200 negatives to **training only** | `0`: source data only |
| `ARCH` | `convnext_tiny` (27.8M params, most accurate) | `efficientnet_b0` (4.0M, about 2× faster) · `mobilenet_v3_large` (4.2M, fastest) |
| `PRETRAINED` | `1`: start from ImageNet weights | `0`: random init |
| `DEVICE` | auto: CUDA, then Apple MPS, then CPU | force one, e.g. `cpu` |
| `SMOKE` | `0` | `1`: 2 epochs on synthetic images, no downloads (see below) |
| `DATA_DIR` | `data` (relative to `cnn/`) | where datasets are downloaded / read |
| `CNN_ARTIFACTS` | `artifacts` (relative to `cnn/`) | where the model and report are written |

**No account needed** with the default source. Food-101 is a one-time ~5 GB download into `cnn/data/`. Training uses
its train split: all 750 hot dogs plus ~2 200 other dishes, two thirds of them look-alikes (fries, prime rib, tacos …).
The test set is 250 hot dogs + 250 other dishes from its separate test split, the same size and mix as Kaggle's SeeFood
test set (which was cut from Food-101). With `DATA_SOURCE=kaggle`, `./run.sh train` asks for a Kaggle API token if
none is found (`~/.kaggle/access_token` or `~/.kaggle/kaggle.json`), and Food-101 images that duplicate any Kaggle
image are removed before training.

### Expected result

On an Apple Silicon Mac, training takes roughly 15–25 min. A run on the Kaggle SeeFood set (500 test images) scored:

| Accuracy | Precision | Recall | ROC-AUC | ECE |
|---|---|---|---|---|
| 94.0 % | 0.99 | 0.89 | 0.989 | 0.017 |

(Positive class = hotdog. ECE is the calibration error after temperature scaling.)

The notebook writes to `artifacts/`:

* `hotdog_cnn.pt`: state_dict (primary)
* `hotdog_cnn.ts.pt`: TorchScript (fallback)
* `model_meta.json`: arch, classes, preprocessing, calibration `temperature`, metrics, data composition, latency

`SMOKE=1` is a quick end-to-end check of the pipeline. It overwrites `artifacts/` with a useless model, so retrain
afterwards.

## 2. Serve

`./run.sh` starts the server on port 8002 together with everything else. To run it on its own, from the project root:

```bash
.venv/bin/python -m uvicorn --app-dir cnn/server app:app --port 8002
```

| Env | Default | Meaning |
|---|---|---|
| `CNN_ARTIFACTS` | `artifacts` (relative to `cnn/`) | directory with `model_meta.json` and the weights |
| `DEVICE` | auto (cuda, then mps, then cpu) | force a device |
| `MAX_UPLOAD_MB` | `25` | larger uploads → 413 |
| `PORT_UI` | `8080` | the UI port allowed by CORS |

Without trained artifacts, `/health` reports `"ready": false` with
`"detail": "CNN not trained yet — run ./run.sh train (or open cnn/train_hotdog_cnn.ipynb)"`, and `/classify` returns 503.

- **Preprocessing** matches evaluation: EXIF transpose, RGB, Resize(256), CenterCrop(224), ImageNet normalize. Big
  JPEGs are decoded directly at a reduced scale (still at least 256 px), which is much faster for phone photos.
- **Probabilities** are `softmax(logits / temperature)`.
- **Speed:** at load time the server measures both memory layouts (channels_last and contiguous; MPS uses contiguous
  only) and keeps the faster one. One forward pass runs at a time. `latency_ms` is the forward pass only
  (device-synchronized); `total_ms` includes decoding and preprocessing.
- **Uploads** are limited to 25 MB (`MAX_UPLOAD_MB`); broken images get 400.

```bash
curl -s localhost:8002/health
# {"status":"ok","model":"cnn","device":"mps","ready":true,"arch":"convnext_tiny","weights":"state_dict","temperature":1.31}

curl -s -F "file=@hotdog.jpg" localhost:8002/classify
# {"model":"cnn","label":"hotdog","is_hotdog":true,"confidence":0.9412,
#  "probabilities":{"hotdog":0.9412,"not_hotdog":0.0588},"latency_ms":3.1,"total_ms":14.7}
```

## 3. Test

`./run.sh test` runs these together with the other suites. Just the CNN:

```bash
cd cnn && ../.venv/bin/python -m pytest
```

The tests cover missing or outdated artifacts, CORS, the architectures, test-leak de-duplication, and a real model:
the notebook runs in SMOKE mode (`ARCH=mobilenet_v3_large`) into a temporary directory, so your `artifacts/` stays
untouched. Set `CNN_TEST_ARTIFACTS=artifacts` to test against your trained model instead.
