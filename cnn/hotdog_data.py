"""Data loading for the hotdog notebook.

Lives in a module (not in the notebook) so that DataLoader worker processes can
import it — on macOS workers are *spawned*, and classes defined inside a notebook
cannot be pickled, which forced num_workers=0 and made decoding single-threaded.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import Dataset

from hotdog_model import CLASSES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Food-101 classes that look most like a hotdog (meaty, fried or bun-shaped dishes).
# They get oversampled as negatives so the model learns "sausage in a bun", not "warm-coloured food".
FOOD101_HARD_NEGATIVES = [
    "french_fries", "prime_rib", "steak", "filet_mignon", "tacos", "spring_rolls", "pad_thai", "pizza",
    "ice_cream", "hamburger", "lobster_roll_sandwich", "pulled_pork_sandwich", "club_sandwich",
    "grilled_cheese_sandwich", "breakfast_burrito", "churros", "chicken_wings", "baby_back_ribs",
    "pork_chop", "croque_madame", "nachos", "onion_rings", "poutine", "chicken_quesadilla", "eggs_benedict",
]


# ---------------------------------------------------------------- images & labels
def open_image(src) -> Image.Image:
    img = Image.open(src) if isinstance(src, (str, Path)) else src
    return ImageOps.exif_transpose(img).convert("RGB")


def normalize_label(name) -> int:
    """Map assorted label spellings to 0 (hotdog) / 1 (not_hotdog)."""
    s = re.sub(r"[^a-z]", "", str(name).lower())
    if s in {"hotdog", "hotdogs"}:
        return 0
    if s.startswith("not") or s.startswith("non") or s in {"other", "nothotdog"}:
        return 1
    raise ValueError(f"cannot map label {name!r} to {CLASSES}")


class HotdogDataset(Dataset):
    def __init__(self, items, labels, transform):
        self.items, self.labels, self.transform = list(items), list(labels), transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.transform(open_image(self.items[i])), self.labels[i]


# ---------------------------------------------------------------- sources
def scan_split(split_dir: Path):
    paths, labels = [], []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        y = normalize_label(class_dir.name)
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                paths.append(str(f))
                labels.append(y)
    return paths, labels


def load_folder(root: Path):
    root = Path(root)
    if not (root / "train").is_dir():
        raise FileNotFoundError(f"expected {root}/train/{{hot_dog,not_hot_dog}}")
    tr = scan_split(root / "train")
    te = scan_split(root / "test") if (root / "test").is_dir() else ([], [])
    return tr, te


def load_kaggle(dataset_id: str):
    import kagglehub

    base = Path(kagglehub.dataset_download(dataset_id))
    # The archive nests things (e.g. seefood/train/hot_dog); find the dir that contains train/
    candidates = [p.parent for p in base.rglob("train") if p.is_dir() and any(c.is_dir() for c in p.iterdir())]
    if not candidates:
        raise FileNotFoundError(f"no train/ folder found under {base}")
    root = sorted(candidates, key=lambda p: len(p.parts))[0]
    print("kaggle dataset root:", root)
    return load_folder(root)


def load_hf(dataset_id: str, image_col: str, label_col: str, test_split: str):
    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    print(ds)

    def convert(split):
        names = getattr(split.features[label_col], "names", None)
        imgs, ys = [], []
        for row in split:
            raw = row[label_col]
            ys.append(normalize_label(names[raw] if names is not None else raw))
            imgs.append(row[image_col])
        return imgs, ys

    return convert(ds["train"]), (convert(ds[test_split]) if test_split in ds else ([], []))


def load_food101(root: Path, n_hotdog: int, hard_per_class: int, other_per_class: int, seed: int = 0):
    """Sample a hotdog/not-hotdog set from Food-101 (101 classes x 1000 images, ~5 GB download).

    Uses both of Food-101's own splits as a *training pool*; our test set stays the Kaggle one.
    Returns (paths, labels, food_class_per_image).
    """
    from torchvision.datasets import Food101

    Food101(str(root), split="train", download=True)  # downloads + extracts once
    base = Path(root) / "food-101"
    pool: dict[str, list[str]] = {}
    for split in ("train", "test"):
        for cls, ids in json.loads((base / "meta" / f"{split}.json").read_text()).items():
            pool.setdefault(cls, []).extend(ids)

    missing = [c for c in FOOD101_HARD_NEGATIVES if c not in pool]
    if missing:
        print("note: hard-negative classes not in Food-101:", missing)
    hard = [c for c in FOOD101_HARD_NEGATIVES if c in pool]

    rng = random.Random(seed)
    paths, labels, classes = [], [], []

    def take(cls, k, y):
        ids = sorted(pool[cls])
        for i in rng.sample(ids, k=min(k, len(ids))):
            paths.append(str(base / "images" / f"{i}.jpg"))
            labels.append(y)
            classes.append(cls)

    take("hot_dog", n_hotdog, 0)
    for cls in sorted(pool):
        if cls == "hot_dog":
            continue
        take(cls, hard_per_class if cls in hard else other_per_class, 1)
    return paths, labels, classes


# ---------------------------------------------------------------- de-duplication
def dhash(src, size: int = 16) -> int:
    """256-bit difference hash: robust to resizing/re-encoding, cheap to compute.

    16x16 instead of the classic 8x8: 64-bit hashes collide on flat or dark images."""
    im = open_image(src).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(im.tobytes())  # mode "L": one byte per pixel
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def drop_overlaps(paths, labels, extra, protected, max_bits: int = 16):
    """Remove from (paths, labels, extra) any image that also appears in `protected` (e.g. the test set).

    Matches by file stem (Kaggle's SeeFood images *are* Food-101 images, often with the same numeric id)
    and by near-identical dHash (Hamming distance <= max_bits out of 256, i.e. ~94% identical).
    """
    prot_stems = {Path(str(p)).stem for p in protected if isinstance(p, (str, Path))}
    prot_hashes = [dhash(p) for p in protected]

    keep, dropped = [], Counter()
    for i, p in enumerate(paths):
        if Path(p).stem in prot_stems:
            dropped["same file id"] += 1
            continue
        h = dhash(p)
        if any((h ^ q).bit_count() <= max_bits for q in prot_hashes):  # brute force: ~1 s for 500 x 4k
            dropped["near-duplicate image"] += 1
            continue
        keep.append(i)
    return [paths[i] for i in keep], [labels[i] for i in keep], [extra[i] for i in keep], dict(dropped)


# ---------------------------------------------------------------- smoke data
def make_synthetic(n: int, seed: int):
    """Toy 'hotdogs' (sausage on a bun) vs random blobs, just to exercise the pipeline."""
    rng = random.Random(seed)
    imgs, ys = [], []
    for i in range(n):
        y = i % 2
        bg = tuple(rng.randint(120, 255) for _ in range(3))
        im = Image.new("RGB", (256, 256), bg)
        d = ImageDraw.Draw(im)
        if y == 0:
            cx, cy = rng.randint(90, 166), rng.randint(90, 166)
            d.rounded_rectangle([cx - 90, cy - 30, cx + 90, cy + 30], radius=28, fill=(222, 170, 100))
            d.ellipse([cx - 100, cy - 14, cx + 100, cy + 14], fill=(170, 50, 30))
            d.line([cx - 70, cy, cx + 70, cy], fill=(240, 200, 30), width=4)
            im = im.rotate(rng.uniform(-40, 40), fillcolor=bg)
        else:
            for _ in range(rng.randint(2, 5)):
                x, yy, r = rng.randint(0, 256), rng.randint(0, 256), rng.randint(15, 60)
                col = tuple(rng.randint(0, 255) for _ in range(3))
                (d.ellipse if rng.random() < 0.5 else d.rectangle)([x - r, yy - r, x + r, yy + r], fill=col)
        imgs.append(im)
        ys.append(y)
    return imgs, ys
