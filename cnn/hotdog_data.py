"""Data loading for the hotdog notebook.

Lives in a module (not in the notebook) so that DataLoader worker processes can
import it — on macOS workers are *spawned*, and classes defined inside a notebook
cannot be pickled, which forced num_workers=0 and made decoding single-threaded.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
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
]  # fmt: skip

ImageSource = str | Path | Image.Image  # a file path or an in-memory image (Hugging Face, synthetic)
Split = tuple[list, list[int]]  # (images, labels)


# ---------------------------------------------------------------- images & labels
def open_image(src: ImageSource) -> Image.Image:
    """Upright RGB copy of `src` (EXIF orientation applied); never mutates an in-memory image."""
    if isinstance(src, Image.Image):
        img = ImageOps.exif_transpose(src)  # returns a copy
    else:
        img = Image.open(src)
        ImageOps.exif_transpose(img, in_place=True)  # our own file handle: no copy needed
    return img if img.mode == "RGB" else img.convert("RGB")


def normalize_label(name: object) -> int:
    """Map assorted label spellings to 0 (hotdog) / 1 (not_hotdog)."""
    s = re.sub(r"[^a-z]", "", str(name).lower())
    if s in {"hotdog", "hotdogs"}:
        return 0
    if s.startswith(("not", "non")) or s == "other":
        return 1
    raise ValueError(f"cannot map label {name!r} to {CLASSES}")


class HotdogDataset(Dataset):
    """(image, label) pairs; images are paths or PIL images, decoded lazily in the DataLoader workers."""

    def __init__(self, items: Sequence[ImageSource], labels: Sequence[int], transform: Callable) -> None:
        self.items, self.labels, self.transform = list(items), list(labels), transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        return self.transform(open_image(self.items[i])), self.labels[i]


# ---------------------------------------------------------------- sources
def scan_split(split_dir: Path) -> tuple[list[str], list[int]]:
    """Image paths and labels from `split_dir/<class name>/**`."""
    paths, labels = [], []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        y = normalize_label(class_dir.name)
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                paths.append(str(f))
                labels.append(y)
    return paths, labels


def load_folder(root: Path) -> tuple[Split, Split]:
    """((train_paths, train_labels), (test_paths, test_labels)) from `root/{train,test}/<class>/`."""
    root = Path(root)
    if not (root / "train").is_dir():
        raise FileNotFoundError(f"expected {root}/train/{{hot_dog,not_hot_dog}}")
    test_dir = root / "test"
    return scan_split(root / "train"), (scan_split(test_dir) if test_dir.is_dir() else ([], []))


def load_kaggle(dataset_id: str) -> tuple[Split, Split]:
    import kagglehub

    base = Path(kagglehub.dataset_download(dataset_id))
    # The archive nests things (e.g. seefood/train/hot_dog): use the shallowest dir with a train/ of class dirs.
    candidates = [p.parent for p in base.rglob("train") if p.is_dir() and any(c.is_dir() for c in p.iterdir())]
    if not candidates:
        raise FileNotFoundError(f"no train/ folder found under {base}")
    root = min(candidates, key=lambda p: len(p.parts))
    print("kaggle dataset root:", root)
    return load_folder(root)


def load_hf(dataset_id: str, image_col: str, label_col: str, test_split: str) -> tuple[Split, Split]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    print(ds)

    def convert(split) -> Split:
        names = getattr(split.features[label_col], "names", None)  # ClassLabel -> index to name
        imgs, ys = [], []
        for row in split:
            raw = row[label_col]
            ys.append(normalize_label(names[raw] if names is not None else raw))
            imgs.append(row[image_col])
        return imgs, ys

    return convert(ds["train"]), (convert(ds[test_split]) if test_split in ds else ([], []))


def _food101_index(root: Path) -> tuple[Path, dict[str, dict[str, list[str]]]]:
    """Download Food-101 once (~5 GB, free, no account) and return (base_dir, {split: {class: [ids]}})."""
    base = Path(root) / "food-101"
    if not (base / "meta" / "train.json").exists():
        from torchvision.datasets import Food101

        Food101(str(root), split="train", download=True)  # downloads + extracts from ETH Zürich
    index = {split: json.loads((base / "meta" / f"{split}.json").read_text()) for split in ("train", "test")}
    if missing := [c for c in FOOD101_HARD_NEGATIVES if c not in index["train"]]:
        print("note: hard-negative classes not in Food-101:", missing)
    return base, index


def _sample(
    base: Path, ids_by_class: dict[str, list[str]], plan: dict[str, int], rng: random.Random
) -> tuple[list[str], list[int], list[str]]:
    """plan = {class: k}, k capped at the class size. Returns (paths, labels, classes); label 0 = hot_dog."""
    paths, labels, classes = [], [], []
    for cls in sorted(plan):  # sorted + sorted ids: the same seed always draws the same images
        ids = sorted(ids_by_class[cls])
        for i in rng.sample(ids, k=min(plan[cls], len(ids))):
            paths.append(str(base / "images" / f"{i}.jpg"))
            labels.append(0 if cls == "hot_dog" else 1)
            classes.append(cls)
    return paths, labels, classes


def _negative_plan(classes, hard: set[str], n_hard_per_class: int, n_other_per_class: int) -> dict[str, int]:
    return {c: (n_hard_per_class if c in hard else n_other_per_class) for c in classes if c != "hot_dog"}


def _spread(total: int, classes: Sequence[str]) -> dict[str, int]:
    """Split `total` as evenly as possible over `classes` (the first ones get the remainder)."""
    if not classes:
        return {}
    q, r = divmod(total, len(classes))
    return {c: q + (i < r) for i, c in enumerate(classes) if q + (i < r)}


def load_food101_splits(
    root: Path,
    hard_per_class: int,
    other_per_class: int,
    test_hotdogs: int = 250,
    test_negatives: int = 250,
    seed: int = 0,
) -> tuple[tuple[list[str], list[int], list[str]], tuple[list[str], list[int]]]:
    """Hotdog / not-hotdog train AND test sets from Food-101's official, non-overlapping splits.

    Train: every training hot dog (750) plus negatives, with look-alike dishes oversampled.
    Test:  `test_hotdogs` hot dogs + `test_negatives` other dishes from the test split, half of them
           look-alikes: the same size and mix as the Kaggle SeeFood test set (which was cut from Food-101).
    Returns ((train_paths, train_labels, train_classes), (test_paths, test_labels)).
    """
    base, index = _food101_index(root)
    rng = random.Random(seed)
    hard = {c for c in FOOD101_HARD_NEGATIVES if c in index["train"]}

    train_plan = {"hot_dog": sys.maxsize, **_negative_plan(index["train"], hard, hard_per_class, other_per_class)}
    train = _sample(base, index["train"], train_plan, rng)

    others = sorted(c for c in index["test"] if c != "hot_dog" and c not in hard)
    n_hard = test_negatives // 2
    test_plan = {
        "hot_dog": test_hotdogs,
        **_spread(n_hard, sorted(hard)),  # the look-alike half evenly over the hard classes
        **_spread(test_negatives - n_hard, others),  # and the rest over all other classes
    }
    test_x, test_y, _ = _sample(base, index["test"], test_plan, rng)
    return train, (test_x, test_y)


def load_food101(
    root: Path, n_hotdog: int, hard_per_class: int, other_per_class: int, seed: int = 0
) -> tuple[list[str], list[int], list[str]]:
    """Extra *training* images from Food-101 (both splits) for the other data sources.
    Returns (paths, labels, food_class_per_image)."""
    base, index = _food101_index(root)
    pool: dict[str, list[str]] = {}
    for split in ("train", "test"):
        for cls, ids in index[split].items():
            pool.setdefault(cls, []).extend(ids)
    hard = {c for c in FOOD101_HARD_NEGATIVES if c in pool}
    plan = {"hot_dog": n_hotdog, **_negative_plan(pool, hard, hard_per_class, other_per_class)}
    return _sample(base, pool, plan, random.Random(seed))


# ---------------------------------------------------------------- de-duplication
def dhash(src: ImageSource, size: int = 16) -> int:
    """256-bit difference hash: robust to resizing/re-encoding, cheap to compute.

    16x16 instead of the classic 8x8: 64-bit hashes collide on flat or dark images."""
    im = open_image(src).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = np.asarray(im, dtype=np.uint8)  # (size, size + 1)
    bits = np.packbits(px[:, :-1] > px[:, 1:])  # row-major, first pixel = most significant bit
    return int.from_bytes(bits.tobytes(), "big")


def _dhash_all(images: Sequence[ImageSource]) -> list[int]:
    # PIL releases the GIL while decoding and resizing, so threads give a near-linear speedup.
    with ThreadPoolExecutor() as pool:
        return list(pool.map(dhash, images))


def drop_overlaps(paths, labels, extra, protected, max_bits: int = 16):
    """Remove from (paths, labels, extra) any image that also appears in `protected` (e.g. the test set).

    Matches by file stem (Kaggle's SeeFood images *are* Food-101 images, often with the same numeric id)
    and by near-identical dHash (Hamming distance <= max_bits out of 256, i.e. ~94% identical).
    Returns (paths, labels, extra, {reason: n_dropped}).
    """
    prot_stems = {Path(p).stem for p in protected if isinstance(p, (str, Path))}
    dropped = Counter()
    candidates = []
    for i, p in enumerate(paths):
        if Path(p).stem in prot_stems:
            dropped["same file id"] += 1
        else:
            candidates.append(i)

    prot_hashes = _dhash_all(protected)
    keep = []
    for i, h in zip(candidates, _dhash_all([paths[i] for i in candidates]), strict=True):
        if any((h ^ q).bit_count() <= max_bits for q in prot_hashes):  # brute force: ~1 s for 500 x 4k
            dropped["near-duplicate image"] += 1
        else:
            keep.append(i)
    return [paths[i] for i in keep], [labels[i] for i in keep], [extra[i] for i in keep], dict(dropped)


# ---------------------------------------------------------------- smoke data
def make_synthetic(n: int, seed: int) -> tuple[list[Image.Image], list[int]]:
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
