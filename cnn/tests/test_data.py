"""Unit tests for hotdog_data / hotdog_model helpers (no downloads)."""

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageEnhance

CNN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CNN_DIR))

from hotdog_data import (  # noqa: E402  (needs the sys.path entry above)
    FOOD101_HARD_NEGATIVES,
    HotdogDataset,
    dhash,
    drop_overlaps,
    load_food101_splits,
    make_synthetic,
    normalize_label,
    open_image,
)
from hotdog_model import (  # noqa: E402
    ARCHS,
    build_model,
    cam_layer,
    count_params,
    param_groups,
    set_backbone_trainable,
)

EXIF_ORIENTATION = 0x0112


def test_normalize_label():
    assert [normalize_label(x) for x in ("hot_dog", "Hotdog", "not_hot_dog", "nothotdog", "other")] == [0, 0, 1, 1, 1]
    with pytest.raises(ValueError):
        normalize_label("pizza")


def test_open_image_applies_exif_orientation(tmp_path):
    img = Image.new("L", (40, 20), 128)
    exif = img.getexif()
    exif[EXIF_ORIENTATION] = 6  # "rotate 90° CW to display"
    path = tmp_path / "rotated.jpg"
    img.save(path, exif=exif)

    from_file = open_image(path)
    assert (from_file.mode, from_file.size) == ("RGB", (20, 40))

    in_memory = Image.open(path)
    assert open_image(in_memory).size == (20, 40)
    assert in_memory.size == (40, 20)  # the caller's image is left untouched

    rgb = Image.new("RGB", (8, 8))
    assert open_image(rgb) is not rgb


def test_dataset_decodes_paths_and_images(tmp_path):
    imgs, ys = make_synthetic(4, 0)
    imgs[1].save(tmp_path / "a.png")
    ds = HotdogDataset([imgs[0], str(tmp_path / "a.png")], ys[:2], transform=lambda im: im.size)
    assert len(ds) == 2
    assert [ds[0], ds[1]] == [((256, 256), 0), ((256, 256), 1)]


def test_dhash_is_stable_and_discriminative():
    a, b = make_synthetic(2, 3)[0]
    assert dhash(a) == dhash(a.copy())
    assert 0 < dhash(a) < 2**256
    assert (dhash(a) ^ dhash(b)).bit_count() > 16


def test_drop_overlaps_catches_duplicates_keeps_new(tmp_path):
    imgs, _ = make_synthetic(10, 3)
    protected = []
    for i, im in enumerate(imgs):
        p = tmp_path / f"t{i}.jpg"
        im.save(p)
        protected.append(str(p))
    cand_dir = tmp_path / "cand"
    cand_dir.mkdir()
    imgs[0].resize((512, 512)).save(cand_dir / "resized.jpg", quality=60)  # near-dup
    ImageEnhance.Brightness(imgs[3]).enhance(1.1).resize((300, 300)).save(cand_dir / "bright.jpg")  # near-dup
    imgs[1].save(cand_dir / "t1.jpg")  # same file id
    make_synthetic(4, 99)[0][2].save(cand_dir / "new_hotdog.jpg")  # genuinely new
    make_synthetic(4, 77)[0][3].save(cand_dir / "new_blob.jpg")  # genuinely new
    cand = sorted(str(p) for p in cand_dir.iterdir())
    extra = [Path(c).stem for c in cand]
    kept, labels, kept_extra, dropped = drop_overlaps(cand, list(range(len(cand))), extra, protected)
    assert sorted(Path(k).name for k in kept) == ["new_blob.jpg", "new_hotdog.jpg"]
    assert [cand[i] for i in labels] == kept  # labels and extras stay aligned with their paths
    assert kept_extra == [Path(k).stem for k in kept]
    assert dropped == {"near-duplicate image": 2, "same file id": 1}


@pytest.mark.parametrize("arch", ["convnext_tiny", "efficientnet_b0", "mobilenet_v3_large"])
def test_archs_build_and_freeze(arch):
    assert arch in ARCHS
    m = build_model(arch).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 3, 224, 224)).shape == (1, 2)
    assert cam_layer(m) is not None
    total = count_params(m)[0]
    set_backbone_trainable(m, False)
    assert 0 < count_params(m)[1] < total
    head_only = param_groups(m, 1e-3)
    assert len(head_only) == 1 and all(p.requires_grad for p in head_only[0]["params"])
    set_backbone_trainable(m, True)
    assert count_params(m)[1] == total
    groups = param_groups(m, 1e-3, backbone_lr_mult=0.1)
    assert [g["lr"] for g in groups] == pytest.approx([1e-3, 1e-4])
    assert sum(len(g["params"]) for g in groups) == len(list(m.parameters()))


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        build_model("scratch")


def test_food101_splits_are_disjoint_and_balanced(tmp_path):
    """Default data source: no account, official non-overlapping splits, 250 + 250 test images."""
    others = [f"other_{i}" for i in range(10)]
    classes = ["hot_dog", *FOOD101_HARD_NEGATIVES, *others]
    meta = tmp_path / "food-101" / "meta"
    meta.mkdir(parents=True)
    for split, n, n_hot in (("train", 40, 750), ("test", 20, 300)):
        idx = {c: [f"{c}/{split}{i}" for i in range(n_hot if c == "hot_dog" else n)] for c in classes}
        (meta / f"{split}.json").write_text(json.dumps(idx))

    (tr_x, tr_y, tr_cls), (te_x, te_y) = load_food101_splits(tmp_path, hard_per_class=30, other_per_class=5, seed=1)
    assert set(tr_x).isdisjoint(te_x)
    assert tr_y.count(0) == 750 and tr_cls.count("hot_dog") == 750  # every training hot dog
    assert tr_cls.count("french_fries") == 30 and tr_cls.count("other_0") == 5  # look-alikes oversampled
    assert te_y.count(0) == 250 and te_y.count(1) == 250  # SeeFood-sized test set
    hard_in_test = sum(1 for p in te_x if p.split("/")[-2] in FOOD101_HARD_NEGATIVES)
    assert hard_in_test == 125  # half of the negatives
    again = load_food101_splits(tmp_path, hard_per_class=30, other_per_class=5, seed=1)
    assert again[1][0] == te_x  # deterministic
