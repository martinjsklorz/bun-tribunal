"""Unit tests for hotdog_data / hotdog_model helpers (no downloads)."""
import sys
from pathlib import Path

import pytest
from PIL import ImageEnhance

CNN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CNN_DIR))

from hotdog_data import drop_overlaps, make_synthetic, normalize_label  # noqa: E402


def test_normalize_label():
    assert [normalize_label(x) for x in ("hot_dog", "Hotdog", "not_hot_dog", "nothotdog", "other")] == [0, 0, 1, 1, 1]
    with pytest.raises(ValueError):
        normalize_label("pizza")


def test_drop_overlaps_catches_duplicates_keeps_new(tmp_path):
    imgs, _ = make_synthetic(10, 3)
    protected = []
    for i, im in enumerate(imgs):
        p = tmp_path / f"t{i}.jpg"
        im.save(p)
        protected.append(str(p))
    cand_dir = tmp_path / "cand"
    cand_dir.mkdir()
    imgs[0].resize((512, 512)).save(cand_dir / "resized.jpg", quality=60)                      # near-dup
    ImageEnhance.Brightness(imgs[3]).enhance(1.1).resize((300, 300)).save(cand_dir / "bright.jpg")  # near-dup
    imgs[1].save(cand_dir / "t1.jpg")                                                           # same file id
    make_synthetic(4, 99)[0][2].save(cand_dir / "new_hotdog.jpg")                               # genuinely new
    make_synthetic(4, 77)[0][3].save(cand_dir / "new_blob.jpg")                                 # genuinely new
    cand = sorted(str(p) for p in cand_dir.iterdir())
    kept, labels, extra, dropped = drop_overlaps(cand, [1] * len(cand), ["x"] * len(cand), protected)
    assert sorted(Path(k).name for k in kept) == ["new_blob.jpg", "new_hotdog.jpg"]
    assert len(labels) == len(extra) == 2
    assert dropped == {"near-duplicate image": 2, "same file id": 1}


@pytest.mark.parametrize("arch", ["convnext_tiny", "efficientnet_b0", "mobilenet_v3_large"])
def test_archs_build_and_freeze(arch):
    import torch
    from hotdog_model import ARCHS, build_model, cam_layer, count_params, param_groups, set_backbone_trainable

    assert arch in ARCHS
    m = build_model(arch).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 3, 224, 224)).shape == (1, 2)
    assert cam_layer(m) is not None
    total = count_params(m)[0]
    set_backbone_trainable(m, False)
    assert 0 < count_params(m)[1] < total
    set_backbone_trainable(m, True)
    assert count_params(m)[1] == total
    assert sum(len(g["params"]) for g in param_groups(m, 1e-3)) == len(list(m.parameters()))


def test_unknown_arch_rejected():
    from hotdog_model import build_model

    with pytest.raises(ValueError):
        build_model("scratch")


def test_food101_splits_are_disjoint_and_balanced(tmp_path):
    """Default data source: no account, official non-overlapping splits, 250 + 250 test images."""
    import json
    from hotdog_data import FOOD101_HARD_NEGATIVES, load_food101_splits

    others = [f"other_{i}" for i in range(10)]
    classes = ["hot_dog", *FOOD101_HARD_NEGATIVES, *others]
    meta = tmp_path / "food-101" / "meta"
    meta.mkdir(parents=True)
    for split, n, n_hot in (("train", 40, 750), ("test", 20, 300)):
        idx = {c: [f"{c}/{split}{i}" for i in range(n_hot if c == "hot_dog" else n)] for c in classes}
        (meta / f"{split}.json").write_text(json.dumps(idx))

    (tr_x, tr_y, tr_cls), (te_x, te_y) = load_food101_splits(tmp_path, hard_per_class=30, other_per_class=5, seed=1)
    assert set(tr_x).isdisjoint(te_x)
    assert tr_y.count(0) == 750 and tr_cls.count("hot_dog") == 750                     # every training hot dog
    assert tr_cls.count("french_fries") == 30 and tr_cls.count("other_0") == 5         # look-alikes oversampled
    assert te_y.count(0) == 250 and te_y.count(1) == 250                               # SeeFood-sized test set
    hard_in_test = sum(1 for p in te_x if p.split("/")[-2] in FOOD101_HARD_NEGATIVES)
    assert hard_in_test == 125                                                         # half of the negatives
    again = load_food101_splits(tmp_path, hard_per_class=30, other_per_class=5, seed=1)
    assert again[1][0] == te_x                                                         # deterministic
