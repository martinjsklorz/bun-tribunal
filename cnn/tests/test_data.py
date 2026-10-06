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
