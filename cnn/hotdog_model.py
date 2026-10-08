"""Single source of truth for the hotdog CNN architectures (ImageNet-pretrained torchvision CNNs).

Imported by both `train_hotdog_cnn.ipynb` and `server/app.py`, so the weights
saved by the notebook (state_dict) can always be re-instantiated by the server.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torchvision.models as tvm
from torch import nn

CLASSES = ["hotdog", "not_hotdog"]  # index 0 = hotdog
IMG_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _Arch(NamedTuple):
    builder: str  # torchvision.models function name
    weights: str  # torchvision weights enum name
    description: str


_TV_ARCHS = {
    "convnext_tiny": _Arch("convnext_tiny", "ConvNeXt_Tiny_Weights", "ConvNeXt-Tiny, 27.8M params, ImageNet"),
    "efficientnet_b0": _Arch("efficientnet_b0", "EfficientNet_B0_Weights", "EfficientNet-B0, 4.0M params, ImageNet"),
    "mobilenet_v3_large": _Arch(
        "mobilenet_v3_large", "MobileNet_V3_Large_Weights", "MobileNetV3-Large, 4.2M params, ImageNet"
    ),
}
ARCHS = tuple(_TV_ARCHS)
DEFAULT_ARCH = "convnext_tiny"


def build_model(arch: str = DEFAULT_ARCH, num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """Build a torchvision CNN by name with a fresh `num_classes`-way head.

    `pretrained` downloads ImageNet weights via torchvision.
    The server always calls this with pretrained=False and then loads the saved state_dict.
    """
    if arch not in _TV_ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose one of {ARCHS}")
    spec = _TV_ARCHS[arch]
    weights = getattr(tvm, spec.weights).DEFAULT if pretrained else None
    model = getattr(tvm, spec.builder)(weights=weights)
    # All supported archs end in `.classifier[-1]` (a Linear layer): swap it for a fresh head.
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def describe(arch: str) -> str:
    return _TV_ARCHS[arch].description


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.classifier.parameters())


def _split_head_body(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    head_ids = {id(p) for p in head_parameters(model)}
    head, body = [], []
    for p in model.parameters():
        (head if id(p) in head_ids else body).append(p)
    return head, body


def param_groups(model: nn.Module, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
    """Discriminative learning rates: the pretrained backbone moves slower than the new head."""
    head, body = ([p for p in ps if p.requires_grad] for ps in _split_head_body(model))
    groups = [{"params": head, "lr": lr}]
    if body:
        groups.append({"params": body, "lr": lr * backbone_lr_mult})
    return groups


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    for p in _split_head_body(model)[1]:
        p.requires_grad = trainable


def cam_layer(model: nn.Module) -> nn.Module:
    """Last spatial feature layer, used for Grad-CAM."""
    return model.features[-1]


def count_params(model: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts."""
    params = list(model.parameters())
    return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad)


def pick_device(override: str | None = None) -> torch.device:
    """`override` if given, else the fastest available device: cuda -> mps -> cpu."""
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Wait for queued GPU work, so wall-clock timings measure the actual compute."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
