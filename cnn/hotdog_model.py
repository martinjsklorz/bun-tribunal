"""Single source of truth for the hotdog CNN architectures (ImageNet-pretrained torchvision CNNs).

Imported by both `train_hotdog_cnn.ipynb` and `server/app.py`, so the weights
saved by the notebook (state_dict) can always be re-instantiated by the server.
"""
from __future__ import annotations

import torch
from torch import nn

CLASSES = ["hotdog", "not_hotdog"]  # index 0 = hotdog
IMG_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# arch -> (torchvision builder, weights enum name, human description)
_TV_ARCHS = {
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", "ConvNeXt-Tiny, 27.8M params, ImageNet"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", "EfficientNet-B0, 4.0M params, ImageNet"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "MobileNetV3-Large, 4.2M params, ImageNet"),
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
    import torchvision.models as tvm

    fn_name, weights_name, _ = _TV_ARCHS[arch]
    weights = getattr(tvm, weights_name).DEFAULT if pretrained else None
    model = getattr(tvm, fn_name)(weights=weights)
    # All supported archs end in `.classifier[-1]` (a Linear layer): swap it for a 2-way head.
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def describe(arch: str) -> str:
    return _TV_ARCHS[arch][2]


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.classifier.parameters())


def param_groups(model: nn.Module, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
    """Discriminative learning rates: the pretrained backbone moves slower than the new head."""
    head_ids = {id(p) for p in head_parameters(model)}
    head = [p for p in model.parameters() if id(p) in head_ids and p.requires_grad]
    body = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    groups = [{"params": head, "lr": lr}]
    if body:
        groups.append({"params": body, "lr": lr * backbone_lr_mult})
    return groups


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    head_ids = {id(p) for p in head_parameters(model)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def cam_layer(model: nn.Module) -> nn.Module:
    """Last spatial feature layer, used for Grad-CAM."""
    return model.features[-1]


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def pick_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
