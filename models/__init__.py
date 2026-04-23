from __future__ import annotations

from torch import nn

from models.cnn import SimpleCNN
from models.fastcnn import FastCNN
from models.resnet18_baseline import build_resnet18

MODEL_NAMES = ("simplecnn", "fastcnn", "resnet18")


def build_model(
    *,
    name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "fastcnn":
        return FastCNN(num_classes=num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model name: {name}")
