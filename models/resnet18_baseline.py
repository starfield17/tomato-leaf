from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(*, num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    return model
