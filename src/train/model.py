from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class ModelBundle:
    model: nn.Module
    loss_fn: Callable


def build_model(model_type: str, feature_dim: int, hidden_dim: int | None, task: str, hidden_layers: list[int] | None = None) -> ModelBundle:
    layers: list[nn.Module] = []
    if model_type == "mlp":
        dims = [feature_dim] + (hidden_layers if hidden_layers else [hidden_dim or 128]) + [1]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
    else:
        layers.append(nn.Linear(feature_dim, 1))
    model = nn.Sequential(*layers)
    loss = nn.MSELoss()
    return ModelBundle(model=model, loss_fn=loss)


def select_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = [
    "ModelBundle",
    "build_model",
    "select_device",
]
