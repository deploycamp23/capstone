from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


yaml = YAML(typ="rt")


@dataclass
class TrainingConfig:
    seed: int = 42
    task: str = "regression"
    model_type: str = "mlp"
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 64
    train_size: int = 2048
    val_size: int = 512
    feature_dim: int = 32
    hidden_dim: int | None = 128  # deprecated when using hidden_layers
    hidden_layers: list[int] | None = None
    output_dir: str = "models"
    device: str | None = None
    problem: str = "car_price"


def load_config(path: str | Path) -> TrainingConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.load(f) or {}
    return TrainingConfig(**data)


def save_config(cfg: TrainingConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(asdict(cfg), f)


def apply_overrides(cfg: TrainingConfig, overrides: dict[str, Any]) -> TrainingConfig:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown config key: {k}")
        current = getattr(cfg, k)
        new_val = _coerce_type(v, type(current))
        setattr(cfg, k, new_val)
    return cfg


def _coerce_type(value: Any, target_type: type) -> Any:
    if target_type is None:
        return value
    try:
        if target_type is bool:
            if isinstance(value, str):
                lv = value.lower()
                if lv in {"true", "1", "yes", "y"}:  # noqa: PLR2004
                    return True
                if lv in {"false", "0", "no", "n"}:  # noqa: PLR2004
                    return False
            return bool(value)
        return target_type(value)
    except Exception:
        return value


__all__ = [
    "TrainingConfig",
    "load_config",
    "save_config",
    "apply_overrides",
]
