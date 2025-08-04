from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import polars as pl  # optional
except Exception:  # pragma: no cover
    pl = None  # type: ignore


@dataclass
class DatasetTensors:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor


def set_seed(seed: int) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)
    _ = rng  # keep for potential future use


def make_synthetic(
    task: str,
    train_size: int,
    val_size: int,
    feature_dim: int,
    seed: int = 42,
) -> DatasetTensors:
    set_seed(seed)
    n = train_size + val_size
    x = np.random.randn(n, feature_dim).astype(np.float32)

    w = np.random.randn(feature_dim, 1).astype(np.float32)
    b = np.random.randn(1).astype(np.float32)
    lin = x @ w + b  # (n,1)

    if task == "classification":
        logits = lin.squeeze(-1)
        probs = 1 / (1 + np.exp(-logits))
        y = (probs > 0.5).astype(np.int64)
    else:
        y = lin.squeeze(-1) + 0.1 * np.random.randn(n).astype(np.float32)

    x_train = torch.from_numpy(x[:train_size])
    x_val = torch.from_numpy(x[train_size:])

    if task == "classification":
        y_train = torch.from_numpy(y[:train_size])
        y_val = torch.from_numpy(y[train_size:])
    else:
        y_train = torch.from_numpy(y[:train_size]).unsqueeze(-1)
        y_val = torch.from_numpy(y[train_size:]).unsqueeze(-1)

    return DatasetTensors(x_train, y_train, x_val, y_val)


def export_preview(
    ds: DatasetTensors,
    out_path: str | Path,
    limit: int = 100,
) -> Optional[Path]:
    if pl is None:
        return None
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def to_rows(x: torch.Tensor, y: torch.Tensor, split: str) -> list[dict]:
        arr_x = x[:limit].detach().cpu().numpy()
        arr_y = y[:limit].detach().cpu().numpy()
        rows = []
        for i in range(arr_x.shape[0]):
            row = {f"x{j}": float(arr_x[i, j]) for j in range(arr_x.shape[1])}
            target = arr_y[i]
            if target.ndim == 0:
                row["y"] = int(target)
            else:
                row["y"] = float(target.squeeze())
            row["split"] = split
            rows.append(row)
        return rows

    rows = []
    rows += to_rows(ds.x_train, ds.y_train, "train")
    rows += to_rows(ds.x_val, ds.y_val, "val")

    df = pl.DataFrame(rows)
    df.write_parquet(str(p))
    return p


__all__ = [
    "DatasetTensors",
    "set_seed",
    "make_synthetic",
    "export_preview",
]
