from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train.config import TrainingConfig
from train.data import DatasetTensors, make_synthetic
from train.model import build_model, select_device


def make_loaders(ds: DatasetTensors, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(ds.x_train, ds.y_train)
    val_ds = TensorDataset(ds.x_val, ds.y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train(cfg: TrainingConfig) -> Dict:
    device = select_device(cfg.device)
    ds = make_synthetic(
        task=cfg.task,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        feature_dim=cfg.feature_dim,
        seed=cfg.seed,
    )
    train_loader, val_loader = make_loaders(ds, cfg.batch_size)

    bundle = build_model(cfg.model_type, cfg.feature_dim, cfg.hidden_dim, cfg.task, cfg.hidden_layers)
    model, loss_fn = bundle.model.to(device), bundle.loss_fn
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "val_loss": []}
    start = perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optim.step()
            running += float(loss.item()) * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_running += float(loss.item()) * xb.size(0)
        val_loss = val_running / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    elapsed = perf_counter() - start

    metrics = {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": min(history["val_loss"]),
        "epochs": cfg.epochs,
        "elapsed_sec": elapsed,
    }
    return {"model": model, "metrics": metrics, "history": history}


def save_artifacts(out_dir: str | Path, model: nn.Module, metrics: Dict, history: Dict) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    model_path = p / "model.pt"
    info_path = p / "model_info.json"
    metrics_path = p / "metrics.json"

    torch.save(model.state_dict(), model_path)

    info = {
        "model": str(model),
    }
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "history": history}, f, indent=2)


__all__ = [
    "train",
    "save_artifacts",
]
