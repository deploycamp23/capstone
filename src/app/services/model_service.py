from __future__ import annotations
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class RWLock:
    def __init__(self) -> None:
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self) -> None:
        with self._read_ready:
            self._readers += 1

    def release_read(self) -> None:
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self) -> None:
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self) -> None:
        self._read_ready.release()


class ModelService:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._lock = RWLock()
        self._model: Optional[Any] = None
        self._model_path: Optional[Path] = None
        self._loaded_at: Optional[float] = None

    def load_from_config(self) -> None:
        cfg = self._read_config()
        path = Path(cfg.get("model", {}).get("path", "model/path_to_model"))
        new_model = self._load_model(path)
        self._lock.acquire_write()
        try:
            self._model = new_model
            self._model_path = path
            self._loaded_at = time.time()
            print(f"[model] loaded: path={path} at={self._loaded_at}")
        finally:
            self._lock.release_write()

    def predict(self, payload: Dict[str, str]) -> str:
        self._lock.acquire_read()
        try:
            if self._model is None:
                self.load_from_config()
            # placeholder prediction: echo concatenated values length
            text = " ".join(payload.values())
            if hasattr(self._model, "predict"):
                return str(self._model.predict(text))
            return f"len={len(text)}"
        finally:
            self._lock.release_read()

    def info(self) -> Dict[str, Any]:
        return {
            "model_path": str(self._model_path) if self._model_path else None,
            "loaded_at": self._loaded_at,
        }

    def _read_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {"model": {"path": "model/path_to_model"}}
        with self.config_path.open("r") as f:
            return yaml.safe_load(f) or {}

    def _load_model(self, path: Path) -> Any:
        if torch is None:
            class Mock:
                def predict(self, text: str) -> str:  # type: ignore
                    return f"mock:{len(text)}"
            return Mock()
        try:
            return torch.jit.load(str(path))
        except Exception:
            class Fallback:
                def predict(self, text: str) -> str:  # type: ignore
                    return f"fallback:{len(text)}"
            return Fallback()
