"""Atomic, resumable checkpoints (survive disconnects / kills mid-write)."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])


def save(path: str | Path, state: dict[str, Any]) -> None:
    """Write to a temp file then atomically replace, so a crash never leaves a half-written checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def clear(path: str | Path | None) -> None:
    if path is not None:
        Path(path).unlink(missing_ok=True)


class JsonProgress:
    """Incremental results store for scripts: get/put by key, flushed atomically after every put."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict = json.loads(self.path.read_text()) if self.path.exists() else {}

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str):
        return self.data[key]

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)
