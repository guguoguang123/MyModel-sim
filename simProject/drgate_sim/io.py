from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_graphs(path: str | Path) -> Dict[str, dict]:
    data = np.load(path, allow_pickle=True)
    return data["circuits"].item()


def save_behavior_labels(path: str | Path, labels: Dict[str, dict], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=labels, meta=meta)


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

