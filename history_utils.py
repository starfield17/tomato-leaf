from __future__ import annotations

import json
import re
from pathlib import Path

HISTORY_KEYS = ("train_loss", "val_loss", "train_accuracy", "val_accuracy")
EPOCH_PATTERN = re.compile(
    r"Epoch\s+\d+\s+\|\s+"
    r"train_loss=(?P<train_loss>\d+\.\d+)\s+"
    r"train_acc=(?P<train_accuracy>\d+\.\d+)\s+\|\s+"
    r"val_loss=(?P<val_loss>\d+\.\d+)\s+"
    r"val_acc=(?P<val_accuracy>\d+\.\d+)"
)


def save_history(*, history: dict[str, list[float]], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_history(*, experiment_dir: Path) -> dict[str, list[float]]:
    history_path = experiment_dir / "history.json"
    if history_path.exists():
        return validate_history(data=json.loads(history_path.read_text(encoding="utf-8")))
    log_path = experiment_dir / "train.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        return parse_history_from_log(log_text=log_text)
    raise FileNotFoundError(
        f"Missing history sources in {experiment_dir}. Expected history.json or train.log."
    )


def validate_history(*, data: dict[str, object]) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {}
    for key in HISTORY_KEYS:
        values = data.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"History key '{key}' is missing or empty.")
        history[key] = [float(value) for value in values]
    lengths = {len(values) for values in history.values()}
    if len(lengths) != 1:
        raise ValueError("History lists have inconsistent lengths.")
    return history


def parse_history_from_log(*, log_text: str) -> dict[str, list[float]]:
    history = {key: [] for key in HISTORY_KEYS}
    for match in EPOCH_PATTERN.finditer(log_text):
        for key in HISTORY_KEYS:
            history[key].append(float(match.group(key)))
    if not history["train_loss"]:
        raise ValueError("Failed to parse epoch history from train.log.")
    return history
