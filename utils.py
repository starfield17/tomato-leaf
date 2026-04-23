from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

WARMUP_RUNS = 10


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_seed(*, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(*, requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested_device)


def ensure_dir(*, path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(*, model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(*, data: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def save_checkpoint(*, checkpoint: dict[str, Any], output_path: Path) -> None:
    torch.save(checkpoint, output_path)


def load_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def collect_predictions(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[list[int], list[int]]:
    labels: list[int] = []
    predictions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for inputs, targets in tqdm(loader, desc=description, leave=False):
            logits = model(inputs.to(device))
            batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(batch_predictions)
            labels.extend(targets.tolist())
    return labels, predictions


def compute_classification_metrics(
    *,
    labels: list[int],
    predictions: list[int],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    report = classification_report(
        labels,
        predictions,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions)
    per_class = {
        class_name: {
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "f1_score": float(report[class_name]["f1-score"]),
            "support": int(report[class_name]["support"]),
        }
        for class_name in class_names
    }
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1_score": float(report["macro avg"]["f1-score"]),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def plot_training_curves(*, history: dict[str, list[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="train_accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    *,
    confusion: list[list[int]],
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    matrix = np.array(confusion)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def measure_inference_time(
    *,
    model: nn.Module,
    device: torch.device,
    image_size: int,
    runs: int,
) -> float:
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed * 1000 / runs
