from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evaluation_artifacts import plot_confusion_matrix_chart

METRIC_COLUMNS = ("accuracy", "macro_precision", "macro_recall", "macro_f1_score")


def plot_model_comparison(*, records: list[Any], output_path: Path) -> None:
    labels = [record.label for record in records]
    positions = np.arange(len(labels))
    width = 0.2
    figure, axis = plt.subplots(figsize=(12, 6))
    for offset, metric_name in enumerate(METRIC_COLUMNS):
        values = [record.metrics[metric_name] * 100 for record in records]
        label = metric_name.replace("_", " ").title()
        axis.bar(positions + (offset - 1.5) * width, values, width=width, label=label)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0.0, 100.0)
    axis.set_title("Model Comparison on Test Metrics")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_efficiency_comparison(*, records: list[Any], output_path: Path) -> None:
    labels = [record.label for record in records]
    params = [record.metrics["parameter_count"] for record in records]
    inference = [record.metrics["average_inference_time_ms"] for record in records]
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].bar(labels, params, color="#4C72B0")
    axes[0].set_yscale("log")
    axes[0].set_title("Parameter Count")
    axes[0].set_ylabel("Parameters (log scale)")
    axes[1].bar(labels, inference, color="#55A868")
    axes[1].set_title("Single-Image Inference Time")
    axes[1].set_ylabel("Time (ms)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_training_curves_comparison(*, records: list[Any], output_path: Path) -> None:
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for record in records:
        epochs = range(1, len(record.history["val_loss"]) + 1)
        axes[0].plot(epochs, record.history["val_loss"], label=record.label)
        axes[1].plot(epochs, record.history["val_accuracy"], label=record.label)
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def render_report_confusion_matrices(*, records: list[Any], output_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for record in records:
        filename = f"{record.config['model_name']}_normalized_confusion_matrix.png"
        plot_confusion_matrix_chart(
            confusion=record.metrics["confusion_matrix"],
            class_names=tuple(record.metrics["per_class"].keys()),
            output_path=output_dir / filename,
            normalize=True,
            title=f"{record.label} Normalized Confusion Matrix",
        )
        paths[record.label] = filename
    return paths
