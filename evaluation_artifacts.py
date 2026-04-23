from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

MATRIX_FIGSIZE = (12, 10)
BAR_FIGSIZE = (14, 6)


def format_class_name(*, name: str) -> str:
    cleaned = name.replace("Tomato___", "")
    return cleaned.replace("_", " ")


def build_display_names(*, class_names: tuple[str, ...]) -> list[str]:
    return [format_class_name(name=name) for name in class_names]


def build_artifact_path(*, output_dir: Path, prefix: str, filename: str) -> Path:
    target_name = f"{prefix}_{filename}" if prefix else filename
    return output_dir / target_name


def build_per_class_rows(*, metrics: dict[str, Any]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    confusion = np.array(metrics["confusion_matrix"], dtype=int)
    for index, (class_name, values) in enumerate(metrics["per_class"].items()):
        support = int(values["support"])
        correct = int(confusion[index, index])
        rows.append(
            {
                "class_name": format_class_name(name=class_name),
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1_score"]),
                "support": support,
                "errors": support - correct,
            }
        )
    return rows


def save_per_class_metrics_csv(
    *,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    rows = build_per_class_rows(metrics=metrics)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("class_name", "precision", "recall", "f1_score", "support", "errors"),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix_chart(
    *,
    confusion: list[list[int]],
    class_names: tuple[str, ...],
    output_path: Path,
    normalize: bool,
    title: str,
) -> None:
    matrix = np.array(confusion, dtype=float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    labels = build_display_names(class_names=class_names)
    figure, axis = plt.subplots(figsize=MATRIX_FIGSIZE)
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0 if normalize else None, vmax=1.0 if normalize else None)
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    threshold = float(matrix.max()) * 0.6 if matrix.size else 0.0
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            text = f"{value:.2f}" if normalize else str(int(value))
            color = "white" if value > threshold and threshold > 0 else "black"
            axis.text(column_index, row_index, text, ha="center", va="center", color=color, fontsize=8)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_metrics(
    *,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    rows = build_per_class_rows(metrics=metrics)
    positions = np.arange(len(rows))
    width = 0.25
    figure, axis = plt.subplots(figsize=BAR_FIGSIZE)
    axis.bar(positions - width, [row["precision"] for row in rows], width=width, label="Precision")
    axis.bar(positions, [row["recall"] for row in rows], width=width, label="Recall")
    axis.bar(positions + width, [row["f1_score"] for row in rows], width=width, label="F1")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(positions)
    axis.set_xticklabels([row["class_name"] for row in rows], rotation=45, ha="right")
    axis.set_ylabel("Score")
    axis.set_title("Per-Class Precision / Recall / F1")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_class_support_errors(
    *,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    rows = build_per_class_rows(metrics=metrics)
    positions = np.arange(len(rows))
    width = 0.38
    figure, axis = plt.subplots(figsize=BAR_FIGSIZE)
    axis.bar(positions - width / 2, [row["support"] for row in rows], width=width, label="Support")
    axis.bar(positions + width / 2, [row["errors"] for row in rows], width=width, label="Errors")
    axis.set_xticks(positions)
    axis.set_xticklabels([row["class_name"] for row in rows], rotation=45, ha="right")
    axis.set_ylabel("Images")
    axis.set_title("Class Support and Misclassification Counts")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_evaluation_artifacts(
    *,
    metrics: dict[str, Any],
    class_names: tuple[str, ...],
    output_dir: Path,
    prefix: str = "",
) -> None:
    plot_confusion_matrix_chart(
        confusion=metrics["confusion_matrix"],
        class_names=class_names,
        output_path=build_artifact_path(output_dir=output_dir, prefix=prefix, filename="confusion_matrix.png"),
        normalize=False,
        title="Confusion Matrix",
    )
    plot_confusion_matrix_chart(
        confusion=metrics["confusion_matrix"],
        class_names=class_names,
        output_path=build_artifact_path(
            output_dir=output_dir,
            prefix=prefix,
            filename="normalized_confusion_matrix.png",
        ),
        normalize=True,
        title="Normalized Confusion Matrix",
    )
    save_per_class_metrics_csv(
        metrics=metrics,
        output_path=build_artifact_path(output_dir=output_dir, prefix=prefix, filename="per_class_metrics.csv"),
    )
    plot_per_class_metrics(
        metrics=metrics,
        output_path=build_artifact_path(output_dir=output_dir, prefix=prefix, filename="per_class_metrics.png"),
    )
    plot_class_support_errors(
        metrics=metrics,
        output_path=build_artifact_path(output_dir=output_dir, prefix=prefix, filename="class_support_errors.png"),
    )
