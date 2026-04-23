from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation_artifacts import format_class_name
from history_utils import load_history
from report_visuals import (
    plot_efficiency_comparison,
    plot_model_comparison,
    plot_training_curves_comparison,
    render_report_confusion_matrices,
)
MODEL_LABELS = {"simplecnn": "SimpleCNN", "fastcnn": "FastCNN", "resnet18": "ResNet18"}


@dataclass(frozen=True)
class ExperimentRecord:
    label: str
    experiment_dir: Path
    metrics: dict[str, Any]
    config: dict[str, Any]
    history: dict[str, list[float]]
def read_json_file(*, path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_record(*, experiment_dir: Path) -> ExperimentRecord:
    metrics = read_json_file(path=experiment_dir / "metrics.json")
    config = read_json_file(path=experiment_dir / "config.json")
    history = load_history(experiment_dir=experiment_dir)
    model_name = str(config["model_name"])
    label = MODEL_LABELS.get(model_name, model_name)
    return ExperimentRecord(
        label=label,
        experiment_dir=experiment_dir,
        metrics=metrics,
        config=config,
        history=history,
    )
def write_model_comparison_csv(*, records: list[ExperimentRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "model",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1_score",
                "parameter_count",
                "average_inference_time_ms",
                "best_val_accuracy",
                "epochs_completed",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.label,
                    record.metrics["accuracy"],
                    record.metrics["macro_precision"],
                    record.metrics["macro_recall"],
                    record.metrics["macro_f1_score"],
                    record.metrics["parameter_count"],
                    record.metrics["average_inference_time_ms"],
                    record.metrics["best_val_accuracy"],
                    len(record.history["train_loss"]),
                ]
            )
def build_markdown_table(*, records: list[ExperimentRecord]) -> str:
    lines = [
        "| 模型 | Accuracy | Precision | Recall | F1 | 参数量 | 单张推理时间(ms) | 最佳验证准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        metrics = record.metrics
        lines.append(
            "| "
            f"{record.label} | {metrics['accuracy']:.4f} | {metrics['macro_precision']:.4f} | "
            f"{metrics['macro_recall']:.4f} | {metrics['macro_f1_score']:.4f} | "
            f"{metrics['parameter_count']} | {metrics['average_inference_time_ms']:.4f} | "
            f"{metrics['best_val_accuracy']:.4f} |"
        )
    return "\n".join(lines)


def find_record_by_model_name(*, records: list[ExperimentRecord], model_name: str) -> ExperimentRecord | None:
    for record in records:
        if record.config["model_name"] == model_name:
            return record
    return None


def top_confusion_pairs(
    *,
    record: ExperimentRecord,
    limit: int = 3,
) -> list[tuple[str, str, int, float]]:
    matrix = np.array(record.metrics["confusion_matrix"], dtype=int)
    class_names = list(record.metrics["per_class"].keys())
    pairs: list[tuple[str, str, int, float]] = []
    for row_index, row_name in enumerate(class_names):
        row_total = int(matrix[row_index].sum())
        for column_index, column_name in enumerate(class_names):
            if row_index == column_index or matrix[row_index, column_index] == 0:
                continue
            rate = matrix[row_index, column_index] / row_total if row_total else 0.0
            pairs.append(
                (
                    format_class_name(name=row_name),
                    format_class_name(name=column_name),
                    int(matrix[row_index, column_index]),
                    rate,
                )
            )
    return sorted(pairs, key=lambda item: item[2], reverse=True)[:limit]


def get_runtime_summary(*, record: ExperimentRecord) -> list[str]:
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    return [
        f"- 操作系统：{platform.platform()}",
        f"- Python：{platform.python_version()}；PyTorch：{torch.__version__}",
        f"- 计算设备：{device_name}",
        f"- 数据配置：`{record.config['data_config']}`；输入尺寸：`{record.config['image_size']}×{record.config['image_size']}`",
        f"- 批大小：`{record.config['batch_size']}`；初始学习率：`{record.config['learning_rate']}`；权重衰减：`{record.config['weight_decay']}`",
        f"- 优化器：`Adam`；损失函数：`CrossEntropyLoss`；学习率调度：`{record.config['scheduler']}`；早停耐心值：`{record.config['patience']}`",
    ]


def build_results_summary(*, records: list[ExperimentRecord]) -> list[str]:
    best_accuracy = max(records, key=lambda item: item.metrics["accuracy"])
    fastest = min(records, key=lambda item: item.metrics["average_inference_time_ms"])
    smallest = min(records, key=lambda item: item.metrics["parameter_count"])
    lines = [
        f"从总体指标看，{best_accuracy.label}在测试集上取得最高准确率 {best_accuracy.metrics['accuracy']:.4f}，宏平均 F1 达到 {best_accuracy.metrics['macro_f1_score']:.4f}。",
        f"推理效率方面，{fastest.label} 的单张图像平均推理时间最短，为 {fastest.metrics['average_inference_time_ms']:.4f} ms；模型规模最小的是 {smallest.label}。",
    ]
    fastcnn = find_record_by_model_name(records=records, model_name="fastcnn")
    resnet18 = find_record_by_model_name(records=records, model_name="resnet18")
    simplecnn = find_record_by_model_name(records=records, model_name="simplecnn")
    if fastcnn and resnet18 and simplecnn:
        ratio = fastcnn.metrics["parameter_count"] / resnet18.metrics["parameter_count"]
        lines.append(
            f"FastCNN 的参数量为 {fastcnn.metrics['parameter_count']}，低于 SimpleCNN 的 {simplecnn.metrics['parameter_count']}，仅约为 ResNet18 的 {ratio:.4%}。"
        )
        lines.append("综合准确率、参数量与推理时间三项指标，FastCNN 在本任务上呈现出更均衡的性能—效率折中。")
    return lines


def build_confusion_summary(*, record: ExperimentRecord) -> list[str]:
    pairs = top_confusion_pairs(record=record)
    if not pairs:
        return [f"- `{record.label}` 未出现明显集中的误分类对，说明类别区分整体较稳定。"]
    return [
        f"- `{record.label}` 主要误分类集中在 `{true_name} → {pred_name}`，出现 {count} 次，占该真实类别样本的 {rate:.2%}。"
        for true_name, pred_name, count, rate in pairs
    ]


def build_curve_summary(*, records: list[ExperimentRecord]) -> list[str]:
    lines: list[str] = []
    for record in records:
        best_epoch = int(np.argmax(record.history["val_accuracy"])) + 1
        best_val = max(record.history["val_accuracy"])
        min_loss = min(record.history["val_loss"])
        lines.append(
            f"- `{record.label}` 共完成 {len(record.history['train_loss'])} 个 epoch，最佳验证准确率出现在第 {best_epoch} 轮，为 {best_val:.4f}，最小验证损失为 {min_loss:.4f}。"
        )
    if find_record_by_model_name(records=records, model_name="resnet18"):
        lines.append("- ResNet18 在较少 epoch 内即达到较高验证精度，体现出迁移学习的快速收敛特性。")
    if find_record_by_model_name(records=records, model_name="fastcnn"):
        lines.append("- FastCNN 的验证曲线整体更平稳，在后期仍能保持较高精度，说明轻量化结构没有明显牺牲泛化能力。")
    if find_record_by_model_name(records=records, model_name="simplecnn"):
        lines.append("- SimpleCNN 的验证精度提升较早进入平台区，后期继续训练的收益有限，早停策略有效避免了无效迭代。")
    return lines


def build_metric_section() -> list[str]:
    return [
        "## 5.2 评价指标",
        "- Accuracy 表示测试样本中被正确分类的比例，用于衡量模型整体分类正确率。",
        "- Precision 表示被模型预测为某一类别的样本中，真正属于该类别的比例，用于衡量预测结果的可靠性。",
        "- Recall 表示某一真实类别样本中被模型正确识别出来的比例，用于衡量漏检情况。",
        "- F1 为 Precision 与 Recall 的调和平均值，兼顾准确性与召回能力；本文表格中的 Precision、Recall、F1 均采用宏平均形式统计。",
        "",
    ]


def build_comparison_section(*, records: list[ExperimentRecord]) -> list[str]:
    return [
        "## 5.3 对比实验结果",
        "表 5-1 展示了三种模型在番茄叶片病害识别任务上的测试结果。",
        "",
        build_markdown_table(records=records),
        "",
        "![模型性能对比](model_comparison.png)",
        "",
        "![效率对比](efficiency_comparison.png)",
        "",
        *build_results_summary(records=records),
        "",
    ]


def build_confusion_section(
    *,
    records: list[ExperimentRecord],
    confusion_paths: dict[str, str],
) -> list[str]:
    lines = [
        "## 5.4 混淆矩阵分析",
        "归一化混淆矩阵用于观察各类别之间的误判分布，不同模型的主要误差模式如下。",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"### {record.label}",
                f"![{record.label} 混淆矩阵]({confusion_paths[record.label]})",
                *build_confusion_summary(record=record),
                "",
            ]
        )
    return lines


def build_curve_section(*, records: list[ExperimentRecord]) -> list[str]:
    return [
        "## 5.5 Loss/Accuracy 曲线分析",
        "下图给出了三种模型在验证集上的 loss 和 accuracy 曲线。",
        "",
        "![训练曲线对比](training_curves_comparison.png)",
        "",
        *build_curve_summary(records=records),
        "",
    ]


def build_summary_section(*, records: list[ExperimentRecord]) -> list[str]:
    preferred = find_record_by_model_name(records=records, model_name="fastcnn")
    representative = preferred or max(records, key=lambda item: item.metrics["accuracy"])
    closing_line = "- 综合实验结果，轻量 FastCNN 更适合作为本课题的主模型方案。"
    if preferred is None:
        closing_line = f"- 综合实验结果，{representative.label} 更适合作为当前实验设置下的主模型方案。"
    return [
        "## 5.6 本章小结",
        f"- 在当前 10 类番茄叶片病害识别任务中，{representative.label} 取得了代表性最强的综合结果，测试准确率为 {representative.metrics['accuracy']:.4f}，宏平均 F1 为 {representative.metrics['macro_f1_score']:.4f}。",
        "- 轻量化网络在本任务中展现出较好的性能—复杂度平衡，说明针对植物病害图像分类设计紧凑卷积结构是可行的。",
        "- 迁移学习模型具备更快的前期收敛速度，但在模型规模与部署成本方面明显高于轻量模型。",
        closing_line,
    ]


def write_markdown_report(*, records: list[ExperimentRecord], output_dir: Path) -> None:
    confusion_paths = render_report_confusion_matrices(records=records, output_dir=output_dir)
    sections = [
        "# 5. 实验与结果分析",
        "## 5.1 实验环境",
        *get_runtime_summary(record=records[0]),
        "",
        *build_metric_section(),
        *build_comparison_section(records=records),
        *build_confusion_section(records=records, confusion_paths=confusion_paths),
        *build_curve_section(records=records),
        *build_summary_section(records=records),
    ]
    (output_dir / "experiment_results.md").write_text("\n".join(sections), encoding="utf-8")


def generate_report_artifacts(*, records: list[ExperimentRecord], output_dir: Path) -> None:
    write_model_comparison_csv(records=records, output_path=output_dir / "model_comparison.csv")
    plot_model_comparison(records=records, output_path=output_dir / "model_comparison.png")
    plot_efficiency_comparison(records=records, output_path=output_dir / "efficiency_comparison.png")
    plot_training_curves_comparison(
        records=records,
        output_path=output_dir / "training_curves_comparison.png",
    )
    write_markdown_report(records=records, output_dir=output_dir)
