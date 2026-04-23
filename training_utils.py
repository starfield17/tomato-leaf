from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from evaluation_artifacts import write_evaluation_artifacts
from history_utils import save_history
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from dataset import DatasetBundle
from utils import (
    collect_predictions,
    compute_classification_metrics,
    count_parameters,
    load_checkpoint,
    measure_inference_time,
    plot_training_curves,
    save_json,
)


def build_scheduler(*, optimizer: Adam, scheduler_name: str, epochs: int, step_size: int, gamma: float) -> Any | None:
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    if scheduler_name == "step":
        return StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    return None


def run_training_epoch(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
    epoch_label: str,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.train()
    for inputs, targets in tqdm(loader, desc=epoch_label, leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)
    return total_loss / total_samples, total_correct / total_samples


def run_eval_epoch(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_label: str,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    with torch.inference_mode():
        for inputs, targets in tqdm(loader, desc=epoch_label, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / total_samples, total_correct / total_samples


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Any | None,
    config: Any,
    bundle: DatasetBundle,
    epoch: int,
    best_val_accuracy: float,
) -> dict[str, Any]:
    idx_to_class = {index: name for index, name in enumerate(bundle.class_names)}
    return {
        "epoch": epoch,
        "model_name": config.model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "class_names": list(bundle.class_names),
        "class_to_idx": bundle.class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": len(bundle.class_names),
        "data_config": config.data_config,
        "data_root": str(config.data_root),
        "image_size": config.image_size,
        "val_split": config.val_split,
        "test_split": config.test_split,
        "seed": config.seed,
    }


def finalize_training(
    *,
    config: Any,
    model: nn.Module,
    device: torch.device,
    history: dict[str, list[float]],
    bundle: DatasetBundle,
    experiment_dir: Path,
) -> None:
    checkpoint = load_checkpoint(
        checkpoint_path=experiment_dir / "best_model.pth",
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    labels, predictions = collect_predictions(
        model=model,
        loader=bundle.test_loader,
        device=device,
        description="test",
    )
    metrics = compute_classification_metrics(
        labels=labels,
        predictions=predictions,
        class_names=bundle.class_names,
    )
    metrics["parameter_count"] = count_parameters(model=model)
    metrics["average_inference_time_ms"] = measure_inference_time(
        model=model,
        device=device,
        image_size=config.image_size,
        runs=config.timing_runs,
    )
    metrics["best_val_accuracy"] = checkpoint["best_val_accuracy"]
    save_json(data=metrics, output_path=experiment_dir / "metrics.json")
    save_json(
        data={**asdict(config), "class_names": list(bundle.class_names)},
        output_path=experiment_dir / "config.json",
    )
    save_history(history=history, output_path=experiment_dir / "history.json")
    plot_training_curves(history=history, output_path=experiment_dir / "training_curves.png")
    write_evaluation_artifacts(
        metrics=metrics,
        class_names=bundle.class_names,
        output_dir=experiment_dir,
    )
