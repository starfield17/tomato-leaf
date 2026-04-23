from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam

from dataset import DataConfig, DatasetBundle, build_dataloaders
from models import MODEL_NAMES, build_model
from training_utils import (
    build_scheduler,
    checkpoint_payload,
    finalize_training,
    run_eval_epoch,
    run_training_epoch,
)
from utils import (
    ensure_dir,
    get_device,
    parse_bool,
    save_checkpoint,
    set_seed,
)


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    data_config: str
    model_name: str
    experiment_name: str
    output_dir: Path
    image_size: int
    batch_size: int
    num_workers: int
    augment: bool
    val_split: float
    test_split: float
    epochs: int
    learning_rate: float
    weight_decay: float
    scheduler: str
    step_size: int
    gamma: float
    patience: int
    seed: int
    pretrained: bool
    device: str
    timing_runs: int


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train tomato leaf disease models.")
    parser.add_argument("--data-root", type=Path, default=Path("plantvillagedataset"))
    parser.add_argument("--data-config", choices=("color", "segmented", "grayscale"), default="color")
    parser.add_argument("--model", dest="model_name", choices=MODEL_NAMES, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--augment", type=parse_bool, default=True)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("cosine", "step", "none"), default="cosine")
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", type=parse_bool, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--timing-runs", type=int, default=30)
    namespace = parser.parse_args()
    return TrainingConfig(**vars(namespace))


def build_data_config(*, config: TrainingConfig) -> DataConfig:
    return DataConfig(
        data_root=config.data_root,
        data_config=config.data_config,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment=config.augment,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed,
    )


def main() -> None:
    config = parse_args()
    set_seed(seed=config.seed)
    device = get_device(requested_device=config.device)
    experiment_dir = ensure_dir(path=config.output_dir / config.experiment_name)
    bundle = build_dataloaders(config=build_data_config(config=config))
    model = build_model(
        name=config.model_name,
        num_classes=len(bundle.class_names),
        pretrained=config.pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=config.scheduler,
        epochs=config.epochs,
        step_size=config.step_size,
        gamma=config.gamma,
    )
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_accuracy = 0.0
    patience_counter = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_training_epoch(
            model=model,
            loader=bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_label=f"train {epoch}/{config.epochs}",
        )
        val_loss, val_accuracy = run_eval_epoch(
            model=model,
            loader=bundle.val_loader,
            criterion=criterion,
            device=device,
            epoch_label=f"val {epoch}/{config.epochs}",
        )
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            save_checkpoint(
                checkpoint=checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    bundle=bundle,
                    epoch=epoch,
                    best_val_accuracy=best_val_accuracy,
                ),
                output_path=experiment_dir / "best_model.pth",
            )
            continue
        patience_counter += 1
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break
    finalize_training(
        config=config,
        model=model,
        device=device,
        history=history,
        bundle=bundle,
        experiment_dir=experiment_dir,
    )
    print(f"Finished training. Outputs saved to {experiment_dir}")


if __name__ == "__main__":
    main()
