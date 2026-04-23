from __future__ import annotations

import argparse
from pathlib import Path

from dataset import DataConfig, build_dataloaders
from evaluation_artifacts import write_evaluation_artifacts
from models import MODEL_NAMES, build_model
from utils import (
    collect_predictions,
    compute_classification_metrics,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    measure_inference_time,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--data-config", choices=("color", "segmented", "grayscale"), default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--timing-runs", type=int, default=30)
    return parser.parse_args()


def build_data_config(
    *,
    args: argparse.Namespace,
    checkpoint: dict,
) -> DataConfig:
    return DataConfig(
        data_root=args.data_root or Path(checkpoint["data_root"]),
        data_config=args.data_config or checkpoint["data_config"],
        image_size=int(checkpoint["image_size"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        val_split=float(checkpoint["val_split"]),
        test_split=float(checkpoint["test_split"]),
        seed=int(checkpoint["seed"]),
    )


def main() -> None:
    args = parse_args()
    device = get_device(requested_device=args.device)
    checkpoint = load_checkpoint(checkpoint_path=args.checkpoint, device=device)
    model_name = checkpoint["model_name"]
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model in checkpoint: {model_name}")
    data_bundle = build_dataloaders(config=build_data_config(args=args, checkpoint=checkpoint))
    model = build_model(
        name=model_name,
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = data_bundle.val_loader if args.split == "val" else data_bundle.test_loader
    labels, predictions = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        description=args.split,
    )
    metrics = compute_classification_metrics(
        labels=labels,
        predictions=predictions,
        class_names=tuple(checkpoint["class_names"]),
    )
    metrics["parameter_count"] = count_parameters(model=model)
    metrics["average_inference_time_ms"] = measure_inference_time(
        model=model,
        device=device,
        image_size=int(checkpoint["image_size"]),
        runs=args.timing_runs,
    )
    output_dir = ensure_dir(path=args.output_dir or args.checkpoint.parent)
    save_json(data=metrics, output_path=output_dir / f"{args.split}_metrics.json")
    write_evaluation_artifacts(
        metrics=metrics,
        class_names=tuple(checkpoint["class_names"]),
        output_dir=output_dir,
        prefix=args.split,
    )
    print(
        f"{args.split} accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1_score']:.4f} "
        f"saved_to={output_dir}"
    )


if __name__ == "__main__":
    main()
