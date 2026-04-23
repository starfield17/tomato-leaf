from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from dataset import build_transforms
from models import MODEL_NAMES, build_model
from utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image prediction.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def prepare_image(*, image_path: Path, image_size: int) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, eval_transform = build_transforms(image_size=image_size, augment=False)
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image)
    return tensor.unsqueeze(0)


def main() -> None:
    args = parse_args()
    device = get_device(requested_device=args.device)
    checkpoint = load_checkpoint(checkpoint_path=args.checkpoint, device=device)
    model_name = checkpoint["model_name"]
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model in checkpoint: {model_name}")
    model = build_model(
        name=model_name,
        num_classes=int(checkpoint["num_classes"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    inputs = prepare_image(
        image_path=args.image_path,
        image_size=int(checkpoint["image_size"]),
    ).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1)
    top_k = min(args.top_k, probabilities.shape[1])
    scores, indices = torch.topk(probabilities, k=top_k, dim=1)
    class_names = checkpoint["class_names"]
    print(f"image={args.image_path}")
    for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
        class_name = class_names[int(index)]
        print(f"top_{rank}: class={class_name} probability={float(score):.4f}")


if __name__ == "__main__":
    main()
