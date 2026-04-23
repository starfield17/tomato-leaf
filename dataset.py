from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_CONFIGS = ("color", "segmented", "grayscale")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    data_config: str
    image_size: int
    batch_size: int
    num_workers: int
    augment: bool
    val_split: float
    test_split: float
    seed: int


@dataclass(frozen=True)
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: tuple[str, ...]
    class_to_idx: dict[str, int]
    data_dir: Path


class ImageFolderSubset(Dataset):
    def __init__(
        self,
        *,
        samples: Sequence[tuple[str, int]],
        transform: transforms.Compose,
    ) -> None:
        self.samples = tuple(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image_path, label = self.samples[index]
        image = self._load_image(image_path=image_path)
        transformed = self.transform(image)
        return transformed, label

    @staticmethod
    def _load_image(*, image_path: str) -> Image.Image:
        return default_loader(image_path)


def build_dataloaders(*, config: DataConfig) -> DatasetBundle:
    data_dir = _resolve_data_dir(config=config)
    train_transform, eval_transform = build_transforms(
        image_size=config.image_size,
        augment=config.augment,
    )
    if _has_presplit_layout(data_dir=data_dir):
        datasets = _build_presplit_datasets(
            data_dir=data_dir,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    else:
        datasets = _build_split_from_single_root(
            data_dir=data_dir,
            train_transform=train_transform,
            eval_transform=eval_transform,
            val_split=config.val_split,
            test_split=config.test_split,
            seed=config.seed,
        )
    class_names, class_to_idx = _read_class_metadata(data_dir=data_dir)
    train_loader = _build_loader(
        dataset=datasets["train"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val_loader = _build_loader(
        dataset=datasets["val"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    test_loader = _build_loader(
        dataset=datasets["test"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        class_to_idx=class_to_idx,
        data_dir=data_dir,
    )


def build_transforms(
    *,
    image_size: int,
    augment: bool,
) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if not augment:
        return eval_transform, eval_transform
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def _resolve_data_dir(*, config: DataConfig) -> Path:
    if config.data_config not in VALID_CONFIGS:
        raise ValueError(f"Unsupported data config: {config.data_config}")
    data_dir = config.data_root / config.data_config
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    return data_dir


def _has_presplit_layout(*, data_dir: Path) -> bool:
    split_paths = [data_dir / split_name for split_name in SPLIT_NAMES]
    if all(path.is_dir() for path in split_paths):
        return True
    if any(path.exists() for path in split_paths):
        raise ValueError(
            f"Incomplete split layout under {data_dir}. Expected train/val/test."
        )
    return False


def _build_presplit_datasets(
    *,
    data_dir: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> dict[str, Dataset]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    return {
        "train": ImageFolder(root=str(train_dir), transform=train_transform),
        "val": ImageFolder(root=str(val_dir), transform=eval_transform),
        "test": ImageFolder(root=str(test_dir), transform=eval_transform),
    }


def _build_split_from_single_root(
    *,
    data_dir: Path,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    val_split: float,
    test_split: float,
    seed: int,
) -> dict[str, Dataset]:
    _validate_split_ratios(val_split=val_split, test_split=test_split)
    base_dataset = ImageFolder(root=str(data_dir))
    indices = np.arange(len(base_dataset.targets))
    train_indices, val_indices, test_indices = _stratified_indices(
        indices=indices,
        targets=np.array(base_dataset.targets),
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    train_samples = [base_dataset.samples[index] for index in train_indices]
    val_samples = [base_dataset.samples[index] for index in val_indices]
    test_samples = [base_dataset.samples[index] for index in test_indices]
    return {
        "train": ImageFolderSubset(samples=train_samples, transform=train_transform),
        "val": ImageFolderSubset(samples=val_samples, transform=eval_transform),
        "test": ImageFolderSubset(samples=test_samples, transform=eval_transform),
    }


def _stratified_indices(
    *,
    indices: np.ndarray,
    targets: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    holdout_size = val_split + test_split
    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=holdout_size,
        stratify=targets,
        random_state=seed,
    )
    holdout_targets = targets[holdout_indices]
    test_ratio = test_split / holdout_size
    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=test_ratio,
        stratify=holdout_targets,
        random_state=seed,
    )
    return train_indices, val_indices, test_indices


def _validate_split_ratios(*, val_split: float, test_split: float) -> None:
    if val_split <= 0 or test_split <= 0:
        raise ValueError("val_split and test_split must be greater than zero.")
    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split must be less than 1.")


def _read_class_metadata(*, data_dir: Path) -> tuple[tuple[str, ...], dict[str, int]]:
    metadata_root = data_dir / "train" if (data_dir / "train").is_dir() else data_dir
    class_dataset = ImageFolder(root=str(metadata_root))
    class_names = tuple(class_dataset.classes)
    class_to_idx = dict(class_dataset.class_to_idx)
    return class_names, class_to_idx


def _build_loader(
    *,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
