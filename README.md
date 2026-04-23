# Tomato Leaf Disease Classification with FastCNN

## Overview

This project implements a 10-class tomato leaf disease classification pipeline in PyTorch for the topic **“基于轻量级 FastCNN 的番茄叶片病害识别研究”**.

Supported models:

- `simplecnn`: plain CNN baseline
- `fastcnn`: lightweight CNN with depthwise separable convolution
- `resnet18`: transfer-learning baseline

Supported dataset variants:

- `color`
- `segmented`
- `grayscale`

If the selected dataset directory does not already contain `train/`, `val/`, and `test/`, the code creates a deterministic stratified split with ratio `70/15/15`.

## Project Structure

```text
.
├── plantvillagedataset/
├── models/
│   ├── __init__.py
│   ├── cnn.py
│   ├── fastcnn.py
│   └── resnet18_baseline.py
├── dataset.py
├── history_utils.py
├── evaluation_artifacts.py
├── report_utils.py
├── training_utils.py
├── utils.py
├── train.py
├── evaluate.py
├── inference.py
├── report_results.py
├── requirements.txt
└── README.md
```

## Dataset Layout

The default data root is `plantvillagedataset/`, with one subdirectory per image configuration:

```text
plantvillagedataset/
├── color/
├── segmented/
└── grayscale/
```

Each configuration directory should contain one folder per class, for example:

```text
plantvillagedataset/color/Tomato___healthy/
plantvillagedataset/color/Tomato___Late_blight/
```

The current local dataset already matches this layout.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Train `FastCNN` on the default `color` dataset:

```bash
python train.py \
  --model fastcnn \
  --experiment-name fastcnn_color \
  --data-root plantvillagedataset \
  --data-config color
```

Train `SimpleCNN`:

```bash
python train.py \
  --model simplecnn \
  --experiment-name simplecnn_color \
  --data-root plantvillagedataset \
  --data-config color
```

Train `ResNet18`:

```bash
python train.py \
  --model resnet18 \
  --experiment-name resnet18_color \
  --data-root plantvillagedataset \
  --data-config color
```

Disable augmentation:

```bash
python train.py \
  --model fastcnn \
  --experiment-name fastcnn_color_noaug \
  --augment false
```

Run the background-bias comparison on segmented images:

```bash
python train.py \
  --model fastcnn \
  --experiment-name fastcnn_segmented \
  --data-config segmented
```

Useful training arguments:

- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--scheduler {cosine,step,none}`
- `--patience`
- `--device {auto,cpu,cuda}`

## Evaluation

Evaluate a saved checkpoint on the test split:

```bash
python evaluate.py \
  --checkpoint outputs/fastcnn_color/best_model.pth \
  --data-root plantvillagedataset \
  --data-config color \
  --split test
```

Evaluate on the validation split:

```bash
python evaluate.py \
  --checkpoint outputs/fastcnn_color/best_model.pth \
  --split val
```

## Inference

Run single-image prediction:

```bash
python inference.py \
  --checkpoint outputs/fastcnn_color/best_model.pth \
  --image-path "plantvillagedataset/color/Tomato___healthy/0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555.JPG"
```

## Saved Outputs

Each experiment writes to `outputs/<experiment_name>/`:

- `best_model.pth`
- `metrics.json`
- `config.json`
- `training_curves.png`
- `confusion_matrix.png`

Running `evaluate.py` also writes:

- `<split>_metrics.json`
- `<split>_confusion_matrix.png`
- `<split>_normalized_confusion_matrix.png`
- `<split>_per_class_metrics.csv`
- `<split>_per_class_metrics.png`
- `<split>_class_support_errors.png`

Each training run now also saves:

- `history.json`

## Experiment Report Generation

Generate the comparison charts and the expanded Chinese report section for **“5. 实验与结果分析”**:

```bash
python report_results.py \
  --experiments outputs/simplecnn_color_e20 outputs/fastcnn_color_e20 outputs/resnet18_color_e20 \
  --output-dir reports/experiment_results
```

This command writes:

- `reports/experiment_results/model_comparison.csv`
- `reports/experiment_results/model_comparison.png`
- `reports/experiment_results/efficiency_comparison.png`
- `reports/experiment_results/training_curves_comparison.png`
- `reports/experiment_results/experiment_results.md`

## Recommended Experiments

1. Model comparison:
   - `simplecnn`
   - `fastcnn`
   - `resnet18`
2. Augmentation ablation:
   - `fastcnn` with augmentation
   - `fastcnn` without augmentation
3. Bias comparison:
   - `fastcnn` on `color`
   - `fastcnn` on `segmented`

## Notes

- The loaders use ImageNet normalization to stay compatible with `ResNet18`.
- `grayscale` images are converted to 3-channel tensors by the loader pipeline.
- All runs save the best checkpoint according to validation accuracy.
- Errors are surfaced directly when paths, checkpoints, or dataset layouts are invalid.
