# Swin MAE (224 × 224)

A cleaned PyTorch implementation of **Swin MAE: Masked Autoencoders for Small Datasets**, with command-line tools for training, tiled inference, and image-quality evaluation.

This public version intentionally excludes datasets, model checkpoints, experiment outputs, TensorBoard logs, IDE settings, and machine-specific paths.

## Project structure

```text
.
├── train.py                 # MAE pre-training
├── infer.py                 # tiled reconstruction for arbitrary-size images
├── evaluate.py              # MAE, MSE, PSNR, and SSIM evaluation
├── swin_mae.py              # Swin MAE model
├── swin_unet.py             # Swin Transformer building blocks
└── utils/                   # training and position-embedding utilities
```

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a specific CUDA version, install the matching PyTorch build from the official PyTorch instructions before installing the remaining requirements.

## Dataset layout

Training uses `torchvision.datasets.ImageFolder`, so the data directory must contain at least one class subdirectory. Labels are only used by `ImageFolder`; MAE training itself is self-supervised.

```text
path/to/training-data/
└── images/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

Datasets are ignored by Git. Keep them outside the repository or under `data/`, `dataset/`, or `datasets/`.

## Training

```bash
python train.py \
  --data-path path/to/training-data \
  --output-dir outputs/experiment-01 \
  --epochs 1700 \
  --batch-size 4
```

Useful options:

- `--device auto` selects CUDA when available and otherwise uses CPU.
- `--pretrained path/to/checkpoint.pth` loads compatible model weights before training.
- `--resume path/to/checkpoint.pth` restores model, optimizer, scaler, and epoch when available.
- `--mask-ratio 0.75` controls the proportion of masked pixels.
- `--save-freq 100` controls checkpoint frequency.

Run `python train.py --help` for all options.

## Inference

The model is trained on 224 × 224 inputs. `infer.py` reconstructs larger images with overlapping tiles and weighted blending.

```bash
python infer.py \
  --input-dir path/to/images \
  --checkpoint path/to/checkpoint.pth \
  --output-dir outputs/reconstruction \
  --stride 30 \
  --save-masked
```

Only load checkpoints from sources you trust. PyTorch checkpoint loading may deserialize Python objects.

## Evaluation

By default, inference outputs are named `reconstructed_<original-name>`. The evaluation script matches those files against the reference filenames.

```bash
python evaluate.py \
  --reference-dir path/to/reference-images \
  --prediction-dir outputs/reconstruction \
  --prediction-prefix reconstructed_ \
  --csv outputs/reconstruction/metrics.csv
```


## Attribution

The core implementation is based on the official [Swin-MAE repository](https://github.com/Zian-Xu/Swin-MAE) and the paper [Swin MAE: Masked Autoencoders for Small Datasets](https://arxiv.org/abs/2212.13805).



