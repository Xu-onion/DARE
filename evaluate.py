import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_parser():
    parser = argparse.ArgumentParser(description="Compare reconstructed images with references")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--prediction-prefix", default="reconstructed_")
    parser.add_argument("--csv", type=Path, default=None, help="Optional path for per-image metrics")
    return parser


def load_grayscale(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def calculate_metrics(reference, prediction):
    difference = prediction - reference
    mse = float(np.mean(difference**2))
    mae = float(np.mean(np.abs(difference)))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    ssim = float(structural_similarity(reference, prediction, data_range=1.0))
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def main(args):
    if not args.reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {args.reference_dir}")
    if not args.prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {args.prediction_dir}")

    references = sorted(
        path
        for path in args.reference_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    rows = []
    for reference_path in references:
        prediction_path = args.prediction_dir / f"{args.prediction_prefix}{reference_path.name}"
        if not prediction_path.is_file():
            print(f"Skipping {reference_path.name}: prediction not found")
            continue

        reference = load_grayscale(reference_path)
        prediction = load_grayscale(prediction_path)
        if reference.shape != prediction.shape:
            print(
                f"Skipping {reference_path.name}: shape {reference.shape} != {prediction.shape}"
            )
            continue

        metrics = calculate_metrics(reference, prediction)
        row = {"filename": reference_path.name, **metrics}
        rows.append(row)
        print(
            f"{reference_path.name}: MAE={metrics['mae']:.6f}  MSE={metrics['mse']:.6f}  "
            f"PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}"
        )

    if not rows:
        raise RuntimeError("No matching image pairs were evaluated")

    averages = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("mae", "mse", "psnr", "ssim")
    }
    print(
        f"Average over {len(rows)} image(s): MAE={averages['mae']:.6f}  "
        f"MSE={averages['mse']:.6f}  PSNR={averages['psnr']:.2f}  "
        f"SSIM={averages['ssim']:.4f}"
    )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "mae", "mse", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"filename": "AVERAGE", **averages})
        print(f"Saved metrics to {args.csv}")


if __name__ == "__main__":
    main(build_parser().parse_args())

