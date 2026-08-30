import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import swin_mae
from utils.misc import load_checkpoint, extract_model_state


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TILE_SIZE = 224


def build_parser():
    parser = argparse.ArgumentParser(description="Reconstruct images with a Swin MAE checkpoint")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=30, help="Tile stride from 1 to 224")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:1")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--save-masked", action="store_true", help="Also save the masked model input")
    return parser


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or --device auto.")
    return device


def load_model(checkpoint_path, device):
    model = swin_mae.swin_mae()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state = extract_model_state(checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} does not contain a model state dictionary")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            f"Checkpoint loaded with {len(result.missing_keys)} missing and "
            f"{len(result.unexpected_keys)} unexpected keys."
        )
    return model.to(device).eval()


def tile_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def pad_image(image, tile_size=TILE_SIZE):
    height, width = image.shape[:2]
    pad_height = max(tile_size - height, 0)
    pad_width = max(tile_size - width, 0)
    if pad_height == 0 and pad_width == 0:
        return image
    return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def reconstruct_image(image, model, device, stride, return_masked=False):
    original_height, original_width = image.shape[:2]
    image = pad_image(image)
    height, width = image.shape[:2]

    y_positions = tile_positions(height, TILE_SIZE, stride)
    x_positions = tile_positions(width, TILE_SIZE, stride)
    output_sum = np.zeros((height, width, 3), dtype=np.float32)
    masked_sum = np.zeros_like(output_sum) if return_masked else None
    weight_sum = np.zeros((height, width, 1), dtype=np.float32)

    axis = np.linspace(-1.0, 1.0, TILE_SIZE, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    window = np.exp(-(grid_x**2 + grid_y**2))[..., None].astype(np.float32)

    with torch.inference_mode():
        for top in y_positions:
            for left in x_positions:
                tile = image[top : top + TILE_SIZE, left : left + TILE_SIZE]
                tensor = torch.from_numpy(tile / 255.0).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.to(device=device, dtype=torch.float32)

                _, prediction, mask = model(tensor)
                prediction = model.unpatchify(prediction)[0].permute(1, 2, 0).cpu().numpy()

                output_sum[top : top + TILE_SIZE, left : left + TILE_SIZE] += prediction * window
                weight_sum[top : top + TILE_SIZE, left : left + TILE_SIZE] += window

                if return_masked:
                    channels = model.patch_embed.patch_size**2 * 3
                    image_mask = mask.unsqueeze(-1).repeat(1, 1, channels)
                    image_mask = model.unpatchify(image_mask)[0].permute(1, 2, 0)
                    masked = tensor[0].permute(1, 2, 0) * (1.0 - image_mask)
                    masked_sum[top : top + TILE_SIZE, left : left + TILE_SIZE] += (
                        masked.cpu().numpy() * window
                    )

    reconstructed = output_sum / np.maximum(weight_sum, 1e-8)
    reconstructed = reconstructed[:original_height, :original_width]
    masked_result = None
    if return_masked:
        masked_result = masked_sum / np.maximum(weight_sum, 1e-8)
        masked_result = masked_result[:original_height, :original_width]
    return reconstructed, masked_result


def to_uint8(image):
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def main(args):
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if not 1 <= args.stride <= TILE_SIZE:
        raise ValueError(f"stride must be between 1 and {TILE_SIZE}")

    image_paths = sorted(
        path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, device)
    print(f"Processing {len(image_paths)} image(s) on {device}.")

    for image_path in image_paths:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        reconstructed, masked = reconstruct_image(
            image,
            model,
            device,
            args.stride,
            return_masked=args.save_masked,
        )
        reconstructed_path = args.output_dir / f"reconstructed_{image_path.name}"
        Image.fromarray(to_uint8(reconstructed)).save(reconstructed_path)
        if masked is not None:
            Image.fromarray(to_uint8(masked)).save(args.output_dir / f"masked_{image_path.name}")
        print(f"Saved {reconstructed_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())

