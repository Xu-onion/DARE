import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import swin_mae
import utils.misc as misc
from utils.engine_pretrain import train_one_epoch


def build_parser():
    parser = argparse.ArgumentParser(description="Pre-train Swin MAE on 224 x 224 images")
    parser.add_argument("--data-path", type=Path, required=True, help="ImageFolder dataset root")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default"))
    parser.add_argument("--log-dir", type=Path, default=None, help="TensorBoard directory")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1700)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--norm-pix-loss", action="store_true")

    parser.add_argument("--accum-iter", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)

    parser.add_argument("--pretrained", type=Path, default=None, help="Load compatible model weights")
    parser.add_argument("--resume", type=Path, default=None, help="Resume a full training checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-pin-mem", action="store_true")
    return parser


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or --device auto.")
    return device


def validate_args(args):
    if not args.data_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data_path}")
    if args.batch_size < 1 or args.epochs < 1 or args.accum_iter < 1:
        raise ValueError("batch-size, epochs, and accum-iter must be positive")
    if args.save_freq < 1:
        raise ValueError("save-freq must be positive")
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("mask-ratio must be between 0 and 1")
    if args.pretrained is not None and args.resume is not None:
        raise ValueError("Use either --pretrained or --resume, not both")


def main(args):
    validate_args(args)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or args.output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        cudnn.benchmark = True

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_mem,
        drop_last=True,
    )
    if len(data_loader) == 0:
        raise ValueError(
            "No complete batch is available. Add images or reduce --batch-size."
        )

    model = swin_mae.swin_mae(
        norm_pix_loss=args.norm_pix_loss,
        mask_ratio=args.mask_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    loss_scaler = misc.NativeScalerWithGradNormCount(enabled=device.type == "cuda")

    if args.pretrained is not None:
        misc.load_compatible_weights(model, args.pretrained)
    if args.resume is not None:
        args.start_epoch = misc.resume_training(args.resume, model, optimizer, loss_scaler)

    print(f"Training {len(dataset)} images on {device} for epochs {args.start_epoch}..{args.epochs - 1}")
    log_writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for epoch in range(args.start_epoch, args.epochs):
            train_stats = train_one_epoch(
                model,
                data_loader,
                optimizer,
                device,
                epoch,
                loss_scaler,
                log_writer=log_writer,
                args=args,
            )

            completed_epoch = epoch + 1
            if completed_epoch % args.save_freq == 0 or completed_epoch == args.epochs:
                misc.save_model(args, completed_epoch, model, optimizer, loss_scaler)

            record = {**{f"train_{key}": value for key, value in train_stats.items()}, "epoch": epoch}
            with (args.output_dir / "log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            log_writer.flush()
    finally:
        log_writer.close()


if __name__ == "__main__":
    main(build_parser().parse_args())

