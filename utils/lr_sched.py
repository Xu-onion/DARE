# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the terms applicable to the upstream MAE implementation.

import math


def adjust_learning_rate(optimizer, epoch, args):
    """Apply linear warmup followed by half-cycle cosine decay."""
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        decay_epochs = max(args.epochs - args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / decay_epochs
        progress = min(max(progress, 0.0), 1.0)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * scale
    return lr

