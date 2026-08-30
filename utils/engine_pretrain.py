import math

import torch

import utils.lr_sched as lr_sched
import utils.misc as misc


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    accum_iter = args.accum_iter

    optimizer.zero_grad(set_to_none=True)
    for step, (samples, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                step / len(data_loader) + epoch,
                args,
            )

        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            loss, _, _ = model(samples)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite loss {loss_value} at epoch {epoch}, step {step}")

        is_update_step = (step + 1) % accum_iter == 0 or step + 1 == len(data_loader)
        loss_scaler(
            loss / accum_iter,
            optimizer,
            parameters=model.parameters(),
            update_grad=is_update_step,
        )
        if is_update_step:
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        learning_rate = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value, lr=learning_rate)

        reduced_loss = misc.all_reduce_mean(loss_value)
        if log_writer is not None and is_update_step:
            global_step = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train/loss", reduced_loss, global_step)
            log_writer.add_scalar("train/learning_rate", learning_rate, global_step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}

