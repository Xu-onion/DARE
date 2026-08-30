# Parts of this file are adapted from Meta's MAE training utilities.

import datetime
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist


class SmoothedValue:
    """Track recent values and their global average."""

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        values = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.all_reduce(values)
        self.count = int(values[0].item())
        self.total = values[1].item()

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not isinstance(value, (float, int)):
                raise TypeError(f"Metric {name!r} must be numeric, got {type(value).__name__}")
            self.meters[name].update(value)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {attr!r}")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=""):
        length = len(iterable)
        if length == 0:
            raise ValueError("The data loader is empty. Check the dataset and batch size.")

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        digits = len(str(length))

        for index, item in enumerate(iterable):
            data_time.update(time.time() - end)
            yield item
            iter_time.update(time.time() - end)

            if index % print_freq == 0 or index == length - 1:
                remaining = length - index - 1
                eta = str(datetime.timedelta(seconds=int(iter_time.global_avg * remaining)))
                message = (
                    f"{header} [{index:{digits}d}/{length}]  eta: {eta}  {self}  "
                    f"time: {iter_time}  data: {data_time}"
                )
                if torch.cuda.is_available():
                    memory = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    message += f"  max mem: {memory:.0f} MB"
                print(message)
            end = time.time()

        total_time = time.time() - start_time
        print(
            f"{header} Total time: {datetime.timedelta(seconds=int(total_time))} "
            f"({total_time / length:.4f} s / it)"
        )


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if not update_grad:
            return None

        if parameters is None:
            raise ValueError("parameters are required when updating gradients")
        self._scaler.unscale_(optimizer)
        norm = (
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            if clip_grad is not None
            else get_grad_norm_(parameters)
        )
        self._scaler.step(optimizer)
        self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return torch.tensor(0.0)

    norm_type = float(norm_type)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        return max(parameter.grad.detach().abs().max().to(device) for parameter in parameters)
    return torch.norm(
        torch.stack(
            [torch.norm(parameter.grad.detach(), norm_type).to(device) for parameter in parameters]
        ),
        norm_type,
    )


def load_checkpoint(path, map_location="cpu"):
    """Load old and new PyTorch checkpoints while making trust explicit."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_compatible_weights(model, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    candidate = extract_model_state(checkpoint)
    if not isinstance(candidate, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} does not contain a model state dictionary")

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in candidate.items()
        if key in model_state and hasattr(value, "shape") and value.shape == model_state[key].shape
    }
    skipped = len(candidate) - len(compatible)
    result = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded {len(compatible)} tensors from {checkpoint_path}; "
        f"skipped {skipped}, missing {len(result.missing_keys)}, unexpected {len(result.unexpected_keys)}."
    )
    return checkpoint


def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler):
    checkpoint_path = Path(args.output_dir) / f"checkpoint-{epoch}.pth"
    payload = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": vars(args),
    }
    save_on_master(payload, checkpoint_path)


def resume_training(path, model, optimizer, loss_scaler):
    checkpoint = load_checkpoint(path, map_location="cpu")
    model_state = extract_model_state(checkpoint)
    model.load_state_dict(model_state, strict=True)

    start_epoch = 0
    if isinstance(checkpoint, dict):
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
    print(f"Resumed training from {path} at epoch {start_epoch}.")
    return start_epoch


def all_reduce_mean(value):
    if get_world_size() == 1:
        return value
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reduced = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(reduced)
    reduced /= get_world_size()
    return reduced.item()

