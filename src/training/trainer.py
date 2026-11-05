from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def resolve_amp_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """Resolve explicit training precision for CUDA/ROCm or Apple MPS."""
    if not isinstance(precision, str) or precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("training.precision must be fp32, bf16 or fp16")
    if precision == "fp32":
        return None
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS mixed precision requires an available MPS backend")
        # PyTorch 2.8 added MPS GradScaler support. Keep this check off FP32
        # and CUDA/ROCm paths, including older HPC environments.
        from torch.torch_version import TorchVersion

        if TorchVersion(torch.__version__) < "2.8.0":
            raise ValueError("MPS mixed precision requires PyTorch >= 2.8.0; use fp32 or upgrade PyTorch")
        if not torch.amp.autocast_mode.is_autocast_available("mps"):
            raise ValueError("MPS autocast is unavailable in this PyTorch build; use fp32")
        if precision == "bf16" and not torch.backends.mps.is_macos_or_newer(14, 0):
            raise ValueError("BF16 MPS training requires macOS >= 14; use fp32 or fp16")
        return torch.bfloat16 if precision == "bf16" else torch.float16
    if device.type != "cuda":
        raise ValueError(f"training.precision={precision} requires a CUDA/ROCm or MPS GPU; got {device}")
    if precision == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 is not supported by the selected GPU/PyTorch environment")
        return torch.bfloat16
    return torch.float16


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 0.0,
    *,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    if amp_dtype not in (None, torch.bfloat16, torch.float16):
        raise ValueError("amp_dtype must be None, torch.bfloat16 or torch.float16")
    if amp_dtype is not None and device.type not in {"cuda", "mps"}:
        raise ValueError("Mixed precision training requires a CUDA/ROCm or MPS GPU")
    if amp_dtype == torch.float16 and (scaler is None or not scaler.is_enabled()):
        raise ValueError("FP16 training requires an enabled GradScaler shared across epochs")
    if scaler is not None and amp_dtype != torch.float16:
        raise ValueError("GradScaler is only used with FP16 training")
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        K = batch["K"].to(device, non_blocking=True)
        Q = batch["Q"].to(device, non_blocking=True)
        G = batch["G"].to(device, non_blocking=True)
        M = batch["M"].to(device, non_blocking=True)
        y = batch["y_joint"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp_dtype is None:
            # Keep the historical FP32 path, including callers such as GRU.
            logits = model(K, Q, G, M)
            loss = loss_fn(logits, y)
        else:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(K, Q, G, M)
            # FocalLoss includes exp/clamp/pow and a 1e-12 epsilon. Keep it FP32.
            loss = loss_fn(logits.float(), y)

        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)
