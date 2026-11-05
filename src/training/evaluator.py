from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from states import state_id_to_pair_ids
from .metrics import accuracy, macro_f1, confusion_matrix


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_n = 0
    ys = []
    preds = []

    for batch in loader:
        K = batch["K"].to(device, non_blocking=True)
        Q = batch["Q"].to(device, non_blocking=True)
        G = batch["G"].to(device, non_blocking=True)
        M = batch["M"].to(device, non_blocking=True)
        y = batch["y_joint"].to(device, non_blocking=True)

        logits = model(K, Q, G, M)
        loss = loss_fn(logits, y)
        pred = logits.argmax(dim=-1)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)

    true_pairs = np.array([state_id_to_pair_ids(int(x)) for x in y_true], dtype=np.int64)
    pred_pairs = np.array([state_id_to_pair_ids(int(x)) for x in y_pred], dtype=np.int64)
    # print("true counts:", np.bincount(y_true, minlength=9))
    # print("pred counts:", np.bincount(y_pred, minlength=9))
    return {
        "loss": total_loss / max(total_n, 1),
        "joint_acc": accuracy(y_true, y_pred),
        "joint_macro_f1": macro_f1(y_true, y_pred, num_classes=9),
        "current_acc_from_joint": accuracy(true_pairs[:, 0], pred_pairs[:, 0]) if len(y_true) else 0.0,
        "future_acc_from_joint": accuracy(true_pairs[:, 1], pred_pairs[:, 1]) if len(y_true) else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes=9),
    }


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, list[str], list[str]]:
    model.eval()
    logits_all = []
    symbols: list[str] = []
    dates: list[str] = []
    for batch in loader:
        K = batch["K"].to(device, non_blocking=True)
        Q = batch["Q"].to(device, non_blocking=True)
        G = batch["G"].to(device, non_blocking=True)
        M = batch["M"].to(device, non_blocking=True)
        logits = model(K, Q, G, M)
        logits_all.append(logits.detach().cpu().numpy())
        symbols.extend(batch["symbol"])
        dates.extend(batch["date"])
    arr = np.concatenate(logits_all, axis=0) if logits_all else np.empty((0, 9), dtype=np.float32)
    return arr, symbols, dates
