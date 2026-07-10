#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

class ArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self.signals = signals.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        return torch.from_numpy(self.signals[sample_idx]).float(), torch.tensor(int(self.labels[sample_idx]), dtype=torch.long)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device).unsqueeze(1))
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            preds.extend(int(v) for v in pred.tolist())
            labels.extend(int(v) for v in y.numpy().tolist())
    if not labels:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan")}
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    from p0.models import create_model

    set_seed(args.seed)
    with np.load(args.dataset, allow_pickle=True) as data:
        signals = data["signals"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.int64, copy=False)
        split = data["split"].astype(str)
        class_names = [str(v) for v in data["class_names"].tolist()]

    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("Dataset must contain non-empty train and val splits")
    if test_idx.size == 0:
        test_idx = val_idx

    device = torch.device(args.device)
    model = create_model(args.model_name, input_length=int(signals.shape[1]), num_classes=len(class_names)).to(device)
    train_loader = DataLoader(ArrayDataset(signals, labels, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ArrayDataset(signals, labels, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ArrayDataset(signals, labels, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = -float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device).unsqueeze(1))
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val": val}
        history.append(row)
        if val["macro_f1"] > best_score:
            best_score = val["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "model_name": args.model_name,
        "input_length": int(signals.shape[1]),
        "class_names": class_names,
        "training_source": str(args.dataset),
        "best_val_macro_f1": float(best_score),
        "test_metrics": test,
    }
    torch.save(checkpoint, args.output_dir / "best_model.pt")
    metrics = {
        "model_name": args.model_name,
        "input_length": int(signals.shape[1]),
        "class_names": class_names,
        "best_val_macro_f1": float(best_score),
        "test": test,
        "history": history,
        "run_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with (args.output_dir / "training_history.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Conv1D-GAP control model on isolated yeast event crops.")
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "yeast_conv1dgap_dataset" / "yeast_conv1dgap_dataset.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "yeast_conv1dgap")
    parser.add_argument("--model-name", default="Conv1DGAP-L")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    metrics = train(build_parser().parse_args())
    print(json.dumps({"best_val_macro_f1": metrics["best_val_macro_f1"], "test": metrics["test"]}, sort_keys=True))


if __name__ == "__main__":
    main()
