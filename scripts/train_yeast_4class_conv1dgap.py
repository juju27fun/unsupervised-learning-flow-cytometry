#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from p3_ssl.yeast_4class_classifier import (
    CLASS_NAMES,
    INPUT_LENGTH,
    BalancedBatchSampler,
    IndexedArrayDataset,
    augment_training_batch,
    classification_metrics,
    create_yeast_classifier_model,
    encode_signals,
    load_dataset,
    load_frozen_split,
    set_seed,
    sha256_file,
    supervised_contrastive_loss,
)


def _write_predictions(path: Path, rows: list[dict[str, str]], indices: np.ndarray, labels: np.ndarray, output: dict[str, np.ndarray]) -> None:
    probabilities = output["probabilities"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "record_id", "development_split", "source_group_original", "class_name", "class_id", "predicted_class", *[f"p_{name}" for name in CLASS_NAMES]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, source_index in enumerate(indices):
            row = rows[int(source_index)]
            writer.writerow({
                "sample_id": row["sample_id"], "record_id": row["record_id"],
                "development_split": row["development_split"], "source_group_original": row["source_group_original"],
                "class_name": row["class_name"], "class_id": int(labels[int(source_index)]),
                "predicted_class": CLASS_NAMES[int(probabilities[position].argmax())],
                **{f"p_{name}": float(probabilities[position, class_id]) for class_id, name in enumerate(CLASS_NAMES)},
            })


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    data = load_dataset(args.dataset_root)
    frozen_split = load_frozen_split(args.split_manifest, data) if args.split_manifest else None
    train_indices = frozen_split.train_indices if frozen_split else data.train_indices
    validation_indices = frozen_split.validation_indices if frozen_split else data.validation_indices
    device = torch.device(args.device)
    model = create_yeast_classifier_model(
        args.model_name,
        normalization=args.normalization,
        head_type=args.head_type,
        pretrained_cache_dir=args.pretrained_cache_dir,
    ).to(device)
    dataset = IndexedArrayDataset(data.signals, data.labels)
    sampler = BalancedBatchSampler(
        data.labels,
        train_indices,
        batch_size=args.batch_size,
        seed=args.seed,
        epoch_size_policy=args.epoch_size_policy,
        batches_per_epoch=args.batches_per_epoch or None,
    )
    train_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    validation_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=validation_indices.tolist(), num_workers=0)
    if args.encoder_lr > 0.0 and hasattr(model, "encoder_parameters"):
        encoder_parameters = list(model.encoder_parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": args.encoder_lr},
                {"params": head_parameters, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    def lr_multiplier(epoch_index: int) -> float:
        epoch_number = epoch_index + 1
        if epoch_number <= args.warmup_epochs:
            return epoch_number / max(1, args.warmup_epochs)
        progress = (epoch_number - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
        if args.scheduler == "cosine"
        else torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    )
    background_indices = train_indices[data.labels[train_indices] == 0]
    background_bank = np.asarray(data.signals[background_indices], dtype=np.float32)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        classification_losses: list[float] = []
        contrastive_losses: list[float] = []
        augmentation_rng = np.random.default_rng(args.seed + 100_003 * epoch)
        for batch_index, (inputs, labels, _) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            if args.augment:
                inputs = augment_training_batch(
                    inputs,
                    labels,
                    background_bank=background_bank,
                    rng=augmentation_rng,
                    max_shift_points=args.max_shift_points,
                    amplitude_scale_min=args.amplitude_scale_min,
                    amplitude_scale_max=args.amplitude_scale_max,
                    real_noise_fraction_max=args.real_noise_fraction_max,
                )
            labels_device = labels.to(device)
            logits, features = model(inputs.to(device).unsqueeze(1), return_features=True)
            if args.head_type == "hierarchical":
                event_logits, condition_logits = model.classifier.component_logits(features)
                event_targets = (labels_device > 0).to(torch.float32)
                eventness_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    event_logits,
                    event_targets,
                )
                event_rows = labels_device > 0
                condition_loss = criterion(condition_logits[event_rows], labels_device[event_rows] - 1)
                classification_loss = (
                    args.eventness_loss_weight * eventness_loss
                    + args.condition_loss_weight * condition_loss
                )
            else:
                classification_loss = criterion(logits, labels_device)
            contrastive_loss = supervised_contrastive_loss(
                features,
                labels_device,
                temperature=args.supcon_temperature,
            ) if args.supcon_weight > 0.0 else features.sum() * 0.0
            loss = classification_loss + args.supcon_weight * contrastive_loss
            loss.backward()
            if args.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            classification_losses.append(float(classification_loss.detach().cpu()))
            contrastive_losses.append(float(contrastive_loss.detach().cpu()))
            if args.max_train_batches and batch_index + 1 >= args.max_train_batches:
                break
        scheduler.step()
        validation_output = encode_signals(model, data.signals[validation_indices], device=device, batch_size=args.batch_size)
        metrics = classification_metrics(data.labels[validation_indices], validation_output["probabilities"])
        history.append({
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)),
            "train_classification_loss": float(np.mean(classification_losses)),
            "train_contrastive_loss": float(np.mean(contrastive_losses)),
            "validation": metrics,
        })
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "seed": args.seed,
                    "train_loss": float(np.mean(losses)),
                    "validation_event_macro_f1": float(metrics["event_only"]["macro_f1"]),
                    "validation_event_balanced_accuracy": float(metrics["event_only"]["balanced_accuracy"]),
                    "selection_metric": args.selection_metric,
                    "best_selection_score_before_epoch": best_score,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        score_key = "macro_f1" if args.selection_metric == "event_macro_f1" else "balanced_accuracy"
        score = float(metrics["event_only"][score_key])
        if score > best_score + 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    validation_output = encode_signals(model, data.signals[validation_indices], device=device, batch_size=args.batch_size)
    metrics = classification_metrics(data.labels[validation_indices], validation_output["probabilities"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_manifest = args.dataset_root / "dataset-manifest.json"
    checkpoint = {
        "schema_version": 2, "classifier_schema_version": 2, "model_name": args.model_name,
        "normalization": args.normalization, "head_type": args.head_type, "input_length": INPUT_LENGTH,
        "pretrained_cache_dir": str(args.pretrained_cache_dir),
        "class_names": list(CLASS_NAMES), "latent_dimension": 512, "seed": args.seed,
        "model_state_dict": best_state, "best_epoch": best_epoch,
        "selection_metric": args.selection_metric, "best_validation_selection_score": best_score,
        "dataset_id": data.contract["dataset_id"],
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "input_contract": data.contract,
        "split_id": frozen_split.manifest["split_id"] if frozen_split else "historical-development-split",
        "method_evidence_id": args.method_evidence_id,
    }
    checkpoint_path = args.output_dir / "best_model.pt"
    torch.save(checkpoint, checkpoint_path)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps({"validation": metrics, "best_epoch": best_epoch, "epochs_ran": len(history)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_predictions(args.output_dir / "validation_predictions.csv", data.rows, validation_indices, data.labels, validation_output)
    np.savez_compressed(
        args.output_dir / "validation_embeddings.npz",
        indices=validation_indices, labels=data.labels[validation_indices],
        logits=validation_output["logits"], probabilities=validation_output["probabilities"],
        embeddings=validation_output["embeddings"], embeddings_l2=validation_output["embeddings_l2"],
    )
    run = {
        "schema_version": 1, "run_id": args.run_id, "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset": checkpoint["dataset_id"],
        "method_evidence_id": checkpoint["method_evidence_id"], "seed": args.seed,
        "command": "train_yeast_4class_conv1dgap.py", "checkpoint_sha256": sha256_file(checkpoint_path),
        "sealed_holdout_accessed": False,
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "config": {
            "model_name": args.model_name,
            "split_manifest": str(args.split_manifest) if args.split_manifest else None,
            "split_id": checkpoint["split_id"],
            "normalization": args.normalization,
            "head_type": args.head_type,
            "eventness_loss_weight": args.eventness_loss_weight,
            "condition_loss_weight": args.condition_loss_weight,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "encoder_lr": args.encoder_lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "scheduler": args.scheduler,
            "label_smoothing": args.label_smoothing,
            "epoch_size_policy": args.epoch_size_policy,
            "batches_per_epoch": args.batches_per_epoch,
            "augment": args.augment,
            "max_shift_points": args.max_shift_points,
            "amplitude_scale_min": args.amplitude_scale_min,
            "amplitude_scale_max": args.amplitude_scale_max,
            "real_noise_fraction_max": args.real_noise_fraction_max,
            "supcon_weight": args.supcon_weight,
            "supcon_temperature": args.supcon_temperature,
            "grad_clip_norm": args.grad_clip_norm,
            "max_train_batches": args.max_train_batches,
            "selection_metric": f"validation.event_only.{score_key}",
        },
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"run_id": args.run_id, "validation": metrics, "checkpoint_sha256": run["checkpoint_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the four-class yeast Conv1D-GAP-L classifier.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-name", default="Conv1DGAP-L")
    parser.add_argument("--normalization", choices=("batch", "group"), default="batch")
    parser.add_argument("--head-type", choices=("flat", "hierarchical"), default="flat")
    parser.add_argument(
        "--selection-metric",
        choices=("event_macro_f1", "event_balanced_accuracy"),
        default="event_macro_f1",
    )
    parser.add_argument("--eventness-loss-weight", type=float, default=1.0)
    parser.add_argument("--condition-loss-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--scheduler", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--epoch-size-policy", choices=("largest", "minority"), default="minority")
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-shift-points", type=int, default=16)
    parser.add_argument("--amplitude-scale-min", type=float, default=0.90)
    parser.add_argument("--amplitude-scale-max", type=float, default=1.10)
    parser.add_argument("--real-noise-fraction-max", type=float, default=0.08)
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.10)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Bounded CPU smoke only; zero uses the full epoch.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method-evidence-id", default="yeast-4class-event-balanced-benchmark-method-r1")
    parser.add_argument("--pretrained-cache-dir", type=Path, default=Path("../.cache/huggingface"))
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
