from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Sampler


CLASS_NAMES = ("background", "budding", "mix", "shmoo")
INPUT_LENGTH = 4096
LATENT_DIMENSION = 512
FROZEN_STFT_CONFIG = {
    "n_fft": 256,
    "win_length": 256,
    "hop_length": 64,
    "window": "hann",
    "center": False,
    "magnitude_transform": "log1p",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class YeastClassificationData:
    signals: np.ndarray
    rows: list[dict[str, str]]
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    contract: dict[str, Any]


@dataclass(frozen=True)
class FrozenDevelopmentSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    manifest: dict[str, Any]


def build_source_disjoint_80_20_split(
    rows: Sequence[dict[str, str]],
    labels: np.ndarray,
    eligible_indices: Sequence[int],
    *,
    seed: int = 20260805,
    validation_fraction: float = 0.20,
    candidates: int = 512,
) -> FrozenDevelopmentSplit:
    """Choose one deterministic record-disjoint split with near-stratified classes."""
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    if eligible.size == 0:
        raise ValueError("eligible_indices must not be empty")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    groups = np.asarray([rows[int(index)]["record_id"] for index in eligible])
    y = np.asarray(labels, dtype=np.int64)[eligible]
    overall = np.bincount(y, minlength=len(CLASS_NAMES)).astype(np.float64)
    overall /= overall.sum()
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(candidates):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=seed + offset,
        )
        train_local, validation_local = next(splitter.split(eligible, y, groups))
        train_labels = y[train_local]
        validation_labels = y[validation_local]
        if len(set(train_labels.tolist())) != len(CLASS_NAMES) or len(set(validation_labels.tolist())) != len(CLASS_NAMES):
            continue
        validation_distribution = np.bincount(validation_labels, minlength=len(CLASS_NAMES)).astype(np.float64)
        validation_distribution /= validation_distribution.sum()
        size_error = abs(validation_local.size / eligible.size - validation_fraction)
        class_error = float(np.abs(validation_distribution - overall).sum())
        score = class_error + 2.0 * size_error
        if best is None or score < best[0] - 1e-15:
            best = (score, train_local, validation_local)
    if best is None:
        raise ValueError("Could not construct a source-disjoint split containing all classes")
    _, train_local, validation_local = best
    train_indices = np.sort(eligible[train_local])
    validation_indices = np.sort(eligible[validation_local])
    train_groups = {rows[int(index)]["record_id"] for index in train_indices}
    validation_groups = {rows[int(index)]["record_id"] for index in validation_indices}
    overlap = train_groups & validation_groups
    if overlap:
        raise AssertionError(f"Source leakage in frozen split: {sorted(overlap)[:3]}")

    def counts(indices: np.ndarray) -> dict[str, int]:
        values = np.bincount(np.asarray(labels)[indices], minlength=len(CLASS_NAMES))
        return {name: int(values[class_id]) for class_id, name in enumerate(CLASS_NAMES)}

    manifest = {
        "schema_version": 1,
        "split_id": "yeast-4class-source-disjoint-80-20-s20260805-r1",
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "selection_candidates": candidates,
        "group_key": "record_id",
        "source_partition": "development_train",
        "sealed_holdout_accessed": False,
        "train": {"rows": int(train_indices.size), "groups": len(train_groups), "class_counts": counts(train_indices)},
        "validation": {
            "rows": int(validation_indices.size),
            "groups": len(validation_groups),
            "class_counts": counts(validation_indices),
        },
    }
    return FrozenDevelopmentSplit(train_indices, validation_indices, manifest)


def load_frozen_split(path: Path, data: YeastClassificationData) -> FrozenDevelopmentSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Frozen split manifest is missing assignments")
    sample_to_index = {row["sample_id"]: index for index, row in enumerate(data.rows)}
    train_ids = assignments.get("train", [])
    validation_ids = assignments.get("validation", [])
    if set(train_ids) & set(validation_ids):
        raise ValueError("Frozen split contains overlapping sample IDs")
    unknown = (set(train_ids) | set(validation_ids)) - set(sample_to_index)
    if unknown:
        raise ValueError(f"Frozen split references unknown samples: {sorted(unknown)[:3]}")
    train = np.asarray(sorted(sample_to_index[value] for value in train_ids), dtype=np.int64)
    validation = np.asarray(sorted(sample_to_index[value] for value in validation_ids), dtype=np.int64)
    allowed = set(data.train_indices.tolist())
    if not set(train.tolist()).issubset(allowed) or not set(validation.tolist()).issubset(allowed):
        raise ValueError("Frozen split may only use the original development_train partition")
    train_groups = {data.rows[int(index)]["record_id"] for index in train}
    validation_groups = {data.rows[int(index)]["record_id"] for index in validation}
    if train_groups & validation_groups:
        raise ValueError("Frozen split is not record-disjoint")
    if set(data.labels[train].tolist()) != set(range(len(CLASS_NAMES))):
        raise ValueError("Frozen training split does not contain every class")
    if set(data.labels[validation].tolist()) != set(range(len(CLASS_NAMES))):
        raise ValueError("Frozen validation split does not contain every class")
    return FrozenDevelopmentSplit(train, validation, payload)


def load_dataset(root: Path) -> YeastClassificationData:
    rows: list[dict[str, str]]
    with (root / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    signals = np.load(root / "signals.npy", mmap_mode="r")
    contract = json.loads((root / "dataset-contract.json").read_text(encoding="utf-8"))
    if signals.shape != (len(rows), INPUT_LENGTH) or signals.dtype != np.float32:
        raise ValueError(f"Unexpected signal tensor: {signals.shape} {signals.dtype}")
    if tuple(contract["classes"]) != CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {contract['classes']}")
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    if set(labels.tolist()) != set(range(len(CLASS_NAMES))):
        raise ValueError("Dataset must contain all four classes")
    splits = np.asarray([row["development_split"] for row in rows])
    if np.any(~np.isin(splits, ["development_train", "development_validation"])):
        raise ValueError("Forbidden split present")
    train = np.flatnonzero(splits == "development_train")
    validation = np.flatnonzero(splits == "development_validation")
    if train.size == 0 or validation.size == 0:
        raise ValueError("Train and validation must both be non-empty")
    return YeastClassificationData(signals, rows, labels, train, validation, contract)


class IndexedArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        self.signals = signals
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return (
            torch.from_numpy(np.asarray(self.signals[index], dtype=np.float32).copy()),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
            int(index),
        )


class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: np.ndarray,
        indices: Sequence[int],
        *,
        batch_size: int,
        seed: int,
        epoch_size_policy: Literal["largest", "minority"] = "largest",
        batches_per_epoch: int | None = None,
    ) -> None:
        if batch_size % len(CLASS_NAMES):
            raise ValueError("batch_size must be divisible by four")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch_size_policy = epoch_size_policy
        self.epoch = 0
        self.by_class = [self.indices[self.labels[self.indices] == class_id] for class_id in range(len(CLASS_NAMES))]
        if any(values.size == 0 for values in self.by_class):
            raise ValueError("Every class must be represented")
        self.per_class = self.batch_size // len(CLASS_NAMES)
        if batches_per_epoch is not None:
            if batches_per_epoch <= 0:
                raise ValueError("batches_per_epoch must be positive")
            self.batch_count = int(batches_per_epoch)
        else:
            if epoch_size_policy == "largest":
                reference_size = max(values.size for values in self.by_class)
            elif epoch_size_policy == "minority":
                reference_size = min(values.size for values in self.by_class)
            else:
                raise ValueError(f"Unsupported epoch size policy: {epoch_size_policy}")
            self.batch_count = math.ceil(reference_size / self.per_class)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        draws: list[np.ndarray] = []
        needed = self.batch_count * self.per_class
        for values in self.by_class:
            repeats = math.ceil(needed / values.size)
            pieces = [rng.permutation(values) for _ in range(repeats)]
            draws.append(np.concatenate(pieces)[:needed])
        for batch_index in range(self.batch_count):
            start = batch_index * self.per_class
            stop = start + self.per_class
            batch = np.concatenate([values[start:stop] for values in draws])
            rng.shuffle(batch)
            yield [int(value) for value in batch]


def replace_batch_norm_with_group_norm(module: nn.Module, *, max_groups: int = 16) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            groups = min(max_groups, child.num_features)
            while child.num_features % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            replace_batch_norm_with_group_norm(child, max_groups=max_groups)


class ProjectedYeastClassifier(nn.Module):
    """Model-zoo backbone with a stable 512-D pre-logit interface."""

    def __init__(
        self,
        model_name: str,
        *,
        input_length: int = INPUT_LENGTH,
        num_classes: int = len(CLASS_NAMES),
        normalization: Literal["batch", "group"] = "batch",
        head_type: Literal["flat", "hierarchical"] = "flat",
    ) -> None:
        super().__init__()
        from p0.models import create_model

        self.model_name = model_name
        self.head_type = head_type
        self.backbone = create_model(model_name, input_length=input_length, num_classes=num_classes)
        self.normalization = normalization
        if normalization == "group":
            replace_batch_norm_with_group_norm(self.backbone)
        elif normalization != "batch":
            raise ValueError(f"Unsupported normalization: {normalization}")
        if model_name.startswith("Conv1DGAP"):
            feature_dim = int(self.backbone.fc2.in_features)
            self.backbone.fc2 = nn.Identity()
        elif hasattr(self.backbone, "classifier") and isinstance(self.backbone.classifier, nn.Linear):
            feature_dim = int(self.backbone.classifier.in_features)
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone does not expose a supported classifier head: {model_name}")
        self.projection = nn.Identity() if feature_dim == LATENT_DIMENSION else nn.Linear(feature_dim, LATENT_DIMENSION)
        if head_type == "flat":
            self.classifier = nn.Linear(LATENT_DIMENSION, num_classes)
        elif head_type == "hierarchical":
            self.classifier = HierarchicalYeastHead(LATENT_DIMENSION)
        else:
            raise ValueError(f"Unsupported classifier head: {head_type}")

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        return torch.relu(self.projection(features))

    def forward(self, inputs: torch.Tensor, *, return_features: bool = False):
        features = self.forward_features(inputs)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class FrozenSTFT(nn.Module):
    """Deterministic log-magnitude STFT shared by spectral and fused models."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(FROZEN_STFT_CONFIG["win_length"]), persistent=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (1, INPUT_LENGTH):
            raise ValueError(f"Expected [B,1,{INPUT_LENGTH}], got {tuple(inputs.shape)}")
        spectrum = torch.stft(
            inputs[:, 0],
            n_fft=FROZEN_STFT_CONFIG["n_fft"],
            hop_length=FROZEN_STFT_CONFIG["hop_length"],
            win_length=FROZEN_STFT_CONFIG["win_length"],
            window=self.window,
            center=FROZEN_STFT_CONFIG["center"],
            return_complex=True,
        )
        return torch.log1p(spectrum.abs()).unsqueeze(1)


class SpectrogramEncoder(nn.Module):
    def __init__(self, output_dimension: int) -> None:
        super().__init__()
        self.stft = FrozenSTFT()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, output_dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.network(self.stft(inputs)).flatten(1)
        return torch.relu(self.projection(features))


class SpectrogramYeastClassifier(nn.Module):
    model_name = "STFT-CNN"
    normalization = "frozen-stft-groupnorm"

    def __init__(self, *, head_type: Literal["flat", "hierarchical"] = "hierarchical") -> None:
        super().__init__()
        self.head_type = head_type
        self.encoder = SpectrogramEncoder(LATENT_DIMENSION)
        self.classifier = (
            nn.Linear(LATENT_DIMENSION, len(CLASS_NAMES))
            if head_type == "flat"
            else HierarchicalYeastHead(LATENT_DIMENSION)
        )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor, *, return_features: bool = False):
        features = self.forward_features(inputs)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class DualBranchYeastClassifier(nn.Module):
    model_name = "DualBranch-ResNet1D-STFT"
    normalization = "group+frozen-stft-groupnorm"

    def __init__(self, *, head_type: Literal["flat", "hierarchical"] = "hierarchical") -> None:
        super().__init__()
        self.head_type = head_type
        temporal = ProjectedYeastClassifier(
            "ResNet1D-XS",
            normalization="group",
            head_type="flat",
        )
        self.temporal_encoder = temporal
        self.spectral_encoder = SpectrogramEncoder(256)
        self.fusion = nn.Sequential(
            nn.Linear(LATENT_DIMENSION + 256, LATENT_DIMENSION),
            nn.LayerNorm(LATENT_DIMENSION),
            nn.GELU(),
        )
        self.classifier = (
            nn.Linear(LATENT_DIMENSION, len(CLASS_NAMES))
            if head_type == "flat"
            else HierarchicalYeastHead(LATENT_DIMENSION)
        )

    def forward_branch_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.temporal_encoder.forward_features(inputs), self.spectral_encoder(inputs)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        temporal, spectral = self.forward_branch_features(inputs)
        return self.fusion(torch.cat((temporal, spectral), dim=1))

    def forward(self, inputs: torch.Tensor, *, return_features: bool = False):
        features = self.forward_features(inputs)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def create_yeast_classifier_model(
    model_name: str,
    *,
    normalization: Literal["batch", "group"] = "batch",
    head_type: Literal["flat", "hierarchical"] = "flat",
    pretrained_cache_dir: Path | None = None,
) -> nn.Module:
    if model_name == "STFT-CNN":
        return SpectrogramYeastClassifier(head_type=head_type)
    if model_name == "DualBranch-ResNet1D-STFT":
        return DualBranchYeastClassifier(head_type=head_type)
    if model_name == "PatchTST-pretrained":
        if normalization != "batch":
            raise ValueError("PatchTST uses its native normalization; group normalization is unsupported")
        return PatchTSTYeastClassifier(
            cache_dir=pretrained_cache_dir or Path(".cache/huggingface"),
            head_type=head_type,
        )
    return ProjectedYeastClassifier(
        model_name,
        input_length=INPUT_LENGTH,
        num_classes=len(CLASS_NAMES),
        normalization=normalization,
        head_type=head_type,
    )


class HierarchicalYeastHead(nn.Module):
    """Joint four-class distribution factored into eventness and event condition."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.event_classifier = nn.Linear(feature_dim, 1)
        self.condition_classifier = nn.Linear(feature_dim, len(CLASS_NAMES) - 1)

    def component_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.event_classifier(features).squeeze(1), self.condition_classifier(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        event_logits, condition_logits = self.component_logits(features)
        log_background = torch.nn.functional.logsigmoid(-event_logits).unsqueeze(1)
        log_event = torch.nn.functional.logsigmoid(event_logits).unsqueeze(1)
        log_condition = torch.nn.functional.log_softmax(condition_logits, dim=1)
        return torch.cat((log_background, log_event + log_condition), dim=1)


class PatchTSTYeastClassifier(nn.Module):
    """Public pretrained PatchTST with the same 512-D yeast inference contract."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        head_type: Literal["flat", "hierarchical"] = "flat",
    ) -> None:
        super().__init__()
        from p3_ssl.pretrained_backbones import PATCHTST_DEFAULT_ID, load_patchtst_1ch_model

        self.model_name = "PatchTST-pretrained"
        self.model_id = PATCHTST_DEFAULT_ID
        self.head_type = head_type
        self.normalization = "pretrained"
        self.encoder, self.transfer_report = load_patchtst_1ch_model(
            model_id=self.model_id,
            cache_dir=cache_dir,
            device="cpu",
            context_length=INPUT_LENGTH,
        )
        feature_dim = int(self.encoder.config.d_model)
        self.projection = nn.Linear(feature_dim, LATENT_DIMENSION)
        if head_type == "flat":
            self.classifier = nn.Linear(LATENT_DIMENSION, len(CLASS_NAMES))
        elif head_type == "hierarchical":
            self.classifier = HierarchicalYeastHead(LATENT_DIMENSION)
        else:
            raise ValueError(f"Unsupported classifier head: {head_type}")

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (1, INPUT_LENGTH):
            raise ValueError(f"Expected [B,1,{INPUT_LENGTH}], got {tuple(inputs.shape)}")
        hidden = self.encoder(past_values=inputs.squeeze(1).unsqueeze(-1)).last_hidden_state
        pooled = hidden.mean(dim=tuple(range(1, hidden.ndim - 1)))
        return torch.relu(self.projection(pooled))

    def forward(self, inputs: torch.Tensor, *, return_features: bool = False):
        features = self.forward_features(inputs)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits

    def encoder_parameters(self):
        return self.encoder.parameters()


def augment_training_batch(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    background_bank: np.ndarray,
    rng: np.random.Generator,
    max_shift_points: int = 16,
    amplitude_scale_min: float = 0.90,
    amplitude_scale_max: float = 1.10,
    real_noise_fraction_max: float = 0.08,
) -> torch.Tensor:
    """Apply bounded train-only nuisances without altering event frequency or duration."""
    if inputs.ndim != 2 or inputs.shape[1] != INPUT_LENGTH:
        raise ValueError(f"Expected [B,{INPUT_LENGTH}] inputs, got {tuple(inputs.shape)}")
    if background_bank.ndim != 2 or background_bank.shape[1] != INPUT_LENGTH:
        raise ValueError("Background bank must contain full-length classifier inputs")
    output = inputs.clone()
    scales = torch.from_numpy(
        rng.uniform(amplitude_scale_min, amplitude_scale_max, size=(inputs.shape[0], 1)).astype(np.float32)
    )
    output *= scales
    if max_shift_points:
        shifts = rng.integers(-max_shift_points, max_shift_points + 1, size=inputs.shape[0])
        shifted = torch.zeros_like(output)
        for index, shift in enumerate(shifts.tolist()):
            if shift > 0:
                shifted[index, shift:] = output[index, :-shift]
            elif shift < 0:
                shifted[index, :shift] = output[index, -shift:]
            else:
                shifted[index] = output[index]
        output = shifted
    event_mask = labels != 0
    if real_noise_fraction_max > 0.0 and bool(event_mask.any()):
        event_count = int(event_mask.sum())
        selected = rng.integers(0, background_bank.shape[0], size=event_count)
        carriers = torch.from_numpy(np.asarray(background_bank[selected], dtype=np.float32).copy())
        event_values = output[event_mask]
        event_std = event_values.std(dim=1, keepdim=True).clamp_min(1.0e-6)
        carrier_std = carriers.std(dim=1, keepdim=True).clamp_min(1.0e-6)
        fractions = torch.from_numpy(
            rng.uniform(0.0, real_noise_fraction_max, size=(event_count, 1)).astype(np.float32)
        )
        output[event_mask] = event_values + carriers * (event_std / carrier_std) * fractions
    return output


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.10) -> torch.Tensor:
    """Class-supervised contrastive loss used only as a screening ablation."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("Features and labels have incompatible shapes")
    normalized = torch.nn.functional.normalize(features, dim=1)
    logits = normalized @ normalized.T / temperature
    identity = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~identity)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return features.sum() * 0.0
    mean_log_prob = (log_prob * positive).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob[valid].mean()


def encode_features(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    x = torch.relu(model.bn1(model.conv1(inputs)))
    x = model.drop1(model.pool1(x))
    x = torch.relu(model.bn2(model.conv2(x)))
    x = model.drop2(model.pool2(x))
    x = torch.relu(model.bn3(model.conv3(x)))
    x = model.drop3(model.pool3(x))
    x = model.flatten(model.gap(x))
    return torch.relu(model.fc1(x))


def encode_signals(
    model: nn.Module,
    signals: np.ndarray,
    *,
    device: torch.device | str,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    if signals.ndim != 2 or signals.shape[1] != INPUT_LENGTH:
        raise ValueError(f"Expected float32 [N,{INPUT_LENGTH}] signals, got {signals.shape}")
    model.eval()
    logits_chunks: list[np.ndarray] = []
    feature_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            batch = torch.from_numpy(np.asarray(signals[start : start + batch_size], dtype=np.float32)).to(device).unsqueeze(1)
            if hasattr(model, "forward_features"):
                features = model.forward_features(batch)
                logits = model.classifier(features)
            else:
                features = encode_features(model, batch)
                logits = model.fc2(features)
            feature_chunks.append(features.cpu().numpy().astype(np.float32))
            logits_chunks.append(logits.cpu().numpy().astype(np.float32))
    features = np.concatenate(feature_chunks, axis=0)
    logits = np.concatenate(logits_chunks, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    if features.shape[1] != LATENT_DIMENSION or not np.isfinite(normalized).all():
        raise ValueError("Invalid latent output")
    return {"logits": logits, "probabilities": probabilities, "embeddings": features, "embeddings_l2": normalized.astype(np.float32)}


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[labels]
    confidence = probabilities.max(axis=1)
    correct = (predictions == labels).astype(np.float64)
    calibration_bins: list[dict[str, float | int]] = []
    expected_calibration_error = 0.0
    bin_edges = np.linspace(0.0, 1.0, 11)
    for bin_index in range(10):
        lower = float(bin_edges[bin_index])
        upper = float(bin_edges[bin_index + 1])
        selected = (confidence >= lower) & (confidence <= upper if bin_index == 9 else confidence < upper)
        count = int(selected.sum())
        mean_confidence = float(confidence[selected].mean()) if count else 0.0
        empirical_accuracy = float(correct[selected].mean()) if count else 0.0
        expected_calibration_error += (count / labels.size) * abs(mean_confidence - empirical_accuracy)
        calibration_bins.append({
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_confidence": mean_confidence,
            "empirical_accuracy": empirical_accuracy,
        })
    per_class_auroc = {
        name: float(roc_auc_score(one_hot[:, index], probabilities[:, index]))
        for index, name in enumerate(CLASS_NAMES)
    }
    event_mask = labels != 0
    event_labels = labels[event_mask]
    event_predictions = predictions[event_mask]
    event_class_ids = np.arange(1, len(CLASS_NAMES))
    event_recalls = precision_recall_fscore_support(
        event_labels,
        event_predictions,
        labels=event_class_ids,
        zero_division=0,
    )[1]
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "ovr_macro_auroc": float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
        "ovr_auroc_per_class": per_class_auroc,
        "multiclass_brier": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "expected_calibration_error_10bin": float(expected_calibration_error),
        "calibration_bins": calibration_bins,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=np.arange(len(CLASS_NAMES))).tolist(),
        "per_class": {
            name: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
            for index, name in enumerate(CLASS_NAMES)
        },
    }
    result["event_only"] = {
        "classes": list(CLASS_NAMES[1:]),
        "support": int(event_mask.sum()),
        "accuracy": float(accuracy_score(event_labels, event_predictions)),
        "balanced_accuracy": float(np.mean(event_recalls)),
        "macro_f1": float(
            f1_score(event_labels, event_predictions, labels=event_class_ids, average="macro", zero_division=0)
        ),
    }
    return result


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
    *,
    pretrained_cache_dir: Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    from p0.models import create_model

    payload = torch.load(path, map_location="cpu")
    if tuple(payload["class_names"]) != CLASS_NAMES or int(payload["input_length"]) != INPUT_LENGTH:
        raise ValueError("Checkpoint contract mismatch")
    if int(payload.get("classifier_schema_version", 1)) >= 2:
        model = create_yeast_classifier_model(
            payload["model_name"],
            normalization=payload.get("normalization", "batch"),
            head_type=payload.get("head_type", "flat"),
            pretrained_cache_dir=pretrained_cache_dir,
        )
    else:
        model = create_model(payload["model_name"], input_length=INPUT_LENGTH, num_classes=len(CLASS_NAMES))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
