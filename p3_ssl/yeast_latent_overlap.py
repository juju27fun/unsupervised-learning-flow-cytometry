#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

from p3_ssl.signal_preprocessing import (
    PREPROCESS_MODES,
    PREPROCESS_NONE,
    PREPROCESS_P1,
    P1PreprocessConfig,
    preprocess_signal,
    signal_quality_report,
    summarize_quality_reports,
)

MODEL_KEYS = ("moment_official", "patchtst_pretrained", "conv1dgap_same_input_3class")
MODEL_DISPLAY = {
    "moment_official": "MOMENT",
    "patchtst_pretrained": "PatchTST",
    "conv1dgap_same_input_3class": "Conv1D-GAP",
}
PARAM_COLUMNS = (
    "delta_t0",
    "amplitude_ratio",
    "delta_fD",
    "delta_phi",
    "tau_ratio",
    "snr_proxy",
)
DEFAULT_SYNTHETIC_ROOT = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "particle_equation_latent_sweeps" / "yeast_budded_template_proof"
DEFAULT_SYNTHETIC_ROOT_P1_FILTERED = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "particle_equation_latent_sweeps" / "yeast_budded_template_proof_p1_filtered"
DEFAULT_REAL_EMBEDDING_ROOT = (
    ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones-4096_20260701" / "particles2snr_f_3class_plus_yeast_quick_offline"
)
DEFAULT_REAL_EVENT_ROOT = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones-4096_20260701" / "yeast_passage_events_p3_4096"
DEFAULT_REAL_EVENT_ROOT_P1_FILTERED = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones-4096_20260701" / "yeast_passage_events_p3_4096_p1_filtered"
DEFAULT_BUDDING_RAW_DIR = Path(
    os.environ.get("YEAST_BUDDING_RAW_DIR", REPO_ROOT / "datasets" / "raw" / "yeast-budding" / "v1")
)
DEFAULT_YEAST_METADATA = (
    DEFAULT_REAL_EVENT_ROOT / "events_metadata.csv"
)
DEFAULT_OUTPUT_DIR = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "yeast_latent_overlap_validation" / "template_budding_v1"
DEFAULT_OUTPUT_DIR_P1_FILTERED = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "yeast_latent_overlap_validation" / "template_budding_v1_p1_filtered"
DEFAULT_CONV_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "unsupervised-learning-flow-cytometry"
    / "pretrained_backbones-4096_20260701"
    / "particles2snr_f_3class_moment_patchtst_conv1dgap_quick_offline"
    / "conv1dgap_same_input_3class"
    / "best_model.pt"
)
MOMENT_DEFAULT_ID = "AutonLab/MOMENT-1-large"
PATCHTST_DEFAULT_ID = "ibm-granite/granite-timeseries-patchtst"
DEFAULT_CONTROL_SYNTHETIC_ROOTS = (
    ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "particle_equation_latent_sweeps" / "yeast_budded_two_particle_proof",
    ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "particle_equation_latent_sweeps" / "yeast_budded_two_particle_proof_budding_ranges",
    ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "particle_equation_latent_sweeps" / "yeast_budded_two_particle_proof_budding_realistic",
)


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def import_encoding_stack():
    try:
        import torch
        from p3_ssl import particle_equation_sweeps as latent_sweeps
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Embedding encoding requires torch and the model stack. "
            "Install those dependencies, use the P0 venv, or provide cached real/synthetic embeddings."
        ) from exc
    return torch, latent_sweeps


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_available": False,
        "torch_version": None,
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
    }
    try:
        import torch

        info.update(
            {
                "torch_available": True,
                "torch_version": str(torch.__version__),
                "torch_cuda_build": str(getattr(torch.version, "cuda", None)),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_device_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available():
            info["torch_cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def preflight_device(args: argparse.Namespace) -> dict[str, Any]:
    info = runtime_info()
    if args.require_cuda and args.device == "cuda" and not info["torch_cuda_available"]:
        raise RuntimeError(
            "CUDA was required but is not available in this Python environment. "
            f"Python executable: {info['python_executable']}. "
            "Use a CUDA-visible environment or run without --require-cuda to allow CPU/cached validation."
        )
    return info


@dataclass(frozen=True)
class EmbeddingGroup:
    name: str
    embeddings: np.ndarray
    metadata: pd.DataFrame
    signals: np.ndarray | None = None
    provenance: dict[str, Any] | None = None


def parse_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    bad = [item for item in models if item not in MODEL_KEYS]
    if bad:
        raise ValueError(f"Unsupported models {bad}; expected any of {MODEL_KEYS}")
    return models


def parse_paths(raw: str | None) -> list[Path]:
    if not raw:
        return []
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def preprocessing_config_from_args(args: argparse.Namespace) -> P1PreprocessConfig:
    return P1PreprocessConfig(
        mode=str(getattr(args, "preprocess_mode", PREPROCESS_NONE)),
        sampling_frequency_hz=float(getattr(args, "preprocess_sampling_frequency_hz", 2_000_000.0)),
        low_khz=float(getattr(args, "preprocess_low_khz", 5.0)),
        high_khz_max=float(getattr(args, "preprocess_high_khz_max", 100.0)),
        saturation_fmin_hz=float(getattr(args, "saturation_fmin_hz", 7_000.0)),
        saturation_fmax_hz=float(getattr(args, "saturation_fmax_hz", 80_000.0)),
        saturation_min_flat=int(getattr(args, "saturation_min_flat", 500)),
        saturation_zero_threshold=float(getattr(args, "saturation_zero_threshold", 1.0e-4)),
        saturation_guard_before=int(getattr(args, "saturation_guard_before", 0)),
        saturation_guard_after=int(getattr(args, "saturation_guard_after", 0)),
        normalization="window_zscore",
    )


def preprocessing_cache_label(args: argparse.Namespace) -> str:
    cfg = preprocessing_config_from_args(args)
    return "raw" if cfg.mode == PREPROCESS_NONE else "p1_filtered"


def maybe_preprocess_signals(
    signals: np.ndarray,
    metadata: pd.DataFrame,
    args: argparse.Namespace,
    *,
    reject_saturation: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    cfg = preprocessing_config_from_args(args)
    if cfg.mode == PREPROCESS_NONE:
        return signals, metadata.reset_index(drop=True), {"preprocessing": cfg.to_dict(), "applied": False}
    input_ids = set(metadata.get("__input_preprocessing_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    if input_ids == {cfg.preprocessing_id}:
        return (
            signals,
            metadata.reset_index(drop=True),
            {
                "preprocessing": cfg.to_dict(),
                "applied": False,
                "already_applied": True,
                "kept_rows": int(signals.shape[0]),
                "rejected_rows": 0,
            },
        )
    if input_ids:
        raise ValueError(
            f"Input signals carry preprocessing_id={sorted(input_ids)}, expected {cfg.preprocessing_id!r}. "
            "Regenerate the real/synthetic p1-filtered inputs or use --preprocess-mode none."
        )
    raise ValueError(
        "P1 validation requires input signal NPZ files with a matching preprocessing_id. "
        "Regenerate real events and synthetic sweeps with --preprocess-mode p1_bandpass_saturation before validation."
    )
    processed: list[np.ndarray] = []
    kept_rows: list[int] = []
    reports: list[dict[str, Any]] = []
    for idx, row in enumerate(np.asarray(signals, dtype=np.float32)):
        report = signal_quality_report(row, cfg)
        reports.append(report)
        ok = bool(report["ok"]) or (report["reject_reason"] == "flat_saturation_interval" and not reject_saturation)
        if not ok:
            continue
        processed.append(preprocess_signal(row, output_length=row.shape[0], cfg=cfg))
        kept_rows.append(idx)
    if not processed:
        raise ValueError(f"Preprocessing rejected all rows: {summarize_quality_reports(reports)}")
    summary = summarize_quality_reports(reports)
    summary["preprocessing"] = cfg.to_dict()
    summary["applied"] = True
    summary["kept_rows"] = int(len(kept_rows))
    summary["rejected_rows"] = int(len(reports) - len(kept_rows))
    return (
        np.stack(processed).astype(np.float32),
        metadata.iloc[kept_rows].reset_index(drop=True),
        summary,
    )


def validate_cache_preprocessing(cache_path: Path, data: np.lib.npyio.NpzFile, args: argparse.Namespace) -> None:
    expected = preprocessing_config_from_args(args).preprocessing_id
    if expected == PREPROCESS_NONE:
        return
    cached = str(data["preprocessing_id"]) if "preprocessing_id" in data.files else ""
    if cached != expected:
        raise ValueError(
            f"Cached embeddings at {cache_path} use preprocessing_id={cached!r}; expected {expected!r}. "
            "Force re-encoding or use a matching p1-filtered output/cache directory."
        )


def discover_control_roots(raw: str | None) -> list[Path]:
    if raw is not None:
        return parse_paths(raw)
    return [path for path in DEFAULT_CONTROL_SYNTHETIC_ROOTS if (path / "synthetic_metadata.csv").is_file()]


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_real_groups(
    real_embedding_root: Path,
    yeast_metadata_csv: Path,
    model_key: str,
    source_group: str,
    min_real: int,
    expected_input_length: int,
) -> tuple[EmbeddingGroup, dict[str, EmbeddingGroup]]:
    emb_path = real_embedding_root / model_key / "all_embeddings.npz"
    metadata_path = real_embedding_root / "events_metadata.csv"
    if not emb_path.is_file():
        raise FileNotFoundError(f"Missing real embeddings for {model_key}: {emb_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing embedding event metadata: {metadata_path}")
    if not yeast_metadata_csv.is_file():
        raise FileNotFoundError(f"Missing full yeast metadata for source_group join: {yeast_metadata_csv}")
    aligned_4096 = real_embedding_root / "aligned_inputs.npz"
    aligned_512 = real_embedding_root / "aligned_512_inputs.npz"
    aligned_path = aligned_4096 if aligned_4096.is_file() else aligned_512
    if aligned_path.is_file():
        with np.load(aligned_path, allow_pickle=True) as aligned:
            actual_length = int(aligned["signals"].shape[1])
        if actual_length != int(expected_input_length):
            raise ValueError(
                f"Real embedding root {real_embedding_root} uses {actual_length}-sample inputs, "
                f"but this validation requires {expected_input_length}-sample inputs."
            )

    with np.load(emb_path) as data:
        embeddings = data["embeddings"].astype(np.float32)
        event_ids = data["event_id"].astype(str)

    events = pd.read_csv(metadata_path)
    full_yeast = pd.read_csv(yeast_metadata_csv)
    join_cols = [
        col
        for col in (
            "event_id",
            "source_group",
            "quality",
            "width_ms",
            "snr_proxy",
            "doppler_peak_hz",
            "phase_coherence",
            "energy_concentration",
        )
        if col in full_yeast.columns
    ]
    events = events.merge(full_yeast[join_cols].drop_duplicates("event_id"), on="event_id", how="left")
    order = pd.DataFrame({"event_id": event_ids, "embedding_index": np.arange(event_ids.size)})
    meta = order.merge(events, on="event_id", how="left", validate="one_to_one")
    if len(meta) != embeddings.shape[0]:
        raise ValueError(f"Metadata/embedding length mismatch for {model_key}")

    real_mask = meta["source_group"].fillna("").eq(source_group).to_numpy()
    if int(real_mask.sum()) < min_real:
        counts = meta["source_group"].fillna("unjoined_or_particle").value_counts().to_dict()
        raise ValueError(
            f"Only {int(real_mask.sum())} real source_group={source_group!r} events found in {real_embedding_root}; "
            f"need at least {min_real}. Joined source_group counts: {counts}"
        )
    real = EmbeddingGroup(f"real_{source_group}", embeddings[real_mask], meta.loc[real_mask].reset_index(drop=True))

    controls: dict[str, EmbeddingGroup] = {}
    yeast_mask = meta["class_name"].fillna("").eq("yeast").to_numpy()
    non_budding_mask = yeast_mask & ~real_mask
    if int(non_budding_mask.sum()) >= min_real:
        controls["non_budding_yeast"] = EmbeddingGroup(
            "non_budding_yeast",
            embeddings[non_budding_mask],
            meta.loc[non_budding_mask].reset_index(drop=True),
        )
    return real, controls


def load_event_root_groups(
    event_root: Path,
    source_group: str,
    min_real: int,
    expected_input_length: int,
) -> tuple[dict[str, tuple[np.ndarray, pd.DataFrame]], dict[str, tuple[np.ndarray, pd.DataFrame]]]:
    aligned_path = event_root / "aligned_inputs.npz"
    metadata_path = event_root / "events_metadata.csv"
    if not aligned_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing real event root inputs at {event_root}. Expected aligned_inputs.npz and events_metadata.csv. "
            "Regenerate budding events with: "
            f"python3 scripts/build_yeast_event_dataset.py --input-dir {DEFAULT_BUDDING_RAW_DIR.parent} "
            f"--include-groups budding --output-dir {event_root} --output-length 4096 --quality strict"
        )
    meta = pd.read_csv(metadata_path)
    with np.load(aligned_path, allow_pickle=True) as data:
        signals = np.asarray(data["signals"], dtype=np.float32)
        event_ids = data["event_id"].astype(str) if "event_id" in data.files else meta["event_id"].astype(str).to_numpy()
        input_preprocessing_id = str(data["preprocessing_id"]) if "preprocessing_id" in data.files else ""
    if signals.shape[1] != int(expected_input_length):
        raise ValueError(
            f"Real event root {event_root} uses {signals.shape[1]}-sample inputs; expected {expected_input_length}."
        )
    if signals.shape[0] != len(meta):
        raise ValueError(f"Event metadata/input row mismatch in {event_root}: {len(meta)} rows vs {signals.shape[0]} signals")
    meta = meta.reset_index(drop=True).copy()
    if "event_id" not in meta.columns:
        meta.insert(0, "event_id", event_ids)
    elif not np.array_equal(meta["event_id"].astype(str).to_numpy(), event_ids):
        meta["event_id_from_aligned_inputs"] = event_ids
    if input_preprocessing_id:
        meta["__input_preprocessing_id"] = input_preprocessing_id
    meta.insert(0, "signal_index", np.arange(event_ids.size))
    real_mask = meta["source_group"].fillna("").eq(source_group).to_numpy()
    if int(real_mask.sum()) < min_real:
        counts = meta["source_group"].fillna("unknown").value_counts().to_dict()
        raise ValueError(f"Only {int(real_mask.sum())} source_group={source_group!r} events in {event_root}; need {min_real}. Counts: {counts}")
    real = {f"real_{source_group}": (signals[real_mask], meta.loc[real_mask].reset_index(drop=True))}
    controls: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    non_budding = meta["class_name"].fillna("").eq("yeast").to_numpy() & ~real_mask
    if int(non_budding.sum()) >= min_real:
        controls["non_budding_yeast"] = (signals[non_budding], meta.loc[non_budding].reset_index(drop=True))
    return real, controls


def encode_signal_group(
    args: argparse.Namespace,
    model_key: str,
    signals: np.ndarray,
    metadata: pd.DataFrame,
    cache_path: Path,
    model_dir: Path,
    group_name: str,
    provenance: dict[str, Any],
) -> EmbeddingGroup:
    is_synthetic_group = str(provenance.get("source", "")).startswith("synthetic")
    force_encode = bool(args.force_encode_synthetic) if is_synthetic_group else bool(getattr(args, "force_encode_real", False))
    if signals.ndim != 2:
        raise ValueError(f"{group_name} signals must be a 2D array, got shape {signals.shape}")
    if cache_path.is_file() and not force_encode:
        with np.load(cache_path, allow_pickle=True) as data:
            validate_cache_preprocessing(cache_path, data, args)
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            cached_length = int(data["input_length"]) if "input_length" in data.files else None
        if embeddings.shape[0] != signals.shape[0]:
            raise ValueError(f"Cached embedding row mismatch for {group_name}: {embeddings.shape[0]} vs {signals.shape[0]}")
        if cached_length is not None and cached_length != signals.shape[1]:
            raise ValueError(f"Cached input length mismatch for {group_name}: {cached_length} vs {signals.shape[1]}")
        return EmbeddingGroup(
            group_name,
            embeddings,
            metadata,
            signals=signals,
            provenance={**provenance, "embedding_source": "cache", "cache_path": str(cache_path), "encoding_exercised": False},
        )
    if args.no_encode_synthetic and is_synthetic_group:
        raise FileNotFoundError(f"Missing cached synthetic/control embeddings for {model_key}: {cache_path}")

    torch, latent_sweeps = import_encoding_stack()
    device = torch.device(args.device)
    model_dir.mkdir(parents=True, exist_ok=True)
    encode_args = argparse.Namespace(
        input_length=int(signals.shape[1]),
        moment_model_id=args.moment_model_id,
        patchtst_model_id=args.patchtst_model_id,
        cache_dir=args.cache_dir,
        conv1dgap_checkpoint=args.conv1dgap_checkpoint,
    )
    encoder, model_metadata = latent_sweeps.load_encoder_for_model(model_key, encode_args, device, model_dir)
    if model_key == "conv1dgap_same_input_3class":
        embeddings = latent_sweeps.encode_conv_features_all(encoder, signals, args.batch_size, device)
    else:
        embeddings = latent_sweeps.backbone.encode_all_events(model_key, encoder, signals, args.batch_size, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings.astype(np.float32),
        event_id=metadata["event_id"].astype(str).to_numpy() if "event_id" in metadata.columns else np.arange(len(metadata)).astype(str),
        input_length=np.asarray(int(signals.shape[1]), dtype=np.int64),
        preprocessing_id=np.asarray(preprocessing_config_from_args(args).preprocessing_id),
    )
    with (model_dir / f"{group_name}_embedding_metadata.json").open("w") as f:
        json.dump({**model_metadata, **provenance, "encoding_exercised": True}, f, indent=2, sort_keys=True)
    return EmbeddingGroup(
        group_name,
        embeddings.astype(np.float32),
        metadata,
        signals=signals,
        provenance={**provenance, "embedding_source": "encoded", "cache_path": str(cache_path), "encoding_exercised": True},
    )


def load_or_encode_real(
    args: argparse.Namespace,
    model_key: str,
    output_dir: Path,
) -> tuple[EmbeddingGroup, dict[str, EmbeddingGroup]]:
    if preprocessing_config_from_args(args).mode == PREPROCESS_NONE:
        try:
            return load_real_groups(
                args.real_embedding_root,
                args.yeast_metadata_csv,
                model_key=model_key,
                source_group=args.source_group,
                min_real=args.min_real,
                expected_input_length=args.expected_input_length,
            )
        except Exception as exc:
            embedding_error = f"{type(exc).__name__}: {exc}"
    else:
        embedding_error = "real_embedding_root skipped because --preprocess-mode requires re-encoding from event signals"

    real_signal_groups, control_signal_groups = load_event_root_groups(
        args.real_event_root,
        args.source_group,
        args.min_real,
        args.expected_input_length,
    )
    real_signals, real_meta = real_signal_groups[f"real_{args.source_group}"]
    real_signals, real_meta, real_preprocessing = maybe_preprocess_signals(real_signals, real_meta, args, reject_saturation=True)
    real = encode_signal_group(
        args,
        model_key,
        real_signals,
        real_meta,
        output_dir / "real_embeddings" / model_key / f"real_{args.source_group}_embeddings.npz",
        output_dir / "real_embeddings" / model_key,
        f"real_{args.source_group}",
        {
            "source": "real_event_root",
            "event_root": str(args.real_event_root),
            "fallback_from_real_embedding_root_error": embedding_error,
            "input_length": int(real_signals.shape[1]),
            "preprocessing": real_preprocessing,
        },
    )
    controls: dict[str, EmbeddingGroup] = {}
    for name, (signals, meta) in control_signal_groups.items():
        signals, meta, control_preprocessing = maybe_preprocess_signals(signals, meta, args, reject_saturation=True)
        controls[name] = encode_signal_group(
            args,
            model_key,
            signals,
            meta,
            output_dir / "real_embeddings" / model_key / f"{name}_embeddings.npz",
            output_dir / "real_embeddings" / model_key,
            name,
            {
                "source": "real_event_root_control",
                "event_root": str(args.real_event_root),
                "input_length": int(signals.shape[1]),
                "preprocessing": control_preprocessing,
            },
        )
    return real, controls


def load_synthetic_metadata(synthetic_root: Path) -> pd.DataFrame:
    metadata_path = synthetic_root / "synthetic_metadata.csv"
    events_path = synthetic_root / "synthetic_events_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing synthetic metadata: {metadata_path}")
    meta = pd.read_csv(metadata_path)
    if events_path.is_file():
        events = pd.read_csv(events_path)
        if len(events) == len(meta):
            meta = pd.concat([events.add_prefix("event_"), meta], axis=1)
            meta["event_id"] = meta["event_event_id"]
    if "event_id" not in meta.columns:
        meta["event_id"] = [f"synthetic/{i:06d}" for i in range(len(meta))]
    return meta


def load_synthetic_signals(synthetic_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    signals_path = synthetic_root / "synthetic_signals_encoded.npz"
    if not signals_path.is_file():
        raise FileNotFoundError(f"Missing encoded synthetic signals: {signals_path}")
    meta = load_synthetic_metadata(synthetic_root)
    chunks: list[np.ndarray] = []
    input_preprocessing_id = ""
    with np.load(signals_path) as data:
        input_preprocessing_id = str(data["preprocessing_id"]) if "preprocessing_id" in data.files else ""
        panels = meta["panel"].drop_duplicates().tolist() if "panel" in meta.columns else []
        for panel in panels:
            key = f"{panel}_signals"
            if key in data:
                chunks.append(data[key].astype(np.float32))
    if not chunks:
        raise ValueError(f"No '*_signals' arrays found in {signals_path}")
    signals = np.concatenate(chunks, axis=0)
    if signals.shape[0] != len(meta):
        raise ValueError(f"Synthetic signal count {signals.shape[0]} does not match metadata rows {len(meta)}")
    if input_preprocessing_id:
        meta["__input_preprocessing_id"] = input_preprocessing_id
    return signals, meta


def load_optional_synthetic_signals(synthetic_root: Path) -> tuple[np.ndarray | None, pd.DataFrame]:
    meta = load_synthetic_metadata(synthetic_root)
    signals_path = synthetic_root / "synthetic_signals_encoded.npz"
    if not signals_path.is_file():
        return None, meta
    signals, meta = load_synthetic_signals(synthetic_root)
    return signals, meta


def load_or_encode_synthetic(
    args: argparse.Namespace,
    model_key: str,
    synthetic_root: Path,
    output_dir: Path,
    group_name: str = "template_budding_v1",
    cache_label: str | None = None,
) -> EmbeddingGroup:
    if cache_label is None:
        cache_path = output_dir / "synthetic_embeddings" / model_key / "synthetic_embeddings.npz"
        model_dir = output_dir / "synthetic_embeddings" / model_key
    else:
        cache_path = output_dir / "synthetic_embeddings" / cache_label / model_key / "synthetic_embeddings.npz"
        model_dir = output_dir / "synthetic_embeddings" / cache_label / model_key
    signals, meta = load_optional_synthetic_signals(synthetic_root)
    if cache_path.is_file() and not args.force_encode_synthetic:
        with np.load(cache_path) as data:
            validate_cache_preprocessing(cache_path, data, args)
            return EmbeddingGroup(
                group_name,
                data["embeddings"].astype(np.float32),
                meta,
                signals=signals,
                provenance={"source": "synthetic_cache", "root": str(synthetic_root), "cache_path": str(cache_path), "encoding_exercised": False},
            )

    if args.no_encode_synthetic:
        raise FileNotFoundError(f"Missing cached synthetic embeddings for {model_key}: {cache_path}")
    if signals is None:
        raise FileNotFoundError(f"Missing encoded synthetic signals for {group_name}: {synthetic_root / 'synthetic_signals_encoded.npz'}")
    signals, meta, synth_preprocessing = maybe_preprocess_signals(
        signals,
        meta,
        args,
        reject_saturation=bool(getattr(args, "preprocess_reject_synthetic_saturation", False)),
    )

    return encode_signal_group(
        args,
        model_key,
        signals,
        meta,
        cache_path,
        model_dir,
        group_name,
        {
            "source": "synthetic_root",
            "root": str(synthetic_root),
            "input_length": int(signals.shape[1]),
            "preprocessing": synth_preprocessing,
        },
    )


def standardize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    combined = scaler.fit_transform(np.vstack([a, b])).astype(np.float32)
    return combined[: len(a)], combined[len(a) :]


def nearest_self_distances(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.full(len(x), np.nan, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=2).fit(x)
    dist, _ = nn.kneighbors(x)
    return dist[:, 1].astype(np.float32)


def kth_self_radius(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < 2:
        return np.full(len(x), np.nan, dtype=np.float32)
    n_neighbors = min(max(2, k + 1), len(x))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
    dist, _ = nn.kneighbors(x)
    return dist[:, -1].astype(np.float32)


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=1).fit(reference)
    dist, idx = nn.kneighbors(query)
    return dist[:, 0].astype(np.float32), idx[:, 0].astype(np.int64)


def mmd_rbf(a: np.ndarray, b: np.ndarray) -> float:
    max_fit = min(1500, len(a), len(b))
    if len(a) > max_fit:
        a = a[np.linspace(0, len(a) - 1, max_fit, dtype=int)]
    if len(b) > max_fit:
        b = b[np.linspace(0, len(b) - 1, max_fit, dtype=int)]
    sample = np.vstack([a, b])
    nn = NearestNeighbors(n_neighbors=min(2, len(sample))).fit(sample)
    dist, _ = nn.kneighbors(sample)
    sigma = float(np.median(dist[:, -1]))
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    aa = np.exp(-gamma * np.square(np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1))).mean()
    bb = np.exp(-gamma * np.square(np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1))).mean()
    ab = np.exp(-gamma * np.square(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1))).mean()
    return float(aa + bb - 2.0 * ab)


def domain_auc(a: np.ndarray, b: np.ndarray, seed: int, repeats: int = 3) -> tuple[float, str]:
    x = np.vstack([a, b])
    y = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
    min_class = int(np.bincount(y).min())
    if min_class < 3:
        return float("nan"), "too_few_samples"
    n_splits = min(5, min_class)
    n_repeats = max(1, int(repeats)) if min_class >= 10 else 1
    aucs: list[float] = []
    for repeat in range(n_repeats):
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=seed + repeat, n_jobs=-1)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        proba = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(float(roc_auc_score(y, proba)))
    reliability = "ok" if min_class >= 50 and n_repeats > 1 else "low"
    return float(np.mean(aucs)), reliability


def support_metrics(real: np.ndarray, other: np.ndarray, k: int, seed: int) -> dict[str, Any]:
    r, s = standardize_pair(real, other)
    real_nn = nearest_self_distances(r)
    baseline = float(np.nanmedian(real_nn))
    if not math.isfinite(baseline) or baseline <= 0:
        baseline = 1.0
    real_radius = kth_self_radius(r, k)
    real_to_synth, _ = nearest_distances(r, s)
    synth_to_real, synth_nn_idx = nearest_distances(s, r)
    real_cover_radius = np.nanmedian(real_radius)
    covered_by_synth = float(np.mean(real_to_synth <= real_radius)) if len(real_radius) else float("nan")
    inside_real = float(np.mean(synth_to_real <= real_radius[synth_nn_idx])) if len(synth_to_real) else float("nan")
    auc, auc_reliability = domain_auc(r, s, seed=seed)
    return {
        "n_real": float(len(real)),
        "n_other": float(len(other)),
        "sample_count_warning": "low_n_real" if len(real) < 50 else "",
        "real_real_nn_median": float(np.nanmedian(real_nn)),
        "real_to_synth_nn_median": float(np.median(real_to_synth)),
        "synth_to_real_nn_median": float(np.median(synth_to_real)),
        "real_to_synth_nn_ratio": float(np.median(real_to_synth) / baseline),
        "synth_to_real_nn_ratio": float(np.median(synth_to_real) / baseline),
        "real_covered_by_synth": covered_by_synth,
        "synth_inside_real_support": inside_real,
        "real_knn_radius_median": float(real_cover_radius),
        "domain_auc": auc,
        "domain_auc_reliability": auc_reliability,
        "mmd_rbf": mmd_rbf(r, s),
    }


def split_real_baseline(real: EmbeddingGroup, seed: int, k: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(real.embeddings))
    half = len(idx) // 2
    if half < 3:
        return {}
    return support_metrics(real.embeddings[idx[:half]], real.embeddings[idx[half:]], k=k, seed=seed)


def convex_hull_iou(a: np.ndarray, b: np.ndarray, bins: int = 80) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    points = np.vstack([a, b])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1.0e-6)
    gx, gy = np.meshgrid(
        np.linspace(mins[0], maxs[0], bins),
        np.linspace(mins[1], maxs[1], bins),
        indexing="xy",
    )
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    try:
        from matplotlib.path import Path as MplPath

        ha = ConvexHull(a)
        hb = ConvexHull(b)
        ma = MplPath(a[ha.vertices]).contains_points(grid)
        mb = MplPath(b[hb.vertices]).contains_points(grid)
    except Exception:
        return float("nan")
    union = np.logical_or(ma, mb).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(ma, mb).sum() / union)


def projection_coords(real: np.ndarray, synth: np.ndarray, seed: int, skip_tsne: bool) -> dict[str, np.ndarray]:
    r, s = standardize_pair(real, synth)
    x = np.vstack([r, s])
    pca = PCA(n_components=2, random_state=seed).fit_transform(x).astype(np.float32)
    if skip_tsne or len(x) < 5:
        tsne = pca.copy()
    else:
        pre_dim = min(50, x.shape[1], len(x) - 1)
        x_pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(x) if pre_dim < x.shape[1] else x
        perplexity = min(30, max(2, (len(x) - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(x_pre).astype(np.float32)
    return {
        "pca_real": pca[: len(real)],
        "pca_synth": pca[len(real) :],
        "tsne_real": tsne[: len(real)],
        "tsne_synth": tsne[len(real) :],
    }


def plot_latent_overlap(
    output_base: Path,
    coords_by_model: dict[str, dict[str, np.ndarray]],
) -> None:
    n = len(coords_by_model)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.6), squeeze=False)
    for col, (model_key, coords) in enumerate(coords_by_model.items()):
        for row, prefix in enumerate(("pca", "tsne")):
            ax = axes[row, col]
            ax.scatter(coords[f"{prefix}_real"][:, 0], coords[f"{prefix}_real"][:, 1], s=10, c="#0072B2", alpha=0.65, linewidths=0, label="real budding")
            ax.scatter(coords[f"{prefix}_synth"][:, 0], coords[f"{prefix}_synth"][:, 1], s=10, c="#D55E00", alpha=0.55, linewidths=0, label="template_budding_v1")
            ax.set_title(f"{MODEL_DISPLAY.get(model_key, model_key)} {prefix.upper()}", fontsize=10)
            ax.tick_params(labelsize=7, length=2)
            if col == 0 and row == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


def plot_parameter_bins(output_base: Path, bin_rows: pd.DataFrame) -> None:
    if bin_rows.empty:
        return
    models = bin_rows["model"].drop_duplicates().tolist()
    params = bin_rows["parameter"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(models), len(params), figsize=(3.3 * len(params), 2.4 * len(models)), squeeze=False)
    for r, model in enumerate(models):
        for c, param in enumerate(params):
            ax = axes[r, c]
            sub = bin_rows[(bin_rows["model"] == model) & (bin_rows["parameter"] == param)].sort_values("bin_mid")
            if not sub.empty:
                ax.plot(sub["bin_mid"], sub["median_synth_to_real_ratio"], marker="o", ms=3, lw=1.1, c="#009E73")
                ax.set_title(f"{MODEL_DISPLAY.get(model, model)}\n{param}", fontsize=8)
            ax.tick_params(labelsize=6, length=2)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


def parameter_bin_rows(
    model_key: str,
    real: np.ndarray,
    synth: EmbeddingGroup,
    k: int,
    n_bins: int,
) -> list[dict[str, Any]]:
    r, s = standardize_pair(real, synth.embeddings)
    real_nn = nearest_self_distances(r)
    baseline = float(np.nanmedian(real_nn))
    if not math.isfinite(baseline) or baseline <= 0:
        baseline = 1.0
    dist, _ = nearest_distances(s, r)
    rows: list[dict[str, Any]] = []
    meta = synth.metadata.reset_index(drop=True)
    params = [p for p in PARAM_COLUMNS if p in set(meta.get("panel", pd.Series(dtype=str)))]
    if not params and "panel" in meta.columns:
        params = meta["panel"].dropna().unique().tolist()
    for param in params:
        mask = meta["panel"].eq(param).to_numpy() if "panel" in meta.columns else np.zeros(len(meta), dtype=bool)
        if not mask.any():
            continue
        values = meta.loc[mask, "color_value"].astype(float).to_numpy()
        edges = np.linspace(np.nanmin(values), np.nanmax(values), n_bins + 1)
        if np.unique(edges).size < 2:
            continue
        bin_id = np.clip(np.digitize(values, edges[1:-1], right=False), 0, n_bins - 1)
        for b in range(n_bins):
            bmask_local = bin_id == b
            if not bmask_local.any():
                continue
            global_idx = np.where(mask)[0][bmask_local]
            rows.append(
                {
                    "model": model_key,
                    "parameter": param,
                    "bin": b,
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "bin_mid": float((edges[b] + edges[b + 1]) / 2.0),
                    "n": int(global_idx.size),
                    "median_synth_to_real_distance": float(np.median(dist[global_idx])),
                    "median_synth_to_real_ratio": float(np.median(dist[global_idx]) / baseline),
                }
            )
    return rows


def plot_nearest_examples(
    output_base: Path,
    real: EmbeddingGroup,
    synth: EmbeddingGroup,
    coords_by_model: dict[str, dict[str, np.ndarray]],
    max_examples: int,
) -> None:
    if not coords_by_model:
        return
    model_key = next(iter(coords_by_model))
    r, s = standardize_pair(real.embeddings, synth.embeddings)
    baseline = float(np.nanmedian(nearest_self_distances(r)))
    if not math.isfinite(baseline) or baseline <= 0:
        baseline = 1.0
    dist, idx = nearest_distances(s, r)
    half = max(1, max_examples // 2)
    order = np.unique(np.concatenate([np.argsort(dist)[:half], np.argsort(dist)[-half:]])).astype(np.int64)
    coords = coords_by_model[model_key]

    rows: list[dict[str, Any]] = []
    for rank, synth_i in enumerate(order.tolist()):
        real_i = int(idx[synth_i])
        synth_row = synth.metadata.iloc[int(synth_i)] if int(synth_i) < len(synth.metadata) else pd.Series(dtype=object)
        real_row = real.metadata.iloc[real_i] if real_i < len(real.metadata) else pd.Series(dtype=object)
        rows.append(
            {
                "rank": int(rank),
                "model": model_key,
                "synthetic_index": int(synth_i),
                "real_index": real_i,
                "synthetic_event_id": str(synth_row.get("event_id", synth_row.get("event_event_id", synth_i))),
                "real_event_id": str(real_row.get("event_id", real_i)),
                "panel": str(synth_row.get("panel", "")),
                "color_value": finite_float(synth_row.get("color_value", float("nan"))),
                "nearest_distance": float(dist[synth_i]),
                "nearest_distance_ratio": float(dist[synth_i] / baseline),
                "group": "closest" if rank < half else "farthest",
            }
        )
    pd.DataFrame(rows).to_csv(output_base.with_suffix(".csv"), index=False)

    if real.signals is not None and synth.signals is not None:
        n = len(order)
        fig, axes = plt.subplots(n, 3, figsize=(10.8, max(2.0 * n, 4.8)), squeeze=False)
        for rank, synth_i in enumerate(order.tolist()):
            real_i = int(idx[synth_i])
            color = "#009E73" if rank < half else "#D55E00"
            axes[rank, 0].plot(synth.signals[int(synth_i)], color="#D55E00", linewidth=0.75)
            axes[rank, 0].set_title(f"synthetic {int(synth_i)}", fontsize=8)
            axes[rank, 1].plot(real.signals[real_i], color="#0072B2", linewidth=0.75)
            axes[rank, 1].set_title(f"nearest real {real_i} | ratio {dist[synth_i] / baseline:.2f}", fontsize=8)
            ax = axes[rank, 2]
            ax.scatter(coords["pca_real"][:, 0], coords["pca_real"][:, 1], s=5, c="#B8B8B8", alpha=0.35, linewidths=0)
            ax.scatter(coords["pca_synth"][:, 0], coords["pca_synth"][:, 1], s=5, c="#E6A27A", alpha=0.25, linewidths=0)
            ax.scatter(coords["pca_synth"][synth_i, 0], coords["pca_synth"][synth_i, 1], s=30, c=color, marker="x")
            ax.scatter(coords["pca_real"][real_i, 0], coords["pca_real"][real_i, 1], s=28, facecolors="none", edgecolors=color)
            ax.plot(
                [coords["pca_synth"][synth_i, 0], coords["pca_real"][real_i, 0]],
                [coords["pca_synth"][synth_i, 1], coords["pca_real"][real_i, 1]],
                c=color,
                lw=0.6,
                alpha=0.75,
            )
            ax.set_title("PCA link", fontsize=8)
            for col in range(3):
                axes[rank, col].tick_params(labelsize=6, length=2)
        fig.tight_layout()
        fig.savefig(output_base.with_suffix(".pdf"))
        fig.savefig(output_base.with_suffix(".png"), dpi=220)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(coords["pca_real"][:, 0], coords["pca_real"][:, 1], s=8, c="#B8B8B8", alpha=0.5, linewidths=0)
    ax.scatter(coords["pca_synth"][:, 0], coords["pca_synth"][:, 1], s=8, c="#E6A27A", alpha=0.35, linewidths=0)
    for rank, synth_i in enumerate(order.tolist()):
        real_i = int(idx[synth_i])
        color = "#009E73" if rank < half else "#D55E00"
        ax.scatter(coords["pca_synth"][synth_i, 0], coords["pca_synth"][synth_i, 1], s=32, c=color, marker="x")
        ax.scatter(coords["pca_real"][real_i, 0], coords["pca_real"][real_i, 1], s=30, facecolors="none", edgecolors=color)
        ax.plot(
            [coords["pca_synth"][synth_i, 0], coords["pca_real"][real_i, 0]],
            [coords["pca_synth"][synth_i, 1], coords["pca_real"][real_i, 1]],
            c=color,
            lw=0.6,
            alpha=0.75,
        )
    ax.set_title(f"Nearest real examples in {MODEL_DISPLAY.get(model_key, model_key)} PCA", fontsize=10)
    ax.tick_params(labelsize=7, length=2)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


def verdict_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [r for r in rows if r["comparison"] == "real_budding_vs_template_budding_v1"]
    if not target:
        return {"decision": "not_evaluable", "reason": "No target comparison rows were computed."}
    aucs = [finite_float(r.get("domain_auc")) for r in target]
    aucs = [x for x in aucs if x is not None]
    ratios = [finite_float(r.get("synth_to_real_nn_ratio")) for r in target]
    ratios = [x for x in ratios if x is not None]
    convincing_auc = bool(aucs) and float(np.median(aucs)) < 0.75
    ratio_ok = bool(ratios) and float(np.median(ratios)) <= 2.0
    return {
        "decision": "pilot_ready" if convincing_auc and ratio_ok else "needs_range_restriction_or_generator_tuning",
        "median_domain_auc": float(np.median(aucs)) if aucs else None,
        "median_synth_to_real_nn_ratio": float(np.median(ratios)) if ratios else None,
        "rule": "pilot_ready requires median domain_auc < 0.75 and median synth_to_real_nn_ratio <= 2.0; compare controls manually.",
    }


def validate_matching_input_lengths(real: EmbeddingGroup, synth: EmbeddingGroup) -> None:
    if real.signals is None or synth.signals is None:
        return
    if real.signals.shape[1] != synth.signals.shape[1]:
        raise ValueError(
            f"Input length mismatch between {real.name} ({real.signals.shape[1]}) and "
            f"{synth.name} ({synth.signals.shape[1]}). Do not compare 512-sample artifacts with 4096-sample template signals."
        )


def best_bin_summary(bin_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if bin_rows.empty:
        return []
    best = (
        bin_rows.sort_values(["model", "parameter", "median_synth_to_real_ratio"])
        .groupby(["model", "parameter"], as_index=False)
        .head(1)
    )
    return best.to_dict(orient="records")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = preflight_device(args)
    models = parse_models(args.models)
    control_roots = discover_control_roots(args.control_synthetic_roots)
    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    coords_by_model: dict[str, dict[str, np.ndarray]] = {}
    provenance: dict[str, Any] = {}
    first_real: EmbeddingGroup | None = None
    first_synth: EmbeddingGroup | None = None

    for model_key in models:
        real, real_controls = load_or_encode_real(args, model_key, output_dir)
        synth = load_or_encode_synthetic(args, model_key, args.synthetic_root, output_dir)
        validate_matching_input_lengths(real, synth)
        provenance[f"{model_key}:real"] = real.provenance or {"source": "real_embedding_root", "root": str(args.real_embedding_root)}
        provenance[f"{model_key}:template_budding_v1"] = synth.provenance or {}
        if first_real is None:
            first_real = real
            first_synth = synth

        target_metrics = support_metrics(real.embeddings, synth.embeddings, k=args.knn_k, seed=args.seed)
        coords = projection_coords(real.embeddings, synth.embeddings, seed=args.seed, skip_tsne=args.skip_tsne)
        target_metrics["viz_pca_iou"] = convex_hull_iou(coords["pca_real"], coords["pca_synth"])
        target_metrics["viz_tsne_iou"] = convex_hull_iou(coords["tsne_real"], coords["tsne_synth"]) if not args.skip_tsne else float("nan")
        metric_rows.append({"model": model_key, "comparison": "real_budding_vs_template_budding_v1", **target_metrics})
        coords_by_model[model_key] = coords
        bin_rows.extend(parameter_bin_rows(model_key, real.embeddings, synth, args.knn_k, args.parameter_bins))

        upper = split_real_baseline(real, seed=args.seed, k=args.knn_k)
        if upper:
            metric_rows.append({"model": model_key, "comparison": "upper_bound_real_split", **upper})
        for name, control in real_controls.items():
            metrics = support_metrics(real.embeddings, control.embeddings, k=args.knn_k, seed=args.seed)
            metric_rows.append({"model": model_key, "comparison": f"lower_bound_real_vs_{name}", **metrics})
        for control_root in control_roots:
            label = control_root.name
            control = load_or_encode_synthetic(
                args,
                model_key,
                control_root,
                output_dir,
                group_name=label,
                cache_label=label,
            )
            validate_matching_input_lengths(real, control)
            provenance[f"{model_key}:{label}"] = control.provenance or {}
            metrics = support_metrics(real.embeddings, control.embeddings, k=args.knn_k, seed=args.seed)
            metric_rows.append({"model": model_key, "comparison": f"lower_bound_real_vs_{label}", **metrics})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "overlap_metrics.csv", index=False)
    bins_df = pd.DataFrame(bin_rows)
    if not bins_df.empty:
        bins_df.to_csv(output_dir / "parameter_overlap_by_bin.csv", index=False)

    plot_latent_overlap(output_dir / "latent_overlap_pca_tsne", coords_by_model)
    plot_parameter_bins(output_dir / "parameter_overlap_by_bin", bins_df)
    if first_real is not None and first_synth is not None:
        plot_nearest_examples(output_dir / "nearest_real_examples", first_real, first_synth, coords_by_model, args.nearest_examples)

    summary = {
        "config": {
            "synthetic_root": str(args.synthetic_root),
            "real_embedding_root": str(args.real_embedding_root),
            "real_event_root": str(args.real_event_root),
            "budding_raw_dir": str(args.budding_raw_dir),
            "yeast_metadata_csv": str(args.yeast_metadata_csv),
            "source_group": args.source_group,
            "models": models,
            "control_synthetic_roots": [str(path) for path in control_roots],
            "knn_k": int(args.knn_k),
            "min_real": int(args.min_real),
            "expected_input_length": int(args.expected_input_length),
            "preprocessing": preprocessing_config_from_args(args).to_dict(),
            "skip_tsne": bool(args.skip_tsne),
            "device": args.device,
            "require_cuda": bool(args.require_cuda),
        },
        "runtime": runtime,
        "provenance": provenance,
        "thresholds": {
            "target_domain_auc": 0.75,
            "target_real_vs_real_distance_multiple": "roughly 1.5x-2x",
        },
        "metrics": metrics_df.to_dict(orient="records"),
        "best_parameter_bins": best_bin_summary(bins_df),
        "decision": verdict_from_rows(metric_rows),
    }
    with (output_dir / "overlap_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Wrote yeast latent overlap validation to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate latent overlap between real budding yeast and template_budding_v1 synthetic events.")
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--real-embedding-root", type=Path, default=DEFAULT_REAL_EMBEDDING_ROOT)
    parser.add_argument("--real-event-root", type=Path, default=DEFAULT_REAL_EVENT_ROOT)
    parser.add_argument("--budding-raw-dir", type=Path, default=DEFAULT_BUDDING_RAW_DIR)
    parser.add_argument("--yeast-metadata-csv", type=Path, default=DEFAULT_YEAST_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default="moment_official,patchtst_pretrained,conv1dgap_same_input_3class")
    parser.add_argument(
        "--control-synthetic-roots",
        default=None,
        help="Optional comma-separated previous proof folders, e.g. original or budding_realistic, to compare as lower-bound controls.",
    )
    parser.add_argument("--source-group", default="budding")
    parser.add_argument("--min-real", type=int, default=20)
    parser.add_argument("--expected-input-length", type=int, default=4096)
    parser.add_argument("--preprocess-mode", choices=PREPROCESS_MODES, default=PREPROCESS_NONE)
    parser.add_argument("--preprocess-sampling-frequency-hz", type=float, default=2_000_000.0)
    parser.add_argument("--preprocess-low-khz", type=float, default=5.0)
    parser.add_argument("--preprocess-high-khz-max", type=float, default=100.0)
    parser.add_argument("--saturation-fmin-hz", type=float, default=7_000.0)
    parser.add_argument("--saturation-fmax-hz", type=float, default=80_000.0)
    parser.add_argument("--saturation-min-flat", type=int, default=500)
    parser.add_argument("--saturation-zero-threshold", type=float, default=1.0e-4)
    parser.add_argument("--saturation-guard-before", type=int, default=0)
    parser.add_argument("--saturation-guard-after", type=int, default=0)
    parser.add_argument("--preprocess-reject-synthetic-saturation", action="store_true")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--parameter-bins", type=int, default=6)
    parser.add_argument("--nearest-examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / ".cache" / "huggingface")
    parser.add_argument("--moment-model-id", default=MOMENT_DEFAULT_ID)
    parser.add_argument("--patchtst-model-id", default=PATCHTST_DEFAULT_ID)
    parser.add_argument("--conv1dgap-checkpoint", type=Path, default=DEFAULT_CONV_CHECKPOINT)
    parser.add_argument("--force-encode-synthetic", action="store_true")
    parser.add_argument("--force-encode-real", action="store_true")
    parser.add_argument("--no-encode-synthetic", action="store_true", help="Require cached synthetic embeddings in the output directory.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail early if --device cuda is requested but unavailable to this Python environment.")
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()
    if args.preprocess_mode == PREPROCESS_P1:
        if args.synthetic_root == DEFAULT_SYNTHETIC_ROOT and DEFAULT_SYNTHETIC_ROOT_P1_FILTERED.exists():
            args.synthetic_root = DEFAULT_SYNTHETIC_ROOT_P1_FILTERED
        if args.real_event_root == DEFAULT_REAL_EVENT_ROOT and DEFAULT_REAL_EVENT_ROOT_P1_FILTERED.exists():
            args.real_event_root = DEFAULT_REAL_EVENT_ROOT_P1_FILTERED
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = DEFAULT_OUTPUT_DIR_P1_FILTERED
    run(args)


if __name__ == "__main__":
    main()
