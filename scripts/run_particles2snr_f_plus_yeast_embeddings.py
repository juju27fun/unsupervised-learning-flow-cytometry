#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
P0_ROOT = REPO_ROOT / "P0"
for path_entry in (ROOT, P0_ROOT, Path(__file__).resolve().parent):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

import run_pretrained_backbone_embeddings as backbone
from p3_ssl.pretrained_backbones import MOMENT_DEFAULT_ID, PATCHTST_DEFAULT_ID, ParticleEvent

from run_particles2snr_f_3class_aligned_backbones import CONV_MODEL_KEY, balanced_visual_indices, encode_conv_features_all


CLASS_LABELS = {0: "2um", 1: "4um", 2: "10um", 3: "yeast"}
MODEL_KEYS = ("moment_official", "patchtst_pretrained", CONV_MODEL_KEY)


def configure_display_names() -> None:
    backbone.CLASS_LABELS.update(CLASS_LABELS)
    backbone.CLASS_COLORS[3] = "#CC79A7"
    backbone.MODEL_DISPLAY["moment_official"] = "MOMENT frozen pretrained"
    backbone.MODEL_DISPLAY["patchtst_pretrained"] = "PatchTST frozen pretrained"
    backbone.MODEL_DISPLAY[CONV_MODEL_KEY] = "Conv1D-GAP supervised 3-class"
    if CONV_MODEL_KEY not in backbone.COMPARISON_MODEL_ORDER:
        backbone.COMPARISON_MODEL_ORDER.insert(2, CONV_MODEL_KEY)


def read_particle_events(path: Path) -> list[ParticleEvent]:
    events: list[ParticleEvent] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            events.append(
                ParticleEvent(
                    event_id=row["event_id"],
                    sample_id=row["sample_id"],
                    split=row["split"],
                    signal_path=row["signal_path"],
                    label_path=row.get("label_path") or "",
                    class_id=int(row["class_id"]),
                    class_name=row["class_name"],
                    center_norm=float(row["center_norm"]),
                    width_norm=float(row["width_norm"]),
                    center_index=int(float(row["center_index"])),
                    crop_start=int(float(row["crop_start"])),
                    crop_end=int(float(row["crop_end"])),
                )
            )
    return events


def load_aligned_bundle(root: Path) -> tuple[list[ParticleEvent], np.ndarray, np.ndarray, np.ndarray]:
    events = read_particle_events(root / "events_metadata.csv")
    with np.load(root / "aligned_512_inputs.npz", allow_pickle=True) as data:
        signals = np.asarray(data["signals"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        split = np.asarray(data["split"]).astype(str)
    if signals.shape[0] != len(events) or labels.shape[0] != len(events):
        raise ValueError(f"Aligned bundle row mismatch in {root}")
    return events, signals, labels, split


def load_existing_embeddings(root: Path, model_key: str) -> np.ndarray:
    path = root / model_key / "all_embeddings.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing existing particle embeddings: {path}")
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data["embeddings"], dtype=np.float32)


def load_conv1dgap_3class_model(checkpoint_path: Path, model_name: str, input_length: int, device: torch.device) -> torch.nn.Module:
    from models import create_model

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {key: value for key, value in state.items() if key not in {"total_ops", "total_params"}}
    model = create_model(model_name, input_length=input_length, num_classes=3)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def encode_yeast_embeddings(
    model_key: str,
    yeast_signals: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = torch.device(args.device)
    if model_key == "moment_official":
        encoder, metadata = backbone.load_encoder(model_key, args.moment_model_id, args.cache_dir, device, output_dir, args)
        return backbone.encode_all_events(model_key, encoder, yeast_signals, args.batch_size, device), metadata
    if model_key == "patchtst_pretrained":
        encoder, metadata = backbone.load_encoder(model_key, args.patchtst_model_id, args.cache_dir, device, output_dir, args)
        return backbone.encode_all_events(model_key, encoder, yeast_signals, args.batch_size, device), metadata
    if model_key == CONV_MODEL_KEY:
        model = load_conv1dgap_3class_model(args.conv_checkpoint, args.conv_model_name, yeast_signals.shape[1], device)
        metadata = {
            "source_model_id": str(args.conv_checkpoint),
            "model_name": args.conv_model_name,
            "input_representation": "same 512-sample aligned tensor as MOMENT/PatchTST",
            "input_length": int(yeast_signals.shape[1]),
            "supervised_same_input_checkpoint": True,
            "public_pretrained": False,
            "ood_note": "Checkpoint was trained only on 2um/4um/10um Particles2SNR_F classes; yeast is an external OOD group.",
        }
        return encode_conv_features_all(model, yeast_signals, args.batch_size, device), metadata
    raise ValueError(f"Unsupported model key: {model_key}")


def write_rows(path: Path, events: list[ParticleEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(events[0]).keys()))
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def run(args: argparse.Namespace) -> None:
    configure_display_names()
    args.event_length = int(args.input_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    particle_events, particle_signals, particle_labels, particle_split = load_aligned_bundle(args.particle_root)
    yeast_events, yeast_signals, yeast_labels, yeast_split = load_aligned_bundle(args.yeast_root)

    events = particle_events + yeast_events
    signals = np.concatenate([particle_signals, yeast_signals], axis=0).astype(np.float32)
    labels = np.concatenate([particle_labels, yeast_labels], axis=0).astype(np.int64)
    split = np.concatenate([particle_split, yeast_split], axis=0)
    write_rows(args.output_dir / "events_metadata.csv", events)
    np.savez_compressed(
        args.output_dir / "aligned_512_inputs.npz",
        signals=signals,
        labels=labels,
        split=split,
        event_id=np.asarray([event.event_id for event in events]),
    )

    visual_idx = balanced_visual_indices(labels, args.max_plot_per_class, args.seed)
    visual_events = [events[int(i)] for i in visual_idx]
    visual_labels = labels[visual_idx]
    write_rows(args.output_dir / "visual_events_metadata.csv", visual_events)

    model_dirs: dict[str, Path] = {}
    summary: dict[str, Any] = {
        "particle_root": str(args.particle_root),
        "yeast_root": str(args.yeast_root),
        "n_particle_events": int(particle_labels.shape[0]),
        "n_yeast_events": int(yeast_labels.shape[0]),
        "n_total_events": int(labels.shape[0]),
        "visual_n_events": int(visual_idx.size),
        "class_labels": {str(k): v for k, v in CLASS_LABELS.items()},
        "input_representation_all_models": "centered event crop raw 4096 -> mean decimate by 8 -> 512 -> window_zscore",
        "models": {},
    }

    for model_key in MODEL_KEYS:
        model_dir = args.output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        particle_embeddings = load_existing_embeddings(args.particle_root, model_key)
        if particle_embeddings.shape[0] != particle_labels.shape[0]:
            raise ValueError(f"Particle embedding row mismatch for {model_key}")
        yeast_embeddings, metadata = encode_yeast_embeddings(model_key, yeast_signals, args, model_dir)
        embeddings = np.concatenate([particle_embeddings, yeast_embeddings], axis=0).astype(np.float32)
        np.savez_compressed(
            model_dir / "all_embeddings.npz",
            embeddings=embeddings,
            labels=labels,
            split=split,
            event_id=np.asarray([event.event_id for event in events]),
        )
        visual_metadata = {
            **metadata,
            "model_key": model_key,
            "display_name": backbone.MODEL_DISPLAY[model_key],
            "feature_dim": int(embeddings.shape[1]),
            "n_classes": len(CLASS_LABELS),
            "class_labels": {str(k): v for k, v in CLASS_LABELS.items()},
            "actual_input_length": int(signals.shape[1]),
            "stage": "particles2snr_f_3class_plus_yeast_external_zero_shot",
            "particle_embeddings_reused_from": str(args.particle_root / model_key / "all_embeddings.npz"),
            "yeast_embeddings_encoded_from": str(args.yeast_root),
        }
        backbone.save_embedding_outputs(
            output_dir=model_dir / "zero_shot",
            events=visual_events,
            embeddings=embeddings[visual_idx],
            labels=visual_labels,
            seed=args.seed,
            title=f"{backbone.MODEL_DISPLAY[model_key]} - Particles2SNR_F + yeast",
            extra_metadata=visual_metadata,
        )
        model_dirs[model_key] = model_dir / "zero_shot"
        summary["models"][model_key] = {
            "feature_dim": int(embeddings.shape[1]),
            "yeast_embeddings": int(yeast_embeddings.shape[0]),
            "metadata": visual_metadata,
        }

    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    backbone.plot_pretrained_model_comparison(
        output_pdf=args.output_dir / "particles2snr_f_3class_plus_yeast_pca_tsne_fig7_style.pdf",
        output_png=args.output_dir / "particles2snr_f_3class_plus_yeast_pca_tsne_fig7_style.png",
        model_output_dirs=model_dirs,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "n_total_events": int(labels.shape[0])}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine Particles2SNR_F 3-class embeddings with detected yeast events for P3 zero-shot figures.")
    parser.add_argument("--particle-root", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap")
    parser.add_argument("--yeast-root", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "yeast_passage_events_p3_512")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "particles2snr_f_3class_plus_yeast_moment_patchtst_conv1dgap")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "hf_cache")
    parser.add_argument("--moment-model-id", default=MOMENT_DEFAULT_ID)
    parser.add_argument("--patchtst-model-id", default=PATCHTST_DEFAULT_ID)
    parser.add_argument("--conv-checkpoint", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap" / CONV_MODEL_KEY / "best_model.pt")
    parser.add_argument("--conv-model-name", default="Conv1DGAP-L")
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--max-plot-per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
