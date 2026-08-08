#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from p3_ssl.yeast_budding_sweep_domain import (
    PARAMETERS,
    canonical_parameter_row,
    domain_statistics,
    finalize_phase,
    fingerprint,
    read_csv,
    robust_medoid,
    sensitivity_rows,
    sha256_file,
    snr_db_from_signal,
)
from particles2snr.yeast_budding_simulation import compare_budding_models
from particles2snr.yeast_representation_dataset import clamped_crop, preprocess_crop


METHOD_EVIDENCE_ID = "yeast-budding-physical-sweep-domain-method-r1"


def _record(workspace: Workspace, dataset_id: str) -> dict[str, Any]:
    matches = [
        record.payload
        for record in load_records(workspace)
        if f"{record.payload['id']}@{record.payload['version']}" == dataset_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one registered dataset: {dataset_id}")
    if matches[0]["status"] not in {"active", "reference"}:
        raise ValueError(f"Dataset is not usable: {dataset_id}")
    return matches[0]


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_fit(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_id": payload["event_id"],
        "delta_bic_m1_minus_m2": payload["delta_bic_m1_minus_m2"],
        "resolvability_score": payload["resolvability_score"],
    }
    for model_name in ("m1", "m2"):
        fit = payload[model_name]
        row[f"{model_name}_bic"] = fit["bic"]
        row[f"{model_name}_envelope_residual_fraction"] = fit[
            "envelope_residual_fraction"
        ]
        row[f"{model_name}_waveform_residual_fraction"] = fit[
            "waveform_residual_fraction"
        ]
        for component_index, component in enumerate(fit["components"], start=1):
            for name, value in component.items():
                row[f"{model_name}_c{component_index}_{name}"] = value
    return row


def _gold_fit_task(
    payload: tuple[str, str, int, int, int, float, float],
) -> tuple[dict[str, Any], float]:
    event_id, raw_path, center, start, end, fit_mean, fit_std = payload
    raw = np.load(raw_path, allow_pickle=False)
    crop, crop_start = clamped_crop(raw, center, 8192)
    processed = preprocess_crop(crop)
    normalized = (processed - fit_mean) / fit_std
    event_start = (start - crop_start) / 2.0
    event_end = (end - crop_start) / 2.0
    fit = compare_budding_models(
        event_id,
        normalized,
        event_start_index=event_start,
        event_end_index=event_end,
    ).to_dict()
    target_signal = processed
    snr = snr_db_from_signal(
        target_signal,
        event_start_index=event_start,
        event_end_index=event_end,
    )
    return _flatten_fit(fit), snr


def _primary_rows(
    *,
    dataset_root: Path,
    fit_rows: list[dict[str, str]],
    amplitude_scale: float,
) -> tuple[list[dict[str, Any]], int]:
    samples = read_csv(dataset_root / "samples.csv")
    primary = [
        row
        for row in samples
        if row["class_name"] == "budding"
        and row["development_split"] == "development_train"
        and row["quality"] == "strict"
    ]
    if len(primary) != 680 or len({row["event_id"] for row in primary}) != 680:
        raise ValueError("Primary budding population must contain exactly 680 events")
    fit_by_id = {row["event_id"]: row for row in fit_rows}
    if not {row["event_id"] for row in primary} <= set(fit_by_id):
        raise ValueError("Historical fit table does not cover the current primary population")
    signals = np.load(dataset_root / "signals.npy", mmap_mode="r")
    rows: list[dict[str, Any]] = []
    for sample in primary:
        start = (float(sample["event_start"]) - float(sample["crop_start"])) / 2.0
        end = (float(sample["event_end"]) - float(sample["crop_start"])) / 2.0
        snr = snr_db_from_signal(
            np.asarray(signals[int(sample["signal_row"])]),
            event_start_index=start,
            event_end_index=end,
        )
        rows.append(
            canonical_parameter_row(
                fit_by_id[sample["event_id"]],
                snr_db=snr,
                amplitude_scale=amplitude_scale,
                population="primary_current_train",
            )
        )
    return rows, len(primary)


def _gold_rows(
    *,
    workspace: Workspace,
    gold_root: Path,
    fit_mean: float,
    fit_std: float,
    amplitude_scale: float,
    workers: int,
    max_events: int,
) -> tuple[list[dict[str, Any]], int]:
    events = read_csv(gold_root / "events.csv")
    if len(events) != 146 or any(row["training_allowed"] != "False" for row in events):
        raise ValueError("Gold inventory contract mismatch")
    selected = events[:max_events] if max_events > 0 else events
    tasks = [
        (
            row["human_event_id"],
            str((workspace.root / row["raw_signal_path"]).resolve()),
            int(row["center_sample"]),
            int(row["start_sample"]),
            int(row["end_sample"]),
            fit_mean,
            fit_std,
        )
        for row in selected
    ]
    if workers == 1:
        fitted = [_gold_fit_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fitted = list(executor.map(_gold_fit_task, tasks, chunksize=2))
    rows = [
        canonical_parameter_row(
            fit,
            snr_db=snr,
            amplitude_scale=amplitude_scale,
            population="gold_sensitivity_only",
        )
        for fit, snr in fitted
    ]
    return rows, len(events)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yeast-budding-mix-shmoo-background-classification@v2")
    parser.add_argument("--fit-dataset", default="yeast-events-representation@v3")
    parser.add_argument("--gold-dataset", default="yeast-budding-reviewed-event-inventory@v1")
    parser.add_argument("--fit-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gold-max-events", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.workers < 1 or args.gold_max_events < 0:
        raise ValueError("Invalid worker or gold event count")

    workspace = Workspace.load()
    primary_record = _record(workspace, args.dataset)
    fit_record = _record(workspace, args.fit_dataset)
    gold_record = _record(workspace, args.gold_dataset)
    primary_root = workspace.datasets_root / primary_record["path"]
    fit_root = workspace.datasets_root / fit_record["path"]
    gold_root = workspace.datasets_root / gold_record["path"]
    primary_contract = json.loads((primary_root / "dataset-contract.json").read_text())
    fit_contract = json.loads((fit_root / "input_contract.json").read_text())
    target_std = float(primary_contract["normalization"]["std"])
    fit_mean = float(fit_contract["normalization"]["mean"])
    fit_std = float(fit_contract["normalization"]["std"])
    if float(primary_contract["raw_crop_length"]) != 8192 or int(primary_contract["output_length"]) != 4096:
        raise ValueError("Unexpected current input contract")
    if float(fit_contract["output_duration_ms"]) != 4.096:
        raise ValueError("Historical fit contract is not the expected 4.096 ms contract")
    amplitude_scale = fit_std / target_std
    fit_path = args.fit_run_dir / "fit_summaries.csv"
    fit_rows = read_csv(fit_path)
    primary_rows, primary_total = _primary_rows(
        dataset_root=primary_root,
        fit_rows=fit_rows,
        amplitude_scale=amplitude_scale,
    )
    gold_rows, gold_total = _gold_rows(
        workspace=workspace,
        gold_root=gold_root,
        fit_mean=fit_mean,
        fit_std=fit_std,
        amplitude_scale=amplitude_scale,
        workers=args.workers,
        max_events=args.gold_max_events,
    )
    primary_phase_center = finalize_phase(primary_rows)
    gold_observed_phase_center = float(
        np.angle(
            np.mean(
                np.exp(
                    1j
                    * np.asarray(
                        [row["delta_phi_rad"] for row in gold_rows if row["fit_valid"]],
                        dtype=np.float64,
                    )
                )
            )
        )
    )
    finalize_phase(gold_rows, reference_center=primary_phase_center)
    primary_stats, grids = domain_statistics(primary_rows)
    gold_stats, _gold_grids = domain_statistics(gold_rows)
    sensitivity = sensitivity_rows(primary_stats, gold_stats)
    anchor_id = robust_medoid(primary_rows)

    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "primary_parameter_values.csv", primary_rows)
    _write_csv(args.output_dir / "gold_parameter_values.csv", gold_rows)
    _write_csv(args.output_dir / "parameter_domain.csv", primary_stats)
    _write_csv(args.output_dir / "parameter_quantile_grid.csv", grids)
    _write_csv(args.output_dir / "gold_sensitivity.csv", sensitivity)
    anchor = next(row for row in primary_rows if row["event_id"] == anchor_id)
    (args.output_dir / "anchor_event.json").write_text(
        json.dumps(anchor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = {
        "schema_version": 1,
        "method_evidence_id": METHOD_EVIDENCE_ID,
        "population": {
            "primary_total": primary_total,
            "primary_valid": sum(bool(row["fit_valid"]) for row in primary_rows),
            "gold_inventory_total": gold_total,
            "gold_fitted": len(gold_rows),
            "gold_valid": sum(bool(row["fit_valid"]) for row in gold_rows),
        },
        "phase_center_rad": {
            "primary_reference": primary_phase_center,
            "gold_observed_diagnostic": gold_observed_phase_center,
            "gold_unwrap_reference": primary_phase_center,
        },
        "anchor_event_id": anchor_id,
        "parameters": list(PARAMETERS),
        "quantile_probabilities": np.linspace(0.01, 0.99, 31).tolist(),
        "amplitude_scale_fit_to_v2": amplitude_scale,
        "window_duration_ms": 4.096,
        "gold_in_primary_domain": False,
        "sealed_test_accessed": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "code": {
            "analysis": "scripts/analyze_yeast_budding_sweep_domain.py",
            "analysis_sha256": sha256_file(Path(__file__)),
            "domain_library": "p3_ssl/yeast_budding_sweep_domain.py",
            "domain_library_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "p3_ssl"
                / "yeast_budding_sweep_domain.py"
            ),
        },
        "datasets": {
            args.dataset: primary_record["manifest_sha256"],
            args.fit_dataset: fit_record["manifest_sha256"],
            args.gold_dataset: gold_record["manifest_sha256"],
        },
        "inputs": {
            "fit_summaries_sha256": sha256_file(fit_path),
            "fit_run_id": args.fit_run_dir.name,
        },
        "metric_definitions": {
            "primary_domain": (
                "Empirical q01, q50 and q99 among valid M2 fits from the 680 "
                "strict budding development_train events only."
            ),
            "quantile_grid": (
                "Thirty-one empirical quantiles uniformly spaced from q01 to q99 "
                "in each parameter's declared transformed coordinate."
            ),
            "gold_sensitivity": (
                "Human-certain budding events fitted identically and compared with "
                "the primary q01-q99 interval; never included in primary bounds."
            ),
            "fit_valid": "M2 delta-BIC >= 10 and resolvability >= 0.1.",
            "anchor": "Observed primary event nearest the robust scaled median vector.",
        },
        "parameters": {
            "resolved_rule": "delta-BIC >= 10 and resolvability >= 0.1",
            "component_order": "earlier component A, later component B",
            "quantiles": "31 empirical values from q01 to q99",
            "gold_policy": "sensitivity-only",
            "gold_max_events": args.gold_max_events,
        },
        "git_revision": {
            "workspace": _git_revision(workspace.root),
            "particles2SNR-pipeline": _git_revision(workspace.root / "particles2SNR-pipeline"),
            "unsupervised-learning-flow-cytometry": _git_revision(
                workspace.root / "unsupervised-learning-flow-cytometry"
            ),
        },
    }
    computation_fingerprint = fingerprint(provenance)
    manifest = {
        "schema_version": 1,
        "analysis_run_id": args.run_id,
        "computation_fingerprint": computation_fingerprint,
        "computation_provenance": provenance,
        "metrics": [
            {
                "path": name,
                "sha256": sha256_file(args.output_dir / name),
                "computation_fingerprint": computation_fingerprint,
            }
            for name in (
                "metrics.json",
                "primary_parameter_values.csv",
                "gold_parameter_values.csv",
                "parameter_domain.csv",
                "parameter_quantile_grid.csv",
                "gold_sensitivity.csv",
                "anchor_event.json",
            )
        ],
    }
    (args.output_dir / "metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method_evidence_id": METHOD_EVIDENCE_ID,
        "source_run_ids": [args.fit_run_dir.name],
        "dataset_ids": [args.dataset, args.fit_dataset, args.gold_dataset],
        "sealed_test_accessed": False,
        "outputs": [
            "primary_parameter_values.csv",
            "gold_parameter_values.csv",
            "parameter_domain.csv",
            "parameter_quantile_grid.csv",
            "gold_sensitivity.csv",
            "anchor_event.json",
            "metrics.json",
            "metrics_manifest.json",
        ],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
