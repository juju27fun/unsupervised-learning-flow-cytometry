#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-"$ROOT/../.venv/bin/python"}"
ARTIFACT_ROOT="${P3_SSL_ARTIFACT_ROOT:-"$ROOT/../artifacts/unsupervised-learning-flow-cytometry"}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_PER_PANEL="${N_PER_PANEL:-1800}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-"$ARTIFACT_ROOT/particle_equation_latent_sweeps"}"
MODELS="${MODELS:-moment_official,patchtst_pretrained,conv1dgap_same_input_3class}"
SKIP_TSNE="${SKIP_TSNE:-0}"
SINGLE_SWEEP_SOURCE="${SINGLE_SWEEP_SOURCE:-legacy}"
PHASE_PROFILE="${PHASE_PROFILE:-table_mean}"
SIGNAL_WINDOW_DURATION_MS="${SIGNAL_WINDOW_DURATION_MS:-2.048}"
REALISTIC_FIGURE_BASED_SWEEPS="${REALISTIC_FIGURE_BASED_SWEEPS:-0}"
YEAST_RANGE_SUMMARY="${YEAST_RANGE_SUMMARY:-"$ARTIFACT_ROOT/yeast_particle_range_analysis_budding/yeast_two_component_range_summary.json"}"
SIGNALS_ONLY="${SIGNALS_ONLY:-0}"
YEAST_BUDDED_STYLE="${YEAST_BUDDED_STYLE:-budding_noisy}"
YEAST_BACKGROUND_NOISE_SCALE="${YEAST_BACKGROUND_NOISE_SCALE:--1}"

extra_args=()
if [[ "$SKIP_TSNE" == "1" ]]; then
  extra_args+=(--skip-tsne)
fi
if [[ "$REALISTIC_FIGURE_BASED_SWEEPS" == "1" ]]; then
  extra_args+=(--realistic-figure-based-sweeps)
fi
if [[ "$SIGNALS_ONLY" == "1" ]]; then
  extra_args+=(--signals-only)
fi

run_scenario() {
  local scenario="$1"
  local output_name="$2"
  "$PYTHON" "$ROOT/scripts/run_particle_equation_latent_sweeps.py" \
    --scenario "$scenario" \
    --models "$MODELS" \
    --n-per-panel "$N_PER_PANEL" \
    --input-length 4096 \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --single-sweep-source "$SINGLE_SWEEP_SOURCE" \
    --phase-profile "$PHASE_PROFILE" \
    --signal-window-duration-ms "$SIGNAL_WINDOW_DURATION_MS" \
    --yeast-range-summary "$YEAST_RANGE_SUMMARY" \
    --yeast-budded-style "$YEAST_BUDDED_STYLE" \
    --yeast-background-noise-scale "$YEAST_BACKGROUND_NOISE_SCALE" \
    --output-dir "$BASE_OUTPUT_DIR/$output_name" \
    "${extra_args[@]}"
}

case "$MODE" in
  smoke)
    N_PER_PANEL="${SMOKE_N_PER_PANEL:-3}"
    BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
    DEVICE="${SMOKE_DEVICE:-cpu}"
    SKIP_TSNE=1
    extra_args=(--skip-tsne)
    run_scenario single_particle smoke_single
    ;;
  single)
    run_scenario single_particle single_n${N_PER_PANEL}
    ;;
  two)
    run_scenario two_particles two_n${N_PER_PANEL}
    ;;
  yeast-proof)
    SIGNALS_ONLY=1
    extra_args+=(--signals-only)
    run_scenario yeast_budded_two_particles yeast_budded_two_particle_proof
    ;;
  yeast-two)
    run_scenario yeast_budded_two_particles yeast_budded_two_n${N_PER_PANEL}
    ;;
  all)
    run_scenario single_particle single_n${N_PER_PANEL}
    run_scenario two_particles two_n${N_PER_PANEL}
    ;;
  *)
    echo "Usage: $0 {smoke|single|two|yeast-proof|yeast-two|all}" >&2
    echo "Optional env: N_PER_PANEL, BATCH_SIZE, DEVICE, SKIP_TSNE, MODELS, BASE_OUTPUT_DIR, SINGLE_SWEEP_SOURCE, PHASE_PROFILE, SIGNAL_WINDOW_DURATION_MS, REALISTIC_FIGURE_BASED_SWEEPS, YEAST_RANGE_SUMMARY, YEAST_BUDDED_STYLE, YEAST_BACKGROUND_NOISE_SCALE, SIGNALS_ONLY, PYTHON" >&2
    exit 2
    ;;
esac
