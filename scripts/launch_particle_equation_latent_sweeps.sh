#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-"$ROOT/../P0/venv/bin/python"}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_PER_PANEL="${N_PER_PANEL:-1800}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-"$ROOT/outputs/particle_equation_latent_sweeps"}"
MODELS="${MODELS:-moment_official,patchtst_pretrained,conv1dgap_same_input_3class}"
SKIP_TSNE="${SKIP_TSNE:-1}"

extra_args=()
if [[ "$SKIP_TSNE" == "1" ]]; then
  extra_args+=(--skip-tsne)
fi

run_scenario() {
  local scenario="$1"
  local output_name="$2"
  "$PYTHON" "$ROOT/scripts/run_particle_equation_latent_sweeps.py" \
    --scenario "$scenario" \
    --models "$MODELS" \
    --n-per-panel "$N_PER_PANEL" \
    --input-length 512 \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
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
  all)
    run_scenario single_particle single_n${N_PER_PANEL}
    run_scenario two_particles two_n${N_PER_PANEL}
    ;;
  *)
    echo "Usage: $0 {smoke|single|two|all}" >&2
    echo "Optional env: N_PER_PANEL, BATCH_SIZE, DEVICE, SKIP_TSNE, MODELS, BASE_OUTPUT_DIR, PYTHON" >&2
    exit 2
    ;;
esac
