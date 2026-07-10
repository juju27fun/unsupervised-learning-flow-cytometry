# P3_SSL - Self-Supervised Particle Signal Modeling

P3_SSL is isolated from the supervised P0/P1/P2 pipelines. It may read signals
and optional labels from those folders, but it writes every manifest, audit PDF,
checkpoint, metric file, and export under `P3_SSL/outputs/`.

The first experiment is a MOMENT-like masked reconstruction model:

- raw input length: 16384 samples
- decimation factor: 4
- SSL input length: 4096 samples
- patch size: 4
- patch stride: 4
- token count: 1024
- masking: learned `[MASK]` token, time-block masks, guard bands
- loss: signal MSE + derivative Huber + energy Huber

Labels are never used as reconstruction targets or class supervision. When
configured, they are used only to reject incoherent masks and to report
event/background reconstruction diagnostics and visualization.

## Canonical 4096 Pipeline

The current P3 representation is a centered 1D event input of `4096` samples,
normalized with `window_zscore`. New event datasets write `aligned_inputs.npz`
with `signals.shape == (n_events, 4096)`.

Older `aligned_512_inputs.npz` artifacts are legacy comparison outputs. They
remain readable where scripts explicitly support the fallback, but they should
not be treated as the canonical P3 input.

See [`docs/p3_4096_pipeline.md`](docs/p3_4096_pipeline.md) for the current
end-to-end commands for yeast, Particles2SNR_F, particles+yeast, latent sweeps,
and GPU execution from Codex.

## First Run

Build an independent manifest:

```bash
P0/venv/bin/python P3_SSL/scripts/build_ssl_manifest.py \
  --source P1/yolo_dataset_Particles2SNR_F_c1_4class_lim10_trainval \
  --output P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv
```

Create the patch/stride/masking audit PDF:

```bash
P0/venv/bin/python P3_SSL/scripts/visualize_patch_stride_masking.py \
  --manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --output P3_SSL/outputs/patch_stride_audit/patch_stride_masking_audit.pdf \
  --max-samples 8
```

Quantify whether sampled masks leave enough particle-passage context visible:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/audit_mask_coherence.py \
  --config P3_SSL/configs/p3_ssl_moment_v0.yaml \
  --manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --output-dir P3_SSL/outputs/mask_coherence_audit \
  --max-samples 500 \
  --masks-per-sample 2
```

Run a smoke training pass:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/train_ssl_reconstruction.py \
  --config P3_SSL/configs/p3_ssl_moment_v0.yaml \
  --manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --output-dir P3_SSL/outputs/runs/smoke \
  --epochs 1 \
  --batch-size 4 \
  --num-workers 0
```

Evaluate a checkpoint:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/evaluate_ssl_reconstruction.py \
  --config P3_SSL/configs/p3_ssl_moment_v0.yaml \
  --manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --checkpoint P3_SSL/outputs/runs/smoke/checkpoints/best.pt \
  --output-dir P3_SSL/outputs/eval/smoke
```

Export the encoder:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/export_ssl_encoder.py \
  --checkpoint P3_SSL/outputs/runs/smoke/checkpoints/best.pt \
  --output P3_SSL/outputs/exports/ssl_encoder_v0.pt
```

## Hybrid Physical Validation Pipeline

The canonical hybrid pipeline makes physical latent fidelity the primary
criterion. It generates synthetic P3 signals with known parameters, merges them
with real manifest rows, optionally enriches real rows from the
`particles2SNR_pipeline` event manifest, trains with masked reconstruction plus
physical contrastive and augmentation-invariance losses, and writes a physical
assessment first.

Smoke check on CPU:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/run_hybrid_physics_pipeline.py \
  --config P3_SSL/configs/p3_ssl_hybrid_physics.yaml \
  --real-manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --simulation-source internal,particles2snr_pipeline \
  --output-root P3_SSL/outputs/runs \
  --profile smoke \
  --device cpu
```

Full approval run on CUDA:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/preflight_hybrid_physics_run.py \
  --config P3_SSL/configs/p3_ssl_hybrid_physics.yaml \
  --real-manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --simulation-source internal,particles2snr_pipeline \
  --output-root P3_SSL/outputs/runs \
  --profile full \
  --device cuda
```

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/run_hybrid_physics_pipeline.py \
  --config P3_SSL/configs/p3_ssl_hybrid_physics.yaml \
  --real-manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --simulation-source internal,particles2snr_pipeline \
  --output-root P3_SSL/outputs/runs \
  --profile full \
  --device cuda
```

Re-run the assessment for an existing run:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/assess_hybrid_physics_run.py \
  --run-dir P3_SSL/outputs/runs/hybrid_physics_full_YYYYMMDD_HHMMSS
```

Each run writes `synthetic_manifest.csv`, `hybrid_manifest.csv`,
`training_history.json`, `physical_metrics.json`,
`physical_dashboard.{png,pdf}`, `reconstruction_metrics_val.json`,
`reconstruction_metrics_test.json`, `robustness_metrics.json`,
`real_estimated_physics_metrics.json`, `reconstruction_reference_comparison.json`,
`run_summary.{json,md}`, and `run_assessment.{json,md}`.
`real_estimated_physics_metrics.json` is a diagnostic correlation report for
partial particles2SNR-derived real-parameter estimates; `physical_metrics.json`
remains the primary physical validation artifact from known synthetic
parameters. Physical baseline rankings, including raw signal, random embedding,
pretrained backbone embeddings, and the reconstruction-only P3 SSL checkpoint,
live under `classic_assessment/physical_baselines/`. Classic label, retrieval,
and manifold diagnostics are secondary and live under `classic_assessment/`,
including `representation_manifold.{pdf,png}`,
`label_efficiency_summary.json`, `label_efficiency_metrics.csv`,
`retrieval_metrics.json`, `retrieval_purity.{pdf,png}`, and
`assessment_dashboard.json`.

`run_summary.json` includes `hybrid_manifest_summary`, which reports the
synthetic/real mix, split counts, physics-parameter source counts, and finite
coverage for `A`, `fD_khz`, `phi_rad`, `t0_fraction`, `tau_ms`, and `snr_db`.
Rows enriched from `particles2SNR_pipeline` usually have estimated `fD`,
`t0`, `tau`, and `snr_db`; `A` and `phi_rad` remain blank unless a reliable
source is added.

Smoke runs validate plumbing and strict artifact creation. They are not expected
to satisfy every approval gate, especially the raw-baseline and reconstruction
reference gates. A full run is considered approved only when
`run_assessment.json` reports `assessment_pass: true`, including the gate that
the hybrid model beats the older reconstruction-only P3 SSL checkpoint on
physical latent-space score.

## Isolation Contract

- Do not write into `P0/`, `P1/`, or `P2/`.
- Do not modify supervised labels or caches in place.
- Keep all generated SSL artifacts under `P3_SSL/outputs/`.
- Treat labels as diagnostics only.

## GPU Access From Codex

GPU jobs must run outside the default Codex sandbox. Inside the sandbox,
`/proc/driver/nvidia/version` may exist while `/dev/nvidia*` is hidden, so
`nvidia-smi` reports that it cannot communicate with the driver and PyTorch sees
`torch.cuda.is_available() == False`.

Check the current process with:

```bash
P0/venv/bin/python P3_SSL/scripts/check_gpu_access.py
```

If this reports no CUDA device from Codex but the host has a GPU, rerun training
or embedding commands with escalated/out-of-sandbox execution. A healthy host
check should show the NVIDIA device in both `nvidia-smi -L` and PyTorch.


## Embedding-Space Figure

Generate a MOMENT-style PCA/t-SNE figure with one point per labeled event:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/visualize_embedding_space.py \
  --config P3_SSL/configs/p3_ssl_moment_v0.yaml \
  --manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --moment-checkpoint P3_SSL/outputs/runs/p3_ssl_moment_v0/checkpoints/best.pt \
  --output-dir P3_SSL/outputs/embedding_space/p3_ssl_moment_vs_patchtst_random \
  --max-events-per-class 500
```

For a quick plumbing check, use the smoke checkpoint and a signal limit:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/visualize_embedding_space.py \
  --config P3_SSL/configs/p3_ssl_moment_v0.yaml \
  --manifest P3_SSL/outputs/manifests/p3_ssl_smoke_manifest.csv \
  --moment-checkpoint P3_SSL/outputs/runs/smoke/checkpoints/best.pt \
  --output-dir P3_SSL/outputs/embedding_space/smoke \
  --max-events-per-class 20 \
  --max-signals 20 \
  --batch-size 4 \
  --device cpu
```

Outputs include `embeddings_all.npz`, `embeddings_balanced.csv`,
`embedding_metrics.json`, `embedding_space_pca_tsne.pdf`, and
`embedding_space_by_width.pdf`. PatchTST is intentionally instantiated with
random weights as an architecture control; labels are used only for event
pooling, coloring, and metrics.

## Synthetic MOMENT Figure 7 Pipeline

`scripts/reproduce_moment_fig7_style.py` is only a visual layout mockup. It
does not pass generated signals through the model.

To validate the real pipeline on synthetic signals generated from the displayed
Figure 7 equations, run:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/run_synthetic_moment_fig7_pipeline.py \
  --config P3_SSL/configs/p3_ssl_moment_v0.yaml \
  --checkpoint P3_SSL/outputs/runs/smoke/checkpoints/best.pt \
  --output-dir P3_SSL/outputs/synthetic_fig7/smoke \
  --n-per-panel 80 \
  --batch-size 16 \
  --device cpu
```

Outputs include `synthetic_signals.npz`, `synthetic_metadata.csv`,
`embeddings.npz`, `reduction_metrics.json`,
`synthetic_moment_fig7_pca_tsne.pdf`, and
`synthetic_moment_fig7_pca_tsne.png`.

### Official MOMENT Backend

The official research code is vendored at `P3_SSL/vendor/moment-research`, and
minimal Python dependencies are installed locally under `P3_SSL/vendor/python`.
Hugging Face model files are cached under `P3_SSL/outputs/hf_cache`, so the
supervised P0/P1/P2 folders are not modified.

Smoke-test the real `AutonLab/MOMENT-1-large` loader:

```bash
P0/venv/bin/python P3_SSL/scripts/check_official_moment.py \
  --model-id AutonLab/MOMENT-1-large \
  --cache-dir P3_SSL/outputs/hf_cache \
  --seq-len 4096 \
  --batch-size 1 \
  --device cpu
```

Generate the dense synthetic Figure 7 pipeline with official MOMENT embeddings:

```bash
P0/venv/bin/python P3_SSL/scripts/run_synthetic_moment_fig7_pipeline.py \
  --backend official_moment \
  --model-id AutonLab/MOMENT-1-large \
  --cache-dir P3_SSL/outputs/hf_cache \
  --output-dir P3_SSL/outputs/synthetic_fig7/official_moment_n1800 \
  --n-per-panel 1800 \
  --input-length 4096 \
  --batch-size 8 \
  --device cuda
```

The official representation path mirrors the MOMENT zero-shot script: the model
is loaded in `pre-training` mode and embeddings are extracted with
`model.embed(..., reduction="mean")`. The vendored `MOMENTPipeline.__init__` has
a one-line compatibility patch so that `model_kwargs["task_name"]` overrides the
HF config value `reconstruction` instead of being popped before model creation.

## Public-Pretrained PatchTST and Swin

The historical P1 `patchtst` and `swin1d` runs are local 1D architectures
trained from scratch. They must not be reported as pretrained baselines. The
P3 embedding-space script also used `PatchTST random` as an architecture
control only.

Use the public-pretrained pipeline for the corrected comparison:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
P0/venv/bin/python P3_SSL/scripts/run_pretrained_backbone_embeddings.py \
  --manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --output-dir P3_SSL/outputs/pretrained_backbones-4096_YYYYMMDD/zero_shot_n100_all \
  --models moment_official,patchtst_pretrained,swin2d_pretrained \
  --max-events-per-class 100 \
  --event-length 4096 \
  --finetune-mode zero_shot \
  --batch-size 16 \
  --device cuda
```

For a transfer-learning smoke run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/matplotlib-cache \
P0/venv/bin/python P3_SSL/scripts/run_pretrained_backbone_embeddings.py \
  --manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --output-dir P3_SSL/outputs/pretrained_backbones-4096_YYYYMMDD/full_finetune \
  --models patchtst_pretrained,swin2d_pretrained \
  --event-length 4096 \
  --finetune-mode full \
  --linear-epochs 20 \
  --full-epochs 5 \
  --batch-size 16 \
  --train-batch-size 16 \
  --device cuda
```

The corrected pretrained names are:

- `moment_official`: `AutonLab/MOMENT-1-large`, extracted with `model.embed(..., reduction="mean")` on the configured P3 event inputs.
- `patchtst_pretrained`: `namctin/patchtst_etth1_pretrain`, converted from
  seven ETTh1 channels to one signal channel by compatible weight transfer and
  instantiated on the configured P3 event length.
- `swin2d_pretrained`: `microsoft/swin-tiny-patch4-window7-224`, fed with
  224x224 log-magnitude spectrograms and ImageNet normalization.

Outputs include event crops, event metadata, weight-transfer reports,
zero-shot embeddings, optional classifier checkpoints, full-finetuned
embeddings, PCA/t-SNE PDFs/PNGs, and metric JSON files.

## SSL Assessment Figure Suite

Generate the unsupervised-learning assessment figures from the cached
same-input embeddings:

```bash
PYTHONPATH=P3_SSL P0/venv/bin/python P3_SSL/scripts/run_ssl_assessment_figures.py \
  --embedding-root P3_SSL/outputs/pretrained_backbones-4096_YYYYMMDD/particles2snr_f_3class_moment_patchtst_conv1dgap \
  --output-dir P3_SSL/outputs/ssl_assessment \
  --include-raw-baseline \
  --include-random-baseline \
  --models moment_official,patchtst_pretrained,conv1dgap_same_input_3class
```

The suite writes representation manifold, label-efficiency, nearest-neighbor
retrieval, reconstruction-diagnostic, and robustness/invariance figures, plus
CSV/JSON metric files. Robustness uses a lightweight placeholder by default;
add `--run-robustness` when the pretrained backbones should be re-encoded on
perturbed signals.
