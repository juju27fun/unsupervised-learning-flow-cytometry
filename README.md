# P3_SSL - Self-Supervised Particle Signal Modeling

P3_SSL is isolated from the supervised P0/P1/P2 pipelines. It may read signals
and optional labels from those folders, but it writes every manifest, audit PDF,
checkpoint, metric file, and export under `P3_SSL/outputs/`.

The first experiment is a MOMENT-like masked reconstruction model:

- raw input length: 16384 samples
- decimation factor: 8
- SSL input length: 2048 samples
- patch size: 4
- patch stride: 4
- token count: 512
- masking: learned `[MASK]` token, time-block masks, guard bands
- loss: signal MSE + derivative Huber + energy Huber

Labels are never used in the SSL loss. They are optional audit metadata for
event/background reconstruction diagnostics and visualization.

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

## Isolation Contract

- Do not write into `P0/`, `P1/`, or `P2/`.
- Do not modify supervised labels or caches in place.
- Keep all generated SSL artifacts under `P3_SSL/outputs/`.
- Treat labels as diagnostics only.


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
  --seq-len 512 \
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
  --input-length 512 \
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
P0/venv/bin/python P3_SSL/scripts/run_pretrained_backbone_embeddings.py \
  --manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --output-dir P3_SSL/outputs/pretrained_backbones/zero_shot_n100_all \
  --models moment_official,patchtst_pretrained,swin2d_pretrained \
  --max-events-per-class 100 \
  --finetune-mode zero_shot \
  --batch-size 16 \
  --device cuda
```

For a transfer-learning smoke run:

```bash
P0/venv/bin/python P3_SSL/scripts/run_pretrained_backbone_embeddings.py \
  --manifest P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv \
  --output-dir P3_SSL/outputs/pretrained_backbones/full_finetune \
  --models patchtst_pretrained,swin2d_pretrained \
  --finetune-mode full \
  --linear-epochs 20 \
  --full-epochs 5 \
  --batch-size 16 \
  --train-batch-size 16 \
  --device cuda
```

The corrected pretrained names are:

- `moment_official`: `AutonLab/MOMENT-1-large`, extracted with `model.embed(..., reduction="mean")` on the same 512-sample event crops.
- `patchtst_pretrained`: `namctin/patchtst_etth1_pretrain`, converted from
  seven ETTh1 channels to one signal channel by compatible weight transfer.
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
  --embedding-root P3_SSL/outputs/pretrained_backbones/particles2snr_f_3class_same_input_moment_patchtst_conv1dgap \
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

