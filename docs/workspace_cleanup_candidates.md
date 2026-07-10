# Workspace cleanup candidates

Date: 2026-06-16

This report separates what was cleaned from what should only be archived or
deleted after explicit validation. The current cleanup policy is conservative:
keep datasets, Python environments, Hugging Face cache, source code, and useful
reference outputs.

## Already cleaned

Removed P3_SSL smoke, visual-design, and superseded outputs:

| Path | Previous size | Reason |
| --- | ---: | --- |
| `P3_SSL/outputs/pretrained_backbones_smoke_full` | 216M | Full smoke output superseded by `zero_shot_n100_all` |
| `P3_SSL/outputs/runs/smoke` | 6.2M | SSL smoke run |
| `P3_SSL/outputs/pretrained_backbones/zero_shot_n100` | 2.8M | Intermediate duplicate |
| `P3_SSL/outputs/pretrained_backbones/zero_shot_n100_moment` | 2.6M | Intermediate duplicate |
| `P3_SSL/outputs/figure_design` | 1.9M | Non-model visual mockup, not a valid reference result |
| `P3_SSL/outputs/embedding_space/c1_smoke` | 252K | Old smoke embedding output |
| `P3_SSL/outputs/pretrained_backbones_smoke_moment` | 240K | Smoke output |
| `P3_SSL/outputs/eval/smoke` | 216K | Smoke evaluation |
| `P3_SSL/outputs/pretrained_backbones_smoke_swin` | 220K | Smoke output |
| `P3_SSL/outputs/pretrained_backbones_smoke_patchtst` | 168K | Smoke output |
| `P3_SSL/outputs/embedding_space/smoke` | 84K | Old smoke embedding output |
| `P3_SSL/outputs/manifests/p3_ssl_smoke_manifest.csv` | 8K | Smoke manifest |
| `P3_SSL/outputs/official_moment_smoke.json` | 4K | Smoke diagnostic |

Also removed regenerable Python bytecode caches:

- all `__pycache__` directories
- all `.pyc` and `.pyo` files

## Explicitly preserved

These paths are intentionally kept:

| Path | Reason |
| --- | --- |
| `P3_SSL/outputs/synthetic_fig7/official_moment_n1800` | Legacy 512-era official MOMENT synthetic Figure 7 reference |
| `P3_SSL/outputs/pretrained_backbones/zero_shot_n100_all` | Legacy pretrained comparison output; 4096 runs should live under `outputs/pretrained_backbones-4096_YYYYMMDD/` |
| `P3_SSL/outputs/hf_cache` | Hugging Face model cache; expensive to redownload |
| `P3_SSL/vendor` | Local dependencies / cloned research code |
| `P3_SSL/outputs/patch_stride_audit` | Useful patch/stride audit artifact |
| `P3_SSL/outputs/manifests/p3_ssl_c1_manifest.csv` | Main P3_SSL manifest |
| `P3_SSL/p3_ssl`, `P3_SSL/scripts`, `P3_SSL/tests`, `P3_SSL/configs`, `P3_SSL/docs` | Source, tests, configs, and documentation |

## Archive candidates, not deleted

These are large or legacy-looking directories. They should be reviewed before
any deletion because they may contain datasets, supervised baselines, or final
experiment results.

| Path | Size | Suggested action |
| --- | ---: | --- |
| `P0/results` | 7.1G | Review and archive externally if no longer active |
| `P0/data` | 2.9G | Dataset; do not delete without dataset policy |
| `P0/output` | 1.4G | Review for old generated outputs |
| `P1/yolo_dataset_long_wave8_postbandpass` | 2.0G | Dataset variant; archive-list only |
| `P1/yolo_dataset_long_wave4` | 2.0G | Dataset variant; archive-list only |
| `P1/yolo_dataset_long` | 2.0G | Dataset variant; archive-list only |
| `P1/yolo_dataset_v3_source_named` | 340M | Dataset variant; compare with `v3` before deletion |
| `P1/yolo_dataset_v3` | 339M | Dataset variant; archive-list only |
| `P1/yolo_dataset` | 284M | Dataset variant; archive-list only |
| `P1/detseg_output_phase3` | 773M | Supervised output; review before archive/delete |
| `P1/detseg_output_decim_long` | 663M | Supervised output; review before archive/delete |
| `P1/detseg_output_wave8_postbandpass_50ep` | 286M | Supervised output; review before archive/delete |
| `P1/detseg_output_phase4_v3` | 100M | Supervised output; review before archive/delete |
| `P1/detseg_output_Particles2SNR_F_c1_3class` | 88M | Supervised output; review before archive/delete |
| `P1/detseg_output_Particles2SNR_F_c1_4class_lim10` | 81M | Supervised output; review before archive/delete |
| `P2/data/p2_c1_multimodal_v0` | 2.4G | Dataset; do not delete without dataset policy |
| `P2/results` | 228M | Review P2 experiment outputs |
| `particle_detector/test` | 562M | Legacy-looking test/data folder; review ownership |
| `particles2SNR_pipeline/output` | 173M | Generated pipeline output; review before deletion |
| `wandb` | 356K | Small; keep unless experiment logs are intentionally pruned |

## Next cleanup pass

If more disk space is needed, the next high-impact pass should not target
P3_SSL first. It should compare and deduplicate dataset variants in `P1`, then
archive old supervised outputs from `P0` and `P1` after checking which runs are
referenced by current reports or notebooks.
