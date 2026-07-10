"""Canonical workspace paths used by P3-SSL code and command-line tools."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.environ.get("INTERNSHIP_WORKSPACE_ROOT", PROJECT_ROOT.parent)).resolve()
DATASETS_ROOT = Path(os.environ.get("INTERNSHIP_DATASETS_ROOT", WORKSPACE_ROOT / "datasets")).resolve()
ARTIFACT_ROOT = Path(
    os.environ.get(
        "P3_SSL_ARTIFACT_ROOT",
        WORKSPACE_ROOT / "artifacts" / "unsupervised-learning-flow-cytometry",
    )
).resolve()
HF_CACHE = Path(os.environ.get("HF_HOME", WORKSPACE_ROOT / ".cache" / "huggingface")).resolve()

