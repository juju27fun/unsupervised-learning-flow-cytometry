#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch


def main() -> None:
    devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    payload: dict[str, object] = {
        "dev_nvidia": devices,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "torch_devices": [],
    }
    if torch.cuda.is_available():
        payload["torch_devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": torch.cuda.get_device_capability(idx),
            }
            for idx in range(torch.cuda.device_count())
        ]
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload["nvidia_smi_returncode"] = int(result.returncode)
        payload["nvidia_smi_stdout"] = result.stdout.strip()
        payload["nvidia_smi_stderr"] = result.stderr.strip()
    except Exception as exc:
        payload["nvidia_smi_error"] = repr(exc)

    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["torch_cuda_available"]:
        print(
            "\nGPU is not visible from this process. In Codex this usually means the sandbox "
            "did not expose /dev/nvidia*. Re-run GPU jobs with escalated/out-of-sandbox execution.",
            flush=True,
        )


if __name__ == "__main__":
    main()
