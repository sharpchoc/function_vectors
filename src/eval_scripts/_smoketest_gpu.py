"""Workflow smoke test — trivial GPU job to validate the CPU-pod -> GPU-pod -> shared-volume
-> commit/push loop. Safe to delete after the workflow is validated.

Runs on the GPU pod: checks CUDA, does a small matmul on the GPU, writes a JSON result to the
shared volume so the CPU pod can see/commit it. No model weights loaded (kept fast/cheap).
"""
import json
import os
import platform
import socket
from datetime import datetime, timezone

import torch

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "general", "_workflow_smoketest",
)
os.makedirs(OUT_DIR, exist_ok=True)

result = {
    "hostname": socket.gethostname(),
    "runpod_pod_id": os.environ.get("RUNPOD_POD_ID", "<unset>"),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "utc_time": datetime.now(timezone.utc).isoformat(),
}

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    result["device_name"] = torch.cuda.get_device_name(0)
    result["device_capability"] = list(torch.cuda.get_device_capability(0))
    result["total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    # small GPU compute to confirm the card actually works
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    c = a @ b
    torch.cuda.synchronize()
    result["matmul_4096_checksum"] = float(c.float().mean().item())
    result["matmul_ok"] = True
else:
    result["matmul_ok"] = False

out_path = os.path.join(OUT_DIR, "smoketest_result.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"\nWROTE: {out_path}")
