"""
CPU/GPU detection for TTS engine.

Set VERIFY_ME_FORCE_CPU=1 to override GPU detection and force CPU-only mode.
"""

import os


def _force_cpu():
    """Check if the user has requested CPU-only mode via env var."""
    return os.environ.get("VERIFY_ME_FORCE_CPU", "").strip() in ("1", "true", "yes")


def get_device():
    """Return 'cuda' if a GPU is available (and not forced to CPU), otherwise 'cpu'."""
    if _force_cpu():
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_device_info():
    """Return a dict with device type, name, memory info, and force_cpu flag."""
    forced = _force_cpu()
    info = {"device": "cpu", "name": "CPU", "memory": None, "force_cpu": forced}

    try:
        import torch
        if torch.cuda.is_available():
            if forced:
                info["name"] = "CPU (GPU available but forced to CPU)"
            else:
                info["device"] = "cuda"
                info["name"] = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory
                info["memory"] = {
                    "total_bytes": mem,
                    "total_gb": round(mem / (1024 ** 3), 2),
                }
    except ImportError:
        pass

    return info
