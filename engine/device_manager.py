"""
CPU/GPU detection for TTS engine.
"""


def get_device():
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_device_info():
    """Return a dict with device type, name, and memory info."""
    info = {"device": "cpu", "name": "CPU", "memory": None}

    try:
        import torch
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem
            info["memory"] = {
                "total_bytes": mem,
                "total_gb": round(mem / (1024 ** 3), 2),
            }
    except ImportError:
        pass

    return info
