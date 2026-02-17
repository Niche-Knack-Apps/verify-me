"""
CPU/GPU detection for TTS engine.

Set VERIFY_ME_FORCE_CPU=1 to override GPU detection and force CPU-only mode.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _force_cpu():
    """Check if the user has requested CPU-only mode via env var."""
    raw = os.environ.get("VERIFY_ME_FORCE_CPU", "")
    forced = raw.strip() in ("1", "true", "yes")
    if forced:
        logger.info("VERIFY_ME_FORCE_CPU=%s — forcing CPU mode", raw)
    return forced


def get_device():
    """Return 'cuda' if a GPU is available (and not forced to CPU), otherwise 'cpu'."""
    if _force_cpu():
        return "cpu"
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        logger.info(
            "torch.cuda.is_available() = %s (torch %s, CUDA %s)",
            cuda_available,
            torch.__version__,
            torch.version.cuda or "N/A",
        )
        if cuda_available:
            logger.info("GPU: %s", torch.cuda.get_device_name(0))
            return "cuda"
        return "cpu"
    except ImportError:
        logger.warning("torch not installed — defaulting to CPU")
        return "cpu"
    except Exception as e:
        logger.error("CUDA detection failed: %s — defaulting to CPU", e)
        return "cpu"


def get_device_info():
    """Return a dict with device type, name, memory info, and force_cpu flag."""
    forced = _force_cpu()
    info = {"device": "cpu", "name": "CPU", "memory": None, "force_cpu": forced}

    try:
        import torch

        logger.info(
            "torch %s, CUDA build: %s, CUDA available: %s",
            torch.__version__,
            torch.version.cuda or "N/A",
            torch.cuda.is_available(),
        )

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory
            mem_gb = round(mem / (1024**3), 2)

            if forced:
                info["name"] = f"CPU (GPU available: {gpu_name} {mem_gb}GB, forced to CPU)"
                logger.info("GPU detected but forced to CPU: %s (%s GB)", gpu_name, mem_gb)
            else:
                info["device"] = "cuda"
                info["name"] = gpu_name
                info["memory"] = {
                    "total_bytes": mem,
                    "total_gb": mem_gb,
                }
                logger.info("Using GPU: %s (%s GB)", gpu_name, mem_gb)
        else:
            logger.info("No CUDA GPU detected — using CPU")
    except ImportError:
        logger.warning("torch not installed — cannot detect GPU")

    return info
