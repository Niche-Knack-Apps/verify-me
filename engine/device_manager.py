"""
CPU/GPU detection for TTS engine.

Set VERIFY_ME_FORCE_CPU=1 to override GPU detection and force CPU-only mode.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def _force_cpu():
    """Check if the user has requested CPU-only mode via env var."""
    raw = os.environ.get("VERIFY_ME_FORCE_CPU", "")
    forced = raw.strip() in ("1", "true", "yes")
    if forced:
        logger.info("VERIFY_ME_FORCE_CPU=%s — forcing CPU mode", raw)
    return forced


def _run_nvidia_smi():
    """Run nvidia-smi and return its output, or an error string."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"nvidia-smi exited with code {result.returncode}: {result.stderr.strip()}"
    except FileNotFoundError:
        return "nvidia-smi not found on PATH"
    except subprocess.TimeoutExpired:
        return "nvidia-smi timed out"
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def _build_cuda_diagnostic():
    """Build diagnostic info when CUDA build is present but GPU is unavailable."""
    try:
        import torch
    except ImportError:
        return None

    cuda_build = torch.version.cuda
    if not cuda_build or torch.cuda.is_available():
        return None

    smi_output = _run_nvidia_smi()
    diagnostic = (
        f"PyTorch has CUDA {cuda_build} support but cannot access GPU. "
        f"nvidia-smi says: {smi_output.splitlines()[0] if smi_output else 'N/A'}. "
        "Try: sudo modprobe nvidia, reboot after kernel update, or check hybrid graphics settings."
    )
    logger.warning(diagnostic)
    return {"cuda_build": cuda_build, "nvidia_smi": smi_output, "diagnostic": diagnostic}


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

        # Log diagnostics when CUDA build exists but GPU isn't available
        if torch.version.cuda:
            _build_cuda_diagnostic()

        return "cpu"
    except ImportError:
        logger.warning("torch not installed — defaulting to CPU")
        return "cpu"
    except Exception as e:
        logger.error("CUDA detection failed: %s — defaulting to CPU", e)
        return "cpu"


def get_device_info():
    """Return a dict with device type, name, memory info, diagnostics, and force_cpu flag."""
    forced = _force_cpu()
    info = {
        "device": "cpu",
        "name": "CPU",
        "memory": None,
        "force_cpu": forced,
        "cuda_build": None,
        "cuda_available": False,
        "nvidia_smi": None,
        "diagnostic": None,
    }

    try:
        import torch

        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()

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
            # Capture diagnostics when CUDA build is present but unavailable
            diag = _build_cuda_diagnostic()
            if diag:
                info["nvidia_smi"] = diag["nvidia_smi"]
                info["diagnostic"] = diag["diagnostic"]
            else:
                logger.info("No CUDA GPU detected — using CPU")
    except ImportError:
        logger.warning("torch not installed — cannot detect GPU")

    return info
