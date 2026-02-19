"""
Checkpoint logger for dev-mode side-by-side comparison.

When VERIFY_ME_CHECKPOINT_LOGGING=1, emits structured checkpoint data
to stderr as CHECKPOINT:{json} lines. The Rust python_engine.rs stderr
reader parses these and forwards them to the frontend via tts-checkpoint
events.

Usage:
    from checkpoint_logger import emit_checkpoint

    emit_checkpoint("tokenization", {"text": "hello", "voice": "Aiden"})
"""

import json
import os
import sys
import time

_enabled = os.environ.get("VERIFY_ME_CHECKPOINT_LOGGING") == "1"

# Use the REAL stderr (not redirected stdout).
# In main.py, sys.stdout is redirected to stderr, but we need the
# original stderr for checkpoint lines that Rust parses.
_stderr = sys.__stderr__ or sys.stderr


def emit_checkpoint(stage: str, data: dict, engine: str = "safetensors"):
    """Emit a structured checkpoint event to stderr for Rust to capture.

    Args:
        stage: Pipeline stage name (e.g. "tokenization", "model_load", "complete")
        data: Arbitrary data dict for this checkpoint
        engine: Engine identifier (default "safetensors")
    """
    if not _enabled:
        return

    checkpoint = {
        "engine": engine,
        "stage": stage,
        "timestamp": int(time.time() * 1000),
        "data": data,
    }

    try:
        line = "CHECKPOINT:" + json.dumps(checkpoint, default=str)
        _stderr.write(line + "\n")
        _stderr.flush()
    except Exception:
        pass  # Never let checkpoint logging break the engine
