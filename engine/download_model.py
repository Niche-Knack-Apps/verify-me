"""
Standalone model download script.

Spawned as a subprocess by the Rust backend for downloading
HuggingFace model repositories to the app's models directory.

Usage:
    python download_model.py '{"repo_id": "...", "local_dir": "..."}'

The HuggingFace token is read from the HF_TOKEN environment variable
(not passed via CLI args, to avoid leaking in process listings).

Outputs JSON status lines to stdout:
    {"status": "downloading", "repo_id": "...", "local_dir": "..."}
    {"status": "complete", "path": "..."}
    {"status": "error", "message": "..."}
"""

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


def emit(status, **kwargs):
    """Print a JSON status line to stdout."""
    data = {"status": status, **kwargs}
    print(json.dumps(data), flush=True)


def main():
    if len(sys.argv) < 2:
        emit("error", message="Missing arguments")
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        emit("error", message=f"Invalid JSON arguments: {e}")
        sys.exit(1)

    repo_id = args.get("repo_id")
    local_dir = args.get("local_dir")
    # Read token from environment (safer than CLI args which are visible in ps)
    token = os.environ.get("HF_TOKEN") or args.get("token")

    if not repo_id or not local_dir:
        emit("error", message="repo_id and local_dir are required")
        sys.exit(1)

    logger.info("Downloading %s to %s", repo_id, local_dir)
    emit("downloading", repo_id=repo_id, local_dir=local_dir)

    try:
        from huggingface_hub import snapshot_download

        os.makedirs(local_dir, exist_ok=True)

        path = snapshot_download(
            repo_id,
            local_dir=local_dir,
            token=token if token else None,
        )

        logger.info("Download complete: %s", path)
        emit("complete", path=str(path))

    except Exception as e:
        logger.exception("Download failed")
        emit("error", message=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
