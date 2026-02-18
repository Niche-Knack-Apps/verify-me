#!/usr/bin/env python3
"""
ONNX Model Validation Script for Verify Me
=============================================

Loads each exported ONNX model with ONNX Runtime, runs a simple forward pass
with dummy inputs to verify shapes are correct, and reports model sizes.

Usage:
  python validate_onnx.py                              # Validate all variants
  python validate_onnx.py --dir resources/onnx-models/qwen3-tts-customvoice/
  python validate_onnx.py --verbose                    # Show input/output details
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_onnx")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_BASE = PROJECT_ROOT / "resources" / "onnx-models"

# Session options: disable GPU, use minimal threads for validation
SESSION_OPTS = ort.SessionOptions()
SESSION_OPTS.inter_op_num_threads = 1
SESSION_OPTS.intra_op_num_threads = 1
SESSION_OPTS.log_severity_level = 3  # Suppress warnings


# ================================================================================
#  Model-specific dummy input generators
# ================================================================================

# Each function returns a dict of {input_name: numpy_array} matching the
# ONNX model's expected input signature.

def dummy_text_project() -> dict[str, np.ndarray]:
    """text_project.onnx: text_ids [batch=1, seq=8] -> projected [1, 8, 2048]"""
    return {"text_ids": np.random.randint(0, 1000, size=(1, 8), dtype=np.int64)}


def dummy_codec_embed() -> dict[str, np.ndarray]:
    """codec_embed.onnx: codec_ids [batch=1, seq=4] -> embedded [1, 4, 2048]"""
    return {"codec_ids": np.random.randint(0, 3072, size=(1, 4), dtype=np.int64)}


def dummy_talker_prefill() -> dict[str, np.ndarray]:
    """talker_prefill.onnx: inputs_embeds [1, 16, 2048], attention_mask [1, 16]"""
    return {
        "inputs_embeds": np.random.randn(1, 16, 2048).astype(np.float32),
        "attention_mask": np.ones((1, 16), dtype=np.int64),
    }


def dummy_talker_decode() -> dict[str, np.ndarray]:
    """talker_decode.onnx: inputs_embeds [1, 1, 2048]"""
    return {"inputs_embeds": np.random.randn(1, 1, 2048).astype(np.float32)}


def dummy_code_predictor() -> dict[str, np.ndarray]:
    """code_predictor.onnx: inputs_embeds [1, 2, 2048]"""
    return {"inputs_embeds": np.random.randn(1, 2, 2048).astype(np.float32)}


def dummy_code_predictor_embed() -> dict[str, np.ndarray]:
    """code_predictor_embed.onnx: group_idx scalar, token_ids [1, 1]"""
    return {
        "group_idx": np.array(0, dtype=np.int64),
        "token_ids": np.random.randint(0, 2048, size=(1, 1), dtype=np.int64),
    }


def dummy_tokenizer_encode() -> dict[str, np.ndarray]:
    """tokenizer12hz_encode.onnx: input_values [1, 1, 24000] (1 second @ 24kHz)"""
    return {"input_values": np.random.randn(1, 1, 24000).astype(np.float32)}


def dummy_tokenizer_decode() -> dict[str, np.ndarray]:
    """tokenizer12hz_decode.onnx: audio_codes [1, 16, 13] (16 quantizers, 13 frames)"""
    return {"audio_codes": np.random.randint(0, 2048, size=(1, 16, 13), dtype=np.int64)}


def dummy_speaker_encoder() -> dict[str, np.ndarray]:
    """speaker_encoder.onnx: mel_spectrogram [1, 281, 128]"""
    return {"mel_spectrogram": np.random.randn(1, 281, 128).astype(np.float32)}


# Map filename stems to their dummy input generators
DUMMY_GENERATORS: dict[str, callable] = {
    "text_project": dummy_text_project,
    "codec_embed": dummy_codec_embed,
    "talker_prefill": dummy_talker_prefill,
    "talker_decode": dummy_talker_decode,
    "code_predictor": dummy_code_predictor,
    "code_predictor_embed": dummy_code_predictor_embed,
    "tokenizer12hz_encode": dummy_tokenizer_encode,
    "tokenizer12hz_decode": dummy_tokenizer_decode,
    "speaker_encoder": dummy_speaker_encoder,
}

# Quantized variants share the same dummy inputs
for stem in list(DUMMY_GENERATORS.keys()):
    DUMMY_GENERATORS[f"{stem}_q"] = DUMMY_GENERATORS[stem]


# ================================================================================
#  Validation logic
# ================================================================================

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024**2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def validate_model(
    onnx_path: Path,
    verbose: bool = False,
) -> bool:
    """Load an ONNX model, run inference with dummy inputs, report results.

    Returns True if validation passed, False otherwise.
    """
    stem = onnx_path.stem
    size = format_size(onnx_path.stat().st_size)

    # Find dummy input generator
    gen = DUMMY_GENERATORS.get(stem)
    if gen is None:
        logger.warning("  SKIP  %-40s  %8s  (no dummy inputs defined)", stem + ".onnx", size)
        return True  # Not a failure, just unknown

    try:
        # Load the model
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=SESSION_OPTS,
            providers=["CPUExecutionProvider"],
        )

        # Get model metadata
        input_info = session.get_inputs()
        output_info = session.get_outputs()

        if verbose:
            logger.info("  Model: %s (%s)", onnx_path.name, size)
            for inp in input_info:
                logger.info("    Input:  %-25s  type=%-10s  shape=%s", inp.name, inp.type, inp.shape)
            for out in output_info:
                logger.info("    Output: %-25s  type=%-10s  shape=%s", out.name, out.type, out.shape)

        # Generate dummy inputs
        dummy = gen()

        # Run inference
        output_names = [o.name for o in output_info]
        results = session.run(output_names, dummy)

        # Report output shapes
        output_shapes = []
        for name, result in zip(output_names, results):
            shape = result.shape if hasattr(result, "shape") else "scalar"
            dtype = result.dtype if hasattr(result, "dtype") else type(result).__name__
            output_shapes.append(f"{name}={shape}({dtype})")

        if verbose:
            for desc in output_shapes:
                logger.info("    Result: %s", desc)

        # Check for NaN/Inf in outputs
        has_issues = False
        for name, result in zip(output_names, results):
            if hasattr(result, "shape"):
                if np.any(np.isnan(result)):
                    logger.warning("    WARNING: %s contains NaN values", name)
                    has_issues = True
                if np.any(np.isinf(result)):
                    logger.warning("    WARNING: %s contains Inf values", name)
                    has_issues = True

        status = "WARN" if has_issues else "OK"
        shapes_summary = ", ".join(
            f"{r.shape}" if hasattr(r, "shape") else "scalar"
            for r in results
        )
        logger.info("  %-5s %-40s  %8s  -> %s", status, onnx_path.name, size, shapes_summary)

        return not has_issues

    except Exception as e:
        logger.error("  FAIL  %-40s  %8s  Error: %s", onnx_path.name, size, e)
        return False


def validate_directory(
    onnx_dir: Path,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """Validate all ONNX files in a directory.

    Returns (passed, failed, skipped) counts.
    """
    onnx_files = sorted(onnx_dir.glob("*.onnx"))

    if not onnx_files:
        logger.warning("No ONNX files found in: %s", onnx_dir)
        return (0, 0, 0)

    logger.info("")
    logger.info("Validating: %s (%d ONNX files)", onnx_dir.name, len(onnx_files))
    logger.info("-" * 80)

    passed = 0
    failed = 0
    skipped = 0

    # Calculate total size
    total_bytes = sum(f.stat().st_size for f in onnx_files)

    for onnx_file in onnx_files:
        stem = onnx_file.stem
        if stem not in DUMMY_GENERATORS:
            skipped += 1
            size = format_size(onnx_file.stat().st_size)
            logger.info("  SKIP  %-40s  %8s  (unknown model)", onnx_file.name, size)
        elif validate_model(onnx_file, verbose=verbose):
            passed += 1
        else:
            failed += 1

    logger.info("-" * 80)
    logger.info(
        "  Total: %d files, %s | Passed: %d | Failed: %d | Skipped: %d",
        len(onnx_files),
        format_size(total_bytes),
        passed,
        failed,
        skipped,
    )

    return (passed, failed, skipped)


# ================================================================================
#  CLI
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate exported ONNX models for Verify Me TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all variants in the default output directory:
  python validate_onnx.py

  # Validate a specific directory:
  python validate_onnx.py --dir resources/onnx-models/qwen3-tts-customvoice/

  # Verbose output (show input/output metadata):
  python validate_onnx.py --verbose
""",
    )

    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"Directory containing ONNX files to validate. Default: all under {DEFAULT_ONNX_BASE}/",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed input/output information for each model.",
    )

    args = parser.parse_args()

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    if args.dir is not None:
        # Validate a single directory
        if not args.dir.exists():
            logger.error("Directory not found: %s", args.dir)
            sys.exit(1)
        p, f, s = validate_directory(args.dir, verbose=args.verbose)
        total_passed += p
        total_failed += f
        total_skipped += s
    else:
        # Validate all variant directories under the default base
        if not DEFAULT_ONNX_BASE.exists():
            logger.error("ONNX output directory not found: %s", DEFAULT_ONNX_BASE)
            logger.error("Run the conversion scripts first.")
            sys.exit(1)

        variant_dirs = sorted(d for d in DEFAULT_ONNX_BASE.iterdir() if d.is_dir())
        if not variant_dirs:
            logger.error("No variant directories found in: %s", DEFAULT_ONNX_BASE)
            sys.exit(1)

        for variant_dir in variant_dirs:
            p, f, s = validate_directory(variant_dir, verbose=args.verbose)
            total_passed += p
            total_failed += f
            total_skipped += s

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info(
        " VALIDATION SUMMARY: Passed=%d  Failed=%d  Skipped=%d",
        total_passed,
        total_failed,
        total_skipped,
    )
    logger.info("=" * 80)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
