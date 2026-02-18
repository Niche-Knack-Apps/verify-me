#!/usr/bin/env bash
# ============================================================================
# Pocket TTS ONNX Conversion Wrapper
# ============================================================================
#
# Pocket TTS ONNX models are produced via the KevinAHM/pocket-tts-onnx-export
# tool. This script automates the process:
#
#   1. Creates a Python virtual environment
#   2. Clones the KevinAHM/pocket-tts-onnx-export repository
#   3. Installs dependencies
#   4. Runs the export with --quantize
#
# NOTE: Pocket TTS ONNX files already exist in the Android assets directory:
#   android/app/src/main/assets/models/pocket-tts/
#
# Files present:
#   - flow_lm_flow_int8.onnx
#   - flow_lm_main_int8.onnx
#   - mimi_decoder_int8.onnx
#   - mimi_encoder.onnx
#   - text_conditioner.onnx
#   - tokenizer.model
#   - embeddings_v2/*.safetensors
#
# This script is provided for reproducibility and re-export if needed.
#
# Usage:
#   ./convert_pocket_tts.sh [--output-dir <path>] [--quantize]
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
OUTPUT_DIR="${PROJECT_ROOT}/resources/onnx-models/pocket-tts"
QUANTIZE=false
WORK_DIR="${SCRIPT_DIR}/.pocket-tts-export"
REPO_URL="https://github.com/KevinAHM/pocket-tts-onnx-export.git"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quantize)
            QUANTIZE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--output-dir <path>] [--quantize]"
            echo ""
            echo "Options:"
            echo "  --output-dir  Output directory for ONNX files (default: resources/onnx-models/pocket-tts/)"
            echo "  --quantize    Apply INT8 quantization to exported models"
            echo ""
            echo "NOTE: Pocket TTS ONNX files already exist at:"
            echo "  ${PROJECT_ROOT}/android/app/src/main/assets/models/pocket-tts/"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Pocket TTS ONNX Export"
echo "============================================"
echo "Output:    ${OUTPUT_DIR}"
echo "Quantize:  ${QUANTIZE}"
echo "Work dir:  ${WORK_DIR}"
echo ""

# Check if ONNX files already exist in Android assets
ANDROID_ASSETS="${PROJECT_ROOT}/android/app/src/main/assets/models/pocket-tts"
if [[ -d "$ANDROID_ASSETS" ]] && ls "$ANDROID_ASSETS"/*.onnx 1>/dev/null 2>&1; then
    echo "NOTE: Pocket TTS ONNX files already exist in Android assets:"
    echo "  ${ANDROID_ASSETS}/"
    ls -lh "$ANDROID_ASSETS"/*.onnx 2>/dev/null || true
    echo ""
    read -p "Continue with re-export? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Step 1: Create work directory and venv
echo "[1/4] Setting up virtual environment..."
mkdir -p "$WORK_DIR"

if [[ ! -d "$WORK_DIR/.venv" ]]; then
    python3 -m venv "$WORK_DIR/.venv"
fi
source "$WORK_DIR/.venv/bin/activate"

pip install --quiet --upgrade pip

# Step 2: Clone the export tool
echo "[2/4] Cloning pocket-tts-onnx-export..."
if [[ -d "$WORK_DIR/pocket-tts-onnx-export" ]]; then
    echo "  Repository already cloned, pulling latest..."
    cd "$WORK_DIR/pocket-tts-onnx-export"
    git pull --quiet
else
    cd "$WORK_DIR"
    git clone --quiet "$REPO_URL"
    cd "$WORK_DIR/pocket-tts-onnx-export"
fi

# Step 3: Install dependencies
echo "[3/4] Installing dependencies..."
if [[ -f "requirements.txt" ]]; then
    pip install --quiet -r requirements.txt
else
    # Fallback: install expected dependencies
    pip install --quiet torch transformers onnx onnxruntime safetensors numpy
fi

# Step 4: Run the export
echo "[4/4] Running ONNX export..."
mkdir -p "$OUTPUT_DIR"

EXPORT_ARGS=("--output-dir" "$OUTPUT_DIR")
if [[ "$QUANTIZE" == "true" ]]; then
    EXPORT_ARGS+=("--quantize")
fi

# The export tool entry point varies by repo structure
if [[ -f "export.py" ]]; then
    python export.py "${EXPORT_ARGS[@]}"
elif [[ -f "convert.py" ]]; then
    python convert.py "${EXPORT_ARGS[@]}"
elif [[ -f "main.py" ]]; then
    python main.py "${EXPORT_ARGS[@]}"
else
    echo "ERROR: Could not find export script in pocket-tts-onnx-export repo."
    echo "Available files:"
    ls -la *.py 2>/dev/null || echo "  (no .py files found)"
    echo ""
    echo "Please check the repository structure and update this script."
    deactivate
    exit 1
fi

deactivate

echo ""
echo "============================================"
echo "Pocket TTS ONNX export complete."
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Files:"
ls -lh "$OUTPUT_DIR"/*.onnx 2>/dev/null || echo "  (no ONNX files found)"
echo "============================================"
