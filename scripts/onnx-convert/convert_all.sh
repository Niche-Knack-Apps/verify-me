#!/usr/bin/env bash
# ============================================================================
# Master ONNX Conversion Script for Verify Me
# ============================================================================
#
# Converts all TTS models to ONNX format:
#   1. Qwen3 TTS CustomVoice  (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
#   2. Qwen3 TTS Base         (Qwen/Qwen3-TTS-12Hz-1.7B-Base)
#   3. Qwen3 TTS VoiceDesign  (Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
#   4. Pocket TTS              (already in Android assets, optional re-export)
#
# Source model directories (PyTorch safetensors):
#   ~/.local/share/com.niche-knack.verify-me/models/qwen3-tts/
#   ~/.local/share/com.niche-knack.verify-me/models/qwen3-tts-base/
#   ~/.local/share/com.niche-knack.verify-me/models/qwen3-tts-voice-design/
#
# Output ONNX directories (safe holding area):
#   <project>/resources/onnx-models/qwen3-tts-customvoice/
#   <project>/resources/onnx-models/qwen3-tts-base/
#   <project>/resources/onnx-models/qwen3-tts-voicedesign/
#
# Requirements:
#   - Python 3.10+
#   - The engine venv with qwen_tts installed, OR a separate venv with
#     the dependencies from requirements.txt
#
# Usage:
#   ./convert_all.sh               # Convert all Qwen3 variants
#   ./convert_all.sh --quantize    # Also produce INT8 quantized variants
#   ./convert_all.sh --pocket-tts  # Also re-export Pocket TTS
#   ./convert_all.sh --help        # Show usage
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
QUANTIZE=false
INCLUDE_POCKET_TTS=false
USE_ENGINE_VENV=true
VENV_DIR="${PROJECT_ROOT}/engine/.venv"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quantize|-q)
            QUANTIZE=true
            shift
            ;;
        --pocket-tts)
            INCLUDE_POCKET_TTS=true
            shift
            ;;
        --venv)
            VENV_DIR="$2"
            USE_ENGINE_VENV=false
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quantize, -q   Produce INT8 quantized variants alongside FP32"
            echo "  --pocket-tts     Also re-export Pocket TTS ONNX models"
            echo "  --venv <path>    Use a specific venv instead of engine/.venv"
            echo "  --help, -h       Show this help"
            echo ""
            echo "Output: ${PROJECT_ROOT}/resources/onnx-models/"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo " Verify Me -- ONNX Model Conversion"
echo "============================================================"
echo ""
echo " Project root:  ${PROJECT_ROOT}"
echo " Quantize:      ${QUANTIZE}"
echo " Pocket TTS:    ${INCLUDE_POCKET_TTS}"
echo " Python venv:   ${VENV_DIR}"
echo ""

# ---- Check prerequisites ----

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment not found at: ${VENV_DIR}"
    echo ""
    if [[ "$USE_ENGINE_VENV" == "true" ]]; then
        echo "The engine venv is expected at engine/.venv."
        echo "Run 'cd engine && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt' first."
    else
        echo "Create a venv and install dependencies from scripts/onnx-convert/requirements.txt."
    fi
    exit 1
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# Check for required packages
echo "Checking dependencies..."
python3 -c "import torch; import onnx; import onnxruntime" 2>/dev/null || {
    echo ""
    echo "ERROR: Missing required Python packages (torch, onnx, onnxruntime)."
    echo "Install them with:"
    echo "  pip install -r ${SCRIPT_DIR}/requirements.txt"
    deactivate
    exit 1
}

# Check for qwen_tts
python3 -c "import qwen_tts" 2>/dev/null || {
    echo ""
    echo "WARNING: qwen_tts package not found. Qwen3 conversion will fail."
    echo "Install it with: pip install qwen-tts"
    echo ""
}

echo "Dependencies OK."
echo ""

# ---- Convert Qwen3 TTS variants ----

QWEN3_ARGS=("--variant" "all")
if [[ "$QUANTIZE" == "true" ]]; then
    QWEN3_ARGS+=("--quantize")
fi

echo "============================================================"
echo " Converting Qwen3 TTS (all 3 variants)..."
echo "============================================================"
echo ""

python3 "${SCRIPT_DIR}/convert_qwen3.py" "${QWEN3_ARGS[@]}"

echo ""

# ---- Optionally convert Pocket TTS ----

if [[ "$INCLUDE_POCKET_TTS" == "true" ]]; then
    echo "============================================================"
    echo " Converting Pocket TTS..."
    echo "============================================================"
    echo ""

    POCKET_ARGS=()
    if [[ "$QUANTIZE" == "true" ]]; then
        POCKET_ARGS+=("--quantize")
    fi

    bash "${SCRIPT_DIR}/convert_pocket_tts.sh" "${POCKET_ARGS[@]}"
fi

deactivate

echo ""
echo "============================================================"
echo " All conversions complete!"
echo "============================================================"
echo ""
echo " Output directory: ${PROJECT_ROOT}/resources/onnx-models/"
echo ""

# List output
if [[ -d "${PROJECT_ROOT}/resources/onnx-models" ]]; then
    echo " Contents:"
    for dir in "${PROJECT_ROOT}/resources/onnx-models"/*/; do
        if [[ -d "$dir" ]]; then
            variant_name=$(basename "$dir")
            onnx_count=$(find "$dir" -name "*.onnx" 2>/dev/null | wc -l)
            total_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "   ${variant_name}: ${onnx_count} ONNX files, ${total_size}"
        fi
    done
fi

echo ""
echo "Done."
