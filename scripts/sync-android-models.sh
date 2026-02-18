#!/usr/bin/env bash
# sync-android-models.sh — Validate and sync model assets for Android builds.
# Copies shared files (tokenizer.model, embeddings_v2/) from the desktop model
# directory and validates all required ONNX files exist in the Android assets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DESKTOP_MODELS="$PROJECT_ROOT/src-tauri/resources/models/pocket-tts"
ANDROID_ASSETS="$PROJECT_ROOT/android/app/src/main/assets/models/pocket-tts"

# Required ONNX model files (Android-specific, not from desktop)
ONNX_FILES=(
  "flow_lm_flow_int8.onnx"
  "flow_lm_main_int8.onnx"
  "mimi_decoder_int8.onnx"
  "mimi_encoder.onnx"
  "text_conditioner.onnx"
)

# Required voice embeddings
EMBEDDINGS=(
  "alba.safetensors"
  "azelma.safetensors"
  "cosette.safetensors"
  "eponine.safetensors"
  "fantine.safetensors"
  "javert.safetensors"
  "jean.safetensors"
  "marius.safetensors"
)

echo "==> Syncing Android model assets..."

# Ensure Android assets directory exists
mkdir -p "$ANDROID_ASSETS/embeddings_v2"

# Sync shared files from desktop models if available
if [ -d "$DESKTOP_MODELS" ]; then
  if [ -f "$DESKTOP_MODELS/tokenizer.model" ]; then
    cp -u "$DESKTOP_MODELS/tokenizer.model" "$ANDROID_ASSETS/tokenizer.model"
    echo "    Synced tokenizer.model"
  fi
  if [ -d "$DESKTOP_MODELS/embeddings_v2" ]; then
    cp -u "$DESKTOP_MODELS/embeddings_v2/"*.safetensors "$ANDROID_ASSETS/embeddings_v2/" 2>/dev/null && \
      echo "    Synced embeddings_v2/" || true
  fi
fi

# Validate all required files
MISSING=()

if [ ! -f "$ANDROID_ASSETS/tokenizer.model" ]; then
  MISSING+=("tokenizer.model")
fi

for f in "${ONNX_FILES[@]}"; do
  if [ ! -f "$ANDROID_ASSETS/$f" ]; then
    MISSING+=("$f")
  fi
done

for f in "${EMBEDDINGS[@]}"; do
  if [ ! -f "$ANDROID_ASSETS/embeddings_v2/$f" ]; then
    MISSING+=("embeddings_v2/$f")
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo ""
  echo "ERROR: Missing required model files in $ANDROID_ASSETS:"
  for f in "${MISSING[@]}"; do
    echo "  - $f"
  done
  echo ""
  echo "ONNX files must be obtained from HuggingFace (pocket-tts ONNX conversion)."
  echo "Shared files (tokenizer.model, embeddings_v2/) can be synced from src-tauri/resources/models/pocket-tts/."
  exit 1
fi

echo "==> All model files validated successfully."
