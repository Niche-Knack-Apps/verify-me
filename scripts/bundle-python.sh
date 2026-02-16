#!/bin/bash
# Bundle Python environment for Verify Me TTS engine
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUNDLE_DIR="$PROJECT_ROOT/src-tauri/resources/python"

echo "Creating bundled Python environment..."

# Create venv
python3 -m venv "$BUNDLE_DIR"

# Install deps
"$BUNDLE_DIR/bin/pip" install --upgrade pip
"$BUNDLE_DIR/bin/pip" install -r "$PROJECT_ROOT/engine/requirements.txt"

# Copy engine files
mkdir -p "$BUNDLE_DIR/engine"
cp -r "$PROJECT_ROOT/engine/"*.py "$BUNDLE_DIR/engine/"
cp -r "$PROJECT_ROOT/engine/models" "$BUNDLE_DIR/engine/"

echo "Python environment bundled to $BUNDLE_DIR"
