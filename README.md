# Verify Me

> "My voice is my passport. Verify me." - Sneakers (1992)

Text-to-Speech and Voice Cloning desktop application by [niche-knack apps](https://nicheknack.app).

## Features

- **Text to Speech** — Convert text to natural-sounding speech
- **Voice Cloning** — Clone voices from reference audio samples
- **Multiple Models** — Pocket TTS (bundled, CPU-friendly) and Qwen 3 TTS (downloadable, high-quality)
- **CPU-First** — Works on any machine, with GPU acceleration when available
- **Cross-Platform** — Linux, Windows, macOS, and Android

## Installation

### Linux
Download the `.AppImage`, `.deb`, or `.rpm` from the releases page.

### Windows
Download the `.msi` installer from the releases page.

### macOS
Download the `.dmg` from the releases page.

### Android
Download the `.apk` from the releases page or install from the Play Store.

### Arch Linux (AUR)
```
yay -S verify-me
```

## Models

### Pocket TTS (Bundled)
A lightweight, CPU-friendly TTS model (~200 MB) that ships with the app. No setup needed.

### Qwen 3 TTS (Downloadable)
A high-quality 1.7B parameter TTS model from Alibaba's Qwen team (~4.5 GB download). Supports 9 premium speakers across 10 languages and instruction-based voice control.

#### Downloading Qwen 3 TTS

Qwen 3 TTS requires a HuggingFace account and access token to download:

1. **Create a HuggingFace account** at [huggingface.co/join](https://huggingface.co/join)
2. **Generate an access token**:
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click **"Create new token"**
   - Give it a name (e.g., "verify-me")
   - Select **"Read"** access (that's all you need)
   - Click **"Generate"** and copy the token (starts with `hf_`)
3. **Enter the token in Verify Me**:
   - Open the app and click the **Settings** gear icon
   - Paste your token into the **HuggingFace Token** field under Models
   - Click **Download** next to Qwen 3 TTS

The model downloads to the app's data directory and is managed automatically. You only need to enter the token once — it's saved for future use.

#### System Requirements for Qwen 3 TTS
- **CPU**: Works on any modern CPU (generation is slower)
- **GPU**: CUDA-compatible GPU with 4+ GB VRAM recommended for faster generation
- **Disk**: ~4.5 GB free space for model files
- **SoX**: Install the SoX audio tool (`sudo apt install sox` on Debian/Ubuntu, `brew install sox` on macOS)

## Development

### Prerequisites
- Node.js 18+
- Rust (latest stable)
- Python 3.10+ (for TTS engine)

### Setup
```bash
npm install

# Set up the Python engine
cd engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Return to project root and run
cd ..
npm run dev
```

### Build
```bash
# Linux
npm run build:linux

# Windows
npm run build:windows

# macOS
npm run build:mac

# Android
npm run build:android
```

### Release (all platforms)
```bash
npm run build:release
```

## Architecture

- **Frontend**: Vue 3 + TypeScript + Tailwind CSS
- **Backend**: Tauri 2 (Rust)
- **TTS Engine**: Python sidecar with JSON-RPC 2.0 over stdio
- **Android**: Capacitor 6 with native plugins
- **Models**: Pocket TTS (bundled), Qwen 3 TTS (downloadable)

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.

---

Part of **niche-knack apps** — Cabinet of Curiosities for Software
