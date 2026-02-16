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

## Development

### Prerequisites
- Node.js 18+
- Rust (latest stable)
- Python 3.10+ (for TTS engine development)

### Setup
```bash
npm install
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
