## Project
Verify Me -- Text-to-Speech and Voice Cloning desktop application.
"My voice is my passport. Verify me." - Sneakers (1992)

## Stack
- Vue 3 + TypeScript (strict), Pinia (3 stores), Vite 6, Tauri 2, Tailwind CSS
- Rust backend: tauri 2, reqwest, serde, tokio, thiserror
- Python engine: torch, transformers, soundfile, scipy, numpy
- Android: Capacitor 6, custom plugins (AudioRecorder, FilePicker)

## Structure
- src/stores/ -- 3 stores: tts, models, settings
- src/components/ -- TTSTab, VoiceCloneTab, AudioPlayer, ModelSelector, settings/, ui/
- src/services/ -- debug-logger
- src-tauri/src/commands/ -- tts, models, engine
- src-tauri/src/services/ -- path_service (OnceLock), engine_manager (JSON-RPC 2.0)
- engine/ -- Python TTS engine (JSON-RPC 2.0 stdio server)
- android/ -- Capacitor Android project with native plugins

## Commands
- Dev: `npm run dev`
- Typecheck: `npm run typecheck`
- Build: `npm run build` (targets: deb, appimage, rpm)
- Release: `npm run build:release` -- builds + copies to _shared/releases/
- Android: `npm run build:android` (APK), `npm run build:android:aab` (AAB)
- Arch pkg: `../_shared/builders/arch/build.sh verify-me` (Podman)

## Verification
After changes, run in order:
1. `npx vue-tsc --noEmit` -- fix type errors
2. `npm run dev` -- verify app launches

## Conventions
- Path aliases: `@` -> src/, `@shared` -> ../_shared/
- Releases output to ../_shared/releases/verify-me/
- Model storage: bundled in src-tauri/resources/models/, downloads to {app_data_dir}/models/
- Engine communication: JSON-RPC 2.0 over stdio (Python sidecar)
- CPU-first: default to CPU, use GPU only if CUDA detected
- Single source icon: src-tauri/icons/icon.png (Tauri generates all sizes)
- Dev server port: 5184
- Identifier: com.niche-knack.verify-me
- Android appId: com.nicheknack.verifyme
- Git branch: main (not master)

## Don't
- Don't use require() -- use ES imports only
- Don't reference master branch -- always use main
- Don't store models in git -- bundled models go in resources/models/, downloads go to app_data_dir
- Don't assume GPU availability -- always default to CPU
