# Verify Me — Developer Notes

## Remediation: JSON-RPC Contract + Engine Stability

### Problems → Fixes

| # | Severity | Problem | File(s) | Fix |
|---|----------|---------|---------|-----|
| 1 | Critical | Rust calls `engine.device_info` but Python has no handler — returns `METHOD_NOT_FOUND` | `engine/main.py`, `src-tauri/src/commands/engine.rs` | Added `handle_engine_device_info` handler in Python using existing `get_device_info()` from `device_manager.py`; registered as `"engine.device_info"` in `METHOD_HANDLERS` |
| 2 | Critical | Voice list type mismatch: Python returns `[{"id":"...","name":"...","language":"..."}]`, Rust `get_voices` does `v.as_str()` — silently returns empty list | `engine/models/pocket_tts.py`, `engine/models/qwen3_tts.py`, `src-tauri/src/commands/tts.rs` | Updated Rust `get_voices` to parse voice objects (`v.as_object()`) and return `Vec<VoiceInfo>` with fallback for plain strings |
| 3 | Critical | `pocket-tts` missing from `requirements.txt`; `torch>=2.0.0` too low (pocket-tts needs ≥2.5); `pyyaml` missing (used by pocket_tts config) | `engine/requirements.txt` | Added `pocket-tts>=1.0.0`, `pyyaml>=6.0`; updated `torch>=2.5` |
| 4 | High | Python logging not configured — could leak to stdout and corrupt JSON-RPC channel | `engine/main.py` | Added `logging.basicConfig(stream=sys.stderr, level=logging.INFO)` before any imports that use logging |
| 5 | High | No way to force CPU even when GPU is available | `engine/device_manager.py`, `src/stores/settings.ts`, `src-tauri/src/commands/engine.rs`, `src/components/settings/SettingsModal.vue` | Added `VERIFY_ME_FORCE_CPU` env var in `device_manager.py`; added `forceCpu` setting with persistence; pass env var from Rust `start_engine`; added Force CPU toggle in Settings UI |
| 6 | Medium | Tauri CSP set to `null` and asset scope allows `**` (everything) | `src-tauri/tauri.conf.json` | Set proper CSP allowing self + Google Fonts + blob/asset protocols; restricted asset scope to `$APPDATA/models/**`, `$APPDATA/output/**`, `$RESOURCE/**` |

### Authoritative JSON-RPC Contract

All methods follow JSON-RPC 2.0 over stdio. Python stdout contains **only** JSON-RPC responses. All logs go to stderr.

| Method | Params | Response (result) |
|--------|--------|-------------------|
| `engine.health` | none | `{"status":"ok","version":"0.1.0","device":"cpu\|cuda"}` |
| `engine.device_info` | none | `{"device":"cpu\|cuda","name":"...","memory":null\|{...},"force_cpu":bool}` |
| `engine.shutdown` | none | `{"status":"shutting_down"}` |
| `models.list` | none | `{"models":[{"id":"...","name":"...","status":"loaded\|available","supports_clone":bool}]}` |
| `models.load` | `{"model_id":"..."}` | `{"status":"loaded","model_id":"..."}` |
| `models.unload` | none | `{"status":"unloaded"}` |
| `tts.generate` | `{"text":"...","model_id":"...","output_path":"...","voice":"...","speed":1.0,"voice_prompt":"..."}` | `{"audio_path":"..."}` |
| `tts.voices` | `{"model_id":"..."}` | `{"voices":[{"id":"...","name":"...","language":"..."}]}` |
| `voice.clone` | `{"text":"...","reference_audio":"...","model_id":"...","output_path":"..."}` | `{"audio_path":"..."}` |

### Voice Object Schema

Both models return the same voice object format:

```json
{"id": "alba", "name": "Alba (Male, Neutral)", "language": "en"}
```

Rust `get_voices` parses objects and also handles plain string fallback.

---

## Test Matrix

### Prerequisites

```bash
cd verify-me
python3 -m venv engine/.venv
source engine/.venv/bin/activate
pip install -r engine/requirements.txt
```

### 1. Engine starts

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"engine.health"}' | \
  VERIFY_ME_MODELS_DIR=src-tauri/resources/models \
  engine/.venv/bin/python3 engine/main.py
```

**Expected:** `{"jsonrpc":"2.0","id":1,"result":{"status":"ok","version":"0.1.0","device":"cpu"}}`

### 2. Device info

```bash
echo '{"jsonrpc":"2.0","id":2,"method":"engine.device_info"}' | \
  VERIFY_ME_MODELS_DIR=src-tauri/resources/models \
  engine/.venv/bin/python3 engine/main.py
```

**Expected:** `{"jsonrpc":"2.0","id":2,"result":{"device":"cpu","name":"CPU","memory":null,"force_cpu":false}}`

### 3. Force CPU mode

```bash
echo '{"jsonrpc":"2.0","id":3,"method":"engine.device_info"}' | \
  VERIFY_ME_MODELS_DIR=src-tauri/resources/models \
  VERIFY_ME_FORCE_CPU=1 \
  engine/.venv/bin/python3 engine/main.py
```

**Expected:** `result.device` = `"cpu"`, `result.force_cpu` = `true`

### 4. List models

```bash
echo '{"jsonrpc":"2.0","id":4,"method":"models.list"}' | \
  VERIFY_ME_MODELS_DIR=src-tauri/resources/models \
  engine/.venv/bin/python3 engine/main.py
```

**Expected:** Array with `pocket-tts` and `qwen3-tts` entries

### 5. Load model + list voices (Pocket TTS)

```bash
printf '{"jsonrpc":"2.0","id":5,"method":"models.load","params":{"model_id":"pocket-tts"}}\n{"jsonrpc":"2.0","id":6,"method":"tts.voices","params":{"model_id":"pocket-tts"}}\n' | \
  VERIFY_ME_MODELS_DIR=src-tauri/resources/models \
  engine/.venv/bin/python3 engine/main.py
```

**Expected:** 8 voice objects with `id`, `name`, `language` fields

### 6. Generate TTS (Pocket TTS)

```bash
printf '{"jsonrpc":"2.0","id":7,"method":"models.load","params":{"model_id":"pocket-tts"}}\n{"jsonrpc":"2.0","id":8,"method":"tts.generate","params":{"text":"Hello world","model_id":"pocket-tts","voice":"alba","speed":1.0,"output_path":"/tmp/test_tts.wav"}}\n' | \
  VERIFY_ME_MODELS_DIR=src-tauri/resources/models \
  engine/.venv/bin/python3 engine/main.py
```

**Expected:** `/tmp/test_tts.wav` exists and is a valid PCM-16 WAV file.
Verify: `file /tmp/test_tts.wav` should show `RIFF (little-endian) data, WAVE audio`

### 7. Voice clone (Pocket TTS)

```bash
# Record or provide a reference audio first, then:
printf '{"jsonrpc":"2.0","id":9,"method":"models.load","params":{"model_id":"pocket-tts"}}\n{"jsonrpc":"2.0","id":10,"method":"voice.clone","params":{"text":"Testing voice clone","reference_audio":"/tmp/test_tts.wav","model_id":"pocket-tts","output_path":"/tmp/test_clone.wav"}}\n' | \
  VERIFY_ME_MODELS_DIR=src-tauri/resources/models \
  engine/.venv/bin/python3 engine/main.py
```

**Expected:** `/tmp/test_clone.wav` exists and is a valid PCM-16 WAV file

### 8. stdout purity check

```bash
printf '{"jsonrpc":"2.0","id":11,"method":"engine.health"}\n{"jsonrpc":"2.0","id":12,"method":"engine.shutdown"}\n' | \
  VERIFY_ME_MODELS_DIR=src-tauri/resources/models \
  engine/.venv/bin/python3 engine/main.py 2>/dev/null | \
  python3 -c "import sys,json; [json.loads(l) for l in sys.stdin if l.strip()]; print('PASS: stdout is pure JSON-RPC')"
```

**Expected:** `PASS: stdout is pure JSON-RPC`

### 9. Rust typecheck

```bash
cd src-tauri && cargo check && cd ..
```

**Expected:** No errors

### 10. Frontend typecheck

```bash
npx vue-tsc --noEmit
```

**Expected:** No errors

### 11. Full app launch

```bash
npm run dev
```

**Expected:** App launches, engine starts, voice list populates, TTS generation works

---

## TEAM NOTES

- Issue: `engine.device_info` METHOD_NOT_FOUND
  - Files/Functions: `engine/main.py:METHOD_HANDLERS`, `engine/device_manager.py:get_device_info()`
  - Proposed Fix: Register `handle_engine_device_info` handler — delegates to existing `get_device_info()`
  - Risk: None — additive change
  - Validation: Test matrix #2

- Issue: Voice list returns empty array
  - Files/Functions: `src-tauri/src/commands/tts.rs:get_voices()`, `engine/models/pocket_tts.py:get_voices()`, `engine/models/qwen3_tts.py:get_voices()`
  - Proposed Fix: Parse voice objects in Rust instead of expecting strings
  - Risk: Return type change from `Vec<String>` to `Vec<VoiceInfo>` — frontend already uses `VoiceInfo` from `list_models`, and `get_voices` is not directly called by UI
  - Validation: Test matrix #5

- Issue: Missing Python dependencies
  - Files/Functions: `engine/requirements.txt`
  - Proposed Fix: Add `pocket-tts>=1.0.0`, `pyyaml>=6.0`, bump `torch>=2.5`
  - Risk: Larger install size due to torch version constraint
  - Validation: Test matrix prerequisites

- Issue: stdout contamination risk
  - Files/Functions: `engine/main.py` (top-level)
  - Proposed Fix: `logging.basicConfig(stream=sys.stderr)` before any model imports
  - Risk: None — logs already go nowhere useful
  - Validation: Test matrix #8

- Issue: No CPU force override
  - Files/Functions: `engine/device_manager.py:get_device()`, `src/stores/settings.ts`, `src-tauri/src/commands/engine.rs:start_engine()`
  - Proposed Fix: `VERIFY_ME_FORCE_CPU` env var + UI toggle + pass-through on engine start
  - Risk: Requires engine restart to take effect (documented in UI)
  - Validation: Test matrix #3
