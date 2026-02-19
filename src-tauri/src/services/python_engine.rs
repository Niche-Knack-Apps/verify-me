//! Dev-only Python JSON-RPC engine for Qwen3-TTS safetensors inference.
//!
//! Spawns `engine/.venv/bin/python3 engine/main.py` and communicates via
//! JSON-RPC 2.0 over stdin/stdout. Stderr is captured for logging and
//! checkpoint events (lines prefixed with `CHECKPOINT:`).
//!
//! This entire module is compiled only in debug builds.
#![cfg(debug_assertions)]

use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// A running Python TTS engine process.
pub struct PythonEngine {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    /// Receives checkpoint events parsed from stderr.
    checkpoint_rx: mpsc::Receiver<Value>,
}

impl PythonEngine {
    /// Spawn the Python engine and initialize it with the given model.
    pub fn initialize(model_dir: &Path, force_cpu: bool) -> Result<Self, String> {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let venv_python = project_root.join("engine/.venv/bin/python3");
        let main_py = project_root.join("engine/main.py");

        if !venv_python.exists() {
            return Err(format!(
                "Python venv not found at {}. Run: python3 -m venv engine/.venv && engine/.venv/bin/pip install -r engine/requirements.txt",
                venv_python.display()
            ));
        }

        if !main_py.exists() {
            return Err(format!("engine/main.py not found at {}", main_py.display()));
        }

        // Resolve models dir: use parent of model_dir (the models/ directory)
        let models_dir = model_dir
            .parent()
            .unwrap_or(model_dir)
            .to_string_lossy()
            .to_string();

        log::info!(
            "[python] Spawning: {} {}",
            venv_python.display(),
            main_py.display()
        );
        log::info!("[python] VERIFY_ME_MODELS_DIR={}", models_dir);

        let mut cmd = Command::new(&venv_python);
        cmd.arg(&main_py)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("VERIFY_ME_MODELS_DIR", &models_dir)
            .env("VERIFY_ME_CHECKPOINT_LOGGING", "1");

        if force_cpu {
            cmd.env("VERIFY_ME_FORCE_CPU", "1");
            log::info!("[python] VERIFY_ME_FORCE_CPU=1");
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn Python engine: {}", e))?;

        let stdin = child
            .stdin
            .take()
            .ok_or("Failed to capture Python stdin")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("Failed to capture Python stdout")?;
        let stderr = child
            .stderr
            .take()
            .ok_or("Failed to capture Python stderr")?;

        // Spawn stderr reader thread: parse CHECKPOINT: lines, log the rest
        let (checkpoint_tx, checkpoint_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        if let Some(json_str) = line.strip_prefix("CHECKPOINT:") {
                            match serde_json::from_str::<Value>(json_str) {
                                Ok(val) => {
                                    let _ = checkpoint_tx.send(val);
                                }
                                Err(e) => {
                                    log::warn!(
                                        "[python] Bad checkpoint JSON: {} — {}",
                                        e,
                                        json_str
                                    );
                                }
                            }
                        } else {
                            log::info!("[python] {}", line);
                        }
                    }
                    Err(e) => {
                        log::warn!("[python] stderr read error: {}", e);
                        break;
                    }
                }
            }
            log::info!("[python] stderr reader thread exiting");
        });

        let mut engine = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            checkpoint_rx,
        };

        // Load the qwen3-tts model
        log::info!("[python] Sending models.load for qwen3-tts...");
        let load_result = engine.rpc_call(
            "models.load",
            serde_json::json!({ "model_id": "qwen3-tts" }),
        )?;
        log::info!("[python] models.load result: {}", load_result);

        Ok(engine)
    }

    /// Send a JSON-RPC request and wait for the response.
    fn rpc_call(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = REQUEST_ID.fetch_add(1, Ordering::Relaxed);

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let line = serde_json::to_string(&request)
            .map_err(|e| format!("JSON serialize error: {}", e))?;

        self.stdin
            .write_all(line.as_bytes())
            .map_err(|e| format!("Failed to write to Python stdin: {}", e))?;
        self.stdin
            .write_all(b"\n")
            .map_err(|e| format!("Failed to write newline: {}", e))?;
        self.stdin
            .flush()
            .map_err(|e| format!("Failed to flush stdin: {}", e))?;

        // Read response line
        let mut response_line = String::new();
        self.stdout
            .read_line(&mut response_line)
            .map_err(|e| format!("Failed to read from Python stdout: {}", e))?;

        if response_line.is_empty() {
            return Err("Python engine closed stdout (process may have crashed)".into());
        }

        let response: Value = serde_json::from_str(response_line.trim())
            .map_err(|e| format!("Invalid JSON-RPC response: {} — raw: {}", e, response_line))?;

        if let Some(error) = response.get("error") {
            let msg = error["message"].as_str().unwrap_or("Unknown error");
            let code = error["code"].as_i64().unwrap_or(-1);
            return Err(format!("Python RPC error {}: {}", code, msg));
        }

        Ok(response["result"].clone())
    }

    /// Generate speech using the Python safetensors engine.
    pub fn generate_speech(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        output_path: &Path,
        voice_prompt: Option<&str>,
        voice_mode: Option<&str>,
        voice_description: Option<&str>,
    ) -> Result<(), String> {
        log::info!(
            "[python] generate_speech: voice={}, speed={}, mode={:?}, text=\"{}\"",
            voice,
            speed,
            voice_mode,
            &text[..std::cmp::min(text.len(), 80)]
        );

        let mut params = serde_json::json!({
            "text": text,
            "model_id": "qwen3-tts",
            "voice": voice,
            "speed": speed,
            "output_path": output_path.to_string_lossy().to_string(),
        });

        if let Some(vp) = voice_prompt {
            params["voice_prompt"] = serde_json::Value::String(vp.to_string());
        }
        if let Some(vm) = voice_mode {
            params["voice_mode"] = serde_json::Value::String(vm.to_string());
        }
        if let Some(vd) = voice_description {
            params["voice_description"] = serde_json::Value::String(vd.to_string());
        }

        let result = self.rpc_call("tts.generate", params)?;

        log::info!("[python] generate_speech result: {}", result);
        Ok(())
    }

    /// Clone a voice from reference audio.
    pub fn clone_voice(
        &mut self,
        text: &str,
        reference_audio: &Path,
        output_path: &Path,
    ) -> Result<(), String> {
        log::info!(
            "[python] clone_voice: ref={}, text=\"{}\"",
            reference_audio.display(),
            &text[..std::cmp::min(text.len(), 80)]
        );

        let result = self.rpc_call(
            "voice.clone",
            serde_json::json!({
                "text": text,
                "reference_audio": reference_audio.to_string_lossy(),
                "model_id": "qwen3-tts",
                "output_path": output_path.to_string_lossy(),
            }),
        )?;

        log::info!("[python] clone_voice result: {}", result);
        Ok(())
    }

    /// Get available voices from the Python engine.
    pub fn get_voices(&mut self) -> Result<Vec<String>, String> {
        let result = self.rpc_call(
            "tts.voices",
            serde_json::json!({ "model_id": "qwen3-tts" }),
        )?;

        let voices = result["voices"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v["id"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(voices)
    }

    /// Drain all pending checkpoint events from the stderr reader.
    pub fn drain_checkpoints(&self) -> Vec<Value> {
        let mut checkpoints = Vec::new();
        while let Ok(cp) = self.checkpoint_rx.try_recv() {
            checkpoints.push(cp);
        }
        checkpoints
    }

    /// Shutdown the Python engine gracefully.
    pub fn shutdown(&mut self) {
        log::info!("[python] Sending engine.shutdown...");
        let _ = self.rpc_call("engine.shutdown", serde_json::json!({}));
        // Give it a moment to exit, then kill if needed
        std::thread::sleep(std::time::Duration::from_millis(500));
        let _ = self.child.kill();
        let _ = self.child.wait();
        log::info!("[python] Engine process terminated");
    }
}

impl Drop for PythonEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}
