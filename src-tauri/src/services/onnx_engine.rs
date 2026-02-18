use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::services::pocket_tts::PocketTTSEngine;
use crate::services::qwen3_tts::Qwen3TTSEngine;

/// Which TTS backend is currently loaded.
enum ActiveEngine {
    PocketTTS(PocketTTSEngine),
    Qwen3TTS(Qwen3TTSEngine),
}

/// Manages the ONNX inference engine lifecycle.
/// Replaces the old Python subprocess EngineManager.
pub struct OnnxEngine {
    engine: Option<ActiveEngine>,
    model_id: Option<String>,
    model_dir: Option<PathBuf>,
}

impl OnnxEngine {
    pub fn new() -> Self {
        Self {
            engine: None,
            model_id: None,
            model_dir: None,
        }
    }

    pub fn is_running(&self) -> bool {
        self.engine.is_some()
    }

    pub fn current_model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    /// Initialize the engine with the given model.
    pub fn initialize(&mut self, model_id: &str, model_dir: &Path) -> Result<(), String> {
        // Shutdown existing engine first
        self.shutdown();

        log::info!(
            "Initializing ONNX engine: model={}, dir={}",
            model_id,
            model_dir.display()
        );

        let engine = if model_id == "pocket-tts" {
            let pocket = PocketTTSEngine::initialize(model_dir)?;
            ActiveEngine::PocketTTS(pocket)
        } else if model_id.starts_with("qwen3-tts") {
            let qwen3 = Qwen3TTSEngine::initialize(model_dir)?;
            ActiveEngine::Qwen3TTS(qwen3)
        } else {
            return Err(format!("Unknown model: {}", model_id));
        };

        self.engine = Some(engine);
        self.model_id = Some(model_id.to_string());
        self.model_dir = Some(model_dir.to_path_buf());

        log::info!("ONNX engine initialized: {}", model_id);
        Ok(())
    }

    /// Generate speech from text.
    pub fn generate_speech(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        output_path: &Path,
    ) -> Result<(), String> {
        match self.engine.as_mut() {
            Some(ActiveEngine::PocketTTS(engine)) => {
                engine.generate_speech(text, voice, speed, output_path)
            }
            Some(ActiveEngine::Qwen3TTS(engine)) => {
                engine.generate_speech(text, voice, speed, output_path)
            }
            None => Err("Engine not initialized".into()),
        }
    }

    /// Clone a voice from reference audio.
    pub fn clone_voice(
        &mut self,
        text: &str,
        reference_audio: &Path,
        output_path: &Path,
    ) -> Result<(), String> {
        match self.engine.as_mut() {
            Some(ActiveEngine::PocketTTS(engine)) => {
                engine.clone_voice(text, reference_audio, output_path)
            }
            Some(ActiveEngine::Qwen3TTS(engine)) => {
                engine.clone_voice(text, reference_audio, output_path)
            }
            None => Err("Engine not initialized".into()),
        }
    }

    /// Get available voices for the current model.
    pub fn get_voices(&self) -> Result<Vec<String>, String> {
        match self.engine.as_ref() {
            Some(ActiveEngine::PocketTTS(engine)) => Ok(engine.get_available_voices()),
            Some(ActiveEngine::Qwen3TTS(engine)) => Ok(engine.get_available_voices()),
            None => Err("Engine not initialized".into()),
        }
    }

    /// Shut down the engine and release all ONNX sessions.
    pub fn shutdown(&mut self) {
        if self.engine.is_some() {
            log::info!("Shutting down ONNX engine");
            self.engine = None;
            self.model_id = None;
            self.model_dir = None;
        }
    }
}

impl Drop for OnnxEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Thread-safe wrapper for Tauri state management.
/// Clone is implemented via Arc internally by Tauri's State, but we need
/// Clone for spawn_blocking moves.
#[derive(Clone)]
pub struct EngineState(pub std::sync::Arc<Mutex<OnnxEngine>>);
