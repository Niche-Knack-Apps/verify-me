use crate::commands::engine::resolve_model_dir;
use crate::commands::models::VoiceInfo;
use crate::services::onnx_engine::{EngineState, OnnxEngine};
use crate::services::path_service;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};

/// Maximum time allowed for a single TTS generation before timeout.
const GENERATION_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

fn ensure_engine_state(app: &AppHandle) {
    if app.try_state::<EngineState>().is_none() {
        app.manage(EngineState(std::sync::Arc::new(std::sync::Mutex::new(OnnxEngine::new()))));
    }
}

fn generate_output_path(prefix: &str) -> Result<String, String> {
    let output_dir = path_service::get_user_data_dir()
        .map_err(|e| format!("Failed to get data dir: {}", e))?
        .join("output");

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Failed to create output dir: {}", e))?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Time error: {}", e))?
        .as_millis();

    let path = output_dir.join(format!("{}_{}.wav", prefix, timestamp));
    path.to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Invalid output path".into())
}

#[tauri::command]
pub async fn generate_speech(
    app: AppHandle,
    text: String,
    model_id: String,
    voice: String,
    speed: Option<f32>,
    voice_prompt: Option<String>,
    voice_mode: Option<String>,
    voice_description: Option<String>,
) -> Result<String, String> {
    ensure_engine_state(&app);

    // Pre-resolve model dir before entering the blocking task (needs AppHandle)
    let model_dir = resolve_model_dir(&app, &model_id)?;
    let output_path = generate_output_path("tts")?;
    let speed = speed.unwrap_or(1.0);

    log::info!(
        "Generating speech: voice={}, speed={}, model={}, text={}...",
        voice,
        speed,
        model_id,
        &text[..std::cmp::min(text.len(), 50)]
    );

    // Create checkpoint channel for debug logging
    let (checkpoint_tx, checkpoint_rx) = std::sync::mpsc::channel::<serde_json::Value>();

    // Single lock acquisition: check/switch model + run inference atomically
    let engine_state = app.state::<EngineState>().inner().clone();
    let output_clone = output_path.clone();

    let task = tokio::task::spawn_blocking(move || {
        let mut engine = engine_state.0.lock().unwrap_or_else(|e| {
            log::warn!("Recovering from poisoned engine lock (blocking)");
            e.into_inner()
        });

        // Ensure correct model is loaded (inside the lock to prevent races)
        let needs_switch = match engine.current_model_id() {
            Some(current) if current == model_id => false,
            _ => true,
        };
        if needs_switch {
            log::info!("Auto-switching engine to model={}", model_id);
            engine.initialize(&model_id, &model_dir)?;
        }

        engine.generate_speech_with_checkpoints(
            &text, &voice, speed,
            std::path::Path::new(&output_clone),
            Some(&checkpoint_tx),
            voice_prompt.as_deref(),
            voice_mode.as_deref(),
            voice_description.as_deref(),
        )
    });

    match tokio::time::timeout(GENERATION_TIMEOUT, task).await {
        Ok(join_result) => {
            join_result.map_err(|e| format!("Speech generation task panicked: {}", e))??;
        }
        Err(_) => {
            return Err(format!(
                "Speech generation timed out after {}s. The model may be too large for CPU inference — try Pocket TTS or quantized models.",
                GENERATION_TIMEOUT.as_secs()
            ));
        }
    }

    // Drain and emit checkpoint events to the frontend
    while let Ok(checkpoint) = checkpoint_rx.try_recv() {
        let _ = app.emit("tts-checkpoint", &checkpoint);
    }

    log::info!("Speech generated: {}", output_path);
    Ok(output_path)
}

#[tauri::command]
pub async fn voice_clone(
    app: AppHandle,
    text: String,
    reference_audio: String,
    model_id: String,
) -> Result<String, String> {
    ensure_engine_state(&app);

    // Pre-resolve model dir before entering the blocking task (needs AppHandle)
    let model_dir = resolve_model_dir(&app, &model_id)?;
    let output_path = generate_output_path("clone")?;

    log::info!("Voice clone request: model={}, ref={}, text={}...", model_id, reference_audio, &text[..std::cmp::min(text.len(), 50)]);

    // Single lock acquisition: check/switch model + run inference atomically
    let engine_state = app.state::<EngineState>().inner().clone();
    let text_clone = text.clone();
    let ref_audio_clone = reference_audio.clone();
    let output_clone = output_path.clone();

    let task = tokio::task::spawn_blocking(move || {
        let mut engine = engine_state.0.lock().unwrap_or_else(|e| {
            log::warn!("Recovering from poisoned engine lock (blocking)");
            e.into_inner()
        });

        // Ensure correct model is loaded (inside the lock to prevent races)
        let needs_switch = match engine.current_model_id() {
            Some(current) if current == model_id => false,
            _ => true,
        };
        if needs_switch {
            log::info!("Auto-switching engine to model={}", model_id);
            engine.initialize(&model_id, &model_dir)?;
        }

        engine.clone_voice(
            &text_clone,
            std::path::Path::new(&ref_audio_clone),
            std::path::Path::new(&output_clone),
        )
    });

    match tokio::time::timeout(GENERATION_TIMEOUT, task).await {
        Ok(join_result) => {
            join_result.map_err(|e| format!("Voice clone task panicked: {}", e))??;
        }
        Err(_) => {
            return Err(format!(
                "Voice cloning timed out after {}s. The model may be too large for CPU inference — try Pocket TTS or quantized models.",
                GENERATION_TIMEOUT.as_secs()
            ));
        }
    }

    log::info!("Voice clone generated: {}", output_path);
    Ok(output_path)
}

#[tauri::command]
pub async fn get_voices(app: AppHandle, _model_id: String) -> Result<Vec<VoiceInfo>, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let mut engine = state.0.lock().unwrap_or_else(|e| {
        log::warn!("Recovering from poisoned engine lock");
        e.into_inner()
    });

    if !engine.is_running() {
        return Err(
            "Engine is not running. Open Settings to start the engine.".into(),
        );
    }

    let voice_ids = engine.get_voices()?;

    let voices = voice_ids
        .iter()
        .map(|id| VoiceInfo {
            id: id.clone(),
            name: format_voice_name(id),
        })
        .collect();

    Ok(voices)
}

/// Convert a voice ID to a display name (capitalize, replace underscores).
fn format_voice_name(id: &str) -> String {
    let mut chars = id.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let rest: String = chars.collect();
            format!("{}{}", first.to_uppercase(), rest.replace('_', " "))
        }
    }
}
