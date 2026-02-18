use crate::commands::models::VoiceInfo;
use crate::services::onnx_engine::{EngineState, OnnxEngine};
use crate::services::path_service;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Manager, State};

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
    _model_id: String,
    voice: String,
    speed: Option<f32>,
    _voice_prompt: Option<String>,
    _voice_mode: Option<String>,
    _voice_description: Option<String>,
) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    // Check engine is running before spawning the blocking task
    {
        let engine = state.0.lock().unwrap_or_else(|e| {
            log::warn!("Recovering from poisoned engine lock");
            e.into_inner()
        });
        if !engine.is_running() {
            return Err(
                "Engine is not running. Open Settings to start the engine.".into(),
            );
        }
    }

    let output_path = generate_output_path("tts")?;
    let speed = speed.unwrap_or(1.0);

    log::info!(
        "Generating speech: voice={}, speed={}, text={}...",
        voice,
        speed,
        &text[..std::cmp::min(text.len(), 50)]
    );

    // Run inference on a blocking thread so the Tauri event loop stays responsive
    let engine_state = app.state::<EngineState>().inner().clone();
    let output_clone = output_path.clone();

    tokio::task::spawn_blocking(move || {
        let mut engine = engine_state.0.lock().unwrap_or_else(|e| {
            log::warn!("Recovering from poisoned engine lock (blocking)");
            e.into_inner()
        });

        engine.generate_speech(&text, &voice, speed, std::path::Path::new(&output_clone))
    })
    .await
    .map_err(|e| format!("Speech generation task panicked: {}", e))??;

    log::info!("Speech generated: {}", output_path);
    Ok(output_path)
}

#[tauri::command]
pub async fn voice_clone(
    app: AppHandle,
    text: String,
    reference_audio: String,
    _model_id: String,
) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    // Check engine is running before spawning the blocking task
    {
        let engine = state.0.lock().unwrap_or_else(|e| {
            log::warn!("Recovering from poisoned engine lock");
            e.into_inner()
        });
        if !engine.is_running() {
            return Err(
                "Engine is not running. Open Settings to start the engine.".into(),
            );
        }
    }

    let output_path = generate_output_path("clone")?;

    log::info!("Voice clone request: ref={}, text={}...", reference_audio, &text[..std::cmp::min(text.len(), 50)]);

    // Run inference on a blocking thread so the Tauri event loop stays responsive.
    // This prevents the UI from freezing during the (potentially long) clone operation.
    let engine_state = app.state::<EngineState>().inner().clone();
    let text_clone = text.clone();
    let ref_audio_clone = reference_audio.clone();
    let output_clone = output_path.clone();

    tokio::task::spawn_blocking(move || {
        let mut engine = engine_state.0.lock().unwrap_or_else(|e| {
            log::warn!("Recovering from poisoned engine lock (blocking)");
            e.into_inner()
        });

        engine.clone_voice(
            &text_clone,
            std::path::Path::new(&ref_audio_clone),
            std::path::Path::new(&output_clone),
        )
    })
    .await
    .map_err(|e| format!("Voice clone task panicked: {}", e))??;

    log::info!("Voice clone generated: {}", output_path);
    Ok(output_path)
}

#[tauri::command]
pub async fn get_voices(app: AppHandle, _model_id: String) -> Result<Vec<VoiceInfo>, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let engine = state.0.lock().unwrap_or_else(|e| {
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
