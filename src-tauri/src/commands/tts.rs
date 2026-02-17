use crate::commands::engine::EngineState;
use crate::commands::models::VoiceInfo;
use crate::services::path_service;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Manager, State};

fn ensure_engine_state(app: &AppHandle) {
    if app.try_state::<EngineState>().is_none() {
        app.manage(EngineState(std::sync::Mutex::new(
            crate::services::engine_manager::EngineManager::new(),
        )));
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
) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let manager = state
        .0
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?;

    if !manager.is_running() {
        return Err("Engine is not running. Start the engine first.".into());
    }

    let output_path = generate_output_path("tts")?;

    let mut params = serde_json::json!({
        "text": text,
        "model_id": model_id,
        "voice": voice,
        "speed": speed.unwrap_or(1.0),
        "output_path": output_path,
    });

    if let Some(ref prompt) = voice_prompt {
        params["voice_prompt"] = serde_json::json!(prompt);
    }

    let result = manager.send_request("tts.generate", Some(params))?;

    result["audio_path"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Engine did not return an audio path".into())
}

#[tauri::command]
pub async fn voice_clone(
    app: AppHandle,
    text: String,
    reference_audio: String,
    model_id: String,
) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let manager = state
        .0
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?;

    if !manager.is_running() {
        return Err("Engine is not running. Start the engine first.".into());
    }

    let output_path = generate_output_path("clone")?;

    let params = serde_json::json!({
        "text": text,
        "reference_audio": reference_audio,
        "model_id": model_id,
        "output_path": output_path,
    });

    let result = manager.send_request("voice.clone", Some(params))?;

    result["audio_path"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Engine did not return an audio path".into())
}

#[tauri::command]
pub async fn get_voices(
    app: AppHandle,
    model_id: String,
) -> Result<Vec<VoiceInfo>, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let manager = state
        .0
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?;

    if !manager.is_running() {
        return Err("Engine is not running. Start the engine first.".into());
    }

    let params = serde_json::json!({
        "model_id": model_id,
    });

    let result = manager.send_request("tts.voices", Some(params))?;

    result["voices"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    // Handle voice objects: {"id": "...", "name": "...", "language": "..."}
                    if let Some(obj) = v.as_object() {
                        let id = obj.get("id")?.as_str()?.to_string();
                        let name = obj
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or_else(|| obj.get("id").unwrap().as_str().unwrap())
                            .to_string();
                        Some(VoiceInfo { id, name })
                    } else if let Some(s) = v.as_str() {
                        // Fallback: plain string voice IDs
                        Some(VoiceInfo {
                            id: s.to_string(),
                            name: s.to_string(),
                        })
                    } else {
                        None
                    }
                })
                .collect()
        })
        .ok_or_else(|| "Engine did not return voices".into())
}
