use crate::commands::engine::EngineState;
use tauri::{AppHandle, Manager, State};

fn ensure_engine_state(app: &AppHandle) {
    if app.try_state::<EngineState>().is_none() {
        app.manage(EngineState(std::sync::Mutex::new(
            crate::services::engine_manager::EngineManager::new(),
        )));
    }
}

#[tauri::command]
pub async fn generate_speech(
    app: AppHandle,
    text: String,
    model_id: String,
    voice: String,
    speed: Option<f32>,
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

    let params = serde_json::json!({
        "text": text,
        "model_id": model_id,
        "voice": voice,
        "speed": speed.unwrap_or(1.0),
    });

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

    let params = serde_json::json!({
        "text": text,
        "reference_audio": reference_audio,
        "model_id": model_id,
    });

    let result = manager.send_request("tts.voice_clone", Some(params))?;

    result["audio_path"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Engine did not return an audio path".into())
}

#[tauri::command]
pub async fn get_voices(
    app: AppHandle,
    model_id: String,
) -> Result<Vec<String>, String> {
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

    let result = manager.send_request("tts.get_voices", Some(params))?;

    result["voices"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .ok_or_else(|| "Engine did not return voices".into())
}
