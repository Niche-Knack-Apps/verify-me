use crate::services::engine_manager::EngineManager;
use serde_json::Value;
use std::sync::Mutex;
use tauri::{AppHandle, Manager, State};

pub struct EngineState(pub Mutex<EngineManager>);

fn ensure_engine_state(app: &AppHandle) {
    if app.try_state::<EngineState>().is_none() {
        app.manage(EngineState(Mutex::new(EngineManager::new())));
    }
}

#[tauri::command]
pub async fn start_engine(app: AppHandle) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let engine_path = app
        .path()
        .resource_dir()
        .map_err(|e| format!("Failed to get resource dir: {}", e))?
        .join("engine")
        .join("main.py");

    let engine_path_str = engine_path
        .to_str()
        .ok_or("Invalid engine path")?
        .to_string();

    let mut manager = state
        .0
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?;

    manager.start(&engine_path_str)?;
    Ok("Engine started successfully".into())
}

#[tauri::command]
pub async fn stop_engine(app: AppHandle) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let mut manager = state
        .0
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?;

    manager.stop()?;
    Ok("Engine stopped".into())
}

#[tauri::command]
pub async fn engine_health(app: AppHandle) -> Result<Value, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let manager = state
        .0
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?;

    if !manager.is_running() {
        return Ok(serde_json::json!({
            "status": "stopped",
            "engine_running": false,
        }));
    }

    match manager.health_check() {
        Ok(health) => Ok(health),
        Err(e) => Ok(serde_json::json!({
            "status": "error",
            "engine_running": true,
            "error": e,
        })),
    }
}

#[tauri::command]
pub async fn get_device_info(app: AppHandle) -> Result<Value, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let manager = state
        .0
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?;

    if !manager.is_running() {
        return Err("Engine is not running".into());
    }

    manager.send_request("engine.device_info", None)
}
