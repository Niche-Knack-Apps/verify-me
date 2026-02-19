use crate::services::onnx_engine::{EngineState, OnnxEngine};
use crate::services::path_service;
use serde_json::Value;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, State};

fn ensure_engine_state(app: &AppHandle) {
    if app.try_state::<EngineState>().is_none() {
        app.manage(EngineState(std::sync::Arc::new(std::sync::Mutex::new(OnnxEngine::new()))));
    }
}

/// Ensure the engine is running with the requested model.
/// If it's already running with a different model, re-initialize.
pub fn ensure_engine_for_model(app: &AppHandle, model_id: &str) -> Result<(), String> {
    ensure_engine_state(app);
    let state: State<'_, EngineState> = app.state();

    let mut engine = state.0.lock().unwrap_or_else(|e| {
        log::warn!("Recovering from poisoned engine lock");
        e.into_inner()
    });

    // Already running with the right model?
    if engine.is_running() {
        if let Some(current) = engine.current_model_id() {
            if current == model_id {
                return Ok(());
            }
        }
    }

    // Need to (re-)initialize
    let model_dir = resolve_model_dir(app, model_id)?;
    log::info!(
        "Auto-switching engine to model={}, dir={}",
        model_id,
        model_dir.display()
    );

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.initialize(model_id, &model_dir)
    }));

    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(panic) => {
            let msg = if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic during engine init".to_string()
            };
            Err(format!("ONNX Runtime failed to load: {}", msg))
        }
    }
}

/// Resolve the models directory for the given model_id.
/// Checks bundled resources first, then user's app data models dir.
fn resolve_model_dir(app: &AppHandle, model_id: &str) -> Result<PathBuf, String> {
    // 1. Dev mode: project-relative bundled resources
    #[cfg(debug_assertions)]
    {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let bundled_dir = project_root
            .join("src-tauri")
            .join("resources")
            .join("models")
            .join(model_id);
        if bundled_dir.is_dir() {
            return Ok(bundled_dir);
        }
    }

    // 2. Bundled in resource dir (production)
    if let Ok(resource_dir) = app.path().resource_dir() {
        let bundled_dir = resource_dir.join("models").join(model_id);
        if bundled_dir.is_dir() {
            return Ok(bundled_dir);
        }
    }

    // 3. User's ONNX exports in app_data_dir/models/onnx/{model_id}/
    let app_data_models = path_service::get_models_dir()?;
    let onnx_dir = app_data_models.join("onnx").join(model_id);
    if onnx_dir.is_dir() {
        return Ok(onnx_dir);
    }

    // 4. User's downloaded models in app_data_dir/models/
    let downloaded_dir = app_data_models.join(model_id);
    if downloaded_dir.is_dir() {
        return Ok(downloaded_dir);
    }

    Err(format!(
        "Model '{}' not found in resources or downloads",
        model_id
    ))
}

#[tauri::command]
pub async fn start_engine(
    app: AppHandle,
    model_id: Option<String>,
    _force_cpu: Option<bool>,
) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    // Default to pocket-tts if no model specified
    let model_id = model_id.unwrap_or_else(|| "pocket-tts".to_string());

    let model_dir = resolve_model_dir(&app, &model_id)?;

    log::info!(
        "Starting ONNX engine: model={}, dir={}",
        model_id,
        model_dir.display()
    );

    // Check ORT_DYLIB_PATH
    match std::env::var("ORT_DYLIB_PATH") {
        Ok(path) => log::info!("ORT_DYLIB_PATH={}", path),
        Err(_) => log::warn!("ORT_DYLIB_PATH not set — ONNX Runtime may fail to load"),
    }

    let mut engine = state.0.lock().unwrap_or_else(|e| {
        log::warn!("Recovering from poisoned engine lock");
        e.into_inner()
    });

    // Catch panics from ort library loading (e.g. missing libonnxruntime.so)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.initialize(&model_id, &model_dir)
    }));

    match result {
        Ok(Ok(())) => Ok(format!("Engine started: {}", model_id)),
        Ok(Err(e)) => {
            log::error!("Engine init failed: {}", e);
            Err(e)
        }
        Err(panic) => {
            let msg = if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic during ONNX init".to_string()
            };
            log::error!("Engine init panicked: {}", msg);
            Err(format!("ONNX Runtime failed to load: {}", msg))
        }
    }
}

#[tauri::command]
pub async fn stop_engine(app: AppHandle) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let mut engine = state.0.lock().unwrap_or_else(|e| {
        log::warn!("Recovering from poisoned engine lock");
        e.into_inner()
    });
    engine.shutdown();

    Ok("Engine stopped".into())
}

#[tauri::command]
pub async fn engine_health(app: AppHandle) -> Result<Value, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let engine = state.0.lock().unwrap_or_else(|e| {
        log::warn!("Recovering from poisoned engine lock");
        e.into_inner()
    });

    if !engine.is_running() {
        return Ok(serde_json::json!({
            "status": "stopped",
            "engine_running": false,
        }));
    }

    Ok(serde_json::json!({
        "status": "ready",
        "engine_running": true,
        "device": "cpu",
        "backend": "onnx",
        "model_id": engine.current_model_id(),
    }))
}

#[tauri::command]
pub async fn get_device_info(app: AppHandle) -> Result<Value, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let engine = state.0.lock().unwrap_or_else(|e| {
        log::warn!("Recovering from poisoned engine lock");
        e.into_inner()
    });

    if !engine.is_running() {
        return Err("Engine is not running".into());
    }

    Ok(serde_json::json!({
        "device": "cpu",
        "name": "ONNX Runtime",
        "model_id": engine.current_model_id(),
    }))
}
