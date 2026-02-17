use crate::services::engine_manager::EngineManager;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{AppHandle, Manager, State};

pub struct EngineState(pub Mutex<EngineManager>);

fn ensure_engine_state(app: &AppHandle) {
    if app.try_state::<EngineState>().is_none() {
        app.manage(EngineState(Mutex::new(EngineManager::new())));
    }
}

struct EnginePaths {
    python: PathBuf,
    script: PathBuf,
    models_dir: PathBuf,
}

fn resolve_engine_paths(app: &AppHandle) -> Result<EnginePaths, String> {
    // In dev mode, use the venv Python and project-relative paths
    #[cfg(debug_assertions)]
    {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let venv_python = project_root.join("engine/.venv/bin/python3");
        let script = project_root.join("engine/main.py");
        let models_dir = project_root.join("src-tauri/resources/models");

        if script.exists() && venv_python.exists() {
            return Ok(EnginePaths {
                python: venv_python,
                script,
                models_dir,
            });
        }

        // Fall back to system python if no venv
        if script.exists() {
            return Ok(EnginePaths {
                python: PathBuf::from("python3"),
                script,
                models_dir,
            });
        }
    }

    // In production, the engine is bundled as a resource
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("Failed to get resource dir: {}", e))?;

    let script = resource_dir.join("engine").join("main.py");
    let models_dir = resource_dir.join("models");

    if !script.exists() {
        return Err(format!("Engine not found at: {}", script.display()));
    }

    Ok(EnginePaths {
        python: PathBuf::from("python3"),
        script,
        models_dir,
    })
}

#[tauri::command]
pub async fn start_engine(app: AppHandle, force_cpu: Option<bool>) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let paths = resolve_engine_paths(&app)?;

    let python_str = paths.python.to_str().ok_or("Invalid python path")?.to_string();
    let script_str = paths.script.to_str().ok_or("Invalid engine path")?.to_string();
    let models_str = paths.models_dir.to_str().unwrap_or("").to_string();

    let mut env_vars = vec![("VERIFY_ME_MODELS_DIR", models_str)];
    if force_cpu.unwrap_or(false) {
        env_vars.push(("VERIFY_ME_FORCE_CPU", "1".to_string()));
    }

    let mut manager = state
        .0
        .lock()
        .map_err(|e| format!("Lock error: {}", e))?;

    manager.start(
        &python_str,
        &script_str,
        env_vars,
    )?;
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
