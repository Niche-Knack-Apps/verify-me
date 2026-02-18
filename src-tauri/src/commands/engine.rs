use crate::services::engine_manager::EngineManager;
use crate::services::python_resolver;
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
pub async fn start_engine(app: AppHandle, force_cpu: Option<bool>) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let python = python_resolver::resolve_python(&app)?;
    let script = python_resolver::resolve_engine_script(&app)?;
    let models_dir = python_resolver::resolve_models_dir(&app)?;

    let python_str = python
        .to_str()
        .ok_or("Invalid python path")?
        .to_string();
    let script_str = script
        .to_str()
        .ok_or("Invalid engine path")?
        .to_string();
    let models_str = models_dir.to_str().unwrap_or("").to_string();

    let mut env_vars = vec![("VERIFY_ME_MODELS_DIR", models_str)];
    if force_cpu.unwrap_or(false) {
        env_vars.push(("VERIFY_ME_FORCE_CPU", "1".to_string()));
    }

    let mut manager = state.0.lock().map_err(|e| format!("Lock error: {}", e))?;

    manager.start(&python_str, &script_str, env_vars)?;
    Ok("Engine started successfully".into())
}

#[tauri::command]
pub async fn stop_engine(app: AppHandle) -> Result<String, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let mut manager = state.0.lock().map_err(|e| format!("Lock error: {}", e))?;

    manager.stop()?;
    Ok("Engine stopped".into())
}

#[tauri::command]
pub async fn engine_health(app: AppHandle) -> Result<Value, String> {
    ensure_engine_state(&app);
    let state: State<'_, EngineState> = app.state();

    let manager = state.0.lock().map_err(|e| format!("Lock error: {}", e))?;

    if !manager.is_running() {
        return Ok(serde_json::json!({
            "status": "stopped",
            "engine_running": false,
        }));
    }

    match manager.health_check() {
        Ok(health) => {
            // Merge engine_running into the Python health response
            let mut obj = health;
            if let Some(map) = obj.as_object_mut() {
                map.insert("engine_running".to_string(), serde_json::json!(true));
            }
            Ok(obj)
        }
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

    let manager = state.0.lock().map_err(|e| format!("Lock error: {}", e))?;

    if !manager.is_running() {
        return Err("Engine is not running".into());
    }

    manager.send_request("engine.device_info", None)
}

#[tauri::command]
pub async fn check_python_environment(app: AppHandle) -> Result<Value, String> {
    let python = python_resolver::resolve_python(&app)?;
    let python_str = python.to_string_lossy().to_string();

    let check_script = r#"
import sys
issues = []
try:
    import torch
except ImportError:
    issues.append('torch')
try:
    import soundfile
except ImportError:
    issues.append('soundfile')
try:
    import scipy
except ImportError:
    issues.append('scipy')
try:
    import numpy
except ImportError:
    issues.append('numpy')
try:
    import yaml
except ImportError:
    issues.append('pyyaml')
if issues:
    print('missing:' + ','.join(issues))
else:
    print('ready')
"#;

    let output = std::process::Command::new(&python_str)
        .arg("-c")
        .arg(check_script)
        .output()
        .map_err(|e| format!("Failed to run Python: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    if stdout.starts_with("ready") {
        Ok(serde_json::json!({
            "ready": true,
            "pythonPath": python_str,
        }))
    } else if stdout.starts_with("missing:") {
        let missing = stdout.strip_prefix("missing:").unwrap_or("");
        Ok(serde_json::json!({
            "ready": false,
            "pythonPath": python_str,
            "issue": format!("Missing packages: {}", missing),
        }))
    } else {
        Ok(serde_json::json!({
            "ready": false,
            "pythonPath": python_str,
            "issue": if stderr.is_empty() {
                "Python check failed".to_string()
            } else {
                stderr
            },
        }))
    }
}

#[tauri::command]
pub async fn setup_python_environment(app: AppHandle) -> Result<String, String> {
    let requirements = python_resolver::resolve_requirements(&app)?;
    let requirements_str = requirements.to_string_lossy().to_string();

    let app_data = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {}", e))?;

    let venv_dir = app_data.join("engine").join(".venv");

    // Find system python3 for creating the venv
    let system_python = if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    };

    // Step 1: Create venv if it doesn't exist
    if !venv_dir.exists() {
        log::info!("Creating Python venv at: {}", venv_dir.display());

        let output = std::process::Command::new(system_python)
            .arg("-m")
            .arg("venv")
            .arg(&venv_dir)
            .output()
            .map_err(|e| format!("Failed to create venv: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to create venv: {}", stderr));
        }
    }

    // Step 2: Install requirements
    let pip_path = if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("pip.exe")
    } else {
        venv_dir.join("bin").join("pip")
    };

    log::info!("Installing requirements from: {}", requirements_str);

    let output = std::process::Command::new(&pip_path)
        .arg("install")
        .arg("-r")
        .arg(&requirements_str)
        .output()
        .map_err(|e| format!("Failed to install requirements: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to install requirements: {}", stderr));
    }

    Ok("Python environment set up successfully".into())
}
