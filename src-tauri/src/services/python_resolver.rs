use std::path::PathBuf;
use tauri::{AppHandle, Manager};

/// Resolve the Python interpreter path with a fallback chain:
/// 1. Dev mode: engine/.venv/bin/python3 (via CARGO_MANIFEST_DIR)
/// 2. Bundled venv: {resource_dir}/engine/.venv/bin/python3
/// 3. App-data venv: {app_data_dir}/engine/.venv/bin/python3
/// 4. Flatpak: /app/lib/verify-me/engine/.venv/bin/python3
/// 5. VERIFY_ME_PYTHON env var
/// 6. System python3
pub fn resolve_python(app: &AppHandle) -> Result<PathBuf, String> {
    // 1. Dev mode: project venv
    #[cfg(debug_assertions)]
    {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let venv_python = venv_python_path(&project_root.join("engine"));
        if venv_python.exists() {
            return Ok(venv_python);
        }
    }

    // 2. Bundled venv in resource dir
    if let Ok(resource_dir) = app.path().resource_dir() {
        let venv_python = venv_python_path(&resource_dir.join("engine"));
        if venv_python.exists() {
            return Ok(venv_python);
        }
    }

    // 3. App-data venv
    if let Ok(app_data) = app.path().app_data_dir() {
        let venv_python = venv_python_path(&app_data.join("engine"));
        if venv_python.exists() {
            return Ok(venv_python);
        }
    }

    // 4. Flatpak path
    let flatpak_python = venv_python_path(&PathBuf::from("/app/lib/verify-me/engine"));
    if flatpak_python.exists() {
        return Ok(flatpak_python);
    }

    // 5. VERIFY_ME_PYTHON env var
    if let Ok(custom) = std::env::var("VERIFY_ME_PYTHON") {
        if !custom.is_empty() {
            return Ok(PathBuf::from(custom));
        }
    }

    // 6. System python3
    Ok(system_python())
}

/// Resolve the engine main.py script path.
pub fn resolve_engine_script(app: &AppHandle) -> Result<PathBuf, String> {
    // Dev mode: project-relative
    #[cfg(debug_assertions)]
    {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let script = project_root.join("engine").join("main.py");
        if script.exists() {
            return Ok(script);
        }
    }

    // Bundled in resource dir
    if let Ok(resource_dir) = app.path().resource_dir() {
        let script = resource_dir.join("engine").join("main.py");
        if script.exists() {
            return Ok(script);
        }
    }

    // Flatpak path
    let flatpak_script = PathBuf::from("/app/lib/verify-me/engine/main.py");
    if flatpak_script.exists() {
        return Ok(flatpak_script);
    }

    Err("Engine script (main.py) not found".to_string())
}

/// Resolve the models directory path.
pub fn resolve_models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    // Dev mode: project-relative
    #[cfg(debug_assertions)]
    {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let models_dir = project_root.join("src-tauri").join("resources").join("models");
        if models_dir.exists() {
            return Ok(models_dir);
        }
    }

    // Bundled in resource dir
    if let Ok(resource_dir) = app.path().resource_dir() {
        let models_dir = resource_dir.join("models");
        if models_dir.exists() {
            return Ok(models_dir);
        }
    }

    // Flatpak path
    let flatpak_models = PathBuf::from("/app/lib/verify-me/models");
    if flatpak_models.exists() {
        return Ok(flatpak_models);
    }

    // Fall back to app_data_dir/models (always valid — created by path_service)
    let app_data_models = super::path_service::get_models_dir()?;
    Ok(app_data_models)
}

/// Resolve the download_model.py script path.
pub fn resolve_download_script(app: &AppHandle) -> Result<PathBuf, String> {
    // Dev mode
    #[cfg(debug_assertions)]
    {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let script = project_root.join("engine").join("download_model.py");
        if script.exists() {
            return Ok(script);
        }
    }

    // Bundled in resource dir
    if let Ok(resource_dir) = app.path().resource_dir() {
        let script = resource_dir.join("engine").join("download_model.py");
        if script.exists() {
            return Ok(script);
        }
    }

    // Flatpak path
    let flatpak_script = PathBuf::from("/app/lib/verify-me/engine/download_model.py");
    if flatpak_script.exists() {
        return Ok(flatpak_script);
    }

    Err("Download script (download_model.py) not found".to_string())
}

/// Resolve the requirements.txt path (for setting up Python environment).
pub fn resolve_requirements(app: &AppHandle) -> Result<PathBuf, String> {
    // Dev mode
    #[cfg(debug_assertions)]
    {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let req = project_root.join("engine").join("requirements.txt");
        if req.exists() {
            return Ok(req);
        }
    }

    // Bundled in resource dir
    if let Ok(resource_dir) = app.path().resource_dir() {
        let req = resource_dir.join("engine").join("requirements.txt");
        if req.exists() {
            return Ok(req);
        }
    }

    // Flatpak
    let flatpak_req = PathBuf::from("/app/lib/verify-me/engine/requirements.txt");
    if flatpak_req.exists() {
        return Ok(flatpak_req);
    }

    Err("requirements.txt not found".to_string())
}

/// Platform-specific venv Python path within an engine directory.
fn venv_python_path(engine_dir: &std::path::Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        engine_dir
            .join(".venv")
            .join("Scripts")
            .join("python.exe")
    } else {
        engine_dir.join(".venv").join("bin").join("python3")
    }
}

/// System Python command for the current platform.
fn system_python() -> PathBuf {
    if cfg!(target_os = "windows") {
        PathBuf::from("python")
    } else {
        PathBuf::from("python3")
    }
}
