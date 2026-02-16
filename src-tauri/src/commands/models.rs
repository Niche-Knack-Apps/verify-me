use crate::services::path_service;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub path: String,
    pub source: String, // "bundled" or "downloaded"
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct DownloadProgress {
    filename: String,
    downloaded: u64,
    total: Option<u64>,
    percent: f32,
}

#[tauri::command]
pub async fn list_models(app: AppHandle) -> Result<Vec<ModelInfo>, String> {
    let mut models = Vec::new();

    // Scan bundled models from resource_dir/models/
    if let Ok(resource_dir) = app.path().resource_dir() {
        let bundled_dir = resource_dir.join("models");
        if bundled_dir.exists() {
            scan_models_dir(&bundled_dir, "bundled", &mut models)?;
        }
    }

    // Scan downloaded models from app_data_dir/models/
    if let Ok(models_dir) = path_service::get_models_dir() {
        if models_dir.exists() {
            scan_models_dir(&models_dir, "downloaded", &mut models)?;
        }
    }

    Ok(models)
}

fn scan_models_dir(
    dir: &std::path::Path,
    source: &str,
    models: &mut Vec<ModelInfo>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read models directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            let size_bytes = std::fs::metadata(&path)
                .map(|m| m.len())
                .unwrap_or(0);

            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&filename)
                .to_string();

            models.push(ModelInfo {
                id,
                name: filename,
                path: path.to_string_lossy().into_owned(),
                source: source.to_string(),
                size_bytes,
            });
        }
    }

    Ok(())
}

#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    url: String,
    filename: String,
) -> Result<String, String> {
    let models_dir = path_service::get_models_dir()?;
    let dest_path = models_dir.join(&filename);

    if dest_path.exists() {
        return Err(format!("Model '{}' already exists", filename));
    }

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Download failed: {}", e))?;

    let total = response.content_length();

    let mut file = std::fs::File::create(&dest_path)
        .map_err(|e| format!("Failed to create file: {}", e))?;

    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    use futures_util::StreamExt;
    use std::io::Write;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download stream error: {}", e))?;
        file.write_all(&chunk)
            .map_err(|e| format!("File write error: {}", e))?;
        downloaded += chunk.len() as u64;

        let percent = total
            .map(|t| (downloaded as f32 / t as f32) * 100.0)
            .unwrap_or(0.0);

        let _ = app.emit(
            "model-download-progress",
            DownloadProgress {
                filename: filename.clone(),
                downloaded,
                total,
                percent,
            },
        );
    }

    Ok(dest_path.to_string_lossy().into_owned())
}

#[tauri::command]
pub async fn delete_model(model_id: String) -> Result<(), String> {
    let models_dir = path_service::get_models_dir()?;

    // Find the model file by ID (stem match)
    let entries = std::fs::read_dir(&models_dir)
        .map_err(|e| format!("Failed to read models directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if stem == model_id {
                std::fs::remove_file(&path)
                    .map_err(|e| format!("Failed to delete model: {}", e))?;
                log::info!("Deleted model: {}", model_id);
                return Ok(());
            }
        }
    }

    Err(format!("Model '{}' not found", model_id))
}

#[tauri::command]
pub async fn get_models_directory() -> Result<String, String> {
    let models_dir = path_service::get_models_dir()?;
    Ok(models_dir.to_string_lossy().into_owned())
}
