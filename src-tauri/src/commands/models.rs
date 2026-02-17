use crate::services::path_service;
use serde::Serialize;
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager};

struct StaticVoice {
    id: &'static str,
    name: &'static str,
}

struct CatalogEntry {
    id: &'static str,
    name: &'static str,
    size: &'static str,
    supports_clone: bool,
    supports_voice_prompt: bool,
    bundled: bool,
    download_url: Option<&'static str>,
    hf_repo: Option<&'static str>,
    voices: &'static [StaticVoice],
}

static KNOWN_MODELS: &[CatalogEntry] = &[
    CatalogEntry {
        id: "pocket-tts",
        name: "Pocket TTS",
        size: "~200 MB",
        supports_clone: true,
        supports_voice_prompt: false,
        bundled: true,
        download_url: None,
        hf_repo: None,
        voices: &[
            StaticVoice { id: "alba", name: "Alba (Male, Neutral)" },
            StaticVoice { id: "cosette", name: "Cosette (Female, Gentle)" },
            StaticVoice { id: "fantine", name: "Fantine (Female, Expressive)" },
            StaticVoice { id: "eponine", name: "Eponine (Female, British)" },
            StaticVoice { id: "azelma", name: "Azelma (Female, Youthful)" },
            StaticVoice { id: "jean", name: "Jean (Male, Warm)" },
            StaticVoice { id: "marius", name: "Marius (Male, Casual)" },
            StaticVoice { id: "javert", name: "Javert (Male, Authoritative)" },
        ],
    },
    CatalogEntry {
        id: "qwen3-tts",
        name: "Qwen 3 TTS",
        size: "~4.5 GB",
        supports_clone: false,
        supports_voice_prompt: true,
        bundled: false,
        download_url: None,
        hf_repo: Some("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        voices: &[
            StaticVoice { id: "Aiden", name: "Aiden (Male, American English)" },
            StaticVoice { id: "Ryan", name: "Ryan (Male, English)" },
            StaticVoice { id: "Vivian", name: "Vivian (Female, Chinese)" },
            StaticVoice { id: "Serena", name: "Serena (Female, Chinese)" },
            StaticVoice { id: "Dylan", name: "Dylan (Male, Chinese)" },
            StaticVoice { id: "Eric", name: "Eric (Male, Chinese/Sichuan)" },
            StaticVoice { id: "Uncle_Fu", name: "Uncle Fu (Male, Chinese)" },
            StaticVoice { id: "Ono_Anna", name: "Ono Anna (Female, Japanese)" },
            StaticVoice { id: "Sohee", name: "Sohee (Female, Korean)" },
        ],
    },
];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub size: String,
    pub status: String,
    pub supports_clone: bool,
    pub supports_voice_prompt: bool,
    pub voices: Vec<VoiceInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DownloadProgress {
    filename: String,
    downloaded: u64,
    total: Option<u64>,
    percent: f32,
}

/// Check if a model exists in the given directory (as subdirectory or file with matching stem).
/// For directories, validates that they contain at least one substantial file (>1KB)
/// to avoid false positives from empty dirs or error files.
fn model_exists_in_dir(dir: &std::path::Path, model_id: &str) -> bool {
    let candidate = dir.join(model_id);

    // Check for directory with actual model files
    if candidate.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&candidate) {
            let has_real_files = entries.flatten().any(|e| {
                e.path().is_file()
                    && e.metadata().map(|m| m.len() > 1024).unwrap_or(false)
            });
            return has_real_files;
        }
        return false;
    }

    // Check for exact file match (must be >1KB to be a real model)
    if candidate.is_file() {
        if let Ok(meta) = candidate.metadata() {
            return meta.len() > 1024;
        }
    }

    // Check for files with matching stem (e.g., pocket-tts.onnx)
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if stem == model_id {
                        if let Ok(meta) = path.metadata() {
                            if meta.len() > 1024 {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

#[tauri::command]
pub async fn list_models(app: AppHandle) -> Result<Vec<ModelInfo>, String> {
    let resource_dir = app.path().resource_dir().ok();
    let app_data_models = path_service::get_models_dir().ok();

    let models = KNOWN_MODELS
        .iter()
        .map(|entry| {
            // Bundled models are always available — shipped with the app
            let found = if entry.bundled {
                true
            } else {
                let mut on_disk = false;

                // Check bundled resource dir
                if let Some(ref res_dir) = resource_dir {
                    let bundled_dir = res_dir.join("models");
                    if bundled_dir.exists() && model_exists_in_dir(&bundled_dir, entry.id) {
                        on_disk = true;
                    }
                }

                // Check downloaded models in app_data_dir/models/
                if !on_disk {
                    if let Some(ref models_dir) = app_data_models {
                        if model_exists_in_dir(models_dir, entry.id) {
                            on_disk = true;
                        }
                    }
                }

                on_disk
            };

            ModelInfo {
                id: entry.id.to_string(),
                name: entry.name.to_string(),
                size: entry.size.to_string(),
                status: if found {
                    "available".to_string()
                } else {
                    "downloadable".to_string()
                },
                supports_clone: entry.supports_clone,
                supports_voice_prompt: entry.supports_voice_prompt,
                voices: entry.voices.iter().map(|v| VoiceInfo {
                    id: v.id.to_string(),
                    name: v.name.to_string(),
                }).collect(),
                download_url: entry.download_url.map(|s| s.to_string()),
                hf_repo: entry.hf_repo.map(|s| s.to_string()),
            }
        })
        .collect();

    Ok(models)
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

/// Resolve the Python path for running download scripts.
fn resolve_python_path(app: &AppHandle) -> Result<PathBuf, String> {
    #[cfg(debug_assertions)]
    {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let venv_python = project_root.join("engine/.venv/bin/python3");
        if venv_python.exists() {
            return Ok(venv_python);
        }
    }

    // Fallback to system python
    let _ = app; // suppress unused warning in release
    Ok(PathBuf::from("python3"))
}

/// Resolve the path to the download_model.py script.
fn resolve_download_script(app: &AppHandle) -> Result<PathBuf, String> {
    #[cfg(debug_assertions)]
    {
        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Failed to resolve project root")?
            .to_path_buf();

        let script = project_root.join("engine/download_model.py");
        if script.exists() {
            return Ok(script);
        }
    }

    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("Failed to get resource dir: {}", e))?;

    let script = resource_dir.join("engine").join("download_model.py");
    if script.exists() {
        return Ok(script);
    }

    Err("Download script not found".to_string())
}

#[tauri::command]
pub async fn download_hf_model(
    app: AppHandle,
    repo_id: String,
    model_id: String,
    token: Option<String>,
) -> Result<String, String> {
    use std::io::BufRead;
    use std::process::Stdio;

    let models_dir = path_service::get_models_dir()?;
    let local_dir = models_dir.join(&model_id);

    if local_dir.exists() && local_dir.is_dir() {
        // Check if directory has actual model files
        if let Ok(entries) = std::fs::read_dir(&local_dir) {
            if entries.count() > 2 {
                return Err(format!("Model '{}' already exists", model_id));
            }
        }
    }

    let python_path = resolve_python_path(&app)?;
    let script_path = resolve_download_script(&app)?;

    let args_json = serde_json::json!({
        "repo_id": repo_id,
        "local_dir": local_dir.to_string_lossy(),
        "token": token,
    })
    .to_string();

    log::info!("Starting HF model download: {} -> {}", repo_id, local_dir.display());

    // Emit initial progress
    let _ = app.emit(
        "model-download-progress",
        DownloadProgress {
            filename: model_id.clone(),
            downloaded: 0,
            total: None,
            percent: 0.0,
        },
    );

    let app_clone = app.clone();
    let model_id_clone = model_id.clone();
    let python_str = python_path.to_string_lossy().to_string();
    let script_str = script_path.to_string_lossy().to_string();

    let result = tokio::task::spawn_blocking(move || {
        let mut child = std::process::Command::new(&python_str)
            .arg(&script_str)
            .arg(&args_json)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to start download process: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or("Failed to capture download stdout")?;
        let reader = std::io::BufReader::new(stdout);

        let mut final_path = String::new();

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            log::debug!("Download output: {}", line);

            if let Ok(status) = serde_json::from_str::<serde_json::Value>(&line) {
                let status_str = status["status"].as_str().unwrap_or("");

                match status_str {
                    "downloading" => {
                        let _ = app_clone.emit(
                            "model-download-progress",
                            DownloadProgress {
                                filename: model_id_clone.clone(),
                                downloaded: 0,
                                total: None,
                                percent: 5.0, // Show some initial progress
                            },
                        );
                    }
                    "complete" => {
                        final_path =
                            status["path"].as_str().unwrap_or("").to_string();
                        let _ = app_clone.emit(
                            "model-download-progress",
                            DownloadProgress {
                                filename: model_id_clone.clone(),
                                downloaded: 1,
                                total: Some(1),
                                percent: 100.0,
                            },
                        );
                    }
                    "error" => {
                        let msg = status["message"]
                            .as_str()
                            .unwrap_or("Unknown download error");
                        return Err(format!("Download failed: {}", msg));
                    }
                    _ => {}
                }
            }
        }

        let exit_status = child.wait().map_err(|e| format!("Wait error: {}", e))?;
        if !exit_status.success() {
            return Err("Download process exited with error".to_string());
        }

        if final_path.is_empty() {
            return Err("Download did not return a model path".to_string());
        }

        Ok(final_path)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))??;

    log::info!("HF model download complete: {}", result);
    Ok(result)
}

#[tauri::command]
pub async fn delete_model(model_id: String) -> Result<(), String> {
    let models_dir = path_service::get_models_dir()?;
    let model_path = models_dir.join(&model_id);

    // Check for directory matching model_id
    if model_path.is_dir() {
        std::fs::remove_dir_all(&model_path)
            .map_err(|e| format!("Failed to delete model: {}", e))?;
        log::info!("Deleted model directory: {}", model_id);
        return Ok(());
    }

    // Check for exact file match
    if model_path.is_file() {
        std::fs::remove_file(&model_path)
            .map_err(|e| format!("Failed to delete model: {}", e))?;
        log::info!("Deleted model: {}", model_id);
        return Ok(());
    }

    // Check for file with matching stem (e.g., qwen3-tts.safetensors)
    let entries = std::fs::read_dir(&models_dir)
        .map_err(|e| format!("Failed to read models directory: {}", e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if stem == model_id {
                    std::fs::remove_file(&path)
                        .map_err(|e| format!("Failed to delete model: {}", e))?;
                    log::info!("Deleted model: {}", model_id);
                    return Ok(());
                }
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

#[tauri::command]
pub async fn open_models_directory() -> Result<(), String> {
    let models_dir = path_service::get_models_dir()?;

    let result = if cfg!(target_os = "linux") {
        std::process::Command::new("xdg-open")
            .arg(&models_dir)
            .spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open")
            .arg(&models_dir)
            .spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("explorer")
            .arg(&models_dir)
            .spawn()
    } else {
        return Err("Unsupported platform".to_string());
    };

    result.map_err(|e| format!("Failed to open directory: {}", e))?;
    Ok(())
}
