use crate::services::path_service;
use serde::Serialize;
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
        voices: &[
            StaticVoice { id: "alba", name: "Alba (Female, Neutral)" },
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
        size: "~3.5 GB",
        supports_clone: true,
        supports_voice_prompt: true,
        bundled: false,
        download_url: Some("https://huggingface.co/Qwen/Qwen3-TTS/resolve/main/model.safetensors"),
        voices: &[
            StaticVoice { id: "default", name: "Default" },
            StaticVoice { id: "female-1", name: "Female 1" },
            StaticVoice { id: "male-1", name: "Male 1" },
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
}

#[derive(Debug, Clone, Serialize)]
struct DownloadProgress {
    filename: String,
    downloaded: u64,
    total: Option<u64>,
    percent: f32,
}

/// Check if a model exists in the given directory (as subdirectory or file with matching stem)
fn model_exists_in_dir(dir: &std::path::Path, model_id: &str) -> bool {
    // Check for exact path match (directory or extensionless file)
    if dir.join(model_id).exists() {
        return true;
    }
    // Check for files with matching stem (e.g., pocket-tts.onnx)
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if stem == model_id {
                        return true;
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
