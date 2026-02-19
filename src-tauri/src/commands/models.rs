use crate::services::path_service;
use serde::{Deserialize, Serialize};
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
    supports_voice_design: bool,
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
        supports_voice_design: false,
        bundled: true,
        download_url: None,
        hf_repo: None,
        voices: &[
            StaticVoice {
                id: "alba",
                name: "Alba (Male, Neutral)",
            },
            StaticVoice {
                id: "cosette",
                name: "Cosette (Female, Gentle)",
            },
            StaticVoice {
                id: "fantine",
                name: "Fantine (Female, Expressive)",
            },
            StaticVoice {
                id: "eponine",
                name: "Eponine (Female, British)",
            },
            StaticVoice {
                id: "azelma",
                name: "Azelma (Female, Youthful)",
            },
            StaticVoice {
                id: "jean",
                name: "Jean (Male, Warm)",
            },
            StaticVoice {
                id: "marius",
                name: "Marius (Male, Casual)",
            },
            StaticVoice {
                id: "javert",
                name: "Javert (Male, Authoritative)",
            },
        ],
    },
    CatalogEntry {
        id: "qwen3-tts",
        name: "Qwen 3 TTS",
        size: "~27 GB",
        supports_clone: true,
        supports_voice_prompt: true,
        supports_voice_design: true,
        bundled: false,
        download_url: None,
        hf_repo: Some("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        voices: &[
            StaticVoice {
                id: "Aiden",
                name: "Aiden (Male, American English)",
            },
            StaticVoice {
                id: "Ryan",
                name: "Ryan (Male, English)",
            },
            StaticVoice {
                id: "Vivian",
                name: "Vivian (Female, Chinese)",
            },
            StaticVoice {
                id: "Serena",
                name: "Serena (Female, Chinese)",
            },
            StaticVoice {
                id: "Dylan",
                name: "Dylan (Male, Chinese)",
            },
            StaticVoice {
                id: "Eric",
                name: "Eric (Male, Chinese/Sichuan)",
            },
            StaticVoice {
                id: "Uncle_Fu",
                name: "Uncle Fu (Male, Chinese)",
            },
            StaticVoice {
                id: "Ono_Anna",
                name: "Ono Anna (Female, Japanese)",
            },
            StaticVoice {
                id: "Sohee",
                name: "Sohee (Female, Korean)",
            },
        ],
    },
];

#[cfg(debug_assertions)]
static DEV_MODELS: &[CatalogEntry] = &[CatalogEntry {
    id: "qwen3-tts-safetensors",
    name: "Qwen 3 TTS (Safetensors) [DEV]",
    size: "~27 GB",
    supports_clone: true,
    supports_voice_prompt: true,
    supports_voice_design: true,
    bundled: false,
    download_url: None,
    hf_repo: None,
    voices: &[
        StaticVoice {
            id: "Aiden",
            name: "Aiden (Male, American English)",
        },
        StaticVoice {
            id: "Ryan",
            name: "Ryan (Male, English)",
        },
        StaticVoice {
            id: "Vivian",
            name: "Vivian (Female, Chinese)",
        },
        StaticVoice {
            id: "Serena",
            name: "Serena (Female, Chinese)",
        },
        StaticVoice {
            id: "Dylan",
            name: "Dylan (Male, Chinese)",
        },
        StaticVoice {
            id: "Eric",
            name: "Eric (Male, Chinese/Sichuan)",
        },
        StaticVoice {
            id: "Uncle_Fu",
            name: "Uncle Fu (Male, Chinese)",
        },
        StaticVoice {
            id: "Ono_Anna",
            name: "Ono Anna (Female, Japanese)",
        },
        StaticVoice {
            id: "Sohee",
            name: "Sohee (Female, Korean)",
        },
    ],
}];

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
    pub supports_voice_design: bool,
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

/// Check if a model exists in the given directory.
fn model_exists_in_dir(dir: &std::path::Path, model_id: &str) -> bool {
    let candidate = dir.join(model_id);

    if candidate.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&candidate) {
            let has_real_files = entries.flatten().any(|e| {
                e.path().is_file() && e.metadata().map(|m| m.len() > 1024).unwrap_or(false)
            });
            return has_real_files;
        }
        return false;
    }

    if candidate.is_file() {
        if let Ok(meta) = candidate.metadata() {
            return meta.len() > 1024;
        }
    }

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
    let resource_dir: Option<std::path::PathBuf> = app.path().resource_dir().ok();
    let app_data_models = path_service::get_models_dir().ok();

    #[cfg(debug_assertions)]
    let all_models: Vec<&CatalogEntry> = KNOWN_MODELS.iter().chain(DEV_MODELS.iter()).collect();
    #[cfg(not(debug_assertions))]
    let all_models: Vec<&CatalogEntry> = KNOWN_MODELS.iter().collect();

    let models = all_models
        .iter()
        .map(|entry| {
            let found = if entry.bundled {
                true
            } else {
                let mut on_disk = false;

                if let Some(ref res_dir) = resource_dir {
                    let bundled_dir = res_dir.join("models");
                    if bundled_dir.exists() && model_exists_in_dir(&bundled_dir, entry.id) {
                        on_disk = true;
                    }
                }

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
                supports_voice_design: entry.supports_voice_design,
                voices: entry
                    .voices
                    .iter()
                    .map(|v| VoiceInfo {
                        id: v.id.to_string(),
                        name: v.name.to_string(),
                    })
                    .collect(),
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

    let mut file =
        std::fs::File::create(&dest_path).map_err(|e| format!("Failed to create file: {}", e))?;

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

/// HuggingFace API file listing response
#[derive(Deserialize)]
struct HfFileInfo {
    #[serde(rename = "rfilename")]
    filename: String,
    size: Option<u64>,
}

/// Download all files from a HuggingFace repository using the API (pure Rust, no Python).
async fn download_hf_repo(
    client: &reqwest::Client,
    repo_id: &str,
    local_dir: &std::path::Path,
    token: &Option<String>,
    app: &AppHandle,
    progress_label: &str,
    progress_base: f32,
    progress_span: f32,
) -> Result<(), String> {
    std::fs::create_dir_all(local_dir)
        .map_err(|e| format!("Failed to create dir: {}", e))?;

    // List files in the repo via HF API
    let api_url = format!("https://huggingface.co/api/models/{}", repo_id);
    let mut req = client.get(&api_url);
    if let Some(ref tok) = token {
        if !tok.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", tok));
        }
    }

    let resp = req.send().await.map_err(|e| format!("HF API error: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("HF API returned status {}", resp.status()));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse HF API response: {}", e))?;

    let siblings = body["siblings"]
        .as_array()
        .ok_or("No files found in HF repo")?;

    let files: Vec<HfFileInfo> = siblings
        .iter()
        .filter_map(|s| serde_json::from_value(s.clone()).ok())
        .collect();

    let total_files = files.len() as f32;
    log::info!(
        "Downloading {} files from {} -> {}",
        files.len(),
        repo_id,
        local_dir.display()
    );

    for (i, file_info) in files.iter().enumerate() {
        let file_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, file_info.filename
        );

        let dest_path = local_dir.join(&file_info.filename);

        // Create parent directories for nested files
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create dir: {}", e))?;
        }

        // Skip if already downloaded and same size
        if dest_path.exists() {
            if let (Some(expected), Ok(meta)) = (file_info.size, dest_path.metadata()) {
                if meta.len() == expected {
                    log::debug!("Skipping {} (already exists)", file_info.filename);
                    continue;
                }
            }
        }

        log::info!("Downloading: {}", file_info.filename);

        let mut req = client.get(&file_url);
        if let Some(ref tok) = token {
            if !tok.is_empty() {
                req = req.header("Authorization", format!("Bearer {}", tok));
            }
        }

        let resp = req
            .send()
            .await
            .map_err(|e| format!("Download error for {}: {}", file_info.filename, e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Download failed for {}: status {}",
                file_info.filename,
                resp.status()
            ));
        }

        let total_bytes = resp.content_length();
        let mut downloaded_bytes: u64 = 0;
        let mut stream = resp.bytes_stream();
        let mut file = std::fs::File::create(&dest_path)
            .map_err(|e| format!("Failed to create {}: {}", file_info.filename, e))?;

        use futures_util::StreamExt;
        use std::io::Write;

        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| format!("Stream error for {}: {}", file_info.filename, e))?;
            file.write_all(&chunk)
                .map_err(|e| format!("Write error for {}: {}", file_info.filename, e))?;
            downloaded_bytes += chunk.len() as u64;
        }

        // Emit progress
        let file_progress = (i as f32 + 1.0) / total_files;
        let _ = app.emit(
            "model-download-progress",
            DownloadProgress {
                filename: progress_label.to_string(),
                downloaded: downloaded_bytes,
                total: total_bytes,
                percent: progress_base + progress_span * file_progress,
            },
        );
    }

    Ok(())
}

/// Additional HF repos to download alongside a model.
static COMPANION_DOWNLOADS: &[(&str, &str, &str)] = &[
    (
        "qwen3-tts",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "qwen3-tts-base",
    ),
    (
        "qwen3-tts",
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "qwen3-tts-voice-design",
    ),
];

#[tauri::command]
pub async fn download_hf_model(
    app: AppHandle,
    repo_id: String,
    model_id: String,
    token: Option<String>,
) -> Result<String, String> {
    let models_dir = path_service::get_models_dir()?;
    let local_dir = models_dir.join(&model_id);

    if local_dir.exists() && local_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&local_dir) {
            if entries.count() > 2 {
                return Err(format!("Model '{}' already exists", model_id));
            }
        }
    }

    let client = reqwest::Client::new();

    // Determine how many downloads we need
    let companions: Vec<_> = COMPANION_DOWNLOADS
        .iter()
        .filter(|(mid, _, _)| *mid == model_id)
        .collect();
    let total_downloads = 1 + companions.len();
    let span_per = 100.0 / total_downloads as f32;

    let _ = app.emit(
        "model-download-progress",
        DownloadProgress {
            filename: model_id.clone(),
            downloaded: 0,
            total: None,
            percent: 0.0,
        },
    );

    // Download primary model
    download_hf_repo(
        &client,
        &repo_id,
        &local_dir,
        &token,
        &app,
        &model_id,
        0.0,
        span_per,
    )
    .await?;

    // Download companion models
    for (i, (_, companion_repo, companion_subdir)) in companions.iter().enumerate() {
        let companion_dir = models_dir.join(companion_subdir);
        let already_exists = companion_dir.exists()
            && companion_dir.is_dir()
            && std::fs::read_dir(&companion_dir)
                .map(|e| e.count() > 2)
                .unwrap_or(false);

        if already_exists {
            log::info!(
                "Companion model {} already exists, skipping",
                companion_subdir
            );
            continue;
        }

        log::info!(
            "Downloading companion model: {} -> {}",
            companion_repo,
            companion_dir.display()
        );
        let base_pct = span_per * (i + 1) as f32;
        download_hf_repo(
            &client,
            companion_repo,
            &companion_dir,
            &token,
            &app,
            &model_id,
            base_pct,
            span_per,
        )
        .await?;
    }

    log::info!("HF model download complete: {}", model_id);
    Ok(local_dir.to_string_lossy().into_owned())
}

#[tauri::command]
pub async fn delete_model(model_id: String) -> Result<(), String> {
    let models_dir = path_service::get_models_dir()?;
    let model_path = models_dir.join(&model_id);

    if model_path.is_dir() {
        std::fs::remove_dir_all(&model_path)
            .map_err(|e| format!("Failed to delete model: {}", e))?;
        log::info!("Deleted model directory: {}", model_id);
        return Ok(());
    }

    if model_path.is_file() {
        std::fs::remove_file(&model_path).map_err(|e| format!("Failed to delete model: {}", e))?;
        log::info!("Deleted model: {}", model_id);
        return Ok(());
    }

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
        std::process::Command::new("open").arg(&models_dir).spawn()
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
