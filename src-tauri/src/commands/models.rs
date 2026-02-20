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
    supports_voice_design: bool,
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
        supports_voice_design: false,
        bundled: true,
        download_url: None,
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
        download_url: Some("https://nicheknack.app/downloads/verify-me/models/qwen3-tts.tar.gz"),
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
}

#[derive(Debug, Clone, Serialize)]
struct DownloadProgress {
    filename: String,
    downloaded: u64,
    total: Option<u64>,
    percent: f32,
}

/// Map a catalog model_id to the actual on-disk directory name.
/// Most models use their ID directly; the dev-only safetensors variant
/// shares the qwen3-tts directory.
fn disk_model_id(model_id: &str) -> &str {
    match model_id {
        "qwen3-tts-safetensors" => "qwen3-tts",
        other => other,
    }
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
            // The on-disk directory name may differ from the catalog ID.
            // qwen3-tts-safetensors shares the qwen3-tts directory.
            let disk_id = disk_model_id(entry.id);

            let found = if entry.bundled {
                true
            } else {
                let mut on_disk = false;

                // Check bundled resources
                if let Some(ref res_dir) = resource_dir {
                    let bundled_dir = res_dir.join("models");
                    if bundled_dir.exists() && model_exists_in_dir(&bundled_dir, disk_id) {
                        on_disk = true;
                    }
                }

                // Check app_data/models/{disk_id}/
                if !on_disk {
                    if let Some(ref models_dir) = app_data_models {
                        if model_exists_in_dir(models_dir, disk_id) {
                            on_disk = true;
                        }
                    }
                }

                // Check app_data/models/onnx/{disk_id}/ (ONNX exports)
                if !on_disk {
                    if let Some(ref models_dir) = app_data_models {
                        let onnx_dir = models_dir.join("onnx");
                        if onnx_dir.exists() && model_exists_in_dir(&onnx_dir, disk_id) {
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
            }
        })
        .collect();

    Ok(models)
}

/// Additional tar.gz archives to download alongside a model.
/// (model_id, archive_url, target_subdir)
static COMPANION_DOWNLOADS: &[(&str, &str, &str)] = &[
    (
        "qwen3-tts",
        "https://nicheknack.app/downloads/verify-me/models/qwen3-tts-base.tar.gz",
        "qwen3-tts-base",
    ),
    (
        "qwen3-tts",
        "https://nicheknack.app/downloads/verify-me/models/qwen3-tts-voice-design.tar.gz",
        "qwen3-tts-voice-design",
    ),
];

/// Download a tar.gz archive and extract it to a directory.
async fn download_and_extract(
    app: &AppHandle,
    url: &str,
    extract_dir: &std::path::Path,
    progress_label: &str,
    progress_base: f32,
    progress_span: f32,
) -> Result<(), String> {
    std::fs::create_dir_all(extract_dir)
        .map_err(|e| format!("Failed to create dir: {}", e))?;

    let response = reqwest::get(url)
        .await
        .map_err(|e| format!("Download failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download failed: HTTP {}",
            response.status()
        ));
    }

    let total = response.content_length();
    let temp_path = extract_dir.join(".download.tar.gz.tmp");

    // Stream download to temp file
    {
        let mut file = std::fs::File::create(&temp_path)
            .map_err(|e| format!("Failed to create temp file: {}", e))?;

        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        use futures_util::StreamExt;
        use std::io::Write;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download stream error: {}", e))?;
            file.write_all(&chunk)
                .map_err(|e| format!("File write error: {}", e))?;
            downloaded += chunk.len() as u64;

            let download_pct = total
                .map(|t| (downloaded as f32 / t as f32) * 0.9)
                .unwrap_or(0.0);

            let _ = app.emit(
                "model-download-progress",
                DownloadProgress {
                    filename: progress_label.to_string(),
                    downloaded,
                    total,
                    percent: progress_base + progress_span * download_pct,
                },
            );
        }
    }

    // Extract tar.gz
    log::info!("Extracting archive to {}", extract_dir.display());
    let _ = app.emit(
        "model-download-progress",
        DownloadProgress {
            filename: progress_label.to_string(),
            downloaded: 0,
            total: None,
            percent: progress_base + progress_span * 0.95,
        },
    );

    let file = std::fs::File::open(&temp_path)
        .map_err(|e| format!("Failed to open temp file: {}", e))?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(gz);
    archive
        .unpack(extract_dir)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_path);

    log::info!("Extraction complete: {}", extract_dir.display());
    Ok(())
}

#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    url: String,
    model_id: String,
) -> Result<String, String> {
    let models_dir = path_service::get_models_dir()?;
    let model_dir = models_dir.join(&model_id);

    if model_exists_in_dir(&models_dir, &model_id) {
        return Err(format!("Model '{}' already exists", model_id));
    }

    // Find companion downloads for this model
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

    // Download and extract primary archive
    download_and_extract(&app, &url, &models_dir, &model_id, 0.0, span_per).await?;

    // Download and extract companion archives
    for (i, (_, companion_url, companion_subdir)) in companions.iter().enumerate() {
        if model_exists_in_dir(&models_dir, companion_subdir) {
            log::info!(
                "Companion model {} already exists, skipping",
                companion_subdir
            );
            continue;
        }

        log::info!(
            "Downloading companion model: {} -> {}",
            companion_url,
            companion_subdir
        );
        let base_pct = span_per * (i + 1) as f32;
        download_and_extract(
            &app,
            companion_url,
            &models_dir,
            &model_id,
            base_pct,
            span_per,
        )
        .await?;
    }

    let _ = app.emit(
        "model-download-progress",
        DownloadProgress {
            filename: model_id.clone(),
            downloaded: 0,
            total: None,
            percent: 100.0,
        },
    );

    log::info!("Model download complete: {}", model_id);
    Ok(model_dir.to_string_lossy().into_owned())
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
