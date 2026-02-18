#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod services;

use commands::{engine, models, recording, tts};

/// Set ORT_DYLIB_PATH if not already set, searching common locations.
fn setup_ort_dylib_path() {
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        return;
    }

    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        // User-local install
        format!("{}/.local/lib/onnxruntime/libonnxruntime.so", home),
        // System-wide
        "/usr/lib/libonnxruntime.so".to_string(),
        "/usr/lib/x86_64-linux-gnu/libonnxruntime.so".to_string(),
    ];

    for candidate in &candidates {
        let path = std::path::Path::new(candidate);
        if path.exists() {
            log::info!("Found ONNX Runtime at: {}", path.display());
            std::env::set_var("ORT_DYLIB_PATH", path);
            return;
        }
    }

    log::warn!("ONNX Runtime library not found — set ORT_DYLIB_PATH manually");
}

fn main() {
    env_logger::init();
    setup_ort_dylib_path();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_handle = app.handle().clone();
            services::path_service::init(&app_handle)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            engine::start_engine,
            engine::stop_engine,
            engine::engine_health,
            engine::get_device_info,
            tts::generate_speech,
            tts::voice_clone,
            tts::get_voices,
            models::list_models,
            models::download_model,
            models::download_hf_model,
            models::delete_model,
            models::get_models_directory,
            models::open_models_directory,
            recording::start_recording,
            recording::stop_recording,
            recording::get_recording_level,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
