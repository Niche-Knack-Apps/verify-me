#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod services;

use commands::{engine, models, recording, tts};

fn main() {
    env_logger::init();

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
