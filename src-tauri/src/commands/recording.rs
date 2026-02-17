use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample};
use tauri::AppHandle;
use tauri::Manager;

// ── Global state ────────────────────────────────────────────────

static RECORDING_ACTIVE: AtomicBool = AtomicBool::new(false);
static CURRENT_LEVEL: AtomicU32 = AtomicU32::new(0);
static SAMPLE_RATE: AtomicU32 = AtomicU32::new(0);
static CHANNELS: AtomicU32 = AtomicU32::new(0);

static RECORDING_STREAM: Mutex<Option<StreamHolder>> = Mutex::new(None);
static AUDIO_BUFFER: Mutex<Option<Vec<f32>>> = Mutex::new(None);

// ── StreamHolder (cpal::Stream is !Send) ────────────────────────

struct StreamHolder {
    #[allow(dead_code)] // Held for RAII — dropping stops the CPAL stream
    stream: cpal::Stream,
}
unsafe impl Send for StreamHolder {}

// ── Commands ────────────────────────────────────────────────────

#[tauri::command]
pub async fn start_recording(_app: AppHandle) -> Result<(), String> {
    if RECORDING_ACTIVE.load(Ordering::SeqCst) {
        return Err("Already recording".into());
    }

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    let supported_configs: Vec<_> = device
        .supported_input_configs()
        .map_err(|e| format!("Failed to query input configs: {e}"))?
        .collect();

    if supported_configs.is_empty() {
        return Err("No supported input configurations found".into());
    }

    // Pick best config: prefer f32, mono/stereo, 44100/48000
    let config_range = supported_configs
        .iter()
        .max_by_key(|c| {
            let mut score: i32 = 0;
            // Prefer f32
            if c.sample_format() == cpal::SampleFormat::F32 {
                score += 100;
            } else if c.sample_format() == cpal::SampleFormat::I16 {
                score += 50;
            }
            // Prefer mono for voice cloning
            if c.channels() == 1 {
                score += 20;
            } else if c.channels() == 2 {
                score += 10;
            }
            score
        })
        .ok_or("No suitable input configuration")?;

    // Pick sample rate: prefer 44100 or 48000
    let target_rate = cpal::SampleRate(44100);
    let config = if config_range.min_sample_rate() <= target_rate
        && config_range.max_sample_rate() >= target_rate
    {
        config_range.with_sample_rate(target_rate)
    } else {
        let rate48 = cpal::SampleRate(48000);
        if config_range.min_sample_rate() <= rate48 && config_range.max_sample_rate() >= rate48 {
            config_range.with_sample_rate(rate48)
        } else {
            config_range.with_max_sample_rate()
        }
    };

    let sample_format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();
    let channels = stream_config.channels as u32;
    let rate = stream_config.sample_rate.0;

    SAMPLE_RATE.store(rate, Ordering::SeqCst);
    CHANNELS.store(channels, Ordering::SeqCst);

    // Pre-allocate buffer for ~60s of audio (generous for short clips)
    let pre_alloc = (rate * channels * 60) as usize;
    {
        let mut buf = AUDIO_BUFFER.lock().map_err(|e| e.to_string())?;
        *buf = Some(Vec::with_capacity(pre_alloc));
    }

    RECORDING_ACTIVE.store(true, Ordering::SeqCst);
    CURRENT_LEVEL.store(0, Ordering::SeqCst);

    let err_callback = |err: cpal::StreamError| {
        log::error!("Recording stream error: {err}");
    };

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_input_stream::<f32>(&device, &stream_config, err_callback),
        cpal::SampleFormat::I16 => build_input_stream::<i16>(&device, &stream_config, err_callback),
        cpal::SampleFormat::U16 => build_input_stream::<u16>(&device, &stream_config, err_callback),
        cpal::SampleFormat::I32 => build_input_stream::<i32>(&device, &stream_config, err_callback),
        cpal::SampleFormat::U8 => build_input_stream::<u8>(&device, &stream_config, err_callback),
        _ => return Err(format!("Unsupported sample format: {sample_format:?}")),
    }
    .map_err(|e| format!("Failed to build input stream: {e}"))?;

    stream
        .play()
        .map_err(|e| format!("Failed to start stream: {e}"))?;

    {
        let mut holder = RECORDING_STREAM.lock().map_err(|e| e.to_string())?;
        *holder = Some(StreamHolder { stream });
    }

    log::info!("Recording started: {rate}Hz, {channels}ch, {sample_format:?}");
    Ok(())
}

#[tauri::command]
pub async fn stop_recording(app: AppHandle) -> Result<String, String> {
    if !RECORDING_ACTIVE.load(Ordering::SeqCst) {
        return Err("Not recording".into());
    }

    RECORDING_ACTIVE.store(false, Ordering::SeqCst);

    // Drop the stream to release the audio device
    {
        let mut holder = RECORDING_STREAM.lock().map_err(|e| e.to_string())?;
        *holder = None;
    }

    let channels = CHANNELS.load(Ordering::SeqCst) as u16;
    let sample_rate = SAMPLE_RATE.load(Ordering::SeqCst);

    // Take samples from buffer
    let samples = {
        let mut buf = AUDIO_BUFFER.lock().map_err(|e| e.to_string())?;
        buf.take().unwrap_or_default()
    };

    if samples.is_empty() {
        return Err("No audio recorded".into());
    }

    // Mix to mono if stereo
    let mono_samples = if channels == 2 {
        samples
            .chunks_exact(2)
            .map(|pair| (pair[0] + pair[1]) * 0.5)
            .collect::<Vec<f32>>()
    } else if channels > 2 {
        // Take first channel only
        samples
            .chunks_exact(channels as usize)
            .map(|frame| frame[0])
            .collect::<Vec<f32>>()
    } else {
        samples
    };

    // Write WAV file
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {e}"))?;
    let recordings_dir = data_dir.join("recordings");
    std::fs::create_dir_all(&recordings_dir)
        .map_err(|e| format!("Failed to create recordings dir: {e}"))?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let file_path = recordings_dir.join(format!("reference-{timestamp}.wav"));

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&file_path, spec)
        .map_err(|e| format!("Failed to create WAV file: {e}"))?;

    for &sample in &mono_samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let pcm16 = (clamped * 32767.0) as i16;
        writer
            .write_sample(pcm16)
            .map_err(|e| format!("Failed to write sample: {e}"))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV: {e}"))?;

    let duration_secs = mono_samples.len() as f64 / sample_rate as f64;
    let path_str = file_path.to_string_lossy().to_string();
    log::info!(
        "Recording saved: {path_str} ({:.1}s, {sample_rate}Hz mono)",
        duration_secs
    );

    CURRENT_LEVEL.store(0, Ordering::SeqCst);

    Ok(path_str)
}

#[tauri::command]
pub fn get_recording_level() -> f32 {
    let level_int = CURRENT_LEVEL.load(Ordering::SeqCst);
    level_int as f32 / 1000.0
}

// ── Stream builder ──────────────────────────────────────────────

fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    err_callback: impl Fn(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample + Send + 'static,
    f32: FromSample<T>,
{
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            if !RECORDING_ACTIVE.load(Ordering::Relaxed) {
                return;
            }

            let mut max_level: f32 = 0.0;
            let float_samples: Vec<f32> = data
                .iter()
                .map(|&s| {
                    let f: f32 = f32::from_sample(s);
                    let abs = f.abs();
                    if abs > max_level {
                        max_level = abs;
                    }
                    f
                })
                .collect();

            // Update level atomically (scale 0.0–1.0 → 0–1000)
            let level_int = (max_level.min(1.0) * 1000.0) as u32;
            CURRENT_LEVEL.store(level_int, Ordering::Relaxed);

            // Append to buffer
            if let Ok(mut buf) = AUDIO_BUFFER.lock() {
                if let Some(ref mut vec) = *buf {
                    vec.extend_from_slice(&float_samples);
                }
            }
        },
        err_callback,
        None,
    )
}
