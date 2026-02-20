use crate::services::tokenizers::HfTokenizer;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{DynValue, Tensor};
use rand::Rng;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SAMPLE_RATE: u32 = 24000;
/// Absolute ceiling — no single generation should exceed ~2 min of audio.
const MAX_STEPS_ABSOLUTE: usize = 1500;
/// ~12 Hz codec: frames per second of audio output.
const CODEC_FPS: f64 = 12.0;
/// Generous safety multiplier over the estimated duration.
const SAFETY_MULTIPLIER: f64 = 3.0;
/// Minimum cap so very short texts still get enough room.
const MIN_MAX_STEPS: usize = 48; // ~4s of audio
/// Maximum reference audio duration in seconds for voice cloning.
const MAX_REF_SECONDS: f32 = 15.0;
/// Wall-clock timeout for the decode loop (seconds). Prevents infinite hangs.
const GENERATION_TIMEOUT_SECS: u64 = 120;
/// Minimum available system memory (MB) required before loading a variant.
const MIN_AVAILABLE_MEMORY_MB: u64 = 1500;
/// Lower temperature for instruct mode to preserve voice conditioning signal.
/// With the default 0.9, stochastic sampling can degrade instruct effects
/// (e.g. "whisper" becomes faint rather than pronounced). 0.7 provides a
/// good balance between naturalness and instruction fidelity.
const INSTRUCT_TEMPERATURE: f32 = 0.7;

/// Estimate a reasonable max decode steps based on text length.
///
/// Heuristic: average speaking rate ~150 words/min = 2.5 words/sec.
/// At 12 Hz codec, that's ~4.8 frames per word.
/// We apply a 3× safety multiplier and clamp to [MIN_MAX_STEPS, MAX_STEPS_ABSOLUTE].
fn compute_max_steps(text: &str) -> usize {
    let word_count = text.split_whitespace().count().max(1);
    let estimated_seconds = word_count as f64 / 2.5;
    let estimated_frames = (estimated_seconds * CODEC_FPS * SAFETY_MULTIPLIER) as usize;
    estimated_frames.clamp(MIN_MAX_STEPS, MAX_STEPS_ABSOLUTE)
}

/// Build a natural-language pace instruction based on speed value.
/// Matches the Python adapter's _speed_instruct().
fn speed_instruct(speed: f32) -> &'static str {
    if speed <= 0.7 {
        "Speak very slowly and deliberately. "
    } else if speed <= 0.9 {
        "Speak at a slow, relaxed pace. "
    } else if speed < 1.1 {
        ""
    } else if speed <= 1.3 {
        "Speak at a quick pace. "
    } else {
        "Speak very fast. "
    }
}

/// Sampling parameters parsed from generation_config.json.
struct GenerationConfig {
    temperature: f32,
    top_k: usize,
    repetition_penalty: f32,
    subtalker_temperature: f32,
    subtalker_top_k: usize,
}

impl GenerationConfig {
    fn load(model_dir: &Path) -> Self {
        let path = model_dir.join("generation_config.json");
        let defaults = Self {
            temperature: 0.9,
            top_k: 50,
            repetition_penalty: 1.05,
            subtalker_temperature: 0.9,
            subtalker_top_k: 50,
        };

        let Ok(data) = std::fs::read_to_string(&path) else {
            log::warn!("No generation_config.json, using defaults");
            return defaults;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
            log::warn!("Invalid generation_config.json, using defaults");
            return defaults;
        };

        Self {
            temperature: json["temperature"].as_f64().unwrap_or(0.9) as f32,
            top_k: json["top_k"].as_u64().unwrap_or(50) as usize,
            repetition_penalty: json["repetition_penalty"].as_f64().unwrap_or(1.05) as f32,
            subtalker_temperature: json["subtalker_temperature"].as_f64().unwrap_or(0.9) as f32,
            subtalker_top_k: json["subtalker_top_k"].as_u64().unwrap_or(50) as usize,
        }
    }
}

/// Configuration parsed from the model's config.json.
struct Qwen3TTSConfig {
    // Codec special token IDs
    codec_bos_id: i64,
    codec_eos_token_id: i64,
    codec_pad_id: i64,
    codec_think_id: i64,
    codec_nothink_id: i64,
    codec_think_bos_id: i64,
    codec_think_eos_id: i64,

    // TTS special token IDs
    tts_bos: i64,
    tts_eos: i64,
    tts_pad: i64,

    // Chat token IDs
    im_start: i64,
    assistant: i64,

    // Speaker map: lowercase name → codec token ID
    spk_id: HashMap<String, i64>,

    // Language map: language name → codec language token ID
    codec_language_id: HashMap<String, i64>,

    // Speaker dialect map: speaker name → dialect name (empty if not dialect)
    spk_is_dialect: HashMap<String, String>,

    // Model dimensions
    num_hidden_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    num_code_groups: usize,
    codec_vocab_size: usize,

    // Code predictor dimensions (for KV cache)
    cp_num_layers: usize,
    cp_num_kv_heads: usize,
    cp_head_dim: usize,
}

impl Qwen3TTSConfig {
    fn load(model_dir: &Path) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let talker = &config["talker_config"];

        // Parse speaker IDs
        let mut spk_id = HashMap::new();
        if let Some(spk_obj) = talker["spk_id"].as_object() {
            for (name, id) in spk_obj {
                if let Some(id_val) = id.as_i64() {
                    spk_id.insert(name.clone(), id_val);
                }
            }
        }

        // Parse language IDs
        let mut codec_language_id = HashMap::new();
        if let Some(lang_obj) = talker["codec_language_id"].as_object() {
            for (name, id) in lang_obj {
                if let Some(id_val) = id.as_i64() {
                    codec_language_id.insert(name.clone(), id_val);
                }
            }
        }

        // Parse speaker dialect map
        let mut spk_is_dialect = HashMap::new();
        if let Some(dialect_obj) = talker["spk_is_dialect"].as_object() {
            for (name, val) in dialect_obj {
                if let Some(dialect) = val.as_str() {
                    spk_is_dialect.insert(name.clone(), dialect.to_string());
                }
                // false values are omitted (non-dialect speakers)
            }
        }

        let cp_config = &talker["code_predictor_config"];

        Ok(Self {
            codec_bos_id: talker["codec_bos_id"].as_i64().unwrap_or(2149),
            codec_eos_token_id: talker["codec_eos_token_id"].as_i64().unwrap_or(2150),
            codec_pad_id: talker["codec_pad_id"].as_i64().unwrap_or(2148),
            codec_think_id: talker["codec_think_id"].as_i64().unwrap_or(2154),
            codec_nothink_id: talker["codec_nothink_id"].as_i64().unwrap_or(2155),
            codec_think_bos_id: talker["codec_think_bos_id"].as_i64().unwrap_or(2156),
            codec_think_eos_id: talker["codec_think_eos_id"].as_i64().unwrap_or(2157),
            tts_bos: config["tts_bos_token_id"].as_i64().unwrap_or(151672),
            tts_eos: config["tts_eos_token_id"].as_i64().unwrap_or(151673),
            tts_pad: config["tts_pad_token_id"].as_i64().unwrap_or(151671),
            im_start: config["im_start_token_id"].as_i64().unwrap_or(151644),
            assistant: config["assistant_token_id"].as_i64().unwrap_or(77091),
            spk_id,
            codec_language_id,
            spk_is_dialect,
            num_hidden_layers: talker["num_hidden_layers"].as_u64().unwrap_or(28) as usize,
            num_kv_heads: talker["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            head_dim: talker["head_dim"].as_u64().unwrap_or(128) as usize,
            hidden_size: talker["hidden_size"].as_u64().unwrap_or(2048) as usize,
            num_code_groups: talker["num_code_groups"].as_u64().unwrap_or(16) as usize,
            codec_vocab_size: cp_config["vocab_size"].as_u64().unwrap_or(2048) as usize,
            cp_num_layers: cp_config["num_hidden_layers"].as_u64().unwrap_or(5) as usize,
            cp_num_kv_heads: cp_config["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            cp_head_dim: cp_config["head_dim"].as_u64().unwrap_or(128) as usize,
        })
    }
}

/// Which model variant is currently loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen3Variant {
    CustomVoice,
    VoiceDesign,
    Base,
}

impl Qwen3Variant {
    fn dirname(self) -> &'static str {
        match self {
            Self::CustomVoice => "qwen3-tts",
            Self::VoiceDesign => "qwen3-tts-voice-design",
            Self::Base => "qwen3-tts-base",
        }
    }
}

impl std::fmt::Display for Qwen3Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CustomVoice => write!(f, "CustomVoice"),
            Self::VoiceDesign => write!(f, "VoiceDesign"),
            Self::Base => write!(f, "Base"),
        }
    }
}

/// ONNX inference engine for Qwen3-TTS (0.6B / 1.7B variants).
///
/// Supports three model variants (CustomVoice, VoiceDesign, Base) that share
/// the same ONNX architecture but have different trained weights. Only one
/// variant is loaded at a time; switching reloads all ONNX sessions.
///
/// Pipeline:
///   build_prompt_embeddings → talker_prefill (with KV cache) →
///   talker_decode (autoregressive) + code_predictor →
///   tokenizer12hz_decode → audio output
pub struct Qwen3TTSEngine {
    variant: Qwen3Variant,
    /// Parent directory containing variant subdirs (e.g. models/onnx/)
    models_parent_dir: PathBuf,
    config: Qwen3TTSConfig,
    gen_config: GenerationConfig,
    text_project: Session,
    codec_embed: Session,
    code_predictor_prefill: Session,
    code_predictor_embed: Session,
    talker_prefill: Session,
    talker_decode: Session,
    tokenizer12hz_decode: Session,
    speaker_encoder: Option<Session>,
    tokenizer: HfTokenizer,
    /// Cached tts_pad text embedding [hidden_size], used as trailing_text_hidden in decode loop.
    tts_pad_embed: Vec<f32>,
}

impl Qwen3TTSEngine {
    /// Load all ONNX sessions and the BPE tokenizer from `model_dir`.
    pub fn initialize(model_dir: &Path) -> Result<Self, String> {
        let models_parent_dir = model_dir
            .parent()
            .ok_or_else(|| format!("Cannot determine parent of {}", model_dir.display()))?
            .to_path_buf();

        // Determine which variant this is based on directory name
        let dir_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let variant = match dir_name {
            "qwen3-tts-voice-design" => Qwen3Variant::VoiceDesign,
            "qwen3-tts-base" => Qwen3Variant::Base,
            _ => Qwen3Variant::CustomVoice,
        };

        Self::initialize_variant(model_dir, &models_parent_dir, variant)
    }

    /// Load all ONNX sessions for a specific variant from a directory.
    fn initialize_variant(
        model_dir: &Path,
        models_parent_dir: &Path,
        variant: Qwen3Variant,
    ) -> Result<Self, String> {
        log::info!(
            "Loading Qwen3 TTS ({}) from: {}",
            variant,
            model_dir.display()
        );

        let config = Qwen3TTSConfig::load(model_dir)?;
        let gen_config = GenerationConfig::load(model_dir);
        log::info!(
            "Config: hidden={}, layers={}, kv_heads={}, head_dim={}, speakers={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_kv_heads,
            config.head_dim,
            config.spk_id.len()
        );
        log::info!(
            "Generation: temp={}, top_k={}, rep_penalty={}, sub_temp={}, sub_top_k={}",
            gen_config.temperature,
            gen_config.top_k,
            gen_config.repetition_penalty,
            gen_config.subtalker_temperature,
            gen_config.subtalker_top_k
        );

        let tokenizer = HfTokenizer::load(model_dir)?;

        let mut text_project =
            load_session(&model_dir.join("text_project.onnx"), "text_project")?;
        let codec_embed = load_session(&model_dir.join("codec_embed.onnx"), "codec_embed")?;
        let code_predictor_prefill = load_session(
            &model_dir.join("code_predictor_prefill.onnx"),
            "code_predictor_prefill",
        )?;
        let code_predictor_embed = load_session(
            &model_dir.join("code_predictor_embed.onnx"),
            "code_predictor_embed",
        )?;
        let talker_prefill =
            load_session(&model_dir.join("talker_prefill.onnx"), "talker_prefill")?;
        let talker_decode =
            load_session(&model_dir.join("talker_decode.onnx"), "talker_decode")?;

        // tokenizer12hz_decode may be in model dir or shared from qwen3-tts variant
        let tok_decode_path = model_dir.join("tokenizer12hz_decode.onnx");
        let tok_decode_path = if tok_decode_path.exists() {
            tok_decode_path
        } else {
            let shared = models_parent_dir
                .join("qwen3-tts")
                .join("tokenizer12hz_decode.onnx");
            if shared.exists() {
                log::info!("Using shared tokenizer12hz_decode from qwen3-tts");
                shared
            } else {
                return Err(format!(
                    "tokenizer12hz_decode.onnx not found in {} or shared location",
                    model_dir.display()
                ));
            }
        };
        let tokenizer12hz_decode = load_session(&tok_decode_path, "tokenizer12hz_decode")?;

        // Speaker encoder is optional (only Base variant)
        let speaker_encoder_path = model_dir.join("speaker_encoder.onnx");
        let speaker_encoder = if speaker_encoder_path.exists() {
            Some(load_session(&speaker_encoder_path, "speaker_encoder")?)
        } else {
            log::info!("No speaker_encoder — voice cloning unavailable for this variant");
            None
        };

        // Pre-compute tts_pad text embedding for use in decode loop
        let tts_pad_embed = run_embed_single(&mut text_project, "input_ids", config.tts_pad)?;
        log::info!("Cached tts_pad_embed: {} floats", tts_pad_embed.len());

        log::info!("Qwen3 TTS ({}) initialized successfully", variant);

        Ok(Self {
            variant,
            models_parent_dir: models_parent_dir.to_path_buf(),
            config,
            gen_config,
            text_project,
            codec_embed,
            code_predictor_prefill,
            code_predictor_embed,
            talker_prefill,
            talker_decode,
            tokenizer12hz_decode,
            speaker_encoder,
            tokenizer,
            tts_pad_embed,
        })
    }

    /// Switch to a different model variant if not already loaded.
    ///
    /// Replaces ONNX sessions **one at a time** so peak memory overhead is only
    /// 1 duplicate session (~200-400 MB) instead of an entire duplicate engine
    /// (~1.7 GB). Rust's assignment drops the old session before the new one
    /// takes its place.
    fn ensure_variant(&mut self, target: Qwen3Variant) -> Result<(), String> {
        if self.variant == target {
            return Ok(());
        }

        log::info!(
            "Switching variant: {} -> {}",
            self.variant,
            target
        );

        let variant_dir = self.models_parent_dir.join(target.dirname());
        if !variant_dir.exists() {
            return Err(format!(
                "Variant directory not found: {}. Run the ONNX conversion for {} first.",
                variant_dir.display(),
                target
            ));
        }

        // Check available system memory before loading
        if let Some(avail_mb) = available_memory_mb() {
            log::info!("Available system memory: {} MB", avail_mb);
            if avail_mb < MIN_AVAILABLE_MEMORY_MB {
                return Err(format!(
                    "Insufficient memory to switch model variant. Available: {} MB, required: {} MB. \
                     Close other applications and try again.",
                    avail_mb, MIN_AVAILABLE_MEMORY_MB
                ));
            }
        }

        // Reload config, generation config, and tokenizer from new variant
        let config = Qwen3TTSConfig::load(&variant_dir)?;
        let gen_config = GenerationConfig::load(&variant_dir);
        let tokenizer = HfTokenizer::load(&variant_dir)?;

        log::info!(
            "Config: hidden={}, layers={}, kv_heads={}, head_dim={}, speakers={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_kv_heads,
            config.head_dim,
            config.spk_id.len()
        );

        // Replace sessions one at a time — each assignment drops the old session
        // before allocating the new one, keeping peak memory to ~1 extra session.
        log::info!("Replacing ONNX sessions incrementally...");

        self.text_project = load_session(&variant_dir.join("text_project.onnx"), "text_project")?;
        self.codec_embed = load_session(&variant_dir.join("codec_embed.onnx"), "codec_embed")?;
        self.code_predictor_prefill = load_session(
            &variant_dir.join("code_predictor_prefill.onnx"),
            "code_predictor_prefill",
        )?;
        self.code_predictor_embed = load_session(
            &variant_dir.join("code_predictor_embed.onnx"),
            "code_predictor_embed",
        )?;
        self.talker_prefill =
            load_session(&variant_dir.join("talker_prefill.onnx"), "talker_prefill")?;
        self.talker_decode =
            load_session(&variant_dir.join("talker_decode.onnx"), "talker_decode")?;

        // tokenizer12hz_decode may be in variant dir or shared from qwen3-tts
        let tok_decode_path = variant_dir.join("tokenizer12hz_decode.onnx");
        let tok_decode_path = if tok_decode_path.exists() {
            tok_decode_path
        } else {
            let shared = self
                .models_parent_dir
                .join("qwen3-tts")
                .join("tokenizer12hz_decode.onnx");
            if shared.exists() {
                log::info!("Using shared tokenizer12hz_decode from qwen3-tts");
                shared
            } else {
                return Err(format!(
                    "tokenizer12hz_decode.onnx not found in {} or shared location",
                    variant_dir.display()
                ));
            }
        };
        self.tokenizer12hz_decode = load_session(&tok_decode_path, "tokenizer12hz_decode")?;

        // Speaker encoder is optional (only Base variant)
        let speaker_encoder_path = variant_dir.join("speaker_encoder.onnx");
        self.speaker_encoder = if speaker_encoder_path.exists() {
            Some(load_session(&speaker_encoder_path, "speaker_encoder")?)
        } else {
            log::info!("No speaker_encoder — voice cloning unavailable for this variant");
            None
        };

        // Update non-session fields
        self.config = config;
        self.gen_config = gen_config;
        self.tokenizer = tokenizer;

        // Recompute tts_pad_embed from the new text_project
        self.tts_pad_embed =
            run_embed_single(&mut self.text_project, "input_ids", self.config.tts_pad)?;
        log::info!("Cached tts_pad_embed: {} floats", self.tts_pad_embed.len());

        self.variant = target;
        log::info!("Variant switch to {} complete", target);

        Ok(())
    }

    /// Generate speech from text using a preset voice.
    pub fn generate_speech(
        &mut self,
        text: &str,
        voice_id: &str,
        speed: f32,
        output_path: &Path,
    ) -> Result<(), String> {
        self.generate_speech_with_checkpoints(
            text, voice_id, speed, output_path, None, None, None, None,
        )
    }

    /// Generate speech with optional checkpoint emission for debugging.
    pub fn generate_speech_with_checkpoints(
        &mut self,
        text: &str,
        voice_id: &str,
        speed: f32,
        output_path: &Path,
        checkpoint_tx: Option<&std::sync::mpsc::Sender<serde_json::Value>>,
        voice_prompt: Option<&str>,
        voice_mode: Option<&str>,
        voice_description: Option<&str>,
    ) -> Result<(), String> {
        // Route to voice design mode if requested
        if voice_mode == Some("design") {
            return self.generate_voice_design(
                text,
                speed,
                output_path,
                checkpoint_tx,
                voice_description,
            );
        }

        // CustomVoice mode (default)
        self.ensure_variant(Qwen3Variant::CustomVoice)?;

        let total_start = std::time::Instant::now();

        // Build instruct from speed hint + user voice prompt
        let speed_hint = speed_instruct(speed);
        let user_prompt = voice_prompt
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .unwrap_or("");
        let instruct = format!("{}{}", speed_hint, user_prompt);
        let instruct = instruct.trim();
        let instruct_opt = if instruct.is_empty() {
            None
        } else {
            Some(instruct)
        };

        log::info!(
            "Qwen3 generating: voice={}, speed={}, instruct={:?}, text=\"{}\"",
            voice_id,
            speed,
            instruct_opt.map(|s| &s[..std::cmp::min(s.len(), 60)]),
            &text[..std::cmp::min(text.len(), 80)]
        );

        // 1. Build prompt embeddings
        let t = std::time::Instant::now();
        let (embeddings, seq_len, trailing_text_hidden, text_token_ids) =
            self.build_prompt_embeddings(text, voice_id, instruct_opt)?;
        let trailing_len = trailing_text_hidden.len() / self.config.hidden_size;
        let elapsed_ms = t.elapsed().as_millis();
        log::info!(
            "[1/5] Prompt embeddings: seq_len={}, trailing_text_hidden={} tokens ({} ms)",
            seq_len,
            trailing_len,
            elapsed_ms
        );
        if let Some(tx) = checkpoint_tx {
            let first_8: Vec<f32> = embeddings.iter().take(8).copied().collect();
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "embedding",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": "custom_voice",
                    "text": &text[..std::cmp::min(text.len(), 200)],
                    "text_len": text.len(),
                    "voice": voice_id,
                    "instruct": instruct_opt,
                    "seq_len": seq_len,
                    "trailing_tokens": trailing_len,
                    "text_token_ids": text_token_ids,
                    "first_8_values": first_8,
                    "elapsed_ms": elapsed_ms,
                }
            }));
        }

        // 2-6. Run the shared inference pipeline
        self.run_inference_pipeline(
            text,
            &embeddings,
            seq_len,
            &trailing_text_hidden,
            speed,
            output_path,
            checkpoint_tx,
            "custom_voice",
            total_start,
            instruct_opt.is_some(),
        )
    }

    /// Generate speech using the VoiceDesign model (voice from NL description).
    fn generate_voice_design(
        &mut self,
        text: &str,
        speed: f32,
        output_path: &Path,
        checkpoint_tx: Option<&std::sync::mpsc::Sender<serde_json::Value>>,
        voice_description: Option<&str>,
    ) -> Result<(), String> {
        self.ensure_variant(Qwen3Variant::VoiceDesign)?;

        let total_start = std::time::Instant::now();

        // Build instruct from speed hint + voice description
        let speed_hint = speed_instruct(speed);
        let desc = voice_description
            .map(|d| d.trim())
            .filter(|d| !d.is_empty())
            .unwrap_or("");
        let instruct = format!("{}{}", speed_hint, desc);
        let instruct = instruct.trim();

        if instruct.is_empty() {
            return Err(
                "Voice design requires a voice description (e.g. 'A young woman with a gentle British accent')"
                    .into(),
            );
        }

        log::info!(
            "Qwen3 VoiceDesign: speed={}, instruct=\"{}\", text=\"{}\"",
            speed,
            &instruct[..std::cmp::min(instruct.len(), 80)],
            &text[..std::cmp::min(text.len(), 80)]
        );

        // 1. Build voice design prompt embeddings (no speaker token)
        let t = std::time::Instant::now();
        let (embeddings, seq_len, trailing_text_hidden, text_token_ids) =
            self.build_voice_design_embeddings(text, instruct)?;
        let trailing_len = trailing_text_hidden.len() / self.config.hidden_size;
        let elapsed_ms = t.elapsed().as_millis();
        log::info!(
            "[1/5] VoiceDesign embeddings: seq_len={}, trailing={} tokens ({} ms)",
            seq_len,
            trailing_len,
            elapsed_ms
        );
        if let Some(tx) = checkpoint_tx {
            let first_8: Vec<f32> = embeddings.iter().take(8).copied().collect();
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "embedding",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": "voice_design",
                    "text": &text[..std::cmp::min(text.len(), 200)],
                    "text_len": text.len(),
                    "description": &instruct[..std::cmp::min(instruct.len(), 200)],
                    "seq_len": seq_len,
                    "trailing_tokens": trailing_len,
                    "text_token_ids": text_token_ids,
                    "first_8_values": first_8,
                    "elapsed_ms": elapsed_ms,
                }
            }));
        }

        // 2-6. Shared inference pipeline
        // VoiceDesign uses default temperature (0.9) — the voice description is the
        // primary conditioning signal baked into the model, not an optional modifier
        // like CustomVoice instruct. Lower temperature flattens the design effect.
        self.run_inference_pipeline(
            text,
            &embeddings,
            seq_len,
            &trailing_text_hidden,
            speed,
            output_path,
            checkpoint_tx,
            "voice_design",
            total_start,
            false,
        )
    }

    /// Generate speech using reference audio for voice cloning (Base variant).
    pub fn clone_voice(
        &mut self,
        text: &str,
        reference_audio: &Path,
        speed: f32,
        output_path: &Path,
        checkpoint_tx: Option<&std::sync::mpsc::Sender<serde_json::Value>>,
    ) -> Result<(), String> {
        self.ensure_variant(Qwen3Variant::Base)?;

        if self.speaker_encoder.is_none() {
            return Err(
                "Voice cloning requires the Base model variant with speaker_encoder.onnx".into(),
            );
        }

        let total_start = std::time::Instant::now();
        log::info!(
            "Qwen3 voice clone: ref={}, speed={}, text=\"{}\"",
            reference_audio.display(),
            speed,
            &text[..std::cmp::min(text.len(), 80)]
        );

        // 1. Load and process reference audio
        let t = std::time::Instant::now();
        let (samples, ref_sr) = load_wav(reference_audio)?;
        let samples = resample_to_24k(&samples, ref_sr);
        let max_samples = (MAX_REF_SECONDS * SAMPLE_RATE as f32) as usize;
        let samples = if samples.len() > max_samples {
            log::info!(
                "Trimming ref audio: {} -> {} samples ({:.1}s)",
                samples.len(),
                max_samples,
                MAX_REF_SECONDS
            );
            samples[..max_samples].to_vec()
        } else {
            samples
        };
        let ref_duration_s = samples.len() as f32 / SAMPLE_RATE as f32;
        let ref_load_ms = t.elapsed().as_millis();
        log::info!(
            "Reference audio: {} samples ({:.1}s) at {}Hz ({} ms)",
            samples.len(),
            ref_duration_s,
            SAMPLE_RATE,
            ref_load_ms
        );
        if let Some(tx) = checkpoint_tx {
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "ref_audio_loaded",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": "voice_clone",
                    "ref_path": reference_audio.display().to_string(),
                    "ref_samples": samples.len(),
                    "ref_duration_s": ref_duration_s,
                    "ref_sample_rate": SAMPLE_RATE,
                    "original_sample_rate": ref_sr,
                    "elapsed_ms": ref_load_ms,
                }
            }));
        }

        // 2. Extract mel spectrogram and run speaker encoder
        let t = std::time::Instant::now();
        let mel = extract_mel(&samples, SAMPLE_RATE);
        let mel_frames = mel.len() / 128;
        let mel_ms = t.elapsed().as_millis();
        log::info!(
            "Mel spectrogram: {} frames x 128 bins ({} ms)",
            mel_frames,
            mel_ms
        );
        if let Some(tx) = checkpoint_tx {
            let first_8_mel: Vec<f32> = mel.iter().take(8).copied().collect();
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "mel_spectrogram",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": "voice_clone",
                    "mel_frames": mel_frames,
                    "mel_bins": 128,
                    "first_8_values": first_8_mel,
                    "elapsed_ms": mel_ms,
                }
            }));
        }

        let t = std::time::Instant::now();
        let speaker_embed = self.run_speaker_encoder(&mel, mel_frames)?;
        let spk_ms = t.elapsed().as_millis();
        log::info!(
            "Speaker embedding: {} dims ({} ms)",
            speaker_embed.len(),
            spk_ms
        );
        if let Some(tx) = checkpoint_tx {
            let first_8_spk: Vec<f32> = speaker_embed.iter().take(8).copied().collect();
            let spk_norm: f32 = speaker_embed.iter().map(|x| x * x).sum::<f32>().sqrt();
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "speaker_encoder",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": "voice_clone",
                    "embedding_dims": speaker_embed.len(),
                    "first_8_values": first_8_spk,
                    "l2_norm": spk_norm,
                    "elapsed_ms": spk_ms,
                }
            }));
        }

        // 3. Build clone prompt embeddings with speaker embedding injected
        let t = std::time::Instant::now();
        let (embeddings, seq_len, trailing_text_hidden, text_token_ids) =
            self.build_clone_embeddings(text, &speaker_embed)?;
        let trailing_len = trailing_text_hidden.len() / self.config.hidden_size;
        let embed_ms = t.elapsed().as_millis();
        log::info!(
            "[1/5] Clone embeddings: seq_len={}, trailing={} tokens ({} ms)",
            seq_len,
            trailing_len,
            embed_ms
        );
        if let Some(tx) = checkpoint_tx {
            let first_8: Vec<f32> = embeddings.iter().take(8).copied().collect();
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "embedding",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": "voice_clone",
                    "text": &text[..std::cmp::min(text.len(), 200)],
                    "text_len": text.len(),
                    "seq_len": seq_len,
                    "trailing_tokens": trailing_len,
                    "text_token_ids": text_token_ids,
                    "first_8_values": first_8,
                    "elapsed_ms": embed_ms,
                }
            }));
        }

        // 4-6. Shared inference pipeline (voice clone has no instruct)
        self.run_inference_pipeline(
            text,
            &embeddings,
            seq_len,
            &trailing_text_hidden,
            speed,
            output_path,
            checkpoint_tx,
            "voice_clone",
            total_start,
            false,
        )
    }

    /// Shared inference pipeline: prefill → decode → audio decode → normalize → write WAV.
    fn run_inference_pipeline(
        &mut self,
        text: &str,
        embeddings: &[f32],
        seq_len: usize,
        trailing_text_hidden: &[f32],
        speed: f32,
        output_path: &Path,
        checkpoint_tx: Option<&std::sync::mpsc::Sender<serde_json::Value>>,
        mode: &str,
        total_start: std::time::Instant,
        has_instruct: bool,
    ) -> Result<(), String> {
        // 2. Prefill
        let t = std::time::Instant::now();
        let (last_logits, last_hidden, kv_cache, kv_seq_len) =
            self.run_prefill(embeddings, seq_len)?;
        let elapsed_ms = t.elapsed().as_millis();
        log::info!(
            "[2/5] Prefill done: kv_seq_len={} ({} ms)",
            kv_seq_len,
            elapsed_ms
        );
        if let Some(tx) = checkpoint_tx {
            let argmax = last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut indexed: Vec<(usize, f32)> =
                last_logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top5: Vec<serde_json::Value> = indexed
                .iter()
                .take(5)
                .map(|(i, v)| serde_json::json!({"idx": i, "logit": v}))
                .collect();
            let first_8_hidden: Vec<f32> = last_hidden.iter().take(8).copied().collect();
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "prefill",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": mode,
                    "kv_seq_len": kv_seq_len,
                    "argmax": argmax,
                    "top5_logits": top5,
                    "first_8_hidden": first_8_hidden,
                    "elapsed_ms": elapsed_ms,
                }
            }));
        }

        // 3. Autoregressive decode
        let max_steps = compute_max_steps(text);
        log::info!(
            "Max decode steps for {} words: {}",
            text.split_whitespace().count(),
            max_steps
        );
        let t = std::time::Instant::now();
        let frames = self.run_decode_loop(
            last_logits,
            last_hidden,
            kv_cache,
            kv_seq_len,
            max_steps,
            trailing_text_hidden,
            has_instruct,
        )?;
        let elapsed_ms = t.elapsed().as_millis();
        log::info!(
            "[3/5] Generated {} code frames ({} ms)",
            frames.len(),
            elapsed_ms
        );
        if let Some(tx) = checkpoint_tx {
            let sample_frames: Vec<Vec<i64>> =
                frames.iter().take(5).map(|f| f.to_vec()).collect();
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "decode_summary",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": mode,
                    "total_frames": frames.len(),
                    "max_steps": max_steps,
                    "generation_ms": elapsed_ms,
                    "first_5_frames": sample_frames,
                }
            }));
        }

        // 4. Decode audio
        let t = std::time::Instant::now();
        let mut audio = self.decode_audio(&frames)?;
        let elapsed_ms = t.elapsed().as_millis();
        let duration_sec = audio.len() as f64 / SAMPLE_RATE as f64;
        log::info!(
            "[4/5] Decoded {} audio samples ({} ms)",
            audio.len(),
            elapsed_ms
        );
        if let Some(tx) = checkpoint_tx {
            let rms = {
                let sum_sq: f64 = audio.iter().map(|&s| (s as f64) * (s as f64)).sum();
                (sum_sq / audio.len().max(1) as f64).sqrt()
            };
            let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "audio_decode",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": mode,
                    "num_frames": frames.len(),
                    "num_samples": audio.len(),
                    "duration_sec": duration_sec,
                    "sample_rate": SAMPLE_RATE,
                    "rms": rms,
                    "peak": peak,
                    "elapsed_ms": elapsed_ms,
                }
            }));
        }

        // 5. Clip audio to [-1.0, 1.0] (no peak normalization — preserve
        //    amplitude dynamics so whispers stay quiet and shouts stay loud)
        let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 1.0 {
            let gain = 1.0 / peak;
            for s in &mut audio {
                *s *= gain;
            }
            log::info!("[4.5] Clipped: peak {:.4} -> 1.0", peak);
        }

        // 6. Speed adjustment + write WAV
        if (speed - 1.0).abs() > 0.01 {
            audio = resample_audio(&audio, speed);
        }
        write_wav(&audio, SAMPLE_RATE, output_path)?;

        let total_ms = total_start.elapsed().as_millis();
        let duration_sec = audio.len() as f64 / SAMPLE_RATE as f64;
        log::info!(
            "[5/5] WAV written: {:.1}s audio in {} ms total",
            duration_sec,
            total_ms
        );
        if let Some(tx) = checkpoint_tx {
            let rms = {
                let sum_sq: f64 = audio.iter().map(|&s| (s as f64) * (s as f64)).sum();
                (sum_sq / audio.len().max(1) as f64).sqrt()
            };
            let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            let _ = tx.send(serde_json::json!({
                "engine": "onnx",
                "stage": "complete",
                "timestamp": chrono_millis(),
                "data": {
                    "mode": mode,
                    "num_samples": audio.len(),
                    "duration_sec": duration_sec,
                    "sample_rate": SAMPLE_RATE,
                    "rms": rms,
                    "peak": peak,
                    "total_ms": total_ms,
                }
            }));
        }

        Ok(())
    }

    /// Returns available voice presets from config.
    pub fn get_available_voices(&self) -> Vec<String> {
        let mut voices: Vec<String> = self
            .config
            .spk_id
            .keys()
            .map(|name| {
                // Capitalize first letter
                let mut chars = name.chars();
                match chars.next() {
                    Some(c) => c.to_uppercase().to_string() + chars.as_str(),
                    None => String::new(),
                }
            })
            .collect();
        voices.sort();
        voices
    }

    /// Drop all sessions and release resources.
    pub fn shutdown(self) {
        log::info!("Shutting down Qwen3 TTS engine");
    }

    // ── Prompt Embedding Construction ──────────────────────────────

    /// Determine the codec language token ID for a voice + text combination.
    ///
    /// Priority: speaker dialect → text-based language detection → English default.
    fn determine_language_id(&self, voice: &str, text: &str) -> i64 {
        let default_english = 2050;

        // 1. Check if speaker has a dialect
        if let Some(dialect) = self.config.spk_is_dialect.get(voice) {
            if let Some(&lang_id) = self.config.codec_language_id.get(dialect.as_str()) {
                return lang_id;
            }
        }

        self.detect_language_from_text(text, default_english)
    }

    /// Detect language from text using character range heuristics.
    fn detect_language_from_text(&self, text: &str, default: i64) -> i64 {
        let has_hangul = text.chars().any(|c| {
            matches!(c, '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}')
        });
        let has_japanese = text.chars().any(|c| {
            matches!(c, '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}')
        });
        let has_cjk = text.chars().any(|c| {
            matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}')
        });

        if has_hangul {
            *self
                .config
                .codec_language_id
                .get("korean")
                .unwrap_or(&default)
        } else if has_japanese {
            *self
                .config
                .codec_language_id
                .get("japanese")
                .unwrap_or(&default)
        } else if has_cjk {
            *self
                .config
                .codec_language_id
                .get("chinese")
                .unwrap_or(&default)
        } else {
            *self
                .config
                .codec_language_id
                .get("english")
                .unwrap_or(&default)
        }
    }

    /// Build instruct embeddings by tokenizing and projecting through text_project.
    ///
    /// Wraps instruct text in chat template: `<|im_start|>user\n{instruct}<|im_end|>\n`
    /// Returns flat embeddings [N * hidden_size].
    fn build_instruct_embeddings(&mut self, instruct: &str) -> Result<Vec<f32>, String> {
        let formatted = format!(
            "<|im_start|>user\n{}<|im_end|>\n",
            instruct
        );
        let instruct_ids = self.tokenizer.tokenize_with_special_tokens(&formatted);
        let n = instruct_ids.len();
        if n == 0 {
            return Err("Instruct text produced no tokens".into());
        }
        log::info!(
            "Instruct tokens: {} -> {:?}{}",
            n,
            &instruct_ids[..std::cmp::min(n, 10)],
            if n > 10 { "..." } else { "" }
        );

        run_embed_batch(&mut self.text_project, "input_ids", &instruct_ids)
    }

    /// Build the full prompt embedding sequence for CustomVoice TTS (non-streaming mode).
    ///
    /// Non-streaming mode puts ALL text tokens in the prefill so the model has
    /// full text context when generating speech. This is required for instruct
    /// (voice instructions) to work correctly.
    ///
    /// Non-streaming layout (matches Python `[:, :-1]` which removes streaming's tts_bos+PAD):
    ///   [instruct(N)?, role(3), mixed(5), text_tokens+codec_pad(T), tts_eos+codec_pad(1), tts_pad+codec_bos(1)]
    ///
    /// Returns (flat_embeddings, prefill_len, trailing_text_hidden, text_token_ids).
    /// trailing_text_hidden is tts_pad_embed (constant for all decode steps).
    fn build_prompt_embeddings(
        &mut self,
        text: &str,
        voice_id: &str,
        instruct: Option<&str>,
    ) -> Result<(Vec<f32>, usize, Vec<f32>, Vec<i64>), String> {
        let h = self.config.hidden_size;

        // Copy config values we need before calling &mut self methods
        let im_start = self.config.im_start;
        let assistant = self.config.assistant;
        let tts_pad = self.config.tts_pad;
        let tts_eos = self.config.tts_eos;
        let codec_think_id = self.config.codec_think_id;
        let codec_think_bos_id = self.config.codec_think_bos_id;
        let codec_think_eos_id = self.config.codec_think_eos_id;
        let codec_pad_id = self.config.codec_pad_id;
        let codec_bos_id = self.config.codec_bos_id;

        // Look up speaker codec token ID
        let voice_lower = voice_id.to_lowercase();
        let spk_token = *self.config.spk_id.get(&voice_lower).ok_or_else(|| {
            format!(
                "Unknown voice '{}'. Available: {:?}",
                voice_id,
                self.config.spk_id.keys().collect::<Vec<_>>()
            )
        })?;

        // Determine language token for this voice + text
        let language_id = self.determine_language_id(&voice_lower, text);
        log::info!("Language ID for voice '{}': {}", voice_lower, language_id);

        // Tokenize the text (HuggingFace ByteLevel BPE)
        let text_tokens = self.tokenizer.tokenize(text);
        let text_len = text_tokens.len();
        log::info!(
            "HF tokens for '{}': {} tokens -> {:?}{}",
            &text[..std::cmp::min(text.len(), 40)],
            text_len,
            &text_tokens[..std::cmp::min(text_len, 20)],
            if text_len > 20 { "..." } else { "" }
        );
        if text_len == 0 {
            return Err("Cannot generate speech from empty text".into());
        }

        // Build instruct embeddings if provided
        let instruct_embeds = match instruct {
            Some(inst) if !inst.is_empty() => Some(self.build_instruct_embeddings(inst)?),
            _ => None,
        };
        let instruct_positions = instruct_embeds.as_ref().map_or(0, |e| e.len() / h);

        // Non-streaming layout (Python removes streaming's last position tts_bos+PAD):
        //   [instruct(N)?, role(3), mixed(5), text_tokens+codec_pad(T), tts_eos+codec_pad(1), tts_pad+codec_bos(1)]
        // Total = N + 3 + 5 + T + 1 + 1 = N + T + 10
        let mixed_count = 5;
        let prefill_len = instruct_positions + 3 + mixed_count + text_len + 1 + 1;

        // ── Build text token IDs for role + mixed section (3 + 5 = 8 positions) ──
        let newline: i64 = 198;
        let role_mixed_len = 3 + mixed_count;
        let mut role_mixed_ids: Vec<i64> = Vec::with_capacity(role_mixed_len);
        // Role prefix (3)
        role_mixed_ids.push(im_start);
        role_mixed_ids.push(assistant);
        role_mixed_ids.push(newline);
        // Mixed text side (5): tts_pad × 5 (no tts_bos — removed in non-streaming)
        for _ in 0..mixed_count {
            role_mixed_ids.push(tts_pad);
        }

        let role_mixed_embeds =
            run_embed_batch(&mut self.text_project, "input_ids", &role_mixed_ids)?;

        // ── Build text token embeddings for ALL text tokens + tts_eos ──
        let mut text_and_eos_ids: Vec<i64> = Vec::with_capacity(text_len + 1);
        text_and_eos_ids.extend_from_slice(&text_tokens);
        text_and_eos_ids.push(tts_eos);
        let text_and_eos_embeds =
            run_embed_batch(&mut self.text_project, "input_ids", &text_and_eos_ids)?;

        // ── Build codec token IDs (7 tokens) ──
        // [think, think_bos, language_id, think_eos, spk, pad, bos]
        let codec_ids = vec![
            codec_think_id,
            codec_think_bos_id,
            language_id,
            codec_think_eos_id,
            spk_token,
            codec_pad_id,
            codec_bos_id,
        ];
        let codec_embeds = run_embed_batch(&mut self.codec_embed, "input_ids", &codec_ids)?;
        assert_eq!(codec_embeds.len(), 7 * h);

        // Extract codec_pad and codec_bos embeddings for text positions
        let codec_pad_embed = &codec_embeds[5 * h..6 * h];
        let codec_bos_embed = &codec_embeds[6 * h..7 * h];

        // ── Construct final embedding ──
        let mut final_embeds = Vec::with_capacity(prefill_len * h);

        // 1. Instruct embeddings (pure text, no codec)
        if let Some(ref ie) = instruct_embeds {
            final_embeds.extend_from_slice(ie);
        }

        // 2. Role positions (3): pure text
        for pos in 0..3 {
            let src = &role_mixed_embeds[pos * h..(pos + 1) * h];
            final_embeds.extend_from_slice(src);
        }

        // 3. Mixed positions (5): tts_pad + codec[0..4] = [THINK, THINK_BOS, LANG, THINK_EOS, SPEAKER]
        for pos in 0..mixed_count {
            let text_src = &role_mixed_embeds[(3 + pos) * h..(3 + pos + 1) * h];
            let codec_src = &codec_embeds[pos * h..(pos + 1) * h];
            for j in 0..h {
                final_embeds.push(text_src[j] + codec_src[j]);
            }
        }

        // 4. ALL text tokens + codec_pad (T positions)
        for t in 0..text_len {
            let text_src = &text_and_eos_embeds[t * h..(t + 1) * h];
            for j in 0..h {
                final_embeds.push(text_src[j] + codec_pad_embed[j]);
            }
        }

        // 5. tts_eos + codec_pad (1 position)
        let tts_eos_src = &text_and_eos_embeds[text_len * h..(text_len + 1) * h];
        for j in 0..h {
            final_embeds.push(tts_eos_src[j] + codec_pad_embed[j]);
        }

        // 6. tts_pad + codec_bos (1 position)
        for j in 0..h {
            final_embeds.push(self.tts_pad_embed[j] + codec_bos_embed[j]);
        }

        assert_eq!(final_embeds.len(), prefill_len * h);

        // Non-streaming: trailing_text_hidden is just tts_pad_embed (constant for all decode steps)
        let trailing_text_hidden = self.tts_pad_embed.clone();

        Ok((
            final_embeds,
            prefill_len,
            trailing_text_hidden,
            text_tokens,
        ))
    }

    /// Build prompt embeddings for VoiceDesign mode (non-streaming, no speaker token, no language).
    ///
    /// VoiceDesign uses language=None, so codec = [nothink, think_bos, think_eos, pad, bos] (5 tokens).
    ///
    /// Non-streaming layout (matches Python `[:, :-1]` which removes streaming's tts_bos+PAD):
    ///   [instruct(N), role(3), mixed(3), text_tokens+codec_pad(T), tts_eos+codec_pad(1), tts_pad+codec_bos(1)]
    fn build_voice_design_embeddings(
        &mut self,
        text: &str,
        instruct: &str,
    ) -> Result<(Vec<f32>, usize, Vec<f32>, Vec<i64>), String> {
        let h = self.config.hidden_size;

        // Copy config values before calling &mut self methods
        let im_start = self.config.im_start;
        let assistant = self.config.assistant;
        let tts_pad = self.config.tts_pad;
        let tts_eos = self.config.tts_eos;
        let codec_nothink_id = self.config.codec_nothink_id;
        let codec_think_bos_id = self.config.codec_think_bos_id;
        let codec_think_eos_id = self.config.codec_think_eos_id;
        let codec_pad_id = self.config.codec_pad_id;
        let codec_bos_id = self.config.codec_bos_id;

        // Tokenize the text
        let text_tokens = self.tokenizer.tokenize(text);
        let text_len = text_tokens.len();
        if text_len == 0 {
            return Err("Cannot generate speech from empty text".into());
        }

        // Build instruct embeddings (required for VoiceDesign)
        let instruct_embeds = self.build_instruct_embeddings(instruct)?;
        let instruct_positions = instruct_embeds.len() / h;

        // Non-streaming mixed: Python builds streaming prefill with (codec_len-1) mixed positions
        // then removes the last (tts_bos+PAD). For VoiceDesign (5 codec), mixed = 5-1-1 = 3.
        // Mixed pairs: [tts_pad+NOTHINK, tts_pad+THINK_BOS, tts_pad+THINK_EOS]
        let mixed_count = 3;
        let prefill_len = instruct_positions + 3 + mixed_count + text_len + 1 + 1;

        // Role + mixed text tokens (3 + 3 = 6)
        let newline: i64 = 198;
        let role_mixed_len = 3 + mixed_count;
        let mut role_mixed_ids: Vec<i64> = Vec::with_capacity(role_mixed_len);
        role_mixed_ids.push(im_start);
        role_mixed_ids.push(assistant);
        role_mixed_ids.push(newline);
        for _ in 0..mixed_count {
            role_mixed_ids.push(tts_pad);
        }

        let role_mixed_embeds =
            run_embed_batch(&mut self.text_project, "input_ids", &role_mixed_ids)?;

        // ALL text tokens + tts_eos
        let mut text_and_eos_ids: Vec<i64> = Vec::with_capacity(text_len + 1);
        text_and_eos_ids.extend_from_slice(&text_tokens);
        text_and_eos_ids.push(tts_eos);
        let text_and_eos_embeds =
            run_embed_batch(&mut self.text_project, "input_ids", &text_and_eos_ids)?;

        // Codec tokens needed: mixed [nothink, think_bos, think_eos] + pad + bos
        let codec_ids = vec![
            codec_nothink_id,
            codec_think_bos_id,
            codec_think_eos_id,
            codec_pad_id,
            codec_bos_id,
        ];
        let codec_embeds = run_embed_batch(&mut self.codec_embed, "input_ids", &codec_ids)?;
        assert_eq!(codec_embeds.len(), 5 * h);

        let codec_pad_embed = &codec_embeds[3 * h..4 * h];
        let codec_bos_embed = &codec_embeds[4 * h..5 * h];

        // Construct final embedding
        let mut final_embeds = Vec::with_capacity(prefill_len * h);

        // 1. Instruct (pure text)
        final_embeds.extend_from_slice(&instruct_embeds);

        // 2. Role (3): pure text
        for pos in 0..3 {
            let src = &role_mixed_embeds[pos * h..(pos + 1) * h];
            final_embeds.extend_from_slice(src);
        }

        // 3. Mixed (3): tts_pad + codec[0..2] = [NOTHINK, THINK_BOS, THINK_EOS]
        for pos in 0..mixed_count {
            let text_src = &role_mixed_embeds[(3 + pos) * h..(3 + pos + 1) * h];
            let codec_src = &codec_embeds[pos * h..(pos + 1) * h];
            for j in 0..h {
                final_embeds.push(text_src[j] + codec_src[j]);
            }
        }

        // 4. ALL text tokens + codec_pad
        for t in 0..text_len {
            let text_src = &text_and_eos_embeds[t * h..(t + 1) * h];
            for j in 0..h {
                final_embeds.push(text_src[j] + codec_pad_embed[j]);
            }
        }

        // 5. tts_eos + codec_pad
        let tts_eos_src = &text_and_eos_embeds[text_len * h..(text_len + 1) * h];
        for j in 0..h {
            final_embeds.push(tts_eos_src[j] + codec_pad_embed[j]);
        }

        // 6. tts_pad + codec_bos
        for j in 0..h {
            final_embeds.push(self.tts_pad_embed[j] + codec_bos_embed[j]);
        }

        assert_eq!(final_embeds.len(), prefill_len * h);

        // Non-streaming: trailing is constant tts_pad_embed
        let trailing_text_hidden = self.tts_pad_embed.clone();

        Ok((
            final_embeds,
            prefill_len,
            trailing_text_hidden,
            text_tokens,
        ))
    }

    /// Build prompt embeddings for voice cloning (Base variant, x_vector_only mode).
    ///
    /// Clone uses language=None, so:
    ///   codec_0 = [nothink, think_bos, think_eos] (3 tokens, NO language)
    ///   speaker_embed = [2048D speaker embedding] (1 position, injected directly)
    ///   codec_1 = [pad, bos] (2 tokens)
    ///   total = 6 codec positions (5 embedded + 1 speaker_embed)
    ///
    /// Non-streaming layout (matches Python `[:, :-1]` which removes streaming's tts_bos+PAD):
    ///   [role(3), mixed(4), text_tokens+codec_pad(T), tts_eos+codec_pad(1), tts_pad+codec_bos(1)]
    fn build_clone_embeddings(
        &mut self,
        text: &str,
        speaker_embed: &[f32],
    ) -> Result<(Vec<f32>, usize, Vec<f32>, Vec<i64>), String> {
        let h = self.config.hidden_size;

        assert_eq!(speaker_embed.len(), h, "Speaker embed must be [hidden_size]");

        // Copy config values before calling &mut self methods
        let im_start = self.config.im_start;
        let assistant = self.config.assistant;
        let tts_pad = self.config.tts_pad;
        let tts_eos = self.config.tts_eos;
        let codec_nothink_id = self.config.codec_nothink_id;
        let codec_think_bos_id = self.config.codec_think_bos_id;
        let codec_think_eos_id = self.config.codec_think_eos_id;
        let codec_pad_id = self.config.codec_pad_id;
        let codec_bos_id = self.config.codec_bos_id;

        // Tokenize the text
        let text_tokens = self.tokenizer.tokenize(text);
        let text_len = text_tokens.len();
        if text_len == 0 {
            return Err("Cannot generate speech from empty text".into());
        }

        // Non-streaming mixed: Python removes streaming's last position (tts_bos+PAD).
        // Clone codec = 6 positions → streaming mixed = 5, non-streaming mixed = 4.
        // Mixed pairs: [tts_pad+NOTHINK, tts_pad+THINK_BOS, tts_pad+THINK_EOS, tts_pad+SPEAKER]
        let mixed_count = 4;
        let prefill_len = 3 + mixed_count + text_len + 1 + 1;

        // Role + mixed text tokens (3 + 4 = 7)
        let newline: i64 = 198;
        let role_mixed_len = 3 + mixed_count;
        let mut role_mixed_ids: Vec<i64> = Vec::with_capacity(role_mixed_len);
        role_mixed_ids.push(im_start);
        role_mixed_ids.push(assistant);
        role_mixed_ids.push(newline);
        for _ in 0..mixed_count {
            role_mixed_ids.push(tts_pad);
        }

        let role_mixed_embeds =
            run_embed_batch(&mut self.text_project, "input_ids", &role_mixed_ids)?;

        // ALL text tokens + tts_eos
        let mut text_and_eos_ids: Vec<i64> = Vec::with_capacity(text_len + 1);
        text_and_eos_ids.extend_from_slice(&text_tokens);
        text_and_eos_ids.push(tts_eos);
        let text_and_eos_embeds =
            run_embed_batch(&mut self.text_project, "input_ids", &text_and_eos_ids)?;

        // Codec tokens — [nothink, think_bos, think_eos] for mixed + [pad, bos] for tail
        let codec_ids = vec![
            codec_nothink_id,
            codec_think_bos_id,
            codec_think_eos_id,
            codec_pad_id,
            codec_bos_id,
        ];
        let codec_embeds = run_embed_batch(&mut self.codec_embed, "input_ids", &codec_ids)?;

        let codec_pad_embed = &codec_embeds[3 * h..4 * h];
        let codec_bos_embed = &codec_embeds[4 * h..5 * h];

        // Construct final embedding
        let mut final_embeds = Vec::with_capacity(prefill_len * h);

        // 1. Role (3): pure text
        for pos in 0..3 {
            let src = &role_mixed_embeds[pos * h..(pos + 1) * h];
            final_embeds.extend_from_slice(src);
        }

        // 2. Mixed (4): tts_pad + codec/speaker
        //   pos 0 -> codec[0] (nothink)
        //   pos 1 -> codec[1] (think_bos)
        //   pos 2 -> codec[2] (think_eos)
        //   pos 3 -> speaker_embed (direct injection)
        for pos in 0..mixed_count {
            let text_src = &role_mixed_embeds[(3 + pos) * h..(3 + pos + 1) * h];
            if pos == 3 {
                // Speaker embedding position
                for j in 0..h {
                    final_embeds.push(text_src[j] + speaker_embed[j]);
                }
            } else {
                let codec_src = &codec_embeds[pos * h..(pos + 1) * h];
                for j in 0..h {
                    final_embeds.push(text_src[j] + codec_src[j]);
                }
            }
        }

        // 3. ALL text tokens + codec_pad
        for t in 0..text_len {
            let text_src = &text_and_eos_embeds[t * h..(t + 1) * h];
            for j in 0..h {
                final_embeds.push(text_src[j] + codec_pad_embed[j]);
            }
        }

        // 4. tts_eos + codec_pad
        let tts_eos_src = &text_and_eos_embeds[text_len * h..(text_len + 1) * h];
        for j in 0..h {
            final_embeds.push(tts_eos_src[j] + codec_pad_embed[j]);
        }

        // 5. tts_pad + codec_bos
        for j in 0..h {
            final_embeds.push(self.tts_pad_embed[j] + codec_bos_embed[j]);
        }

        assert_eq!(final_embeds.len(), prefill_len * h);

        // Non-streaming: trailing is constant tts_pad_embed
        let trailing_text_hidden = self.tts_pad_embed.clone();

        Ok((
            final_embeds,
            prefill_len,
            trailing_text_hidden,
            text_tokens,
        ))
    }

    // ── Speaker Encoder ──────────────────────────────────────────

    /// Run the speaker encoder ONNX model on mel spectrogram input.
    /// Input: mel [T, 128] flattened, Output: speaker embedding [2048].
    fn run_speaker_encoder(
        &mut self,
        mel: &[f32],
        mel_frames: usize,
    ) -> Result<Vec<f32>, String> {
        let encoder = self
            .speaker_encoder
            .as_mut()
            .ok_or("Speaker encoder not loaded")?;

        let mel_tensor = Tensor::from_array((
            vec![1i64, mel_frames as i64, 128],
            mel.to_vec(),
        ))
        .map_err(|e| format!("Mel tensor: {}", e))?;

        let outputs = encoder
            .run(ort::inputs!["mels" => mel_tensor])
            .map_err(|e| format!("speaker_encoder error: {}", e))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Speaker encoder output: {}", e))?;

        Ok(data.to_vec())
    }

    // ── Prefill ───────────────────────────────────────────────────

    /// Run talker_prefill and return (last_logits, last_hidden, kv_cache_value, kv_seq_len).
    fn run_prefill(
        &mut self,
        embeddings: &[f32],
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, DynValue, usize), String> {
        let h = self.config.hidden_size;
        let vocab_size = 3072; // talker codec vocab size

        // Build position_ids [3, 1, seq_len]: values 0..seq_len-1 replicated 3x
        let mut pos_ids = vec![0i64; 3 * seq_len];
        for d in 0..3 {
            for s in 0..seq_len {
                pos_ids[d * seq_len + s] = s as i64;
            }
        }

        let kv_output_name = self
            .talker_prefill
            .outputs()
            .get(2)
            .map(|o| o.name().to_string())
            .ok_or_else(|| "Prefill: no output at index 2 for KV cache".to_string())?;

        let embed_tensor = Tensor::from_array((
            vec![1i64, seq_len as i64, h as i64],
            embeddings.to_vec(),
        ))
        .map_err(|e| format!("Prefill embed tensor: {}", e))?;

        let pos_tensor =
            Tensor::from_array((vec![3i64, 1i64, seq_len as i64], pos_ids))
                .map_err(|e| format!("Prefill pos tensor: {}", e))?;

        let mut outputs = self
            .talker_prefill
            .run(ort::inputs![
                "inputs_embeds" => embed_tensor,
                "position_ids" => pos_tensor
            ])
            .map_err(|e| format!("talker_prefill error: {}", e))?;

        // logits: [1, seq_len, 3072]
        let (_shape, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Prefill logits extraction: {}", e))?;
        // hidden: [1, seq_len, 2048]
        let (_shape, hidden_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Prefill hidden extraction: {}", e))?;

        let last_logits_start = (seq_len - 1) * vocab_size;
        let last_logits = logits_data[last_logits_start..last_logits_start + vocab_size].to_vec();

        let last_hidden_start = (seq_len - 1) * h;
        let last_hidden = hidden_data[last_hidden_start..last_hidden_start + h].to_vec();

        let kv_value = outputs
            .remove(&kv_output_name)
            .ok_or_else(|| format!("Prefill: missing '{}' output", kv_output_name))?;

        Ok((last_logits, last_hidden, kv_value, seq_len))
    }

    // ── Autoregressive Decode Loop ────────────────────────────────

    /// Run the autoregressive decode loop, producing codec frames.
    fn run_decode_loop(
        &mut self,
        initial_logits: Vec<f32>,
        initial_hidden: Vec<f32>,
        initial_kv: DynValue,
        initial_kv_len: usize,
        max_steps: usize,
        trailing_text_hidden: &[f32],
        has_instruct: bool,
    ) -> Result<Vec<[i64; 16]>, String> {
        let h = self.config.hidden_size;
        let cfg = &self.config;

        let mut frames: Vec<[i64; 16]> = Vec::new();
        let mut kv_value: DynValue = initial_kv;
        let mut kv_seq_len = initial_kv_len;
        let mut cur_logits = initial_logits;
        let mut cur_hidden = initial_hidden;

        let kv_output_name = self
            .talker_decode
            .outputs()
            .get(2)
            .map(|o| o.name().to_string())
            .ok_or_else(|| "talker_decode: no output at index 2 for KV cache".to_string())?;
        let start = std::time::Instant::now();
        let mut rng = rand::thread_rng();

        let mut generated_codes: Vec<i64> = Vec::new();

        // Use lower temperature for instruct mode to better preserve voice conditioning
        let talker_temperature = if has_instruct {
            log::info!(
                "Instruct mode: using temperature {:.2} (default {:.2})",
                INSTRUCT_TEMPERATURE,
                self.gen_config.temperature,
            );
            INSTRUCT_TEMPERATURE
        } else {
            self.gen_config.temperature
        };

        let timeout = std::time::Duration::from_secs(GENERATION_TIMEOUT_SECS);

        for step in 0..max_steps {
            // Wall-clock timeout check
            if start.elapsed() >= timeout {
                log::warn!(
                    "Decode loop timed out after {}s at step {} — returning {} partial frames",
                    GENERATION_TIMEOUT_SECS,
                    step,
                    frames.len()
                );
                break;
            }

            // 1. Sample first_code with repetition penalty + token suppression
            apply_repetition_penalty(
                &mut cur_logits,
                &generated_codes,
                self.gen_config.repetition_penalty,
            );

            // Token suppression: mask [2048, 3072) except EOS
            for i in 2048..3072 {
                if i as i64 != cfg.codec_eos_token_id {
                    cur_logits[i] = f32::NEG_INFINITY;
                }
            }

            // min_new_tokens=2: suppress EOS during the first 2 steps
            // (matches HF generate's MinNewTokensLengthLogitsProcessor)
            if step < 2 {
                cur_logits[cfg.codec_eos_token_id as usize] = f32::NEG_INFINITY;
            }

            let first_code = sample_top_k(
                &cur_logits,
                talker_temperature,
                self.gen_config.top_k,
                &mut rng,
            ) as i64;

            generated_codes.push(first_code);

            // 2. EOS check
            if first_code == cfg.codec_eos_token_id {
                log::info!("EOS at step {}", step);
                break;
            }

            // 3. Get codec embedding of first_code
            let first_code_embed =
                run_embed_single(&mut self.codec_embed, "input_ids", first_code)?;

            // 4. Autoregressive code_predictor: predict 15 codes
            let cv = cfg.codec_vocab_size;
            let mut predicted_codes = [0i64; 15];

            let mut cp_embeds: Vec<f32> = Vec::with_capacity(17 * h);
            cp_embeds.extend_from_slice(&cur_hidden);
            cp_embeds.extend_from_slice(&first_code_embed);
            let mut cp_seq_len: i64 = 2;

            for group in 0..15i64 {
                let seq = cp_seq_len as usize;

                let cp_tensor = Tensor::from_array(
                    (vec![1i64, cp_seq_len, h as i64], cp_embeds.clone()),
                )
                .map_err(|e| format!("CP prefill tensor: {}", e))?;

                let cp_pos: Vec<i64> = (0..cp_seq_len).collect();
                let cp_pos_tensor =
                    Tensor::from_array((vec![1i64, cp_seq_len], cp_pos))
                        .map_err(|e| format!("CP pos tensor: {}", e))?;

                let mut mask = vec![0.0f32; seq * seq];
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        mask[i * seq + j] = f32::NEG_INFINITY;
                    }
                }
                let mask_tensor = Tensor::from_array(
                    (vec![1i64, 1, cp_seq_len, cp_seq_len], mask),
                )
                .map_err(|e| format!("CP mask tensor: {}", e))?;

                let cp_outputs = self
                    .code_predictor_prefill
                    .run(ort::inputs![
                        "inputs_embeds" => cp_tensor,
                        "position_ids" => cp_pos_tensor,
                        "attention_mask" => mask_tensor
                    ])
                    .map_err(|e| {
                        format!("CP error at step {}, group {}: {}", step, group, e)
                    })?;

                let (_shape, logits_data) = cp_outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("CP logits: {}", e))?;

                let last_pos_offset = (cp_seq_len as usize - 1) * 15 * cv;
                let head_offset = group as usize * cv;
                let head_logits = &logits_data
                    [last_pos_offset + head_offset..last_pos_offset + head_offset + cv];

                predicted_codes[group as usize] = sample_top_k(
                    head_logits,
                    self.gen_config.subtalker_temperature,
                    self.gen_config.subtalker_top_k,
                    &mut rng,
                ) as i64;

                if group < 14 {
                    let code_embed = run_code_predictor_embed(
                        &mut self.code_predictor_embed,
                        group,
                        predicted_codes[group as usize],
                    )?;
                    cp_embeds.extend_from_slice(&code_embed);
                    cp_seq_len += 1;
                }
            }

            // 5. Embed all 16 codes: codec_embed(first_code) + sum(code_predictor_embed)
            let mut total_embed = first_code_embed;

            for group in 0..15i64 {
                let group_embed = run_code_predictor_embed(
                    &mut self.code_predictor_embed,
                    group,
                    predicted_codes[group as usize],
                )?;
                for j in 0..h {
                    total_embed[j] += group_embed[j];
                }
            }

            // 6. Add trailing text hidden
            let trailing_count = trailing_text_hidden.len() / h;
            if step < trailing_count {
                let offset = step * h;
                for j in 0..h {
                    total_embed[j] += trailing_text_hidden[offset + j];
                }
            } else {
                for j in 0..h {
                    total_embed[j] += self.tts_pad_embed[j];
                }
            }

            // 7. Run talker_decode
            let embed_tensor =
                Tensor::from_array((vec![1i64, 1i64, h as i64], total_embed))
                    .map_err(|e| format!("Decode embed tensor: {}", e))?;

            let pos = kv_seq_len as i64;
            let pos_tensor =
                Tensor::from_array((vec![3i64, 1i64, 1i64], vec![pos, pos, pos]))
                    .map_err(|e| format!("Decode pos tensor: {}", e))?;

            let mut outputs = self
                .talker_decode
                .run(ort::inputs![
                    "inputs_embeds" => embed_tensor,
                    "position_ids" => pos_tensor,
                    "past_key_values" => kv_value
                ])
                .map_err(|e| format!("talker_decode error at step {}: {}", step, e))?;

            let (_shape, new_logits) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Decode logits: {}", e))?;
            let (_shape, new_hidden) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Decode hidden: {}", e))?;

            cur_logits = new_logits.to_vec();
            cur_hidden = new_hidden.to_vec();

            kv_value = outputs.remove(&kv_output_name).ok_or_else(|| {
                format!("Decode step {}: missing '{}' output", step, kv_output_name)
            })?;
            kv_seq_len += 1;

            let mut frame = [0i64; 16];
            frame[0] = first_code;
            frame[1..16].copy_from_slice(&predicted_codes);
            frames.push(frame);

            if step < 5 || step % 25 == 0 {
                let elapsed = start.elapsed().as_millis();
                let rate = if elapsed > 0 {
                    (step + 1) as f64 / (elapsed as f64 / 1000.0)
                } else {
                    0.0
                };
                log::info!(
                    "Decode step {}/{}: {:.0} ms elapsed, {:.1} steps/s",
                    step + 1,
                    max_steps,
                    elapsed,
                    rate
                );
            }
        }

        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_millis();
        let audio_sec = frames.len() as f64 / CODEC_FPS;
        let hit_limit = frames.len() >= max_steps;
        let hit_timeout = elapsed >= timeout;
        log::info!(
            "Decode done: {} frames (~{:.1}s audio) in {} ms ({:.2}x real-time){}",
            frames.len(),
            audio_sec,
            elapsed_ms,
            if elapsed_ms > 0 {
                audio_sec / (elapsed_ms as f64 / 1000.0)
            } else {
                0.0
            },
            if hit_timeout {
                " [TIMED OUT -- audio may be truncated]"
            } else if hit_limit {
                " [HIT MAX STEPS -- audio may be truncated]"
            } else {
                ""
            }
        );

        Ok(frames)
    }

    // ── Audio Decode ──────────────────────────────────────────────

    /// Decode codec frames via tokenizer12hz_decode → audio samples.
    fn decode_audio(&mut self, frames: &[[i64; 16]]) -> Result<Vec<f32>, String> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let num_frames = frames.len();
        let num_groups = 16;

        let mut codes = vec![0i64; num_groups * num_frames];
        for f in 0..num_frames {
            for g in 0..num_groups {
                codes[g * num_frames + f] = frames[f][g];
            }
        }

        let codes_tensor = Tensor::from_array((
            vec![1i64, num_groups as i64, num_frames as i64],
            codes,
        ))
        .map_err(|e| format!("Audio codes tensor: {}", e))?;

        let outputs = self
            .tokenizer12hz_decode
            .run(ort::inputs!["audio_codes" => codes_tensor])
            .map_err(|e| format!("tokenizer12hz_decode error: {}", e))?;

        let (_shape, audio_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Audio decode output: {}", e))?;

        Ok(audio_data.to_vec())
    }
}

// ── ONNX Helpers ─────────────────────────────────────────────────

fn run_embed_single(
    session: &mut Session,
    input_name: &str,
    token_id: i64,
) -> Result<Vec<f32>, String> {
    let input = Tensor::from_array((vec![1i64, 1i64], vec![token_id]))
        .map_err(|e| format!("Embed single tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs![input_name => input])
        .map_err(|e| format!("Embed single session error: {}", e))?;

    let (_shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("Embed single extraction: {}", e))?;

    Ok(data.to_vec())
}

fn run_embed_batch(
    session: &mut Session,
    input_name: &str,
    token_ids: &[i64],
) -> Result<Vec<f32>, String> {
    let seq_len = token_ids.len();
    let input = Tensor::from_array((vec![1i64, seq_len as i64], token_ids.to_vec()))
        .map_err(|e| format!("Embed batch tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs![input_name => input])
        .map_err(|e| format!("Embed batch session error: {}", e))?;

    let (_shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("Embed batch extraction: {}", e))?;

    Ok(data.to_vec())
}

fn run_code_predictor_embed(
    session: &mut Session,
    group_idx: i64,
    token_id: i64,
) -> Result<Vec<f32>, String> {
    let group_tensor = Tensor::from_array((vec![1i64], vec![group_idx]))
        .map_err(|e| format!("CP embed group tensor: {}", e))?;
    let id_tensor = Tensor::from_array((vec![1i64, 1i64], vec![token_id]))
        .map_err(|e| format!("CP embed id tensor: {}", e))?;

    let outputs = session
        .run(ort::inputs!["group_idx" => group_tensor, "input_ids" => id_tensor])
        .map_err(|e| format!("code_predictor_embed error (group={}): {}", group_idx, e))?;

    let (_shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("CP embed extraction: {}", e))?;

    Ok(data.to_vec())
}

// ── Utilities ───────────────────────────────────────────────────

/// Read available system memory in MB from /proc/meminfo (Linux only).
/// Returns `None` on non-Linux platforms or if the file can't be read.
fn available_memory_mb() -> Option<u64> {
    let data = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in data.lines() {
        if line.starts_with("MemAvailable:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: u64 = parts[1].parse().ok()?;
                return Some(kb / 1024);
            }
        }
    }
    None
}

fn load_session(path: &std::path::Path, name: &str) -> Result<Session, String> {
    if !path.exists() {
        return Err(format!("Model file not found: {}", path.display()));
    }

    log::debug!("Loading ONNX session: {} from {}", name, path.display());

    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let intra = num_cpus.div_ceil(2).clamp(2, 4);

    Session::builder()
        .map_err(|e| format!("Session builder error for {}: {}", name, e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("Optimization error for {}: {}", name, e))?
        .with_intra_threads(intra)
        .map_err(|e| format!("Thread config error for {}: {}", name, e))?
        .with_inter_threads(2)
        .map_err(|e| format!("Inter-thread config error for {}: {}", name, e))?
        .commit_from_file(path)
        .map_err(|e| format!("Failed to load {}: {}", name, e))
}

fn apply_repetition_penalty(logits: &mut [f32], generated: &[i64], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &token in generated {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn sample_top_k(logits: &[f32], temperature: f32, top_k: usize, rng: &mut impl Rng) -> usize {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let k = top_k.min(indexed.len()).max(1);
    indexed.truncate(k);

    let max_logit = indexed[0].1;
    let mut probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(idx, logit)| {
            let scaled = (logit - max_logit) / temperature;
            (idx, scaled.exp())
        })
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for p in &mut probs {
        p.1 /= sum;
    }

    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for &(idx, prob) in &probs {
        cumulative += prob;
        if r < cumulative {
            return idx;
        }
    }

    probs[0].0
}

fn resample_audio(audio: &[f32], speed: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let new_len = (audio.len() as f32 / speed) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f32 * speed;
        let idx0 = src_idx as usize;
        let idx1 = std::cmp::min(idx0 + 1, audio.len() - 1);
        let frac = src_idx - idx0 as f32;
        resampled.push(audio[idx0] * (1.0 - frac) + audio[idx1] * frac);
    }

    resampled
}

fn chrono_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn write_wav(audio: &[f32], sample_rate: u32, path: &Path) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer =
        hound::WavWriter::create(path, spec).map_err(|e| format!("WAV create error: {}", e))?;

    for &sample in audio {
        let clamped = sample.clamp(-1.0, 1.0);
        writer
            .write_sample((clamped * 32767.0) as i16)
            .map_err(|e| format!("WAV write error: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("WAV finalize error: {}", e))?;

    Ok(())
}

// ── WAV Loading ─────────────────────────────────────────────────

/// Load a WAV file and return (samples as f32 in [-1, 1], sample_rate).
fn load_wav(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let reader =
        hound::WavReader::open(path).map_err(|e| format!("Failed to open WAV: {}", e))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };

    // Convert stereo to mono by averaging channels
    let samples = if spec.channels > 1 {
        let ch = spec.channels as usize;
        samples
            .chunks(ch)
            .map(|chunk| chunk.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        samples
    };

    log::info!(
        "Loaded WAV: {} samples, {}Hz, {}ch, {}bit",
        samples.len(),
        sample_rate,
        spec.channels,
        spec.bits_per_sample
    );

    Ok((samples, sample_rate))
}

/// Simple linear interpolation resample to 24kHz.
fn resample_to_24k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == SAMPLE_RATE {
        return samples.to_vec();
    }

    let ratio = SAMPLE_RATE as f64 / src_rate as f64;
    let new_len = (samples.len() as f64 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = (src_idx - idx0 as f64) as f32;
        resampled.push(samples[idx0] * (1.0 - frac) + samples[idx1] * frac);
    }

    log::info!(
        "Resampled: {}Hz -> {}Hz ({} -> {} samples)",
        src_rate,
        SAMPLE_RATE,
        samples.len(),
        resampled.len()
    );

    resampled
}

// ── Mel Spectrogram Extraction ──────────────────────────────────

/// Extract log-magnitude mel spectrogram matching Python's mel_spectrogram() exactly.
///
/// Python pipeline: reflect pad → STFT → magnitude (sqrt) → mel filterbank (slaney) → log
/// Parameters: n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000, sr=24000
/// Returns flat [T, 128] mel spectrogram.
fn extract_mel(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    let n_fft: usize = 1024;
    let hop_length: usize = 256;
    let n_mels: usize = 128;
    let fmax: f64 = 12000.0;

    // Reflect-pad signal: (n_fft - hop_length) / 2 = 384 samples on each side
    // Matches Python: F.pad(y, (pad, pad), mode="reflect")
    let pad = (n_fft - hop_length) / 2;
    let padded_len = pad + samples.len() + pad;
    let mut padded = Vec::with_capacity(padded_len);
    // Left reflect pad
    for i in (1..=pad).rev() {
        let idx = if i < samples.len() { i } else { samples.len() - 1 };
        padded.push(samples[idx]);
    }
    padded.extend_from_slice(samples);
    // Right reflect pad
    for i in 1..=pad {
        let idx = if samples.len() > i + 1 {
            samples.len() - 1 - i
        } else {
            0
        };
        padded.push(samples[idx]);
    }

    // Build Hann window (periodic, matching torch.hann_window)
    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos())
        })
        .collect();

    // STFT (center=False since we already padded)
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let num_frames = if padded.len() >= n_fft {
        (padded.len() - n_fft) / hop_length + 1
    } else {
        1
    };
    let freq_bins = n_fft / 2 + 1;

    // Magnitude spectrogram (NOT power) — matches Python: sqrt(real^2 + imag^2 + 1e-9)
    let mut mag_spec = vec![0.0f64; num_frames * freq_bins];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        let mut buffer: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let sample = if start + i < padded.len() {
                    padded[start + i]
                } else {
                    0.0
                };
                Complex::new(sample * window[i], 0.0)
            })
            .collect();

        fft.process(&mut buffer);

        for bin in 0..freq_bins {
            let re = buffer[bin].re as f64;
            let im = buffer[bin].im as f64;
            mag_spec[frame_idx * freq_bins + bin] = (re * re + im * im + 1e-9).sqrt();
        }
    }

    // Build mel filterbank with slaney normalization (matches librosa default)
    let mel_filterbank =
        build_mel_filterbank(n_mels, freq_bins, sample_rate as f64, 0.0, fmax);

    // Apply mel filterbank + log compression
    // Python: torch.log(torch.clamp(mel_output, min=1e-5))
    let mut mel = vec![0.0f32; num_frames * n_mels];
    for frame in 0..num_frames {
        for mel_bin in 0..n_mels {
            let mut sum = 0.0f64;
            for freq_bin in 0..freq_bins {
                sum += mel_filterbank[mel_bin * freq_bins + freq_bin]
                    * mag_spec[frame * freq_bins + freq_bin];
            }
            mel[frame * n_mels + mel_bin] = (sum.max(1e-5).ln()) as f32;
        }
    }

    mel
}

/// Build a mel filterbank matrix [n_mels, freq_bins] with slaney normalization.
/// Uses the **Slaney mel scale** (librosa default, htk=False): linear below 1000 Hz,
/// logarithmic above. Matches `librosa.filters.mel(norm='slaney', htk=False)`.
fn build_mel_filterbank(
    n_mels: usize,
    freq_bins: usize,
    sr: f64,
    fmin: f64,
    fmax: f64,
) -> Vec<f64> {
    // Slaney mel scale (librosa default, htk=False)
    // Linear below 1000 Hz, logarithmic above.
    let f_sp: f64 = 200.0 / 3.0;
    let min_log_hz: f64 = 1000.0;
    let min_log_mel: f64 = min_log_hz / f_sp; // 15.0
    let logstep: f64 = (6.4_f64).ln() / 27.0; // ln(6400/1000) / 27

    let hz_to_mel = |hz: f64| -> f64 {
        if hz < min_log_hz {
            hz / f_sp
        } else {
            min_log_mel + (hz / min_log_hz).ln() / logstep
        }
    };
    let mel_to_hz = |mel: f64| -> f64 {
        if mel < min_log_mel {
            mel * f_sp
        } else {
            min_log_hz * (logstep * (mel - min_log_mel)).exp()
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 points evenly spaced in mel scale
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();

    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
    let n_fft = (freq_bins - 1) * 2;
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&hz| hz * n_fft as f64 / sr)
        .collect();

    let mut filterbank = vec![0.0f64; n_mels * freq_bins];

    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        for k in 0..freq_bins {
            let kf = k as f64;
            if kf >= f_left && kf <= f_center && f_center > f_left {
                filterbank[m * freq_bins + k] = (kf - f_left) / (f_center - f_left);
            } else if kf > f_center && kf <= f_right && f_right > f_center {
                filterbank[m * freq_bins + k] = (f_right - kf) / (f_right - f_center);
            }
        }

        // Slaney normalization: normalize each filter by its bandwidth
        // enorm = 2.0 / (hz_points[m+2] - hz_points[m])
        let bandwidth = hz_points[m + 2] - hz_points[m];
        if bandwidth > 0.0 {
            let enorm = 2.0 / bandwidth;
            for k in 0..freq_bins {
                filterbank[m * freq_bins + k] *= enorm;
            }
        }
    }

    filterbank
}
