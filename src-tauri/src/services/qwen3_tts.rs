use crate::services::tokenizers::BpeTokenizer;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{DynValue, Tensor};
use rand::Rng;
use std::collections::HashMap;
use std::path::Path;

const SAMPLE_RATE: u32 = 24000;
/// Absolute ceiling — the model should always hit EOS before this.
const MAX_STEPS_ABSOLUTE: usize = 4096;
/// ~12 Hz codec: frames per second of audio output.
const CODEC_FPS: f64 = 12.0;
/// Generous safety multiplier over the estimated duration.
const SAFETY_MULTIPLIER: f64 = 3.0;
/// Minimum cap so very short texts still get enough room.
const MIN_MAX_STEPS: usize = 48; // ~4s of audio

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

    // Model dimensions
    num_hidden_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    num_code_groups: usize,
    codec_vocab_size: usize,
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

        let cp_config = &talker["code_predictor_config"];

        Ok(Self {
            codec_bos_id: talker["codec_bos_id"].as_i64().unwrap_or(2149),
            codec_eos_token_id: talker["codec_eos_token_id"].as_i64().unwrap_or(2150),
            codec_pad_id: talker["codec_pad_id"].as_i64().unwrap_or(2148),
            codec_nothink_id: talker["codec_nothink_id"].as_i64().unwrap_or(2155),
            codec_think_bos_id: talker["codec_think_bos_id"].as_i64().unwrap_or(2156),
            codec_think_eos_id: talker["codec_think_eos_id"].as_i64().unwrap_or(2157),
            tts_bos: config["tts_bos_token_id"].as_i64().unwrap_or(151672),
            tts_eos: config["tts_eos_token_id"].as_i64().unwrap_or(151673),
            tts_pad: config["tts_pad_token_id"].as_i64().unwrap_or(151671),
            im_start: config["im_start_token_id"].as_i64().unwrap_or(151644),
            assistant: config["assistant_token_id"].as_i64().unwrap_or(77091),
            spk_id,
            num_hidden_layers: talker["num_hidden_layers"].as_u64().unwrap_or(28) as usize,
            num_kv_heads: talker["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            head_dim: talker["head_dim"].as_u64().unwrap_or(128) as usize,
            hidden_size: talker["hidden_size"].as_u64().unwrap_or(2048) as usize,
            num_code_groups: talker["num_code_groups"].as_u64().unwrap_or(16) as usize,
            codec_vocab_size: cp_config["vocab_size"].as_u64().unwrap_or(2048) as usize,
        })
    }
}

/// ONNX inference engine for Qwen3-TTS (0.6B / 1.7B variants).
///
/// Pipeline:
///   build_prompt_embeddings → talker_prefill (with KV cache) →
///   talker_decode (autoregressive) + code_predictor →
///   tokenizer12hz_decode → audio output
pub struct Qwen3TTSEngine {
    config: Qwen3TTSConfig,
    gen_config: GenerationConfig,
    text_project: Session,
    codec_embed: Session,
    code_predictor: Session,
    code_predictor_embed: Session,
    talker_prefill: Session,
    talker_decode: Session,
    tokenizer12hz_decode: Session,
    speaker_encoder: Option<Session>,
    tokenizer: BpeTokenizer,
    /// Cached tts_pad text embedding [hidden_size], used as trailing_text_hidden in decode loop.
    tts_pad_embed: Vec<f32>,
}

impl Qwen3TTSEngine {
    /// Load all ONNX sessions and the BPE tokenizer from `model_dir`.
    pub fn initialize(model_dir: &Path) -> Result<Self, String> {
        log::info!("Loading Qwen3 TTS from: {}", model_dir.display());

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

        let tokenizer = BpeTokenizer::load(
            &model_dir.join("vocab.json"),
            &model_dir.join("merges.txt"),
        )?;

        let mut text_project = load_session(&model_dir.join("text_project.onnx"), "text_project")?;
        let codec_embed = load_session(&model_dir.join("codec_embed.onnx"), "codec_embed")?;
        let code_predictor =
            load_session(&model_dir.join("code_predictor.onnx"), "code_predictor")?;
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
        } else if let Some(parent) = model_dir.parent() {
            let shared = parent.join("qwen3-tts").join("tokenizer12hz_decode.onnx");
            if shared.exists() {
                log::info!("Using shared tokenizer12hz_decode from qwen3-tts");
                shared
            } else {
                return Err(format!(
                    "tokenizer12hz_decode.onnx not found in {} or shared location",
                    model_dir.display()
                ));
            }
        } else {
            return Err("tokenizer12hz_decode.onnx not found".into());
        };
        let tokenizer12hz_decode =
            load_session(&tok_decode_path, "tokenizer12hz_decode")?;

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
        log::info!(
            "Cached tts_pad_embed: {} floats",
            tts_pad_embed.len()
        );

        log::info!("Qwen3 TTS initialized successfully");

        Ok(Self {
            config,
            gen_config,
            text_project,
            codec_embed,
            code_predictor,
            code_predictor_embed,
            talker_prefill,
            talker_decode,
            tokenizer12hz_decode,
            speaker_encoder,
            tokenizer,
            tts_pad_embed,
        })
    }

    /// Generate speech from text using a preset voice.
    pub fn generate_speech(
        &mut self,
        text: &str,
        voice_id: &str,
        speed: f32,
        output_path: &Path,
    ) -> Result<(), String> {
        let total_start = std::time::Instant::now();
        log::info!(
            "Qwen3 generating: voice={}, speed={}, text=\"{}\"",
            voice_id,
            speed,
            &text[..std::cmp::min(text.len(), 80)]
        );

        // 1. Build prompt embeddings
        let t = std::time::Instant::now();
        let (embeddings, seq_len, trailing_text_hidden) = self.build_prompt_embeddings(text, voice_id)?;
        let trailing_len = trailing_text_hidden.len() / self.config.hidden_size;
        log::info!(
            "[1/5] Prompt embeddings: seq_len={}, trailing_text_hidden={} tokens ({} ms)",
            seq_len,
            trailing_len,
            t.elapsed().as_millis()
        );

        // 2. Prefill
        let t = std::time::Instant::now();
        let (last_logits, last_hidden, kv_cache, kv_seq_len) =
            self.run_prefill(&embeddings, seq_len)?;
        log::info!(
            "[2/5] Prefill done: kv_seq_len={} ({} ms)",
            kv_seq_len,
            t.elapsed().as_millis()
        );

        // 3. Autoregressive decode
        let max_steps = compute_max_steps(text);
        log::info!("Max decode steps for {} words: {}", text.split_whitespace().count(), max_steps);
        let t = std::time::Instant::now();
        let frames = self.run_decode_loop(
            last_logits, last_hidden, kv_cache, kv_seq_len, max_steps,
            &trailing_text_hidden,
        )?;
        log::info!(
            "[3/5] Generated {} code frames ({} ms)",
            frames.len(),
            t.elapsed().as_millis()
        );

        // 4. Decode audio
        let t = std::time::Instant::now();
        let mut audio = self.decode_audio(&frames)?;
        log::info!(
            "[4/5] Decoded {} audio samples ({} ms)",
            audio.len(),
            t.elapsed().as_millis()
        );

        // 5. Speed adjustment + write WAV
        if (speed - 1.0).abs() > 0.01 {
            audio = resample_audio(&audio, speed);
        }
        write_wav(&audio, SAMPLE_RATE, output_path)?;

        let duration_sec = audio.len() as f64 / SAMPLE_RATE as f64;
        log::info!(
            "[5/5] WAV written: {:.1}s audio in {} ms total",
            duration_sec,
            total_start.elapsed().as_millis()
        );

        Ok(())
    }

    /// Generate speech using reference audio for voice cloning.
    pub fn clone_voice(
        &mut self,
        _text: &str,
        _reference_audio: &Path,
        _output_path: &Path,
    ) -> Result<(), String> {
        if self.speaker_encoder.is_none() {
            return Err(
                "Voice cloning requires the Base model variant with speaker_encoder".into(),
            );
        }
        Err("Voice cloning is not yet implemented for Qwen3-TTS".into())
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

    /// Build the full prompt embedding sequence for TTS generation.
    ///
    /// Layout: [role(3), mixed(5), text+eos(T+1), final(1)] = T+10 positions
    ///
    /// Returns (flat_embeddings, total_seq_len, trailing_text_hidden).
    ///
    /// `trailing_text_hidden` contains the raw text_project embeddings for the
    /// T text tokens + tts_eos (T+1 total). These are drip-fed one per decode
    /// step to maintain text–audio alignment (per the reference implementation).
    fn build_prompt_embeddings(
        &mut self,
        text: &str,
        voice_id: &str,
    ) -> Result<(Vec<f32>, usize, Vec<f32>), String> {
        let h = self.config.hidden_size;
        let cfg = &self.config;

        // Look up speaker codec token ID
        let voice_lower = voice_id.to_lowercase();
        let spk_token = *cfg.spk_id.get(&voice_lower).ok_or_else(|| {
            format!(
                "Unknown voice '{}'. Available: {:?}",
                voice_id,
                cfg.spk_id.keys().collect::<Vec<_>>()
            )
        })?;

        // BPE-encode the text
        let text_tokens = self.tokenizer.tokenize(text);
        let text_len = text_tokens.len();
        let total_len = text_len + 10;

        // ── Build text token IDs for one batched text_project call ──
        // [im_start, assistant, \n, tts_pad×4, tts_bos, ...text_tokens..., tts_eos, tts_pad]
        let newline: i64 = 198;
        let mut text_ids: Vec<i64> = Vec::with_capacity(total_len);
        // Role prefix (3)
        text_ids.push(cfg.im_start);
        text_ids.push(cfg.assistant);
        text_ids.push(newline);
        // Mixed text side (5): tts_pad×4 + tts_bos
        for _ in 0..4 {
            text_ids.push(cfg.tts_pad);
        }
        text_ids.push(cfg.tts_bos);
        // Text content (T)
        text_ids.extend_from_slice(&text_tokens);
        // TTS EOS (1)
        text_ids.push(cfg.tts_eos);
        // Final: tts_pad (1)
        text_ids.push(cfg.tts_pad);

        assert_eq!(text_ids.len(), total_len);

        // Run text_project on all tokens at once → [1, total_len, hidden_size]
        let text_embeds = run_embed_batch(&mut self.text_project, "input_ids", &text_ids)?;
        assert_eq!(text_embeds.len(), total_len * h);

        // ── Build codec token IDs for one batched codec_embed call ──
        // [nothink, think_bos, think_eos, spk_token, pad, bos]
        let codec_ids = vec![
            cfg.codec_nothink_id,
            cfg.codec_think_bos_id,
            cfg.codec_think_eos_id,
            spk_token,
            cfg.codec_pad_id,
            cfg.codec_bos_id,
        ];
        let codec_embeds = run_embed_batch(&mut self.codec_embed, "input_ids", &codec_ids)?;
        assert_eq!(codec_embeds.len(), 6 * h);

        // Extract individual codec embeddings (as slices into codec_embeds)
        let codec_pad_embed = &codec_embeds[4 * h..5 * h]; // position 4 = pad
        let codec_bos_embed = &codec_embeds[5 * h..6 * h]; // position 5 = bos

        // ── Construct final embedding by element-wise addition ──
        let mut final_embeds = vec![0.0f32; total_len * h];

        for pos in 0..total_len {
            let dst = &mut final_embeds[pos * h..(pos + 1) * h];
            let text_src = &text_embeds[pos * h..(pos + 1) * h];

            if pos < 3 {
                // Role positions: pure text embedding (no codec)
                dst.copy_from_slice(text_src);
            } else if pos < 8 {
                // Mixed positions (3..8): text + codec[pos-3]
                let codec_offset = (pos - 3) * h;
                let codec_src = &codec_embeds[codec_offset..codec_offset + h];
                for j in 0..h {
                    dst[j] = text_src[j] + codec_src[j];
                }
            } else if pos < 8 + text_len + 1 {
                // Text+EOS positions: text + codec_pad
                for j in 0..h {
                    dst[j] = text_src[j] + codec_pad_embed[j];
                }
            } else {
                // Final position: tts_pad + codec_bos
                for j in 0..h {
                    dst[j] = text_src[j] + codec_bos_embed[j];
                }
            }
        }

        // Extract trailing_text_hidden: text_project outputs for text tokens + tts_eos
        // These are positions 8..8+text_len (text) + position 8+text_len (tts_eos) = T+1 embeds
        let trailing_start = 8 * h;
        let trailing_end = (8 + text_len + 1) * h;
        let trailing_text_hidden = text_embeds[trailing_start..trailing_end].to_vec();

        Ok((final_embeds, total_len, trailing_text_hidden))
    }

    // ── Prefill ───────────────────────────────────────────────────

    /// Run talker_prefill and return (last_logits, last_hidden, kv_cache_value, kv_seq_len).
    ///
    /// The KV cache is returned as an owned `DynValue` to avoid copying the
    /// (potentially large) tensor data into a `Vec<f32>` each decode step.
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

        // Look up KV cache output name from session metadata before running
        // (avoids borrow conflict with the mutable session.run() call).
        let kv_output_name = self.talker_prefill.outputs().get(2)
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

        // Extract last position logits and hidden (small copies, needed for processing)
        let last_logits_start = (seq_len - 1) * vocab_size;
        let last_logits = logits_data[last_logits_start..last_logits_start + vocab_size].to_vec();

        let last_hidden_start = (seq_len - 1) * h;
        let last_hidden = hidden_data[last_hidden_start..last_hidden_start + h].to_vec();

        // Extract KV cache as owned DynValue (Arc::clone, no data copy).
        let kv_value = outputs
            .remove(&kv_output_name)
            .ok_or_else(|| format!("Prefill: missing '{}' output", kv_output_name))?;

        Ok((last_logits, last_hidden, kv_value, seq_len))
    }

    // ── Autoregressive Decode Loop ────────────────────────────────

    /// Run the autoregressive decode loop, producing codec frames.
    /// Each frame is [first_code, predicted_code_0..14] = 16 codes.
    ///
    /// `trailing_text_hidden` contains T+1 text_project embeddings (text tokens + tts_eos).
    /// At each decode step i, we add trailing_text_hidden[i] if i < len, else tts_pad_embed.
    /// This "drip-feeds" text context into the decoder to maintain text–audio alignment.
    ///
    /// Optimizations:
    /// - KV cache is kept as an owned `DynValue` (no f32 copy per step).
    /// - code_predictor_embed inner loop reuses pre-allocated tensor buffers.
    fn run_decode_loop(
        &mut self,
        initial_logits: Vec<f32>,
        initial_hidden: Vec<f32>,
        initial_kv: DynValue,
        initial_kv_len: usize,
        max_steps: usize,
        trailing_text_hidden: &[f32],
    ) -> Result<Vec<[i64; 16]>, String> {
        let h = self.config.hidden_size;
        let cfg = &self.config;

        let mut frames: Vec<[i64; 16]> = Vec::new();
        let mut kv_value: DynValue = initial_kv;
        let mut kv_seq_len = initial_kv_len;
        let mut cur_logits = initial_logits;
        let mut cur_hidden = initial_hidden;

        // Pre-allocate reusable buffers for code_predictor_embed inner loop.
        // These are overwritten each iteration to avoid per-group allocations.
        let mut code_buf = vec![0i64; 1];
        let mut group_buf = vec![0i64; 1];

        // Look up actual KV cache output name from session metadata (index 2).
        let kv_output_name = self.talker_decode.outputs().get(2)
            .map(|o| o.name().to_string())
            .ok_or_else(|| "talker_decode: no output at index 2 for KV cache".to_string())?;

        let start = std::time::Instant::now();
        let mut rng = rand::thread_rng();

        // Track generated first_codes for repetition penalty
        let mut generated_codes: Vec<i64> = Vec::new();

        for step in 0..max_steps {
            // 1. Sample first_code from logits with temperature + top-k + repetition penalty
            apply_repetition_penalty(
                &mut cur_logits,
                &generated_codes,
                self.gen_config.repetition_penalty,
            );
            let first_code = sample_top_k(
                &cur_logits,
                self.gen_config.temperature,
                self.gen_config.top_k,
                &mut rng,
            ) as i64;

            // Track for repetition penalty
            generated_codes.push(first_code);

            // 2. EOS check (after at least 2 steps)
            if first_code == cfg.codec_eos_token_id && step > 1 {
                log::info!("EOS at step {}", step);
                break;
            }

            // 3. Get codec embedding of first_code -> [hidden_size]
            let first_code_embed =
                run_embed_single(&mut self.codec_embed, "input_ids", first_code)?;

            // 4. Run code_predictor: cat(hidden, first_code_embed) -> [1, 2, hidden_size]
            let mut cp_input = vec![0.0f32; 2 * h];
            cp_input[..h].copy_from_slice(&cur_hidden);
            cp_input[h..].copy_from_slice(&first_code_embed);

            let cp_tensor =
                Tensor::from_array((vec![1i64, 2, h as i64], cp_input))
                    .map_err(|e| format!("Code predictor tensor: {}", e))?;

            let cp_outputs = self
                .code_predictor
                .run(ort::inputs!["inputs_embeds" => cp_tensor])
                .map_err(|e| format!("code_predictor error at step {}: {}", step, e))?;

            // all_logits: [1, 2, 15, codec_vocab_size]
            let (_shape, cp_logits_data) = cp_outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Code predictor output: {}", e))?;

            // Take last position (pos=1) -> [15, codec_vocab_size]
            let cv = cfg.codec_vocab_size;
            let pos1_offset = 15 * cv; // skip pos 0
            let mut predicted_codes = [0i64; 15];
            for head in 0..15 {
                let head_start = pos1_offset + head * cv;
                let head_logits = &cp_logits_data[head_start..head_start + cv];
                predicted_codes[head] = sample_top_k(
                    head_logits,
                    self.gen_config.subtalker_temperature,
                    self.gen_config.subtalker_top_k,
                    &mut rng,
                ) as i64;
            }

            // 5. Embed all 16 codes: codec_embed(first_code) + sum(code_predictor_embed(code_i, i))
            //    Reuse pre-allocated buffers for the inner loop tensors.
            let mut total_embed = first_code_embed;

            for i in 0..15 {
                code_buf[0] = predicted_codes[i];
                group_buf[0] = i as i64;

                let code_tensor =
                    Tensor::from_array((vec![1i64, 1i64], code_buf.clone()))
                        .map_err(|e| format!("CP embed code tensor: {}", e))?;
                let group_tensor =
                    Tensor::from_array((vec![1i64], group_buf.clone()))
                        .map_err(|e| format!("CP embed group tensor: {}", e))?;

                let outputs = self
                    .code_predictor_embed
                    .run(ort::inputs![
                        "input_ids" => code_tensor,
                        "group_idx" => group_tensor
                    ])
                    .map_err(|e| {
                        format!("code_predictor_embed error step {} group {}: {}", step, i, e)
                    })?;

                let (_shape, embed_data) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("CP embed output: {}", e))?;

                for j in 0..h {
                    total_embed[j] += embed_data[j];
                }
            }

            // 6. Add trailing text hidden: drip-feed text_project embeddings per step
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

            // 7. Run talker_decode — pass KV cache as DynValue directly (no f32 copy)
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

            // Extract logits and hidden state as Vec<f32> (small, needed for processing)
            let (_shape, new_logits) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Decode logits: {}", e))?;
            let (_shape, new_hidden) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Decode hidden: {}", e))?;

            cur_logits = new_logits.to_vec();
            cur_hidden = new_hidden.to_vec();

            // Keep KV cache as DynValue — Arc::clone via remove(), no data copy.
            // This avoids copying the entire KV cache (~115 MB at step 500) every step.
            kv_value = outputs
                .remove(&kv_output_name)
                .ok_or_else(|| {
                    format!("Decode step {}: missing '{}' output", step, kv_output_name)
                })?;
            kv_seq_len += 1;

            // Store frame: [first_code, predicted_codes[0..14]]
            let mut frame = [0i64; 16];
            frame[0] = first_code;
            frame[1..16].copy_from_slice(&predicted_codes);
            frames.push(frame);

            // Log first 5 steps individually, then every 25th step
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

        let elapsed = start.elapsed().as_millis();
        let audio_sec = frames.len() as f64 / CODEC_FPS;
        let hit_limit = frames.len() >= max_steps;
        log::info!(
            "Decode done: {} frames (~{:.1}s audio) in {} ms ({:.2}x real-time){}",
            frames.len(),
            audio_sec,
            elapsed,
            if elapsed > 0 {
                audio_sec / (elapsed as f64 / 1000.0)
            } else {
                0.0
            },
            if hit_limit { " [HIT MAX STEPS — audio may be truncated]" } else { "" }
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

        // Build audio_codes: [1, 16, num_frames] in row-major order
        // Layout: for each group g, then each frame f: data[g * num_frames + f]
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

/// Run an embedding session on a single token ID, returning [hidden_size] flat data.
fn run_embed_single(session: &mut Session, input_name: &str, token_id: i64) -> Result<Vec<f32>, String> {
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

/// Run an embedding session on a batch of token IDs, returning [seq_len * hidden_size] flat data.
fn run_embed_batch(session: &mut Session, input_name: &str, token_ids: &[i64]) -> Result<Vec<f32>, String> {
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

// ── Utilities ───────────────────────────────────────────────────

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

/// Apply repetition penalty to logits for tokens that have been previously generated.
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

/// Sample from logits using temperature scaling and top-k filtering.
///
/// Steps: scale by temperature → top-k filter → softmax → weighted random sample.
fn sample_top_k(logits: &[f32], temperature: f32, top_k: usize, rng: &mut impl Rng) -> usize {
    // Build (index, logit) pairs
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();

    // Sort descending by logit value
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to top-k
    let k = top_k.min(indexed.len()).max(1);
    indexed.truncate(k);

    // Apply temperature and compute softmax
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

    // Weighted random sample
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for &(idx, prob) in &probs {
        cumulative += prob;
        if r < cumulative {
            return idx;
        }
    }

    // Fallback to highest probability
    probs[0].0
}

/// Linear interpolation resample for speed adjustment.
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

/// Write audio samples as 16-bit PCM WAV.
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
