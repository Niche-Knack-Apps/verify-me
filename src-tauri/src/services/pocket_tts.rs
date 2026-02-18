use crate::services::tokenizers::SentencePieceTokenizer;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{Tensor, Value, ValueType};
use std::collections::HashMap;
use std::path::Path;

const SAMPLE_RATE: u32 = 24000;
const LATENT_DIM: usize = 32;
const KV_CACHE_LEN: usize = 1000;
const NUM_HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const NUM_LAYERS: usize = 6;
const EOS_THRESHOLD: f32 = -4.0;

/// Maximum reference audio duration in seconds for voice cloning.
/// Longer clips get truncated to prevent OOM / system freeze.
const MAX_REFERENCE_SECONDS: f32 = 30.0;

/// Maximum samples for reference audio (30s × 24kHz).
const MAX_REFERENCE_SAMPLES: usize = (MAX_REFERENCE_SECONDS * SAMPLE_RATE as f32) as usize;

// ── State value: typed tensor for ONNX state carry-over ─────────

#[derive(Clone)]
enum StateValue {
    F32(Vec<i64>, Vec<f32>),
    I64(Vec<i64>, Vec<i64>),
    Bool(Vec<i64>, Vec<bool>),
}

impl StateValue {
    fn to_session_input(&self) -> Result<ort::session::SessionInputValue<'_>, String> {
        // Use ndarray for tensor creation to support zero-size dimensions
        // (ort's Tensor::from_array((shape,data)) rejects dimensions < 1)
        match self {
            StateValue::F32(shape, data) => {
                let ix: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
                let arr = ndarray::ArrayD::<f32>::from_shape_vec(
                    ndarray::IxDyn(&ix),
                    data.clone(),
                )
                .map_err(|e| format!("F32 state array: {}", e))?;
                let v = Value::from_array(arr)
                    .map_err(|e| format!("F32 state tensor: {}", e))?;
                Ok(v.into())
            }
            StateValue::I64(shape, data) => {
                let ix: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
                let arr = ndarray::ArrayD::<i64>::from_shape_vec(
                    ndarray::IxDyn(&ix),
                    data.clone(),
                )
                .map_err(|e| format!("I64 state array: {}", e))?;
                let v = Value::from_array(arr)
                    .map_err(|e| format!("I64 state tensor: {}", e))?;
                Ok(v.into())
            }
            StateValue::Bool(shape, data) => {
                let ix: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
                let arr = ndarray::ArrayD::<bool>::from_shape_vec(
                    ndarray::IxDyn(&ix),
                    data.clone(),
                )
                .map_err(|e| format!("Bool state array: {}", e))?;
                let v = Value::from_array(arr)
                    .map_err(|e| format!("Bool state tensor: {}", e))?;
                Ok(v.into())
            }
        }
    }
}

// ── Flow LM state: 18 tensors (6 layers × {cache, current_end, step}) ──

fn init_flow_lm_states() -> Vec<StateValue> {
    let mut states = Vec::with_capacity(18);
    for _ in 0..NUM_LAYERS {
        // cache [2, 1, 1000, 16, 64] filled with NaN
        let cache_size = 2 * KV_CACHE_LEN * NUM_HEADS * HEAD_DIM;
        states.push(StateValue::F32(
            vec![2, 1, KV_CACHE_LEN as i64, NUM_HEADS as i64, HEAD_DIM as i64],
            vec![f32::NAN; cache_size],
        ));
        // current_end: empty f32 tensor shape [0]
        states.push(StateValue::F32(vec![0], vec![]));
        // step: i64 [1] = 0
        states.push(StateValue::I64(vec![1], vec![0]));
    }
    states
}

/// Load predefined voice embedding into flow_lm states.
/// Voice embeddings contain cache [2,1,N,16,64] and current_end [N] per layer.
/// We pad cache to [2,1,1000,16,64] with NaN and set step = N.
fn load_voice_into_states(
    states: &mut [StateValue],
    voice_tensors: &HashMap<String, (Vec<i64>, Vec<f32>)>,
) {
    for layer in 0..NUM_LAYERS {
        let cache_key = format!("transformer.layers.{}.self_attn/cache", layer);
        let current_end_key = format!("transformer.layers.{}.self_attn/current_end", layer);

        if let Some((shape, cache_data)) = voice_tensors.get(&cache_key) {
            // Shape is [2, 1, N, 16, 64] — extract N (voice frames)
            let voice_frames = if shape.len() >= 3 {
                shape[2] as usize
            } else {
                // Compute from data size
                cache_data.len() / (2 * NUM_HEADS * HEAD_DIM)
            };

            // Pad cache to [2, 1, 1000, 16, 64] with NaN
            let total = 2 * KV_CACHE_LEN * NUM_HEADS * HEAD_DIM;
            let mut padded = vec![f32::NAN; total];
            let frames_per_kv = voice_frames.min(KV_CACHE_LEN);
            let elems_per_frame = NUM_HEADS * HEAD_DIM; // 1024

            for kv in 0..2 {
                let src_base = kv * voice_frames * elems_per_frame;
                let dst_base = kv * KV_CACHE_LEN * elems_per_frame;
                let copy_len = frames_per_kv * elems_per_frame;
                if src_base + copy_len <= cache_data.len()
                    && dst_base + copy_len <= padded.len()
                {
                    padded[dst_base..dst_base + copy_len]
                        .copy_from_slice(&cache_data[src_base..src_base + copy_len]);
                }
            }

            states[layer * 3] = StateValue::F32(
                vec![2, 1, KV_CACHE_LEN as i64, NUM_HEADS as i64, HEAD_DIM as i64],
                padded,
            );

            // Set step = voice_frames
            states[layer * 3 + 2] = StateValue::I64(vec![1], vec![voice_frames as i64]);

            log::debug!(
                "Layer {}: loaded {} voice frames into KV cache",
                layer,
                voice_frames
            );
        }

        // current_end stays as empty [0] — not used by ONNX patched attention
        if voice_tensors.contains_key(&current_end_key) {
            // Intentionally not loaded — ONNX model uses step instead
        }
    }
}

/// Extract updated flow_lm states from session outputs.
fn extract_flow_lm_states(
    outputs: &ort::session::SessionOutputs,
    offset: usize,
) -> Result<Vec<StateValue>, String> {
    let mut states = Vec::with_capacity(18);
    for i in 0..18 {
        let idx = offset + i;
        let state_type = i % 3; // 0=cache(f32), 1=current_end(f32), 2=step(i64)
        match state_type {
            0 | 1 => {
                let (shape, data) = outputs[idx]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("State {} f32 extraction: {}", i, e))?;
                states.push(StateValue::F32(shape.to_vec(), data.to_vec()));
            }
            2 => {
                let (shape, data) = outputs[idx]
                    .try_extract_tensor::<i64>()
                    .map_err(|e| format!("State {} i64 extraction: {}", i, e))?;
                states.push(StateValue::I64(shape.to_vec(), data.to_vec()));
            }
            _ => unreachable!(),
        }
    }
    Ok(states)
}

// ── Mimi decoder state: 56 tensors, mixed types ─────────────────

/// Initialize mimi decoder states from model input metadata.
fn init_mimi_states(session: &Session) -> Result<Vec<StateValue>, String> {
    let inputs = session.inputs();
    let mut states = Vec::new();
    let mut i = 0;

    loop {
        let name = format!("state_{}", i);
        let input = inputs.iter().find(|inp| inp.name() == name);
        let input = match input {
            Some(inp) => inp,
            None => break,
        };

        let state = match input.dtype() {
            ValueType::Tensor { ty, shape: model_shape, .. } => {
                let shape: Vec<i64> = model_shape
                    .iter()
                    .map(|&d| if d < 0 { 0 } else { d })
                    .collect();
                let num_elements: usize = shape.iter().map(|&d| d.max(0) as usize).product();

                match ty {
                    TensorElementType::Float32 => {
                        StateValue::F32(shape, vec![0.0f32; num_elements])
                    }
                    TensorElementType::Int64 => {
                        StateValue::I64(shape, vec![0i64; num_elements])
                    }
                    TensorElementType::Bool => {
                        // Bool states are "first" flags — initialized to true
                        StateValue::Bool(shape, vec![true; num_elements])
                    }
                    other => {
                        return Err(format!(
                            "Unsupported mimi state type {:?} for state_{}",
                            other, i
                        ));
                    }
                }
            }
            other => {
                return Err(format!(
                    "Expected tensor for state_{}, got {:?}",
                    i, other
                ));
            }
        };

        states.push(state);
        i += 1;
    }

    log::debug!("Initialized {} mimi decoder states", states.len());
    Ok(states)
}

/// Extract updated mimi states from outputs, using input state types as template.
fn extract_mimi_states(
    outputs: &ort::session::SessionOutputs,
    offset: usize,
    template: &[StateValue],
) -> Result<Vec<StateValue>, String> {
    let mut states = Vec::with_capacity(template.len());
    for (i, tmpl) in template.iter().enumerate() {
        let idx = offset + i;
        let state = match tmpl {
            StateValue::F32(..) => {
                let (shape, data) = outputs[idx]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("Mimi state {} f32: {}", i, e))?;
                StateValue::F32(shape.to_vec(), data.to_vec())
            }
            StateValue::I64(..) => {
                let (shape, data) = outputs[idx]
                    .try_extract_tensor::<i64>()
                    .map_err(|e| format!("Mimi state {} i64: {}", i, e))?;
                StateValue::I64(shape.to_vec(), data.to_vec())
            }
            StateValue::Bool(..) => {
                let (shape, data) = outputs[idx]
                    .try_extract_tensor::<bool>()
                    .map_err(|e| format!("Mimi state {} bool: {}", i, e))?;
                StateValue::Bool(shape.to_vec(), data.to_vec())
            }
        };
        states.push(state);
    }
    Ok(states)
}

// ── Engine ──────────────────────────────────────────────────────

pub struct PocketTTSEngine {
    text_conditioner: Session,
    mimi_encoder: Session,
    flow_lm_main: Session,
    flow_lm_flow: Session,
    mimi_decoder: Session,
    tokenizer: SentencePieceTokenizer,
    model_dir: std::path::PathBuf,
}

impl PocketTTSEngine {
    pub fn initialize(model_dir: &Path) -> Result<Self, String> {
        log::info!("=== PocketTTS initialize START === dir={}", model_dir.display());

        // List model files for diagnostics
        if let Ok(entries) = std::fs::read_dir(model_dir) {
            let files: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                    format!("  {} ({:.1} MB)", name, size as f64 / 1_048_576.0)
                })
                .collect();
            log::info!("Model directory contents:\n{}", files.join("\n"));
        }

        log::info!("Loading tokenizer...");
        let tokenizer = SentencePieceTokenizer::load(&model_dir.join("tokenizer.model"))?;
        log::info!("Tokenizer loaded");

        log::info!("Loading ONNX sessions...");
        let text_conditioner =
            load_session(&model_dir.join("text_conditioner.onnx"), "text_conditioner")?;
        log::info!("  text_conditioner loaded");
        let mimi_encoder = load_session(&model_dir.join("mimi_encoder.onnx"), "mimi_encoder")?;
        log::info!("  mimi_encoder loaded");
        let flow_lm_main =
            load_session(&model_dir.join("flow_lm_main_int8.onnx"), "flow_lm_main")?;
        log::info!("  flow_lm_main loaded");
        let flow_lm_flow =
            load_session(&model_dir.join("flow_lm_flow_int8.onnx"), "flow_lm_flow")?;
        log::info!("  flow_lm_flow loaded");
        let mimi_decoder =
            load_session(&model_dir.join("mimi_decoder_int8.onnx"), "mimi_decoder")?;
        log::info!("  mimi_decoder loaded");

        log::info!("=== PocketTTS initialize DONE ===");

        Ok(Self {
            text_conditioner,
            mimi_encoder,
            flow_lm_main,
            flow_lm_flow,
            mimi_decoder,
            tokenizer,
            model_dir: model_dir.to_path_buf(),
        })
    }

    pub fn generate_speech(
        &mut self,
        text: &str,
        voice_id: &str,
        speed: f32,
        output_path: &Path,
    ) -> Result<(), String> {
        let total_start = std::time::Instant::now();
        log::info!(
            "=== PocketTTS generate_speech START === voice={}, speed={}, text=\"{}\"",
            voice_id,
            speed,
            &text[..std::cmp::min(text.len(), 80)]
        );

        // 1. Prepare text
        let prepared = prepare_text(text);
        let max_gen_frames = compute_max_frames(&prepared);
        let frames_after_eos = compute_frames_after_eos(&prepared);
        log::info!(
            "[Step 1/9] Text prepared: \"{}\" | max_frames={}, frames_after_eos={}",
            &prepared[..std::cmp::min(prepared.len(), 80)],
            max_gen_frames,
            frames_after_eos
        );

        // 2. Tokenize
        let token_ids = self.tokenizer.tokenize(&prepared);
        log::info!(
            "[Step 2/9] Tokenized: {} tokens → {:?}",
            token_ids.len(),
            &token_ids[..std::cmp::min(token_ids.len(), 20)]
        );

        // 3. Run text conditioner → text embeddings [1, T, 1024]
        let t = std::time::Instant::now();
        let text_embeddings = self.run_text_conditioner(&token_ids)?;
        let text_len = token_ids.len();
        log::info!(
            "[Step 3/9] Text conditioner: {} tokens → {} floats ({} ms)",
            text_len,
            text_embeddings.len(),
            t.elapsed().as_millis()
        );

        // 4. Initialize flow_lm state and load voice
        let t = std::time::Instant::now();
        let mut flow_states = init_flow_lm_states();
        let voice_tensors = self.load_voice_embedding(voice_id)?;
        log::info!(
            "[Step 4/9] Voice embedding loaded: {} tensors from '{}' ({} ms)",
            voice_tensors.len(),
            voice_id,
            t.elapsed().as_millis()
        );
        load_voice_into_states(&mut flow_states, &voice_tensors);

        // 5. Text conditioning pass: process text through backbone, updating KV cache
        let t = std::time::Instant::now();
        log::info!(
            "[Step 5/9] Text conditioning pass: seq_len=0, text_len={}, 18 states",
            text_len
        );
        flow_states = self
            .run_flow_lm_step(
                &[],  // empty sequence (no backbone input)
                0,    // seq_len = 0
                &text_embeddings,
                text_len,
                flow_states,
            )?
            .2; // only need updated states
        log::info!(
            "[Step 5/9] Text conditioning done ({} ms)",
            t.elapsed().as_millis()
        );

        // 6. Autoregressive generation
        let t = std::time::Instant::now();
        log::info!(
            "[Step 6/9] Starting AR generation: max_frames={}, frames_after_eos={}",
            max_gen_frames,
            frames_after_eos
        );
        let latents = self.autoregressive_generate(flow_states, max_gen_frames, frames_after_eos)?;
        log::info!(
            "[Step 6/9] AR generation done: {} latent frames ({} ms)",
            latents.len(),
            t.elapsed().as_millis()
        );

        // 7. Decode latents to audio
        let t = std::time::Instant::now();
        log::info!("[Step 7/9] Decoding {} latent frames via mimi", latents.len());
        let mut audio = self.decode_latents(&latents)?;
        log::info!(
            "[Step 7/9] Mimi decode done: {} audio samples ({} ms)",
            audio.len(),
            t.elapsed().as_millis()
        );

        // 8. Speed adjustment
        if (speed - 1.0).abs() > 0.01 {
            let original_len = audio.len();
            audio = resample_audio(&audio, speed);
            log::info!(
                "[Step 8/9] Speed adjustment {:.2}x: {} → {} samples",
                speed,
                original_len,
                audio.len()
            );
        } else {
            log::info!("[Step 8/9] Speed adjustment: skipped (1.0x)");
        }

        // 9. Write WAV
        write_wav(&audio, SAMPLE_RATE, output_path)?;
        let duration_sec = audio.len() as f64 / SAMPLE_RATE as f64;
        log::info!(
            "[Step 9/9] WAV written: {:.1}s audio to {}",
            duration_sec,
            output_path.display()
        );

        log::info!(
            "=== PocketTTS generate_speech DONE === {:.1}s audio in {} ms",
            duration_sec,
            total_start.elapsed().as_millis()
        );

        Ok(())
    }

    pub fn clone_voice(
        &mut self,
        text: &str,
        reference_audio: &Path,
        output_path: &Path,
    ) -> Result<(), String> {
        let total_start = std::time::Instant::now();
        log::info!(
            "=== PocketTTS clone_voice START === ref={}, text=\"{}\"",
            reference_audio.display(),
            &text[..std::cmp::min(text.len(), 80)]
        );

        // 1. Prepare text
        let prepared = prepare_text(text);
        let max_gen_frames = compute_max_frames(&prepared);
        let frames_after_eos = compute_frames_after_eos(&prepared);
        log::info!(
            "[Clone 1/8] Text prepared: \"{}\" | max_frames={}, frames_after_eos={}",
            &prepared[..std::cmp::min(prepared.len(), 80)],
            max_gen_frames,
            frames_after_eos
        );

        // 2. Tokenize and get text embeddings
        let t = std::time::Instant::now();
        let token_ids = self.tokenizer.tokenize(&prepared);
        let text_embeddings = self.run_text_conditioner(&token_ids)?;
        let text_len = token_ids.len();
        log::info!(
            "[Clone 2/8] Tokenized + text conditioner: {} tokens ({} ms)",
            text_len,
            t.elapsed().as_millis()
        );

        // 3. Load and validate reference audio
        let t = std::time::Instant::now();
        let mut ref_samples = load_audio_samples(reference_audio)?;
        let original_len = ref_samples.len();
        let original_duration = original_len as f32 / SAMPLE_RATE as f32;

        if ref_samples.is_empty() {
            return Err("Reference audio is empty".into());
        }

        // Truncate overly long reference audio to prevent OOM
        if ref_samples.len() > MAX_REFERENCE_SAMPLES {
            log::warn!(
                "[Clone 3/8] Reference audio too long ({:.1}s, {} samples) — truncating to {:.0}s",
                original_duration,
                original_len,
                MAX_REFERENCE_SECONDS
            );
            ref_samples.truncate(MAX_REFERENCE_SAMPLES);
        }

        let ref_duration = ref_samples.len() as f32 / SAMPLE_RATE as f32;
        let ref_mb = (ref_samples.len() * 4) as f64 / 1_048_576.0;
        log::info!(
            "[Clone 3/8] Reference audio loaded: {:.1}s ({} samples, {:.1} MB) ({} ms)",
            ref_duration,
            ref_samples.len(),
            ref_mb,
            t.elapsed().as_millis()
        );

        // 4. Encode reference audio → conditioning [1, T', 1024]
        let t = std::time::Instant::now();
        log::info!(
            "[Clone 4/8] Encoding reference audio ({} samples) via mimi_encoder...",
            ref_samples.len()
        );
        let (audio_conditioning, audio_cond_len) =
            self.encode_reference_audio(&ref_samples)?;
        log::info!(
            "[Clone 4/8] Mimi encode done: {} conditioning frames, {} floats ({} ms)",
            audio_cond_len,
            audio_conditioning.len(),
            t.elapsed().as_millis()
        );

        // 5. Initialize fresh flow_lm state (no predefined voice)
        let flow_states = init_flow_lm_states();
        log::info!("[Clone 5/8] Initialized fresh flow_lm states (18 tensors)");

        // 6. Voice conditioning pass: audio conditioning through backbone
        let t = std::time::Instant::now();
        log::info!(
            "[Clone 6/8] Voice conditioning pass: {} frames through flow_lm...",
            audio_cond_len
        );
        let flow_states = self.run_flow_lm_step(
            &[],                    // empty sequence
            0,
            &audio_conditioning,    // audio conditioning as text_embeddings
            audio_cond_len,
            flow_states,
        )?
        .2;
        log::info!(
            "[Clone 6/8] Voice conditioning done ({} ms)",
            t.elapsed().as_millis()
        );

        // Drop audio conditioning to free memory before generation
        drop(audio_conditioning);
        drop(ref_samples);

        // 7. Text conditioning pass
        let t = std::time::Instant::now();
        log::info!(
            "[Clone 7/8] Text conditioning pass: {} tokens through flow_lm...",
            text_len
        );
        let flow_states = self.run_flow_lm_step(
            &[],
            0,
            &text_embeddings,
            text_len,
            flow_states,
        )?
        .2;
        log::info!(
            "[Clone 7/8] Text conditioning done ({} ms)",
            t.elapsed().as_millis()
        );

        // Drop text embeddings before generation loop
        drop(text_embeddings);

        // 8. Autoregressive generation + decode + write
        let t = std::time::Instant::now();
        log::info!(
            "[Clone 8/8] AR generation: max_frames={}, frames_after_eos={}",
            max_gen_frames,
            frames_after_eos
        );
        let latents = self.autoregressive_generate(
            flow_states,
            max_gen_frames,
            frames_after_eos,
        )?;
        log::info!(
            "[Clone 8/8] Generated {} latent frames ({} ms)",
            latents.len(),
            t.elapsed().as_millis()
        );

        // Decode and write
        let t = std::time::Instant::now();
        let audio = self.decode_latents(&latents)?;
        let duration_sec = audio.len() as f64 / SAMPLE_RATE as f64;
        log::info!(
            "[Clone] Mimi decode: {:.1}s audio, {} samples ({} ms)",
            duration_sec,
            audio.len(),
            t.elapsed().as_millis()
        );

        write_wav(&audio, SAMPLE_RATE, output_path)?;

        log::info!(
            "=== PocketTTS clone_voice DONE === {:.1}s audio in {} ms, output={}",
            duration_sec,
            total_start.elapsed().as_millis(),
            output_path.display()
        );

        Ok(())
    }

    /// Shut down the engine, releasing all ONNX sessions.
    pub fn shutdown(self) {
        log::info!("PocketTTS engine shutting down");
    }

    pub fn get_available_voices(&self) -> Vec<String> {
        let emb_dir = self.model_dir.join("embeddings_v2");
        let mut voices = Vec::new();

        if let Ok(entries) = std::fs::read_dir(emb_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".safetensors") {
                    voices.push(name.trim_end_matches(".safetensors").to_string());
                }
            }
        }

        voices.sort();
        voices
    }

    // ── Internal inference methods ──────────────────────────────

    /// Run text conditioner: token_ids → text embeddings [1, T, 1024].
    /// Returns flattened embedding data.
    fn run_text_conditioner(&mut self, token_ids: &[i64]) -> Result<Vec<f32>, String> {
        let seq_len = token_ids.len();
        let input_tensor = Tensor::from_array((vec![1i64, seq_len as i64], token_ids.to_vec()))
            .map_err(|e| format!("Token tensor error: {}", e))?;

        let outputs = self
            .text_conditioner
            .run(ort::inputs!["token_ids" => input_tensor])
            .map_err(|e| format!("Text conditioner error: {}", e))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Text conditioner output error: {}", e))?;

        Ok(data.to_vec())
    }

    /// Load voice embedding from safetensors, returning tensors with shapes.
    fn load_voice_embedding(
        &self,
        voice_id: &str,
    ) -> Result<HashMap<String, (Vec<i64>, Vec<f32>)>, String> {
        let emb_path = self
            .model_dir
            .join("embeddings_v2")
            .join(format!("{}.safetensors", voice_id));

        if !emb_path.exists() {
            return Err(format!("Voice embedding not found: {}", voice_id));
        }

        load_safetensors_with_shapes(&emb_path)
    }

    /// Encode reference audio → conditioning embeddings [1, T', 1024].
    /// Returns (flat_data, num_frames).
    fn encode_reference_audio(
        &mut self,
        audio_samples: &[f32],
    ) -> Result<(Vec<f32>, usize), String> {
        let num_samples = audio_samples.len();
        let input_tensor =
            Tensor::from_array((vec![1i64, 1, num_samples as i64], audio_samples.to_vec()))
                .map_err(|e| format!("Audio tensor error: {}", e))?;

        let outputs = self
            .mimi_encoder
            .run(ort::inputs!["audio" => input_tensor])
            .map_err(|e| format!("Mimi encoder error: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Encoder output error: {}", e))?;

        // shape is [1, T', 1024]
        let num_frames = if shape.len() >= 2 {
            shape[1] as usize
        } else {
            1
        };

        Ok((data.to_vec(), num_frames))
    }

    /// Run one step of flow_lm_main.
    /// Returns (conditioning [1024], eos_logit, updated_states).
    fn run_flow_lm_step(
        &mut self,
        sequence_data: &[f32], // flat [1, S, 32]
        seq_len: usize,
        text_emb_data: &[f32], // flat [1, T, 1024]
        text_len: usize,
        states: Vec<StateValue>,
    ) -> Result<(Vec<f32>, f32, Vec<StateValue>), String> {
        let mut inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> =
            Vec::with_capacity(2 + states.len());

        // Sequence input [1, seq_len, 32]
        // Note: ort Tensor::from_array((shape, data)) rejects zero dimensions,
        // but ONNX Runtime supports them. Use ndarray which handles zero-size arrays.
        let seq_array = ndarray::Array3::<f32>::from_shape_vec(
            (1, seq_len, LATENT_DIM),
            sequence_data.to_vec(),
        )
        .map_err(|e| format!("Sequence array error: {}", e))?;
        let seq_value = Value::from_array(seq_array)
            .map_err(|e| format!("Sequence tensor error: {}", e))?;
        inputs.push(("sequence".into(), seq_value.into()));

        // Text embeddings input [1, text_len, 1024]
        let text_array = ndarray::Array3::<f32>::from_shape_vec(
            (1, text_len, 1024),
            text_emb_data.to_vec(),
        )
        .map_err(|e| format!("Text embedding array error: {}", e))?;
        let text_value = Value::from_array(text_array)
            .map_err(|e| format!("Text embedding tensor error: {}", e))?;
        inputs.push(("text_embeddings".into(), text_value.into()));

        // State inputs
        for (i, state) in states.iter().enumerate() {
            let name = format!("state_{}", i);
            let input = state
                .to_session_input()
                .map_err(|e| format!("State {} input error: {}", i, e))?;
            inputs.push((name.into(), input));
        }

        // Run
        let outputs = self
            .flow_lm_main
            .run(inputs)
            .map_err(|e| format!("flow_lm_main error: {}", e))?;

        // Extract conditioning [1, 1024]
        let (_cond_shape, cond_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Conditioning extraction error: {}", e))?;
        let conditioning = cond_data.to_vec();

        // Extract eos_logit [1, 1]
        let (_eos_shape, eos_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("EOS logit extraction error: {}", e))?;
        let eos_logit = eos_data[0];

        // Extract updated states
        let new_states = extract_flow_lm_states(&outputs, 2)?;

        Ok((conditioning, eos_logit, new_states))
    }

    /// Run flow matching: conditioning → latent via LSD decode (1 step).
    /// flow_net(c, s=0.0, t=1.0, x=zeros) → flow_dir
    fn run_flow_matching(&mut self, conditioning: &[f32]) -> Result<Vec<f32>, String> {
        let c_tensor = Tensor::from_array((vec![1i64, 1024i64], conditioning.to_vec()))
            .map_err(|e| format!("Conditioning tensor error: {}", e))?;
        let s_tensor = Tensor::from_array((vec![1i64, 1i64], vec![0.0f32]))
            .map_err(|e| format!("S tensor error: {}", e))?;
        let t_tensor = Tensor::from_array((vec![1i64, 1i64], vec![1.0f32]))
            .map_err(|e| format!("T tensor error: {}", e))?;
        let x_tensor =
            Tensor::from_array((vec![1i64, LATENT_DIM as i64], vec![0.0f32; LATENT_DIM]))
                .map_err(|e| format!("X tensor error: {}", e))?;

        let outputs = self
            .flow_lm_flow
            .run(
                ort::inputs!["c" => c_tensor, "s" => s_tensor, "t" => t_tensor, "x" => x_tensor],
            )
            .map_err(|e| format!("flow_lm_flow error: {}", e))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Flow output error: {}", e))?;

        Ok(data.to_vec())
    }

    /// Autoregressive generation loop.
    /// Returns generated latent frames.
    fn autoregressive_generate(
        &mut self,
        mut flow_states: Vec<StateValue>,
        max_frames: usize,
        frames_after_eos: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut latents: Vec<Vec<f32>> = Vec::new();

        // First backbone input is NaN (signals BOS position)
        let mut backbone_input = vec![f32::NAN; LATENT_DIM];
        let mut eos_step: Option<usize> = None;

        let start = std::time::Instant::now();

        for step in 0..max_frames {
            // Run flow_lm_main with backbone input [1,1,32]
            let (conditioning, eos_logit, new_states) = self.run_flow_lm_step(
                &backbone_input, // [1, 1, 32] flat
                1,               // seq_len = 1
                &[],             // empty text embeddings
                0,               // text_len = 0
                flow_states,
            )?;
            flow_states = new_states;

            // Check EOS
            if eos_logit > EOS_THRESHOLD && eos_step.is_none() {
                eos_step = Some(step);
                log::debug!("EOS detected at step {} (logit={:.2})", step, eos_logit);
            }
            if let Some(eos) = eos_step {
                if step >= eos + frames_after_eos {
                    log::debug!("Stopping after {} frames post-EOS", frames_after_eos);
                    break;
                }
            }

            // Run flow matching: conditioning → latent
            let latent = self.run_flow_matching(&conditioning)?;

            latents.push(latent.clone());
            backbone_input = latent;

            if step % 50 == 0 && step > 0 {
                let elapsed = start.elapsed().as_millis();
                log::debug!("Step {}/{} ({} ms elapsed)", step, max_frames, elapsed);
            }
        }

        if eos_step.is_none() {
            log::warn!(
                "Reached max generation length ({}) without EOS",
                max_frames
            );
        }

        let elapsed = start.elapsed().as_millis();
        let audio_ms = latents.len() as f64 * 80.0; // ~80ms per frame at 12.5 Hz
        let rtf = if elapsed > 0 {
            audio_ms / elapsed as f64
        } else {
            0.0
        };
        log::info!(
            "Generated {:.0} ms audio in {} ms ({:.2}x real-time)",
            audio_ms,
            elapsed,
            rtf
        );

        Ok(latents)
    }

    /// Decode latent frames to audio via mimi decoder (streaming, frame by frame).
    fn decode_latents(&mut self, latents: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        if latents.is_empty() {
            return Ok(Vec::new());
        }

        let mut mimi_states = init_mimi_states(&self.mimi_decoder)?;
        let mut audio_samples: Vec<f32> = Vec::new();

        for (i, latent) in latents.iter().enumerate() {
            // Build inputs: latent [1, 1, 32] + 56 state tensors
            let mut inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> =
                Vec::with_capacity(1 + mimi_states.len());

            let latent_tensor = Tensor::from_array((
                vec![1i64, 1i64, LATENT_DIM as i64],
                latent.clone(),
            ))
            .map_err(|e| format!("Mimi latent tensor error: {}", e))?;
            inputs.push(("latent".into(), latent_tensor.into()));

            for (j, state) in mimi_states.iter().enumerate() {
                let name = format!("state_{}", j);
                let input = state
                    .to_session_input()
                    .map_err(|e| format!("Mimi state {} input error: {}", j, e))?;
                inputs.push((name.into(), input));
            }

            // Run mimi decoder
            let outputs = self
                .mimi_decoder
                .run(inputs)
                .map_err(|e| format!("Mimi decoder error at frame {}: {}", i, e))?;

            // Extract audio frame [1, 1, N]
            let (_shape, audio_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Mimi audio extraction error: {}", e))?;
            audio_samples.extend_from_slice(audio_data);

            // Extract updated states
            mimi_states = extract_mimi_states(&outputs, 1, &mimi_states)?;

            if i % 50 == 0 && i > 0 {
                log::debug!("Decoded frame {}/{}", i, latents.len());
            }
        }

        Ok(audio_samples)
    }
}

// ── Text preparation (matching PyTorch prepare_text_prompt) ─────

fn prepare_text(text: &str) -> String {
    let mut text = text.trim().to_string();
    if text.is_empty() {
        return text;
    }

    // Normalize whitespace
    text = text.replace('\n', " ").replace('\r', " ");
    while text.contains("  ") {
        text = text.replace("  ", " ");
    }

    // Capitalize first letter
    if let Some(first) = text.chars().next() {
        if first.is_lowercase() {
            text = format!(
                "{}{}",
                first.to_uppercase(),
                &text[first.len_utf8()..]
            );
        }
    }

    // Ensure ends with punctuation
    if let Some(last) = text.chars().last() {
        if last.is_alphanumeric() {
            text.push('.');
        }
    }

    // Pad short texts (model performs poorly with very few tokens)
    if text.split_whitespace().count() < 5 {
        text = format!("        {}", text); // 8 spaces
    }

    text
}

fn compute_max_frames(text: &str) -> usize {
    let words = text.split_whitespace().count();
    let gen_len_sec = words as f64 + 2.0;
    (gen_len_sec * 12.5) as usize
}

fn compute_frames_after_eos(text: &str) -> usize {
    let words = text.split_whitespace().count();
    if words <= 4 { 5 } else { 3 }
}

// ── Session loader ──────────────────────────────────────────────

fn load_session(path: &Path, name: &str) -> Result<Session, String> {
    if !path.exists() {
        return Err(format!("Model file not found: {}", path.display()));
    }

    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);
    log::debug!(
        "Loading ONNX session: {} from {} ({:.1} MB)",
        name,
        path.display(),
        file_size as f64 / 1_048_576.0
    );

    // Limit threads to avoid saturating all CPU cores and freezing the system.
    // intra_threads: parallelism within a single op (e.g., matmul)
    // inter_threads: parallelism across independent ops in the graph
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    // Use at most half the cores (min 2, max 4) so the UI stays responsive
    let intra = num_cpus.div_ceil(2).clamp(2, 4);

    Session::builder()
        .map_err(|e| format!("Session builder error for {}: {}", name, e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("Optimization error for {}: {}", name, e))?
        .with_intra_threads(intra)
        .map_err(|e| format!("Intra-thread config error for {}: {}", name, e))?
        .with_inter_threads(2)
        .map_err(|e| format!("Inter-thread config error for {}: {}", name, e))?
        .commit_from_file(path)
        .map_err(|e| format!("Failed to load {}: {}", name, e))
}

// ── Safetensors loader (with shape info) ────────────────────────

fn load_safetensors_with_shapes(
    path: &Path,
) -> Result<HashMap<String, (Vec<i64>, Vec<f32>)>, String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read safetensors: {}", e))?;

    if data.len() < 8 {
        return Err("Safetensors file too small".into());
    }

    let header_len = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    if data.len() < 8 + header_len {
        return Err("Safetensors file truncated".into());
    }

    let header_json = std::str::from_utf8(&data[8..8 + header_len])
        .map_err(|e| format!("Invalid safetensors header: {}", e))?;
    let header: serde_json::Value = serde_json::from_str(header_json)
        .map_err(|e| format!("Failed to parse safetensors header: {}", e))?;

    let data_offset = 8 + header_len;
    let mut tensors = HashMap::new();

    if let Some(obj) = header.as_object() {
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }

            let dtype = meta["dtype"].as_str().unwrap_or("");
            let shape: Vec<i64> = meta["shape"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_i64())
                        .collect()
                })
                .unwrap_or_default();

            let offsets = meta["data_offsets"]
                .as_array()
                .ok_or_else(|| format!("Missing data_offsets for {}", name))?;
            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;

            if data_offset + end > data.len() {
                return Err(format!(
                    "Tensor '{}' data out of bounds",
                    name,
                ));
            }

            let tensor_data = &data[data_offset + start..data_offset + end];

            let floats = match dtype {
                "F32" => {
                    let num_floats = tensor_data.len() / 4;
                    let mut result = vec![0.0f32; num_floats];
                    for i in 0..num_floats {
                        let bytes: [u8; 4] = tensor_data[i * 4..(i + 1) * 4]
                            .try_into()
                            .map_err(|_| "F32 read error".to_string())?;
                        result[i] = f32::from_le_bytes(bytes);
                    }
                    result
                }
                "F16" => {
                    let num_floats = tensor_data.len() / 2;
                    let mut result = vec![0.0f32; num_floats];
                    for i in 0..num_floats {
                        let bytes: [u8; 2] = tensor_data[i * 2..(i + 1) * 2]
                            .try_into()
                            .map_err(|_| "F16 read error".to_string())?;
                        result[i] = half_to_float(u16::from_le_bytes(bytes));
                    }
                    result
                }
                _ => continue,
            };

            tensors.insert(name.clone(), (shape, floats));
        }
    }

    Ok(tensors)
}

fn half_to_float(half: u16) -> f32 {
    let sign = (half >> 15) & 1;
    let exp = (half >> 10) & 0x1F;
    let mant = half & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = f32::powi(2.0, -14) * (mant as f32 / 1024.0);
        return if sign == 1 { -val } else { val };
    } else if exp == 31 {
        if mant == 0 {
            return if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }
        return f32::NAN;
    }

    let val = f32::powi(2.0, exp as i32 - 15) * (1.0 + mant as f32 / 1024.0);
    if sign == 1 { -val } else { val }
}

// ── Audio utilities ─────────────────────────────────────────────

fn load_audio_samples(path: &Path) -> Result<Vec<f32>, String> {
    let reader =
        hound::WavReader::open(path).map_err(|e| format!("Failed to open WAV: {}", e))?;

    let spec = reader.spec();
    log::info!(
        "Loading audio: {}ch, {}Hz, {}bit {:?} from {}",
        spec.channels,
        spec.sample_rate,
        spec.bits_per_sample,
        spec.sample_format,
        path.display()
    );

    let raw_samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Int {
        if spec.bits_per_sample == 16 {
            reader
                .into_samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect()
        } else {
            let divisor = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / divisor)
                .collect()
        }
    } else {
        reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect()
    };

    // Mix to mono if stereo/multi-channel
    let channels = spec.channels as usize;
    let mono = if channels > 1 {
        log::info!("Mixing {} channels to mono", channels);
        raw_samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw_samples
    };

    // Resample to SAMPLE_RATE if needed (simple linear interpolation)
    let samples = if spec.sample_rate != SAMPLE_RATE {
        log::info!(
            "Resampling audio: {}Hz → {}Hz ({} samples)",
            spec.sample_rate,
            SAMPLE_RATE,
            mono.len()
        );
        let ratio = SAMPLE_RATE as f64 / spec.sample_rate as f64;
        let new_len = (mono.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx0 = src_idx as usize;
            let idx1 = (idx0 + 1).min(mono.len() - 1);
            let frac = (src_idx - idx0 as f64) as f32;
            resampled.push(mono[idx0] * (1.0 - frac) + mono[idx1] * frac);
        }
        resampled
    } else {
        mono
    };

    let duration = samples.len() as f32 / SAMPLE_RATE as f32;
    log::info!(
        "Audio loaded: {} samples ({:.1}s at {}Hz)",
        samples.len(),
        duration,
        SAMPLE_RATE
    );

    Ok(samples)
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
