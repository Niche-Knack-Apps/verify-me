use crate::services::tokenizers::BpeTokenizer;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

const SAMPLE_RATE: u32 = 24000;
const MAX_STEPS: usize = 4096;
const EOS_ID: i64 = 1;

/// ONNX inference engine for Qwen3-TTS (0.6B / 1.7B variants).
///
/// Pipeline:
///   tokenize → text_project → talker_prefill →
///   talker_decode (autoregressive) → code_predictor →
///   tokenizer12hz_decode → audio output
pub struct Qwen3TTSEngine {
    codec_embed: Session,
    speaker_encoder: Option<Session>,
    code_predictor_embed: Session,
    code_predictor: Session,
    tokenizer12hz_encode: Session,
    tokenizer12hz_decode: Session,
    text_project: Session,
    talker_decode: Session,
    talker_prefill: Session,
    tokenizer: BpeTokenizer,
}

impl Qwen3TTSEngine {
    /// Load all ONNX sessions and the BPE tokenizer from `model_dir`.
    pub fn initialize(model_dir: &Path) -> Result<Self, String> {
        log::info!("Loading Qwen3 TTS from: {}", model_dir.display());

        let tokenizer = BpeTokenizer::load(
            &model_dir.join("vocab.json"),
            &model_dir.join("merges.txt"),
        )?;

        let codec_embed = load_session(&model_dir.join("codec_embed_q.onnx"), "codec_embed")?;
        let code_predictor_embed = load_session(
            &model_dir.join("code_predictor_embed_q.onnx"),
            "code_predictor_embed",
        )?;
        let code_predictor =
            load_session(&model_dir.join("code_predictor_q.onnx"), "code_predictor")?;
        let tokenizer12hz_encode = load_session(
            &model_dir.join("tokenizer12hz_encode_q.onnx"),
            "tokenizer12hz_encode",
        )?;
        let tokenizer12hz_decode = load_session(
            &model_dir.join("tokenizer12hz_decode_q.onnx"),
            "tokenizer12hz_decode",
        )?;
        let text_project =
            load_session(&model_dir.join("text_project_q.onnx"), "text_project")?;
        let talker_decode =
            load_session(&model_dir.join("talker_decode_q.onnx"), "talker_decode")?;
        let talker_prefill =
            load_session(&model_dir.join("talker_prefill_q.onnx"), "talker_prefill")?;

        // Speaker encoder is optional (only Base variant)
        let speaker_encoder_path = model_dir.join("speaker_encoder_q.onnx");
        let speaker_encoder = if speaker_encoder_path.exists() {
            Some(load_session(&speaker_encoder_path, "speaker_encoder")?)
        } else {
            log::info!("No speaker_encoder — voice cloning unavailable for this variant");
            None
        };

        log::info!("Qwen3 TTS initialized successfully");

        Ok(Self {
            codec_embed,
            speaker_encoder,
            code_predictor_embed,
            code_predictor,
            tokenizer12hz_encode,
            tokenizer12hz_decode,
            text_project,
            talker_decode,
            talker_prefill,
            tokenizer,
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
        log::info!("Qwen3 generating speech: voice={}, speed={}", voice_id, speed);

        // 1. Tokenize text with BPE
        let token_ids = self.tokenizer.tokenize(text);
        log::debug!("Tokenized: {} tokens", token_ids.len());

        // 2. Project text tokens to embeddings
        let (text_emb_shape, text_emb_data) = self.run_text_project(&token_ids)?;

        // 3. Prefill with text embeddings + voice conditioning
        let (prefill_shape, prefill_data) =
            self.run_talker_prefill(&text_emb_shape, &text_emb_data, voice_id)?;

        // 4. Autoregressive generation
        let codes = self.autoregressive_generate(&prefill_shape, &prefill_data)?;
        log::info!("Generated {} code frames", codes.len());

        // 5. Decode audio codes
        let mut audio = self.decode_audio(&codes)?;
        log::info!("Decoded {} audio samples", audio.len());

        // 6. Speed adjustment
        if (speed - 1.0).abs() > 0.01 {
            audio = resample_audio(&audio, speed);
        }

        // 7. Write WAV
        write_wav(&audio, SAMPLE_RATE, output_path)?;
        Ok(())
    }

    /// Generate speech using reference audio for voice cloning.
    pub fn clone_voice(
        &mut self,
        text: &str,
        reference_audio: &Path,
        output_path: &Path,
    ) -> Result<(), String> {
        let has_encoder = self.speaker_encoder.is_some();
        if !has_encoder {
            return Err(
                "Voice cloning not supported by this Qwen3 variant (requires Base model)".into(),
            );
        }

        log::info!("Qwen3 cloning voice from: {}", reference_audio.display());

        // 1. Tokenize text
        let token_ids = self.tokenizer.tokenize(text);

        // 2. Project text tokens
        let (text_emb_shape, text_emb_data) = self.run_text_project(&token_ids)?;

        // 3. Encode reference audio for speaker embedding
        let ref_samples = load_audio_samples(reference_audio)?;
        let (speaker_shape, speaker_data) = self.run_speaker_encoder(&ref_samples)?;

        // 4. Prefill with text + speaker embedding
        let (prefill_shape, prefill_data) = self.run_talker_prefill_with_speaker(
            &text_emb_shape,
            &text_emb_data,
            &speaker_shape,
            &speaker_data,
        )?;

        // 5. Autoregressive generation
        let codes = self.autoregressive_generate(&prefill_shape, &prefill_data)?;

        // 6. Decode audio
        let audio = self.decode_audio(&codes)?;

        // 7. Write WAV
        write_wav(&audio, SAMPLE_RATE, output_path)?;
        Ok(())
    }

    /// Returns hardcoded list of available voice presets.
    pub fn get_available_voices(&self) -> Vec<String> {
        vec![
            "Aiden".into(),
            "Ryan".into(),
            "Vivian".into(),
            "Serena".into(),
            "Dylan".into(),
            "Eric".into(),
            "Uncle_Fu".into(),
            "Ono_Anna".into(),
            "Sohee".into(),
        ]
    }

    /// Drop all sessions and release resources.
    pub fn shutdown(self) {
        log::info!("Shutting down Qwen3 TTS engine");
        // All sessions dropped when `self` is consumed
    }

    // ── ONNX inference steps ────────────────────────────────────

    /// Run text_project: input_ids → text embeddings [1, seq_len, embed_dim]
    fn run_text_project(
        &mut self,
        token_ids: &[i64],
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let seq_len = token_ids.len();
        let input_tensor =
            Tensor::from_array(([1i64, seq_len as i64], token_ids.to_vec()))
                .map_err(|e| format!("Tensor error: {}", e))?;

        let outputs = self
            .text_project
            .run(ort::inputs!["input_ids" => input_tensor])
            .map_err(|e| format!("text_project error: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Output extraction error: {}", e))?;

        Ok((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
    }

    /// Run speaker_encoder: audio [1, 1, num_samples] → speaker embedding.
    fn run_speaker_encoder(
        &mut self,
        audio_samples: &[f32],
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let encoder = self.speaker_encoder.as_mut().unwrap();
        let num_samples = audio_samples.len();
        let input_tensor = Tensor::from_array((
            [1i64, 1, num_samples as i64],
            audio_samples.to_vec(),
        ))
        .map_err(|e| format!("Audio tensor error: {}", e))?;

        let outputs = encoder
            .run(ort::inputs!["audio" => input_tensor])
            .map_err(|e| format!("speaker_encoder error: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Speaker output error: {}", e))?;

        Ok((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
    }

    /// Run talker_prefill with a voice ID token → prefill state.
    fn run_talker_prefill(
        &mut self,
        text_emb_shape: &[usize],
        text_emb_data: &[f32],
        voice_id: &str,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let text_shape: Vec<i64> = text_emb_shape.iter().map(|&d| d as i64).collect();
        let text_tensor = Tensor::from_array((text_shape, text_emb_data.to_vec()))
            .map_err(|e| format!("Text tensor error: {}", e))?;

        // Direct vocab lookup for voice ID (matches Java: vocabMap.get(voiceId))
        let voice_token_id = self
            .tokenizer
            .vocab_lookup(voice_id)
            .unwrap_or(0);

        let voice_tensor = Tensor::from_array(([1i64, 1], vec![voice_token_id]))
            .map_err(|e| format!("Voice tensor error: {}", e))?;

        let inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![
            ("text_embeddings".into(), text_tensor.into()),
            ("speaker_id".into(), voice_tensor.into()),
        ];

        let outputs = self
            .talker_prefill
            .run(inputs)
            .map_err(|e| format!("talker_prefill error: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Prefill output error: {}", e))?;

        Ok((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
    }

    /// Run talker_prefill with a speaker embedding (for voice cloning).
    fn run_talker_prefill_with_speaker(
        &mut self,
        text_emb_shape: &[usize],
        text_emb_data: &[f32],
        speaker_shape: &[usize],
        speaker_data: &[f32],
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let text_shape: Vec<i64> = text_emb_shape.iter().map(|&d| d as i64).collect();
        let text_tensor = Tensor::from_array((text_shape, text_emb_data.to_vec()))
            .map_err(|e| format!("Text tensor error: {}", e))?;

        let spk_shape: Vec<i64> = speaker_shape.iter().map(|&d| d as i64).collect();
        let speaker_tensor = Tensor::from_array((spk_shape, speaker_data.to_vec()))
            .map_err(|e| format!("Speaker tensor error: {}", e))?;

        let inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![
            ("text_embeddings".into(), text_tensor.into()),
            ("speaker_embedding".into(), speaker_tensor.into()),
        ];

        let outputs = self
            .talker_prefill
            .run(inputs)
            .map_err(|e| format!("talker_prefill (speaker) error: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Prefill output error: {}", e))?;

        Ok((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
    }

    /// Autoregressive decode loop: state → logits → argmax → code frames.
    fn autoregressive_generate(
        &mut self,
        prefill_shape: &[usize],
        prefill_data: &[f32],
    ) -> Result<Vec<Vec<i64>>, String> {
        let mut code_frames: Vec<Vec<i64>> = Vec::new();

        // Current state — starts from prefill, updated each step
        let mut state_shape: Vec<i64> = prefill_shape.iter().map(|&d| d as i64).collect();
        let mut state_data: Vec<f32> = prefill_data.to_vec();

        for step in 0..MAX_STEPS {
            let state_tensor =
                Tensor::from_array((state_shape.clone(), state_data.clone()))
                    .map_err(|e| format!("State tensor error: {}", e))?;

            let step_tensor = Tensor::from_array(([1i64, 1], vec![step as i64]))
                .map_err(|e| format!("Step tensor error: {}", e))?;

            let inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![
                ("state".into(), state_tensor.into()),
                ("step".into(), step_tensor.into()),
            ];

            // Run talker_decode → logits
            let decode_outputs = self
                .talker_decode
                .run(inputs)
                .map_err(|e| format!("talker_decode error at step {}: {}", step, e))?;

            let (_logit_shape, logits_data) = decode_outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Decode output error: {}", e))?;

            // Argmax over logits → code index
            let code_idx = argmax(logits_data);

            // Check EOS (code == 1 after step 10)
            if code_idx as i64 == EOS_ID && step > 10 {
                log::debug!("EOS at step {}", step);
                break;
            }

            // Run code_predictor → predicted codes for this frame
            let code_tensor = Tensor::from_array(([1i64, 1], vec![code_idx as i64]))
                .map_err(|e| format!("Code tensor error: {}", e))?;

            let cp_outputs = self
                .code_predictor
                .run(ort::inputs!["code" => code_tensor])
                .map_err(|e| format!("code_predictor error at step {}: {}", step, e))?;

            let (_cp_shape, predicted_data) = cp_outputs[0]
                .try_extract_tensor::<i64>()
                .map_err(|e| format!("Code predictor output error: {}", e))?;
            code_frames.push(predicted_data.to_vec());

            // Codec embed for feedback into next decode step
            let embed_tensor = Tensor::from_array(([1i64, 1], vec![code_idx as i64]))
                .map_err(|e| format!("Embed tensor error: {}", e))?;

            let embed_outputs = self
                .codec_embed
                .run(ort::inputs!["codes" => embed_tensor])
                .map_err(|e| format!("codec_embed error at step {}: {}", step, e))?;

            let (embed_shape, embed_data) = embed_outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Embed output error: {}", e))?;

            // Update state: reshape to [1, 1, embed_dim]
            let embed_dim = embed_shape[embed_shape.len() - 1] as i64;
            state_shape = vec![1, 1, embed_dim];
            state_data = embed_data.to_vec();

            if step % 100 == 0 && step > 0 {
                log::debug!("Qwen3 generation step {}/{}", step, MAX_STEPS);
            }
        }

        Ok(code_frames)
    }

    /// Decode code frames via tokenizer12hz_decode → audio samples.
    fn decode_audio(&mut self, codes: &[Vec<i64>]) -> Result<Vec<f32>, String> {
        if codes.is_empty() {
            return Ok(Vec::new());
        }

        let num_frames = codes.len();
        let num_codes = codes[0].len();

        // Shape: [1, num_codebooks, num_frames]
        let mut data = vec![0i64; num_codes * num_frames];
        for f in 0..num_frames {
            for c in 0..num_codes {
                if c < codes[f].len() {
                    data[c * num_frames + f] = codes[f][c];
                }
            }
        }

        let shape = vec![1i64, num_codes as i64, num_frames as i64];
        let input_tensor = Tensor::from_array((shape, data))
            .map_err(|e| format!("Decoder tensor error: {}", e))?;

        let outputs = self
            .tokenizer12hz_decode
            .run(ort::inputs!["codes" => input_tensor])
            .map_err(|e| format!("tokenizer12hz_decode error: {}", e))?;

        let (_shape, audio_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Audio decode output error: {}", e))?;

        // Output is [1, 1, num_samples] — return flat
        Ok(audio_data.to_vec())
    }
}

// ── Utilities ───────────────────────────────────────────────────

fn load_session(path: &Path, name: &str) -> Result<Session, String> {
    if !path.exists() {
        return Err(format!("Model file not found: {}", path.display()));
    }

    log::debug!("Loading ONNX session: {} from {}", name, path.display());

    Session::builder()
        .map_err(|e| format!("Session builder error for {}: {}", name, e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("Optimization error for {}: {}", name, e))?
        .with_intra_threads(4)
        .map_err(|e| format!("Thread config error for {}: {}", name, e))?
        .commit_from_file(path)
        .map_err(|e| format!("Failed to load {}: {}", name, e))
}

fn argmax(data: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

/// Load PCM16 WAV samples as normalized f32 [-1.0, 1.0].
fn load_audio_samples(path: &Path) -> Result<Vec<f32>, String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read audio: {}", e))?;

    if data.len() < 44 {
        return Err("Audio file too small for WAV header".into());
    }

    let data_size = u32::from_le_bytes(
        data[40..44]
            .try_into()
            .map_err(|_| "WAV header read error".to_string())?,
    ) as usize;

    let num_samples = data_size / 2;
    let audio_data = &data[44..];

    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        if i * 2 + 1 >= audio_data.len() {
            break;
        }
        let bytes: [u8; 2] = audio_data[i * 2..i * 2 + 2]
            .try_into()
            .map_err(|_| "PCM16 read error".to_string())?;
        samples.push(i16::from_le_bytes(bytes) as f32 / 32768.0);
    }

    Ok(samples)
}

/// Linear interpolation resample for speed adjustment.
fn resample_audio(audio: &[f32], speed: f32) -> Vec<f32> {
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
