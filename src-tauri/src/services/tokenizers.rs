use std::collections::HashMap;
use std::path::Path;

// ── SentencePiece Tokenizer (for Pocket TTS) ───────────────────

pub struct SentencePieceTokenizer {
    vocab: HashMap<String, i32>,
    bos_id: i32,
    eos_id: i32,
}

impl SentencePieceTokenizer {
    pub fn load(tokenizer_path: &Path) -> Result<Self, String> {
        let data = std::fs::read(tokenizer_path)
            .map_err(|e| format!("Failed to read tokenizer: {}", e))?;

        let mut vocab = HashMap::new();
        extract_vocab_from_protobuf(&data, &mut vocab);

        if vocab.is_empty() {
            return Err("Failed to extract vocabulary from tokenizer.model".into());
        }

        log::info!("Loaded SentencePiece tokenizer with {} pieces", vocab.len());

        Ok(Self {
            vocab,
            bos_id: 1,
            eos_id: 2,
        })
    }

    pub fn tokenize(&self, text: &str) -> Vec<i64> {
        let mut tokens: Vec<i64> = vec![self.bos_id as i64];

        let normalized = text.replace('\n', " ").trim().to_string();
        // SentencePiece prefix: replace spaces with ▁ and prepend ▁
        let normalized = format!(
            "\u{2581}{}",
            normalized.replace(' ', "\u{2581}")
        );

        let chars: Vec<char> = normalized.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut best_len = 0;
            let mut best_id: Option<i32> = None;

            // Try longest match first (up to 32 chars)
            let max_len = std::cmp::min(chars.len() - i, 32);
            for len in (1..=max_len).rev() {
                let candidate: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.vocab.get(&candidate) {
                    best_len = len;
                    best_id = Some(id);
                    break;
                }
            }

            if let Some(id) = best_id {
                tokens.push(id as i64);
                i += best_len;
            } else {
                // Try single character fallback
                let c = chars[i].to_string();
                if let Some(&id) = self.vocab.get(&c) {
                    tokens.push(id as i64);
                }
                i += 1;
            }
        }

        tokens.push(self.eos_id as i64);
        tokens
    }
}

/// Parse SentencePiece protobuf to extract vocabulary pieces.
fn extract_vocab_from_protobuf(data: &[u8], vocab: &mut HashMap<String, i32>) {
    let mut idx = 0;
    let mut piece_index: i32 = 0;

    while idx < data.len() {
        let Some((tag, tag_size)) = read_varint(data, idx) else {
            break;
        };
        idx += tag_size;

        let field_number = tag >> 3;
        let wire_type = tag & 0x7;

        if field_number == 1 && wire_type == 2 {
            // Length-delimited: piece message
            let Some((msg_len, msg_len_size)) = read_varint(data, idx) else {
                break;
            };
            idx += msg_len_size;

            let msg_end = idx + msg_len as usize;
            if msg_end > data.len() {
                break;
            }

            // Parse inner message to find string piece (field 1)
            let mut piece: Option<String> = None;
            let mut inner = idx;

            while inner < msg_end {
                let Some((inner_tag, inner_tag_size)) = read_varint(data, inner) else {
                    break;
                };
                inner += inner_tag_size;

                let inner_field = inner_tag >> 3;
                let inner_wire = inner_tag & 0x7;

                if inner_field == 1 && inner_wire == 2 {
                    let Some((str_len, str_len_size)) = read_varint(data, inner) else {
                        break;
                    };
                    inner += str_len_size;

                    let str_end = inner + str_len as usize;
                    if str_end <= msg_end {
                        if let Ok(s) = std::str::from_utf8(&data[inner..str_end]) {
                            piece = Some(s.to_string());
                        }
                    }
                    inner = str_end;
                } else if inner_wire == 0 {
                    // Varint — skip
                    if let Some((_, size)) = read_varint(data, inner) {
                        inner += size;
                    } else {
                        break;
                    }
                } else if inner_wire == 2 {
                    if let Some((len, len_size)) = read_varint(data, inner) {
                        inner += len_size + len as usize;
                    } else {
                        break;
                    }
                } else if inner_wire == 5 {
                    inner += 4; // 32-bit fixed
                } else if inner_wire == 1 {
                    inner += 8; // 64-bit fixed
                } else {
                    break;
                }
            }

            if let Some(p) = piece {
                vocab.insert(p, piece_index);
            }
            piece_index += 1;
            idx = msg_end;
        } else if wire_type == 0 {
            if let Some((_, size)) = read_varint(data, idx) {
                idx += size;
            } else {
                break;
            }
        } else if wire_type == 2 {
            if let Some((len, len_size)) = read_varint(data, idx) {
                idx += len_size + len as usize;
            } else {
                break;
            }
        } else if wire_type == 5 {
            idx += 4;
        } else if wire_type == 1 {
            idx += 8;
        } else {
            break;
        }
    }
}

fn read_varint(data: &[u8], mut offset: usize) -> Option<(u64, usize)> {
    let start = offset;
    let mut result: u64 = 0;
    let mut shift = 0;

    while offset < data.len() {
        let b = data[offset];
        offset += 1;
        result |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((result, offset - start));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

// ── HuggingFace Tokenizer (for Qwen3 TTS) ───────────────────────
//
// Uses the `tokenizers` crate (HuggingFace's official Rust implementation)
// to get exact tokenization parity with the Python pipeline.
// Loads tokenizer.json if available, else builds from vocab.json + merges.txt
// with ByteLevel pre-tokenization (matching the Python ONNX trace).

pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        // Try tokenizer.json first (pre-built, most reliable)
        let json_path = model_dir.join("tokenizer.json");
        if json_path.exists() {
            let inner = tokenizers::Tokenizer::from_file(&json_path)
                .map_err(|e| format!("Failed to load tokenizer.json: {}", e))?;
            log::info!("Loaded HF tokenizer from tokenizer.json");
            return Ok(Self { inner });
        }

        // Build from vocab.json + merges.txt (matching Python ONNX trace)
        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");

        if !vocab_path.exists() || !merges_path.exists() {
            return Err(format!(
                "Tokenizer files not found in {}. Need tokenizer.json OR vocab.json + merges.txt",
                model_dir.display()
            ));
        }

        let bpe = tokenizers::models::bpe::BPE::from_file(
            vocab_path.to_str().ok_or("Invalid vocab path encoding")?,
            merges_path.to_str().ok_or("Invalid merges path encoding")?,
        )
        .build()
        .map_err(|e| format!("Failed to build BPE model: {}", e))?;

        let mut inner = tokenizers::Tokenizer::new(bpe);

        // ByteLevel pre-tokenizer with add_prefix_space=false
        // Matches: tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
        inner.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::new(
                false, // add_prefix_space
                true,  // trim_offsets
                true,  // use_regex
            ),
        ));

        log::info!("Built HF tokenizer from vocab.json + merges.txt (ByteLevel BPE)");
        Ok(Self { inner })
    }

    pub fn tokenize(&self, text: &str) -> Vec<i64> {
        match self.inner.encode(text, false) {
            Ok(encoding) => encoding.get_ids().iter().map(|&id| id as i64).collect(),
            Err(e) => {
                log::error!("Tokenization failed: {}", e);
                Vec::new()
            }
        }
    }
}

// ── BPE Tokenizer (for Qwen3 TTS — legacy fallback) ────────────

pub struct BpeTokenizer {
    vocab: HashMap<String, i64>,
    merges: Vec<(String, String)>,
}

impl BpeTokenizer {
    pub fn load(vocab_path: &Path, merges_path: &Path) -> Result<Self, String> {
        // Load vocab.json
        let vocab_str = std::fs::read_to_string(vocab_path)
            .map_err(|e| format!("Failed to read vocab.json: {}", e))?;

        let vocab: HashMap<String, i64> = serde_json::from_str(&vocab_str)
            .map_err(|e| format!("Failed to parse vocab.json: {}", e))?;

        // Load merges.txt
        let merges_str = std::fs::read_to_string(merges_path)
            .map_err(|e| format!("Failed to read merges.txt: {}", e))?;

        let merges: Vec<(String, String)> = merges_str
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with("#version"))
            .filter_map(|line| {
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        log::info!(
            "Loaded BPE tokenizer: {} vocab, {} merges",
            vocab.len(),
            merges.len()
        );

        Ok(Self { vocab, merges })
    }

    /// Direct vocab lookup by key. Returns the token ID if found.
    pub fn vocab_lookup(&self, key: &str) -> Option<i64> {
        self.vocab.get(key).copied()
    }

    /// Encode text with the Qwen3-TTS chat template.
    /// Template: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
    /// Special tokens are hardcoded IDs (not BPE-encoded).
    pub fn encode_chat_prompt(&self, text: &str) -> Vec<i64> {
        const IM_START: i64 = 151644;
        const ASSISTANT: i64 = 77091;
        const IM_END: i64 = 151645;
        const NEWLINE: i64 = 198;

        let text_tokens = self.tokenize(text);

        let mut tokens = Vec::with_capacity(text_tokens.len() + 8);
        // <|im_start|>assistant\n
        tokens.push(IM_START);
        tokens.push(ASSISTANT);
        tokens.push(NEWLINE);
        // {text}
        tokens.extend_from_slice(&text_tokens);
        // <|im_end|>\n<|im_start|>assistant\n
        tokens.push(IM_END);
        tokens.push(NEWLINE);
        tokens.push(IM_START);
        tokens.push(ASSISTANT);
        tokens.push(NEWLINE);

        tokens
    }

    pub fn tokenize(&self, text: &str) -> Vec<i64> {
        let words = self.pre_tokenize(text);

        let mut tokens = Vec::new();
        for word in &words {
            let word_tokens = self.apply_bpe(word);
            for tok in &word_tokens {
                if let Some(&id) = self.vocab.get(tok) {
                    tokens.push(id);
                }
            }
        }

        tokens
    }

    /// GPT-style pre-tokenization: split on whitespace, prefix spaces with \u{0120}
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            if c == ' ' {
                if !current.is_empty() {
                    result.push(current.clone());
                    current.clear();
                }
                current.push('\u{0120}'); // GPT2 space prefix (Ġ)
            } else {
                current.push(c);
            }
        }

        if !current.is_empty() {
            result.push(current);
        }

        result
    }

    fn apply_bpe(&self, word: &str) -> Vec<String> {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        for (first, second) in &self.merges {
            let mut new_symbols = Vec::new();
            let mut i = 0;

            while i < symbols.len() {
                if i < symbols.len() - 1 && symbols[i] == *first && symbols[i + 1] == *second {
                    new_symbols.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_symbols.push(symbols[i].clone());
                    i += 1;
                }
            }

            symbols = new_symbols;
            if symbols.len() == 1 {
                break;
            }
        }

        symbols
    }
}
