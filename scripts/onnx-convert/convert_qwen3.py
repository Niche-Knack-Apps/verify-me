#!/usr/bin/env python3
"""
Qwen3 TTS ONNX Conversion Script
==================================

Exports each submodule of the Qwen3-TTS-12Hz model family to individual ONNX files
for use in ONNX Runtime inference (desktop, mobile, edge).

Architecture (Qwen3TTSForConditionalGeneration):
  - talker (Qwen3TTSTalkerForConditionalGeneration):
      - model.text_embedding    -> text embeddings lookup
      - model.codec_embedding   -> codec token embeddings lookup
      - text_projection         -> ResizeMLP: text_hidden -> talker_hidden
      - model (Qwen3TTSTalkerModel) -> transformer decoder (28 layers)
      - codec_head              -> project hidden to codec vocab logits
      - code_predictor          -> sub-talker for multi-codebook prediction
          - model.codec_embedding (ModuleList) -> per-group embeddings
          - small_to_mtp_projection -> projects talker hidden to predictor hidden
          - model (transformer, 5 layers)
          - lm_head (ModuleList)  -> per-group logit heads
  - speaker_encoder (Base variant only) -> ECAPA-TDNN x-vector encoder
  - speech_tokenizer (shared) -> 12Hz encoder/decoder for audio <-> codes

Output ONNX files per variant:
  text_project.onnx              -- text_embedding + text_projection (text IDs -> hidden)
  talker_prefill.onnx            -- full talker model forward for KV-cache prefill
  talker_decode.onnx             -- single-step talker decode with KV-cache
  code_predictor_prefill.onnx    -- code predictor prefill with KV-cache output
  code_predictor_decode.onnx     -- code predictor single-step decode with KV-cache
  code_predictor_embed.onnx      -- code predictor input embeddings (codec_embedding list)
  codec_embed.onnx               -- main codec_embedding (feedback into talker)
  codec_head.onnx                -- project hidden to codec vocab logits
  tokenizer12hz_encode.onnx      -- speech tokenizer encoder
  tokenizer12hz_decode.onnx      -- speech tokenizer decoder
  speaker_encoder.onnx           -- ECAPA-TDNN speaker encoder (Base variant only)

Usage:
  python convert_qwen3.py --model-dir /path/to/qwen3-tts/ --variant customvoice
  python convert_qwen3.py --model-dir /path/to/qwen3-tts-base/ --variant base --quantize
  python convert_qwen3.py --variant all  # converts all three from default model dir
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert_qwen3")

# -- Project paths ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = Path.home() / ".local/share/com.niche-knack.verify-me/models"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "resources" / "onnx-models"

VARIANT_MAP = {
    "customvoice": "qwen3-tts",
    "base": "qwen3-tts-base",
    "voicedesign": "qwen3-tts-voice-design",
}

# -- Model dimensions (from config.json) ----------------------------------------

# Talker: hidden_size=2048, text_hidden_size=2048, text_vocab_size=151936
# Talker: num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128
# Talker: vocab_size=3072 (codec), num_code_groups=16
# Code predictor: hidden_size=1024, num_hidden_layers=5, vocab_size=2048
# Speech tokenizer: 12Hz, input/output 24kHz, encode_downsample_rate=1920

TALKER_HIDDEN = 2048
TALKER_TEXT_HIDDEN = 2048
TEXT_VOCAB_SIZE = 151936
CODEC_VOCAB_SIZE = 3072
NUM_CODE_GROUPS = 16
NUM_TALKER_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128

CODE_PRED_HIDDEN = 1024
CODE_PRED_VOCAB = 2048
CODE_PRED_LAYERS = 5

_DEFAULT_OPSET = 17
opset_version = _DEFAULT_OPSET


# ================================================================================
#  Wrapper modules for clean ONNX export
# ================================================================================

class TextProjectModule(nn.Module):
    """Wraps text_embedding + text_projection for a single ONNX export.

    Input:  text_ids (int64) [batch, seq_len]
    Output: projected (float32) [batch, seq_len, talker_hidden]
    """

    def __init__(self, text_embedding: nn.Embedding, text_projection: nn.Module):
        super().__init__()
        self.text_embedding = text_embedding
        self.text_projection = text_projection

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.text_projection(self.text_embedding(text_ids))


class CodecEmbedModule(nn.Module):
    """Wraps the main codec_embedding for feedback into the talker.

    Input:  codec_ids (int64) [batch, seq_len]
    Output: embedded (float32) [batch, seq_len, talker_hidden]
    """

    def __init__(self, codec_embedding: nn.Embedding):
        super().__init__()
        self.codec_embedding = codec_embedding

    def forward(self, codec_ids: torch.Tensor) -> torch.Tensor:
        return self.codec_embedding(codec_ids)


class CodePredictorEmbedModule(nn.Module):
    """Wraps the code predictor's per-group embeddings (ModuleList of nn.Embedding).

    The code predictor has (num_code_groups - 1) embeddings, one for each group
    after the first (the first group uses the main codec_embedding from the talker).

    Input:  group_idx (int64 [1]), token_ids (int64 [batch, seq])
    Output: embedded (float32) [batch, seq, embed_dim]

    For ONNX, we concatenate all embedding weight matrices into a single
    [num_groups * vocab, dim] parameter and use arithmetic indexing to select
    the correct group. This avoids the torch.stack + dynamic index pattern
    which causes ONNX trace to fold away unused groups' weights.
    """

    def __init__(self, codec_embeddings: nn.ModuleList):
        super().__init__()
        num_groups = len(codec_embeddings)
        vocab_size = codec_embeddings[0].weight.shape[0]
        embed_dim = codec_embeddings[0].weight.shape[1]

        # Concatenate all embedding weights into a single [num_groups * vocab, dim] matrix
        weights = torch.cat([emb.weight.data for emb in codec_embeddings], dim=0)
        self.all_weights = nn.Parameter(weights)
        self.num_groups = num_groups
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, group_idx: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass selecting the embedding for the given group index.

        Args:
            group_idx: [1] int64 tensor, which embedding group (0 to num_groups-1)
            token_ids: [batch, seq] int64 token IDs
        Returns:
            embedded: [batch, seq, embed_dim]
        """
        # Offset token IDs by group: actual_idx = group_idx * vocab_size + token_ids
        offset = group_idx[0] * self.vocab_size
        offset_ids = token_ids + offset
        # Single embedding lookup from concatenated weight matrix
        return nn.functional.embedding(offset_ids, self.all_weights)


class CodePredictorPrefillModule(nn.Module):
    """Code predictor forward: inputs_embeds + position_ids + attention_mask → logits.

    Used in repeated-prefill mode: run for each autoregressive step with a
    growing input sequence. No KV cache needed since the code predictor is
    tiny (5 layers, max 16 tokens).

    Explicit position_ids and 4D attention_mask bypass all internal shape
    computations (causal mask creation, position derivation) that produce
    hardcoded Reshape/Expand ops in TorchScript tracing.

    Input:  inputs_embeds [batch, seq_len, talker_hidden],
            position_ids [batch, seq_len],
            attention_mask [batch, 1, seq_len, seq_len] (4D causal mask: 0=attend, -inf=mask)
    Output: all_logits [batch, seq_len, 15 * vocab]
    """

    def __init__(self, code_predictor, num_layers):
        super().__init__()
        self.model = code_predictor.model
        self.lm_heads = code_predictor.lm_head
        self.small_to_mtp_projection = code_predictor.small_to_mtp_projection
        self.num_layers = num_layers

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        projected = self.small_to_mtp_projection(inputs_embeds)
        outputs = self.model(
            inputs_embeds=projected,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state
        all_logits = torch.cat([head(hidden) for head in self.lm_heads], dim=-1)
        return all_logits


class CodePredictorDecodeModule(nn.Module):
    """Code predictor single-step decode with KV cache.

    Input:  inputs_embeds [batch, 1, talker_hidden], position_ids [batch, 1],
            past_key_values [layers, 2, batch, heads, prev_len, dim]
    Output: all_logits [batch, 1, 15 * vocab],
            past_key_values_out [layers, 2, batch, heads, prev_len+1, dim]
    """

    def __init__(self, code_predictor, num_layers):
        super().__init__()
        self.model = code_predictor.model
        self.lm_heads = code_predictor.lm_head
        self.small_to_mtp_projection = code_predictor.small_to_mtp_projection
        self.num_layers = num_layers

    def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor,
                past_key_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers import DynamicCache
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_key_values[i, 0], past_key_values[i, 1], i)

        projected = self.small_to_mtp_projection(inputs_embeds)
        outputs = self.model(
            inputs_embeds=projected,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state
        all_logits = torch.cat([head(hidden) for head in self.lm_heads], dim=-1)

        new_kv = outputs.past_key_values
        if hasattr(new_kv, 'to_legacy_cache'):
            new_kv = new_kv.to_legacy_cache()
        kv_flat = torch.stack([torch.stack([k, v]) for k, v in new_kv])

        return all_logits, kv_flat


class SpeakerEncoderModule(nn.Module):
    """Wraps the ECAPA-TDNN speaker encoder for ONNX export.

    Input:  mel_spectrogram [batch, time, 128]
    Output: speaker_embedding [batch, enc_dim]
    """

    def __init__(self, speaker_encoder):
        super().__init__()
        self.encoder = speaker_encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)


# ================================================================================
#  Export helpers
# ================================================================================

def export_onnx(
    module: nn.Module,
    dummy_inputs: tuple | dict,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None = None,
    opset: int | None = None,
    skip_existing: bool = False,
) -> None:
    """Export a PyTorch module to ONNX format."""
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("  SKIP: %s already exists (%.1f MB)", output_path.name, size_mb)
        return

    if opset is None:
        opset = opset_version

    module.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("  Exporting %s ...", output_path.name)

    with torch.no_grad():
        if isinstance(dummy_inputs, dict):
            torch.onnx.export(
                module,
                (),
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes or {},
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )
        else:
            torch.onnx.export(
                module,
                dummy_inputs,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes or {},
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )

    # Verify the exported model
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model, full_check=True)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("  OK: %s (%.1f MB)", output_path.name, size_mb)


def fix_hardcoded_reshape(onnx_path: Path, traced_seq_len: int) -> int:
    """Fix hardcoded Reshape shapes caused by TorchScript tracing.

    When torch.onnx.export traces a model with a concrete seq_len,
    operations like `arange(seq_len).view(-1, 1)` produce Reshape nodes
    with constant target shapes (e.g., [2, 1] for seq_len=2). These
    fail at runtime when seq_len differs.

    This function finds such patterns (Range → Reshape with [traced_len, 1])
    and replaces the target shape with [-1, 1] so the first dimension is
    inferred dynamically.

    Returns the number of Reshape nodes fixed.
    """
    model = onnx.load(str(onnx_path))
    fixed = 0

    # Build maps for both initializers and Constant nodes
    init_map = {init.name: init for init in model.graph.initializer}
    const_nodes = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for out in node.output:
                const_nodes[out] = node

    # Find Range → Reshape patterns with hardcoded shapes
    range_outputs = set()
    for node in model.graph.node:
        if node.op_type == "Range":
            range_outputs.update(node.output)

    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        data_input, shape_input = node.input[0], node.input[1]

        # Only fix Reshape nodes that consume Range output
        if data_input not in range_outputs:
            continue

        # Try initializer first
        if shape_input in init_map:
            init = init_map[shape_input]
            shape_val = np.frombuffer(init.raw_data, dtype=np.int64).copy()

            if len(shape_val) == 2 and shape_val[0] == traced_seq_len and shape_val[1] == 1:
                logger.info(
                    "  Fix Reshape %s (initializer): [%d, 1] → [-1, 1]",
                    node.name, traced_seq_len,
                )
                shape_val[0] = -1
                init.raw_data = shape_val.tobytes()
                fixed += 1

        # Try Constant node
        elif shape_input in const_nodes:
            const_node = const_nodes[shape_input]
            for attr in const_node.attribute:
                if attr.name == "value":
                    shape_val = np.frombuffer(attr.t.raw_data, dtype=np.int64).copy()
                    if len(shape_val) == 2 and shape_val[0] == traced_seq_len and shape_val[1] == 1:
                        logger.info(
                            "  Fix Reshape %s (Constant): [%d, 1] → [-1, 1]",
                            node.name, traced_seq_len,
                        )
                        shape_val[0] = -1
                        attr.t.raw_data = shape_val.tobytes()
                        fixed += 1
                    break

    if fixed > 0:
        onnx.save(model, str(onnx_path))

    return fixed


def quantize_model(input_path: Path, output_path: Path) -> None:
    """Apply INT8 dynamic quantization (MatMul only) to an ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    logger.info("  Quantizing %s -> %s ...", input_path.name, output_path.name)

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        per_channel=True,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("  OK: %s (%.1f MB)", output_path.name, size_mb)


# ================================================================================
#  Main conversion logic
# ================================================================================

def load_qwen3_model(model_dir: Path):
    """Load the Qwen3 TTS model from a local directory using transformers.

    Returns the Qwen3TTSForConditionalGeneration instance (CPU, float32).
    """
    logger.info("Loading Qwen3 TTS model from: %s", model_dir)

    # Register the custom model classes
    from transformers import AutoConfig, AutoModel

    # Import the model classes from qwen_tts
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSConfig,
        Qwen3TTSTalkerConfig,
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSSpeakerEncoderConfig,
    )
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    config = Qwen3TTSConfig.from_pretrained(str(model_dir))
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        str(model_dir),
        config=config,
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    variant = config.tts_model_type
    logger.info("Model loaded: variant=%s, size=%s", variant, config.tts_model_size)

    return model, config


def load_speech_tokenizer(model_dir: Path):
    """Load the 12Hz speech tokenizer from the speech_tokenizer/ subdirectory.

    Returns the Qwen3TTSTokenizerV2Model instance (CPU, float32).
    """
    tokenizer_dir = model_dir / "speech_tokenizer"
    if not tokenizer_dir.exists():
        logger.warning("speech_tokenizer/ not found in %s, skipping", model_dir)
        return None

    logger.info("Loading speech tokenizer from: %s", tokenizer_dir)

    from transformers import AutoConfig, AutoModel
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

    tokenizer_model = AutoModel.from_pretrained(str(tokenizer_dir), torch_dtype=torch.float32)
    tokenizer_model.eval()

    logger.info("Speech tokenizer loaded: type=%s", tokenizer_model.config.model_type)
    return tokenizer_model


def convert_variant(
    model_dir: Path,
    output_dir: Path,
    variant: str,
    do_quantize: bool = False,
    skip_existing: bool = False,
) -> None:
    """Convert a single Qwen3 TTS variant to ONNX files."""

    logger.info("=" * 70)
    logger.info("Converting variant: %s", variant)
    logger.info("  Model dir:  %s", model_dir)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  Quantize:   %s", do_quantize)
    logger.info("  Skip exist: %s", skip_existing)
    logger.info("=" * 70)

    if not model_dir.exists():
        logger.error("Model directory does not exist: %s", model_dir)
        sys.exit(1)

    # Load the main model
    model, config = load_qwen3_model(model_dir)
    talker = model.talker  # Qwen3TTSTalkerForConditionalGeneration

    output_dir.mkdir(parents=True, exist_ok=True)

    _skip = skip_existing  # local alias for export calls

    # ---- 1. text_project.onnx ----
    # Combines text_embedding + text_projection
    logger.info("[1/9] text_project.onnx")
    text_proj = TextProjectModule(
        talker.model.text_embedding,
        talker.text_projection,
    )
    text_proj.eval()

    dummy_text_ids = torch.randint(0, TEXT_VOCAB_SIZE, (1, 16), dtype=torch.int64)
    export_onnx(
        text_proj,
        (dummy_text_ids,),
        output_dir / "text_project.onnx",
        input_names=["text_ids"],
        output_names=["projected"],
        dynamic_axes={
            "text_ids": {0: "batch", 1: "seq_len"},
            "projected": {0: "batch", 1: "seq_len"},
        },
        skip_existing=_skip,
    )

    # ---- 2. codec_embed.onnx ----
    # Main codec embedding (talker.model.codec_embedding)
    logger.info("[2/9] codec_embed.onnx")
    codec_embed = CodecEmbedModule(talker.model.codec_embedding)
    codec_embed.eval()

    dummy_codec_ids = torch.randint(0, CODEC_VOCAB_SIZE, (1, 8), dtype=torch.int64)
    export_onnx(
        codec_embed,
        (dummy_codec_ids,),
        output_dir / "codec_embed.onnx",
        input_names=["codec_ids"],
        output_names=["embedded"],
        dynamic_axes={
            "codec_ids": {0: "batch", 1: "seq_len"},
            "embedded": {0: "batch", 1: "seq_len"},
        },
        skip_existing=_skip,
    )

    # ---- 3. talker_prefill.onnx ----
    # Full talker model forward for KV-cache prefill
    # Inputs: inputs_embeds, position_ids (MRoPE: [3, batch, seq])
    # Outputs: logits, hidden_states, past_key_values (flattened KV cache)
    logger.info("[3/9] talker_prefill.onnx")

    class TalkerPrefillModule(nn.Module):
        """Talker forward pass for prefill with KV cache output."""
        def __init__(self, talker_model, codec_head, num_layers):
            super().__init__()
            self.model = talker_model
            self.codec_head = codec_head
            self.num_layers = num_layers

        def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=False,
            )
            hidden = outputs.last_hidden_state
            logits = self.codec_head(hidden)

            # Flatten KV cache: tuple of (key, value) -> [num_layers, 2, batch, heads, seq, dim]
            past_kv = outputs.past_key_values
            if hasattr(past_kv, 'to_legacy_cache'):
                past_kv = past_kv.to_legacy_cache()
            kv_flat = torch.stack([torch.stack([k, v]) for k, v in past_kv])

            return logits, hidden, kv_flat

    prefill_mod = TalkerPrefillModule(talker.model, talker.codec_head, NUM_TALKER_LAYERS)
    prefill_mod.eval()

    seq_len = 32
    dummy_embeds = torch.randn(1, seq_len, TALKER_HIDDEN)
    # MRoPE position_ids: [3, 1, seq_len] — all 3 dims identical for TTS
    dummy_pos = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).unsqueeze(0).expand(3, 1, seq_len)

    export_onnx(
        prefill_mod,
        (dummy_embeds, dummy_pos),
        output_dir / "talker_prefill.onnx",
        input_names=["inputs_embeds", "position_ids"],
        output_names=["logits", "hidden_states", "past_key_values"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "position_ids": {2: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "seq_len"},
            "past_key_values": {4: "seq_len"},
        },
        skip_existing=_skip,
    )

    # ---- 4. talker_decode.onnx ----
    # Single-step decode with KV cache input/output
    # Inputs: inputs_embeds, position_ids (MRoPE: [3, 1, 1]), past_key_values
    # Outputs: logits, hidden_states, past_key_values (updated)
    logger.info("[4/9] talker_decode.onnx")

    class TalkerDecodeModule(nn.Module):
        """Single-step talker decode with KV cache."""
        def __init__(self, talker_model, codec_head, num_layers):
            super().__init__()
            self.model = talker_model
            self.codec_head = codec_head
            self.num_layers = num_layers

        def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor,
                    past_key_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Unflatten KV cache: [num_layers, 2, batch, heads, seq, dim] -> tuple of (key, value)
            past_kv = []
            for i in range(self.num_layers):
                k = past_key_values[i, 0]  # [batch, heads, seq, dim]
                v = past_key_values[i, 1]
                past_kv.append((k, v))
            past_kv = tuple(past_kv)

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=False,
            )
            hidden = outputs.last_hidden_state
            logits = self.codec_head(hidden)

            # Re-flatten updated KV cache
            new_kv = outputs.past_key_values
            if hasattr(new_kv, 'to_legacy_cache'):
                new_kv = new_kv.to_legacy_cache()
            kv_flat = torch.stack([torch.stack([k, v]) for k, v in new_kv])

            return logits, hidden, kv_flat

    decode_mod = TalkerDecodeModule(talker.model, talker.codec_head, NUM_TALKER_LAYERS)
    decode_mod.eval()

    dummy_step_embeds = torch.randn(1, 1, TALKER_HIDDEN)
    dummy_step_pos = torch.tensor([[[seq_len]], [[seq_len]], [[seq_len]]], dtype=torch.int64)  # [3, 1, 1]
    # Dummy KV cache from prefill: [28, 2, 1, 8, seq_len, 128]
    dummy_kv = torch.randn(NUM_TALKER_LAYERS, 2, 1, NUM_KV_HEADS, seq_len, HEAD_DIM)

    export_onnx(
        decode_mod,
        (dummy_step_embeds, dummy_step_pos, dummy_kv),
        output_dir / "talker_decode.onnx",
        input_names=["inputs_embeds", "position_ids", "past_key_values"],
        output_names=["logits", "hidden_states", "past_key_values_out"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "position_ids": {2: "seq_len"},
            "past_key_values": {4: "past_seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "seq_len"},
            "past_key_values_out": {4: "total_seq_len"},
        },
        skip_existing=_skip,
    )

    # ---- 5a. code_predictor_prefill.onnx ----
    # Autoregressive code predictor: prefill with KV cache output
    logger.info("[5a/10] code_predictor_prefill.onnx")
    cp_prefill = CodePredictorPrefillModule(talker.code_predictor, CODE_PRED_LAYERS)
    cp_prefill.eval()

    cp_prefill_seq = 2
    dummy_pred_input = torch.randn(1, cp_prefill_seq, TALKER_HIDDEN)
    dummy_pred_pos = torch.arange(cp_prefill_seq, dtype=torch.int64).unsqueeze(0)
    # 4D causal mask: [batch, 1, seq_len, seq_len], 0.0=attend, -inf=mask
    min_dtype = torch.finfo(torch.float32).min
    dummy_attn_mask = torch.full((1, 1, cp_prefill_seq, cp_prefill_seq), 0.0)
    dummy_attn_mask += torch.triu(
        torch.full((cp_prefill_seq, cp_prefill_seq), min_dtype), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    cp_prefill_path = output_dir / "code_predictor_prefill.onnx"
    export_onnx(
        cp_prefill,
        (dummy_pred_input, dummy_pred_pos, dummy_attn_mask),
        cp_prefill_path,
        input_names=["inputs_embeds", "position_ids", "attention_mask"],
        output_names=["all_logits"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "position_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {2: "q_len", 3: "kv_len"},
            "all_logits": {0: "batch", 1: "seq_len"},
        },
        skip_existing=_skip,
    )
    # Fix any remaining hardcoded Reshape shapes from tracing
    if cp_prefill_path.exists():
        n = fix_hardcoded_reshape(cp_prefill_path, cp_prefill_seq)
        if n:
            logger.info("  Fixed %d hardcoded Reshape node(s)", n)

    # ---- 5b. code_predictor_decode.onnx ----
    # Autoregressive code predictor: single-step decode with KV cache
    logger.info("[5b/10] code_predictor_decode.onnx")
    cp_decode = CodePredictorDecodeModule(talker.code_predictor, CODE_PRED_LAYERS)
    cp_decode.eval()

    cp_prefill_seq = 2
    dummy_cp_step = torch.randn(1, 1, TALKER_HIDDEN)
    dummy_cp_pos = torch.tensor([[cp_prefill_seq]], dtype=torch.int64)
    dummy_cp_kv = torch.randn(CODE_PRED_LAYERS, 2, 1, NUM_KV_HEADS, cp_prefill_seq, HEAD_DIM)
    export_onnx(
        cp_decode,
        (dummy_cp_step, dummy_cp_pos, dummy_cp_kv),
        output_dir / "code_predictor_decode.onnx",
        input_names=["inputs_embeds", "position_ids", "past_key_values"],
        output_names=["all_logits", "past_key_values_out"],
        dynamic_axes={
            "inputs_embeds": {0: "batch"},
            "past_key_values": {4: "past_seq_len"},
            "all_logits": {0: "batch"},
            "past_key_values_out": {4: "total_seq_len"},
        },
        skip_existing=_skip,
    )

    # ---- 6. code_predictor_embed.onnx ----
    # Code predictor per-group embeddings
    logger.info("[6/10] code_predictor_embed.onnx")
    cp_embed = CodePredictorEmbedModule(talker.code_predictor.model.codec_embedding)
    cp_embed.eval()

    dummy_group_idx = torch.tensor([0], dtype=torch.int64)  # [1] shape, not scalar
    dummy_tok_ids = torch.randint(0, CODE_PRED_VOCAB, (1, 1), dtype=torch.int64)
    export_onnx(
        cp_embed,
        (dummy_group_idx, dummy_tok_ids),
        output_dir / "code_predictor_embed.onnx",
        input_names=["group_idx", "input_ids"],
        output_names=["embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "embeds": {0: "batch", 1: "seq_len"},
        },
        skip_existing=_skip,
    )

    # ---- 7 & 8. Speech tokenizer ONNX ----
    speech_tokenizer = load_speech_tokenizer(model_dir)
    if speech_tokenizer is not None:
        # 7. tokenizer12hz_encode.onnx
        logger.info("[7/10] tokenizer12hz_encode.onnx")

        class TokenizerEncodeModule(nn.Module):
            """Wraps speech tokenizer encoder for ONNX."""
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, input_values: torch.Tensor) -> torch.Tensor:
                # encoder.encode expects [batch, 1, samples]
                output = self.encoder.encode(input_values, return_dict=True)
                return output.audio_codes  # [num_quantizers, batch, codes_len]

        enc_mod = TokenizerEncodeModule(speech_tokenizer.encoder)
        enc_mod.eval()

        # Dummy: 1 second of audio at 24kHz
        dummy_audio = torch.randn(1, 1, 24000)
        export_onnx(
            enc_mod,
            (dummy_audio,),
            output_dir / "tokenizer12hz_encode.onnx",
            input_names=["input_values"],
            output_names=["audio_codes"],
            dynamic_axes={
                "input_values": {0: "batch", 2: "samples"},
                "audio_codes": {1: "batch", 2: "codes_len"},
            },
            skip_existing=_skip,
        )

        # 8. tokenizer12hz_decode.onnx
        logger.info("[8/10] tokenizer12hz_decode.onnx")

        class TokenizerDecodeModule(nn.Module):
            """Wraps speech tokenizer decoder for ONNX."""
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
                # decoder expects [batch, num_quantizers, codes_len]
                return self.decoder.decode(audio_codes)  # [batch, 1, samples]

        dec_mod = TokenizerDecodeModule(speech_tokenizer.decoder)
        dec_mod.eval()

        # Dummy: 13 code frames (1 second at 12.5Hz), 16 quantizers
        dummy_codes = torch.randint(0, 2048, (1, 16, 13), dtype=torch.int64)
        export_onnx(
            dec_mod,
            (dummy_codes,),
            output_dir / "tokenizer12hz_decode.onnx",
            input_names=["audio_codes"],
            output_names=["audio_values"],
            dynamic_axes={
                "audio_codes": {0: "batch", 2: "codes_len"},
                "audio_values": {0: "batch", 2: "samples"},
            },
            skip_existing=_skip,
        )
    else:
        logger.warning("Skipping speech tokenizer export (not found)")

    # ---- 9. speaker_encoder.onnx (Base variant only) ----
    if model.speaker_encoder is not None:
        logger.info("[9/10] speaker_encoder.onnx")
        spk_enc = SpeakerEncoderModule(model.speaker_encoder)
        spk_enc.eval()

        # Dummy: ~3 seconds of mel spectrogram (128 bins, 24kHz, hop=256 -> ~281 frames)
        dummy_mel = torch.randn(1, 281, 128)
        export_onnx(
            spk_enc,
            (dummy_mel,),
            output_dir / "speaker_encoder.onnx",
            input_names=["mel_spectrogram"],
            output_names=["speaker_embedding"],
            dynamic_axes={
                "mel_spectrogram": {0: "batch", 1: "time"},
                "speaker_embedding": {0: "batch"},
            },
            skip_existing=_skip,
        )
    else:
        logger.info("[9/10] speaker_encoder.onnx -- SKIPPED (not present in %s variant)", variant)

    # ---- Copy tokenizer files ----
    logger.info("Copying tokenizer files...")
    for fname in ["vocab.json", "merges.txt", "tokenizer_config.json", "tokenizer.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            logger.info("  Copied: %s", fname)

    # Generate tokenizer.json if not already present (for Rust `tokenizers` crate)
    if not (output_dir / "tokenizer.json").exists():
        vocab_path = output_dir / "vocab.json"
        merges_path = output_dir / "merges.txt"
        if vocab_path.exists() and merges_path.exists():
            try:
                from tokenizers import Tokenizer, models, pre_tokenizers
                tok = Tokenizer(models.BPE.from_file(str(vocab_path), str(merges_path)))
                tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
                tok.save(str(output_dir / "tokenizer.json"))
                logger.info("  Generated: tokenizer.json (from vocab.json + merges.txt)")
            except Exception as e:
                logger.warning("  Could not generate tokenizer.json: %s", e)

    # Also copy config.json and generation_config.json for reference
    for fname in ["config.json", "generation_config.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            logger.info("  Copied: %s", fname)

    # ---- Quantization ----
    if do_quantize:
        logger.info("Applying INT8 dynamic quantization...")
        onnx_files = list(output_dir.glob("*.onnx"))
        for onnx_path in onnx_files:
            q_name = onnx_path.stem + "_q.onnx"
            q_path = output_dir / q_name
            try:
                quantize_model(onnx_path, q_path)
            except Exception as e:
                logger.warning("  Quantization failed for %s: %s", onnx_path.name, e)

    # ---- Summary ----
    logger.info("")
    logger.info("Conversion complete for variant: %s", variant)
    logger.info("Output directory: %s", output_dir)

    total_size = 0
    for f in sorted(output_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        logger.info("  %-40s %8.1f MB", f.name, size_mb)
    logger.info("  %-40s %8.1f MB", "TOTAL", total_size)

    # Free memory
    del model
    del talker
    if speech_tokenizer is not None:
        del speech_tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ================================================================================
#  CLI
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3 TTS model(s) to ONNX format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert CustomVoice variant:
  python convert_qwen3.py --variant customvoice

  # Convert Base variant with quantization:
  python convert_qwen3.py --variant base --quantize

  # Convert all three variants:
  python convert_qwen3.py --variant all --quantize

  # Custom model and output directories:
  python convert_qwen3.py \\
    --model-dir /path/to/qwen3-tts/ \\
    --output-dir /path/to/output/ \\
    --variant customvoice
""",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Path to the Qwen3 TTS model directory. "
            f"Default: {DEFAULT_MODELS_DIR}/qwen3-tts[-base|-voice-design]/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for ONNX files. "
            f"Default: {DEFAULT_OUTPUT_BASE}/qwen3-tts-<variant>/"
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["customvoice", "base", "voicedesign", "all"],
        default="customvoice",
        help="Which model variant to convert. 'all' converts all three. Default: customvoice",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Produce INT8 quantized variants (_q suffix) via dynamic quantization on MatMul ops.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip export of ONNX files that already exist in the output directory. Useful for resuming after OOM crashes.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=_DEFAULT_OPSET,
        help=f"ONNX opset version. Default: {_DEFAULT_OPSET}",
    )

    args = parser.parse_args()

    global opset_version
    opset_version = args.opset

    if args.variant == "all":
        variants = ["customvoice", "base", "voicedesign"]
    else:
        variants = [args.variant]

    for variant in variants:
        subdir = VARIANT_MAP[variant]

        if args.model_dir is not None:
            model_dir = args.model_dir
        else:
            model_dir = DEFAULT_MODELS_DIR / subdir

        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            output_dir = DEFAULT_OUTPUT_BASE / f"qwen3-tts-{variant}"

        convert_variant(
            model_dir=model_dir,
            output_dir=output_dir,
            variant=variant,
            do_quantize=args.quantize,
            skip_existing=args.skip_existing,
        )

    logger.info("")
    logger.info("All conversions complete.")


if __name__ == "__main__":
    main()
