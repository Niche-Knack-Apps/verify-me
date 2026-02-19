"""
Qwen 3 TTS adapter — 1.7B parameter CustomVoice model.

Uses the qwen-tts library (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
for high-quality multilingual TTS with predefined speakers and
instruction-based voice control.

Voice cloning requires the separate Base model variant
(Qwen/Qwen3-TTS-12Hz-1.7B-Base) which supports generate_voice_clone().
"""

import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

from checkpoint_logger import emit_checkpoint

logger = logging.getLogger(__name__)

# HuggingFace repo IDs
HF_REPO_CUSTOM_VOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
HF_REPO_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
HF_REPO_VOICE_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Language mapping: the model expects capitalized full names
LANG_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

# Predefined speakers from the CustomVoice model
VOICES = [
    {"id": "Aiden", "name": "Aiden (Male, American English)", "language": "English"},
    {"id": "Ryan", "name": "Ryan (Male, English)", "language": "English"},
    {"id": "Vivian", "name": "Vivian (Female, Chinese)", "language": "Chinese"},
    {"id": "Serena", "name": "Serena (Female, Chinese)", "language": "Chinese"},
    {"id": "Dylan", "name": "Dylan (Male, Chinese)", "language": "Chinese"},
    {"id": "Eric", "name": "Eric (Male, Chinese/Sichuan)", "language": "Chinese"},
    {"id": "Uncle_Fu", "name": "Uncle Fu (Male, Chinese)", "language": "Chinese"},
    {"id": "Ono_Anna", "name": "Ono Anna (Female, Japanese)", "language": "Japanese"},
    {"id": "Sohee", "name": "Sohee (Female, Korean)", "language": "Korean"},
]


def _resolve_language(lang_str):
    """Map language codes (en, zh, etc.) to capitalized names expected by the model.

    Returns None for auto-detection (the model expects Python None, not "auto").
    """
    if lang_str is None:
        return None
    mapped = LANG_MAP.get(lang_str.lower(), lang_str.capitalize())
    logger.debug("Language resolved: '%s' -> '%s'", lang_str, mapped)
    return mapped


def _apply_speed(audio_array, speed):
    """Resample audio to change playback speed. speed>1 = faster."""
    if abs(speed - 1.0) < 0.05:
        return audio_array
    new_length = int(len(audio_array) / speed)
    return resample(audio_array, new_length).astype(audio_array.dtype)


def _speed_instruct(speed):
    """Build a natural-language pace instruction for the model based on speed value."""
    if speed <= 0.7:
        return "Speak very slowly and deliberately. "
    elif speed <= 0.9:
        return "Speak at a slow, relaxed pace. "
    elif speed < 1.1:
        return ""
    elif speed <= 1.3:
        return "Speak at a quick pace. "
    else:
        return "Speak very fast. "


def _write_wav(wavs, sample_rate, output_path):
    """Write model output to 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    audio = wavs[0]
    if audio.dtype in (np.float32, np.float64):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    duration = len(wavs[0]) / sample_rate
    logger.info("Audio saved: %s (%.1fs, %d Hz)", output_path, duration, sample_rate)


class Qwen3TTSModel:
    def __init__(self):
        self._model = None
        self._base_model = None  # Separate Base model for voice cloning
        self._voice_design_model = None  # Separate VoiceDesign model
        self._model_dir = None

    @staticmethod
    def display_name():
        return "Qwen 3 TTS"

    @staticmethod
    def supports_clone():
        return True

    def _resolve_model_path(self, models_dir, subdir, hf_repo):
        """Find a local model path or fall back to HF repo ID."""
        if models_dir:
            candidate = Path(models_dir) / subdir
            if candidate.exists() and any(candidate.iterdir()):
                logger.info("Found local model at: %s", candidate)
                return str(candidate)
        logger.info("No local model for %s, will use HuggingFace: %s", subdir, hf_repo)
        return hf_repo

    def _load_kwargs(self):
        """Build kwargs for from_pretrained based on device."""
        from device_manager import get_device

        device = get_device()
        logger.info("Device for model loading: %s", device)

        kwargs = {}
        if device == "cuda":
            import torch

            kwargs["device_map"] = "cuda:0"
            kwargs["torch_dtype"] = torch.bfloat16
            logger.info(
                "CUDA kwargs: device_map=%s, dtype=%s",
                kwargs["device_map"],
                kwargs["torch_dtype"],
            )
        else:
            logger.info("Loading on CPU (no CUDA kwargs)")

        return kwargs

    def _load_custom_voice(self):
        """Load the CustomVoice model into self._model."""
        from qwen_tts import Qwen3TTSModel as QwenModel

        model_path = self._resolve_model_path(self._model_dir, "qwen3-tts", HF_REPO_CUSTOM_VOICE)
        kwargs = self._load_kwargs()

        from device_manager import get_device
        emit_checkpoint("model_load", {
            "model": "CustomVoice",
            "model_path": str(model_path),
            "device": get_device(),
            "torch_dtype": str(kwargs.get("torch_dtype", "float32")),
        })
        logger.info("Loading CustomVoice model from: %s", model_path)
        self._model = QwenModel.from_pretrained(model_path, **kwargs)

        tts_type = getattr(self._model.model, "tts_model_type", "unknown")
        tok_type = getattr(self._model.model, "tokenizer_type", "unknown")
        logger.info("Model loaded — tts_model_type=%s, tokenizer_type=%s", tts_type, tok_type)

        speakers = self._model.get_supported_speakers()
        if speakers:
            logger.info("Available speakers: %s", ", ".join(str(s) for s in speakers))
        languages = self._model.get_supported_languages()
        if languages:
            logger.info("Supported languages: %s", ", ".join(str(lang) for lang in languages))

        logger.info("Qwen3 TTS CustomVoice ready")

    def load(self, models_dir=None):
        """Load the Qwen3 TTS CustomVoice model."""
        self._model_dir = models_dir
        self._load_custom_voice()

    def _ensure_custom_voice_model(self):
        """Reload the CustomVoice model if it was swapped out for another variant."""
        if self._model is not None:
            return

        # Unload other models first to free memory
        self._unload_model("base")
        self._unload_model("voice_design")

        logger.info("Re-loading CustomVoice model (was swapped out)")
        self._load_custom_voice()

    def _ensure_base_model(self):
        """Load the Base model for voice cloning, unloading others first to save memory."""
        if self._base_model is not None:
            return

        # Unload other models first — each variant is ~1.7B params,
        # keeping multiple loaded simultaneously causes OOM.
        self._unload_model("custom_voice")
        self._unload_model("voice_design")

        from qwen_tts import Qwen3TTSModel as QwenModel

        model_path = self._resolve_model_path(
            self._model_dir, "qwen3-tts-base", HF_REPO_BASE
        )
        kwargs = self._load_kwargs()

        logger.info("Loading Base model for voice cloning from: %s", model_path)
        self._base_model = QwenModel.from_pretrained(model_path, **kwargs)

        tts_type = getattr(self._base_model.model, "tts_model_type", "unknown")
        logger.info("Base model loaded — tts_model_type=%s", tts_type)
        logger.info("Qwen3 TTS Base ready for voice cloning")

    def _ensure_voice_design_model(self):
        """Load the VoiceDesign model, unloading others first to save memory."""
        if self._voice_design_model is not None:
            return

        # Unload other models first — each variant is ~1.7B params,
        # keeping multiple loaded simultaneously causes OOM.
        self._unload_model("custom_voice")
        self._unload_model("base")

        from qwen_tts import Qwen3TTSModel as QwenModel

        model_path = self._resolve_model_path(
            self._model_dir, "qwen3-tts-voice-design", HF_REPO_VOICE_DESIGN
        )
        kwargs = self._load_kwargs()

        logger.info("Loading VoiceDesign model from: %s", model_path)
        self._voice_design_model = QwenModel.from_pretrained(model_path, **kwargs)

        tts_type = getattr(self._voice_design_model.model, "tts_model_type", "unknown")
        logger.info("VoiceDesign model loaded — tts_model_type=%s", tts_type)
        logger.info("Qwen3 TTS VoiceDesign ready")

    def _unload_model(self, which):
        """Unload a specific model variant and free its memory.

        Args:
            which: one of "custom_voice", "base", "voice_design"
        """
        freed = False
        if which == "custom_voice" and self._model is not None:
            del self._model
            self._model = None
            logger.info("CustomVoice model unloaded (swapping)")
            freed = True
        elif which == "base" and self._base_model is not None:
            del self._base_model
            self._base_model = None
            logger.info("Base model unloaded (swapping)")
            freed = True
        elif which == "voice_design" and self._voice_design_model is not None:
            del self._voice_design_model
            self._voice_design_model = None
            logger.info("VoiceDesign model unloaded (swapping)")
            freed = True

        if freed:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared after unload")
            except ImportError:
                pass

    def unload(self):
        """Unload all models and free memory."""
        self._unload_model("custom_voice")
        self._unload_model("base")
        self._unload_model("voice_design")
        self._model_dir = None

    def generate(self, text, voice="Aiden", speed=1.0, output_path="output.wav",
                 voice_prompt=None, voice_mode=None, voice_description=None):
        """Generate speech using CustomVoice or VoiceDesign model."""
        if voice_mode == "design":
            return self._generate_voice_design(text, speed, output_path, voice_description)
        return self._generate_custom_voice(text, voice, speed, output_path, voice_prompt)

    def _generate_voice_design(self, text, speed, output_path, voice_description):
        """Generate speech using the VoiceDesign model (voice from NL description)."""
        import time as _time

        total_start = _time.monotonic()

        emit_checkpoint("embedding", {
            "mode": "voice_design",
            "text": text[:200],
            "text_len": len(text),
            "description": (voice_description[:200] if voice_description else None),
            "speed": speed,
        })

        self._ensure_voice_design_model()

        logger.info(
            "=== TTS Generate (VoiceDesign) ===\n"
            "  text:        '%s' (%d chars)\n"
            "  description: %s\n"
            "  speed:       %s\n"
            "  output:      %s",
            text[:80],
            len(text),
            (voice_description[:80] + "...") if voice_description and len(voice_description) > 80 else voice_description,
            speed,
            output_path,
        )

        # Build instruct: prepend speed hint, then append voice description
        speed_hint = _speed_instruct(speed)
        desc = voice_description.strip() if voice_description and voice_description.strip() else ""
        instruct = (speed_hint + desc).strip() or None

        gen_kwargs = {}
        if instruct:
            gen_kwargs["instruct"] = instruct
            logger.info("VoiceDesign instruction: '%s'", instruct[:100])

        emit_checkpoint("prefill", {
            "mode": "voice_design",
            "instruct": instruct[:200] if instruct else None,
            "gen_kwargs": list(gen_kwargs.keys()),
        })

        logger.info("Calling generate_voice_design(language=None, non_streaming=True)")

        gen_start = _time.monotonic()
        wavs, sample_rate = self._voice_design_model.generate_voice_design(
            text=text,
            language=None,
            non_streaming_mode=True,
            **gen_kwargs,
        )
        gen_elapsed_ms = int((_time.monotonic() - gen_start) * 1000)

        logger.info("VoiceDesign complete: %d samples, %d Hz", len(wavs[0]), sample_rate)

        audio_arr = wavs[0] if isinstance(wavs[0], np.ndarray) else np.array(wavs[0])
        rms = float(np.sqrt(np.mean(audio_arr.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(audio_arr)))

        emit_checkpoint("decode_summary", {
            "mode": "voice_design",
            "num_samples": len(wavs[0]),
            "sample_rate": sample_rate,
            "generation_ms": gen_elapsed_ms,
        })

        if abs(speed - 1.0) >= 0.05:
            original_len = len(wavs[0])
            wavs[0] = _apply_speed(wavs[0], speed)
            logger.info("Speed %.2fx applied: %d -> %d samples", speed, original_len, len(wavs[0]))

        _write_wav(wavs, sample_rate, output_path)

        total_ms = int((_time.monotonic() - total_start) * 1000)
        emit_checkpoint("complete", {
            "mode": "voice_design",
            "num_samples": len(wavs[0]),
            "duration_sec": len(wavs[0]) / sample_rate,
            "sample_rate": sample_rate,
            "rms": rms,
            "peak": peak,
            "total_ms": total_ms,
            "generation_ms": gen_elapsed_ms,
        })

        return output_path

    def _generate_custom_voice(self, text, voice, speed, output_path, voice_prompt):
        """Generate speech using the CustomVoice model (predefined speaker + instructions)."""
        import time as _time

        total_start = _time.monotonic()

        self._ensure_custom_voice_model()

        logger.info(
            "=== TTS Generate (CustomVoice) ===\n"
            "  text:    '%s' (%d chars)\n"
            "  voice:   %s\n"
            "  speed:   %s\n"
            "  instruct: %s\n"
            "  output:  %s",
            text[:80],
            len(text),
            voice,
            speed,
            (voice_prompt[:60] + "...") if voice_prompt and len(voice_prompt) > 60 else voice_prompt,
            output_path,
        )

        # Map voice to speaker name (case-insensitive)
        supported = self._model.get_supported_speakers()
        speaker = voice
        if supported:
            match = next(
                (s for s in supported if str(s).lower() == voice.lower()), None
            )
            if match:
                speaker = match
                logger.info("Speaker matched: %s", speaker)
            else:
                speaker = list(supported)[0] if supported else voice
                logger.warning(
                    "Speaker '%s' not found in %s, falling back to: %s",
                    voice,
                    list(supported),
                    speaker,
                )

        # Detect language from speaker metadata (None = auto-detect)
        voice_info = next(
            (v for v in VOICES if v["id"].lower() == str(speaker).lower()), None
        )
        language = voice_info["language"] if voice_info else None
        logger.info("Language for speaker '%s': %s", speaker, language)

        # Build instruct: prepend speed hint, then append user's voice prompt
        speed_hint = _speed_instruct(speed)
        user_instruct = voice_prompt.strip() if voice_prompt and voice_prompt.strip() else ""
        instruct = (speed_hint + user_instruct).strip()

        gen_kwargs = {}
        if instruct:
            gen_kwargs["instruct"] = instruct
            logger.info("Voice instruction: '%s'", instruct[:80])

        # Checkpoint: embedding stage (matches ONNX "embedding")
        emit_checkpoint("embedding", {
            "mode": "custom_voice",
            "text": text[:200],
            "text_len": len(text),
            "voice": str(speaker),
            "language": language,
            "speed": speed,
        })

        # Checkpoint: prefill stage (matches ONNX "prefill")
        emit_checkpoint("prefill", {
            "mode": "custom_voice",
            "speaker": str(speaker),
            "language": language,
            "instruct": instruct[:200] if instruct else None,
            "gen_kwargs": list(gen_kwargs.keys()),
        })

        logger.info(
            "Calling generate_custom_voice(speaker=%s, language=%s, non_streaming=True, %s)",
            speaker,
            language,
            ", ".join(f"{k}=..." for k in gen_kwargs) if gen_kwargs else "no extra kwargs",
        )

        gen_start = _time.monotonic()
        wavs, sample_rate = self._model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            non_streaming_mode=True,
            **gen_kwargs,
        )
        gen_elapsed_ms = int((_time.monotonic() - gen_start) * 1000)

        logger.info("Generation complete: %d samples, %d Hz", len(wavs[0]), sample_rate)

        # Compute audio stats
        audio_arr = wavs[0] if isinstance(wavs[0], np.ndarray) else np.array(wavs[0])
        rms = float(np.sqrt(np.mean(audio_arr.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(audio_arr)))

        # Checkpoint: decode_summary (matches ONNX "decode_summary")
        emit_checkpoint("decode_summary", {
            "mode": "custom_voice",
            "num_samples": len(wavs[0]),
            "sample_rate": sample_rate,
            "generation_ms": gen_elapsed_ms,
        })

        # Checkpoint: audio_decode (matches ONNX "audio_decode")
        emit_checkpoint("audio_decode", {
            "mode": "custom_voice",
            "num_samples": len(wavs[0]),
            "duration_sec": len(wavs[0]) / sample_rate,
            "sample_rate": sample_rate,
            "rms": rms,
            "peak": peak,
        })

        # Apply speed via resampling (in addition to instruct hint)
        if abs(speed - 1.0) >= 0.05:
            original_len = len(wavs[0])
            wavs[0] = _apply_speed(wavs[0], speed)
            logger.info(
                "Speed %.2fx applied: %d -> %d samples", speed, original_len, len(wavs[0])
            )

        _write_wav(wavs, sample_rate, output_path)

        total_ms = int((_time.monotonic() - total_start) * 1000)

        # Checkpoint: complete (matches ONNX "complete")
        emit_checkpoint("complete", {
            "mode": "custom_voice",
            "num_samples": len(wavs[0]),
            "duration_sec": len(wavs[0]) / sample_rate,
            "sample_rate": sample_rate,
            "rms": rms,
            "peak": peak,
            "total_ms": total_ms,
            "generation_ms": gen_elapsed_ms,
        })

        return output_path

    def clone_from_audio(self, reference_audio, text, output_path):
        """Clone a voice from reference audio using the Base model's generate_voice_clone.

        Reference audio is trimmed to MAX_REF_SECONDS (default 15s) to prevent
        OOM — the model only needs 3-20s of clean speech for good cloning.
        """
        import time as _time

        MAX_REF_SECONDS = 15
        total_start = _time.monotonic()

        logger.info(
            "=== Voice Clone ===\n"
            "  ref_audio: %s\n"
            "  text:      '%s' (%d chars)\n"
            "  output:    %s",
            reference_audio,
            text[:80],
            len(text),
            output_path,
        )

        # Trim reference audio to avoid OOM on long recordings.
        # Qwen3-TTS only needs 3-20s; longer audio wastes memory and can hang.
        trimmed_path = self._trim_reference_audio(reference_audio, MAX_REF_SECONDS)

        # Log ref audio info for checkpoint
        ref_info = sf.info(trimmed_path)
        emit_checkpoint("embedding", {
            "mode": "voice_clone",
            "text": text[:200],
            "text_len": len(text),
            "ref_audio": os.path.basename(reference_audio),
            "ref_duration_sec": ref_info.duration,
            "ref_sample_rate": ref_info.samplerate,
            "trimmed": trimmed_path != reference_audio,
        })

        # Voice cloning requires the Base model variant
        self._ensure_base_model()

        emit_checkpoint("prefill", {
            "mode": "voice_clone",
            "model": "Base",
            "ref_audio_duration": ref_info.duration,
            "x_vector_only": True,
        })

        logger.info(
            "Calling generate_voice_clone(language=None, x_vector_only=True, non_streaming=True)"
        )

        gen_start = _time.monotonic()
        wavs, sample_rate = self._base_model.generate_voice_clone(
            text=text,
            language=None,
            ref_audio=trimmed_path,
            x_vector_only_mode=True,
            non_streaming_mode=True,
        )
        gen_elapsed_ms = int((_time.monotonic() - gen_start) * 1000)

        logger.info("Clone complete: %d samples, %d Hz", len(wavs[0]), sample_rate)

        # Clean up trimmed temp file if we created one
        if trimmed_path != reference_audio:
            try:
                os.remove(trimmed_path)
            except OSError:
                pass

        # Compute audio stats
        audio_arr = wavs[0] if isinstance(wavs[0], np.ndarray) else np.array(wavs[0])
        rms = float(np.sqrt(np.mean(audio_arr.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(audio_arr)))

        emit_checkpoint("decode_summary", {
            "mode": "voice_clone",
            "num_samples": len(wavs[0]),
            "sample_rate": sample_rate,
            "generation_ms": gen_elapsed_ms,
        })

        _write_wav(wavs, sample_rate, output_path)

        total_ms = int((_time.monotonic() - total_start) * 1000)
        emit_checkpoint("complete", {
            "mode": "voice_clone",
            "num_samples": len(wavs[0]),
            "duration_sec": len(wavs[0]) / sample_rate,
            "sample_rate": sample_rate,
            "rms": rms,
            "peak": peak,
            "total_ms": total_ms,
            "generation_ms": gen_elapsed_ms,
        })

        return output_path

    @staticmethod
    def _trim_reference_audio(audio_path, max_seconds):
        """Trim reference audio to max_seconds. Returns original path if already short enough."""
        import soundfile as sf_read

        info = sf_read.info(audio_path)
        duration = info.duration
        logger.info("Reference audio: %.1fs, %d Hz, %d channels", duration, info.samplerate, info.channels)

        if duration <= max_seconds:
            return audio_path

        logger.info("Trimming reference audio from %.1fs to %ds", duration, max_seconds)
        max_frames = int(max_seconds * info.samplerate)
        data, sr = sf_read.read(audio_path, frames=max_frames, dtype="float32")

        # Write trimmed audio to a temp file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf_read.write(tmp.name, data, sr, subtype="PCM_16")
        logger.info("Trimmed reference saved to: %s (%.1fs)", tmp.name, max_seconds)
        return tmp.name

    def get_voices(self):
        """Return available voices (speakers)."""
        if self._model:
            speakers = self._model.get_supported_speakers()
            if speakers:
                result = []
                for s in speakers:
                    info = next(
                        (v for v in VOICES if v["id"].lower() == str(s).lower()),
                        None,
                    )
                    if info:
                        result.append(info)
                    else:
                        result.append({"id": str(s), "name": str(s), "language": None})
                return result
        return VOICES
