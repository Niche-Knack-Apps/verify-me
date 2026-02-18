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

    def load(self, models_dir=None):
        """Load the Qwen3 TTS CustomVoice model."""
        from qwen_tts import Qwen3TTSModel as QwenModel

        self._model_dir = models_dir
        model_path = self._resolve_model_path(models_dir, "qwen3-tts", HF_REPO_CUSTOM_VOICE)
        kwargs = self._load_kwargs()

        logger.info("Loading CustomVoice model from: %s", model_path)
        self._model = QwenModel.from_pretrained(model_path, **kwargs)

        # Log model metadata
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

    def _ensure_base_model(self):
        """Lazily load the Base model for voice cloning (separate from CustomVoice)."""
        if self._base_model is not None:
            return

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
        """Lazily load the VoiceDesign model for voice-from-description generation."""
        if self._voice_design_model is not None:
            return

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

    def unload(self):
        """Unload models and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("CustomVoice model unloaded")

        if self._base_model is not None:
            del self._base_model
            self._base_model = None
            logger.info("Base model unloaded")

        if self._voice_design_model is not None:
            del self._voice_design_model
            self._voice_design_model = None
            logger.info("VoiceDesign model unloaded")

        self._model_dir = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except ImportError:
            pass

    def generate(self, text, voice="Aiden", speed=1.0, output_path="output.wav",
                 voice_prompt=None, voice_mode=None, voice_description=None):
        """Generate speech using CustomVoice or VoiceDesign model."""
        if voice_mode == "design":
            return self._generate_voice_design(text, speed, output_path, voice_description)
        return self._generate_custom_voice(text, voice, speed, output_path, voice_prompt)

    def _generate_voice_design(self, text, speed, output_path, voice_description):
        """Generate speech using the VoiceDesign model (voice from NL description)."""
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

        logger.info("Calling generate_voice_design(language=None, non_streaming=True)")

        wavs, sample_rate = self._voice_design_model.generate_voice_design(
            text=text,
            language=None,
            non_streaming_mode=True,
            **gen_kwargs,
        )

        logger.info("VoiceDesign complete: %d samples, %d Hz", len(wavs[0]), sample_rate)

        if abs(speed - 1.0) >= 0.05:
            original_len = len(wavs[0])
            wavs[0] = _apply_speed(wavs[0], speed)
            logger.info("Speed %.2fx applied: %d -> %d samples", speed, original_len, len(wavs[0]))

        _write_wav(wavs, sample_rate, output_path)
        return output_path

    def _generate_custom_voice(self, text, voice, speed, output_path, voice_prompt):
        """Generate speech using the CustomVoice model (predefined speaker + instructions)."""
        if not self._model:
            raise RuntimeError("Model not loaded")

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

        logger.info(
            "Calling generate_custom_voice(speaker=%s, language=%s, non_streaming=True, %s)",
            speaker,
            language,
            ", ".join(f"{k}=..." for k in gen_kwargs) if gen_kwargs else "no extra kwargs",
        )

        wavs, sample_rate = self._model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            non_streaming_mode=True,
            **gen_kwargs,
        )

        logger.info("Generation complete: %d samples, %d Hz", len(wavs[0]), sample_rate)

        # Apply speed via resampling (in addition to instruct hint)
        if abs(speed - 1.0) >= 0.05:
            original_len = len(wavs[0])
            wavs[0] = _apply_speed(wavs[0], speed)
            logger.info(
                "Speed %.2fx applied: %d -> %d samples", speed, original_len, len(wavs[0])
            )

        _write_wav(wavs, sample_rate, output_path)
        return output_path

    def clone_from_audio(self, reference_audio, text, output_path):
        """Clone a voice from reference audio using the Base model's generate_voice_clone."""
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

        # Voice cloning requires the Base model variant
        self._ensure_base_model()

        logger.info(
            "Calling generate_voice_clone(language=None, x_vector_only=True, non_streaming=True)"
        )

        wavs, sample_rate = self._base_model.generate_voice_clone(
            text=text,
            language=None,
            ref_audio=reference_audio,
            x_vector_only_mode=True,
            non_streaming_mode=True,
        )

        logger.info("Clone complete: %d samples, %d Hz", len(wavs[0]), sample_rate)
        _write_wav(wavs, sample_rate, output_path)
        return output_path

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
