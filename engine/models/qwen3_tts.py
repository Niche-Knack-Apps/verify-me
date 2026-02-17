"""
Qwen 3 TTS adapter — 1.7B parameter CustomVoice model.

Uses the qwen-tts library (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
for high-quality multilingual TTS with predefined speakers and
instruction-based voice control.
"""

import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# HuggingFace repo ID for model download
HF_REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Predefined speakers from the CustomVoice model
VOICES = [
    {"id": "Aiden", "name": "Aiden (Male, American English)", "language": "en"},
    {"id": "Ryan", "name": "Ryan (Male, English)", "language": "en"},
    {"id": "Vivian", "name": "Vivian (Female, Chinese)", "language": "zh"},
    {"id": "Serena", "name": "Serena (Female, Chinese)", "language": "zh"},
    {"id": "Dylan", "name": "Dylan (Male, Chinese)", "language": "zh"},
    {"id": "Eric", "name": "Eric (Male, Chinese/Sichuan)", "language": "zh"},
    {"id": "Uncle_Fu", "name": "Uncle Fu (Male, Chinese)", "language": "zh"},
    {"id": "Ono_Anna", "name": "Ono Anna (Female, Japanese)", "language": "ja"},
    {"id": "Sohee", "name": "Sohee (Female, Korean)", "language": "ko"},
]


class Qwen3TTSModel:
    def __init__(self):
        self._model = None
        self._model_dir = None

    @staticmethod
    def display_name():
        return "Qwen 3 TTS"

    @staticmethod
    def supports_clone():
        return False  # Use pocket-tts for voice cloning

    def load(self, models_dir=None):
        """Load the Qwen3 TTS CustomVoice model."""
        from qwen_tts import Qwen3TTSModel as QwenModel
        from device_manager import get_device

        # Look for locally downloaded model first
        model_path = None
        if models_dir:
            candidate = Path(models_dir) / "qwen3-tts"
            if candidate.exists() and any(candidate.iterdir()):
                model_path = str(candidate)
                self._model_dir = candidate

        if not model_path:
            # Fall back to HF repo ID (downloads to HF cache)
            model_path = HF_REPO_ID
            logger.info("No local model found, loading from HuggingFace: %s", model_path)
        else:
            logger.info("Loading Qwen3 TTS from local path: %s", model_path)

        device = get_device()
        logger.info("Using device: %s", device)

        kwargs = {}
        if device == "cuda":
            import torch
            kwargs["device_map"] = "cuda:0"
            kwargs["torch_dtype"] = torch.bfloat16

        self._model = QwenModel.from_pretrained(model_path, **kwargs)

        # Log available speakers and languages
        speakers = self._model.get_supported_speakers()
        if speakers:
            logger.info("Available speakers: %s", ", ".join(speakers))
        languages = self._model.get_supported_languages()
        if languages:
            logger.info("Supported languages: %s", ", ".join(languages))

        logger.info("Qwen3 TTS loaded successfully")

    def unload(self):
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_dir = None

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        logger.info("Qwen3 TTS unloaded")

    def generate(self, text, voice="Aiden", speed=1.0, output_path="output.wav", voice_prompt=None):
        """Generate speech using the CustomVoice model.

        Args:
            text: Text to synthesize.
            voice: Speaker name (e.g. 'Aiden', 'Ryan', 'Serena').
            speed: Speech speed multiplier (currently unused by Qwen3).
            output_path: Where to write the output WAV file.
            voice_prompt: Optional instruction for voice style,
                          e.g. "speak warmly and gently".
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        logger.info(
            "Generating speech: voice=%s, instruct=%s, text='%s...'",
            voice,
            voice_prompt[:50] if voice_prompt else None,
            text[:50],
        )

        # Map voice to speaker name (case-insensitive)
        supported = self._model.get_supported_speakers()
        speaker = voice
        if supported:
            match = next(
                (s for s in supported if s.lower() == voice.lower()), None
            )
            if match:
                speaker = match
            else:
                logger.warning(
                    "Speaker '%s' not found, using first available: %s",
                    voice,
                    supported[0],
                )
                speaker = supported[0]

        # Detect language from speaker metadata
        voice_info = next(
            (v for v in VOICES if v["id"].lower() == speaker.lower()), None
        )
        language = voice_info["language"] if voice_info else "en"

        gen_kwargs = {}
        if voice_prompt and voice_prompt.strip():
            gen_kwargs["instruct"] = voice_prompt.strip()
            logger.info("Using voice instruction: '%s'", voice_prompt.strip()[:80])

        wavs, sample_rate = self._model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            non_streaming_mode=True,
            **gen_kwargs,
        )

        # Write WAV output as 16-bit PCM for compatibility
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        audio = wavs[0]
        if audio.dtype in (np.float32, np.float64):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")

        duration = len(wavs[0]) / sample_rate
        logger.info("Audio saved: %s (%.1fs, %dHz)", output_path, duration, sample_rate)
        return output_path

    def clone_from_audio(self, reference_audio, text, output_path):
        """Voice cloning is not supported by CustomVoice model."""
        raise NotImplementedError(
            "Qwen3 TTS CustomVoice does not support voice cloning. "
            "Use Pocket TTS for voice cloning instead."
        )

    def get_voices(self):
        """Return available voices (speakers)."""
        if self._model:
            speakers = self._model.get_supported_speakers()
            if speakers:
                result = []
                for s in speakers:
                    info = next(
                        (v for v in VOICES if v["id"].lower() == s.lower()),
                        None,
                    )
                    if info:
                        result.append(info)
                    else:
                        result.append({"id": s, "name": s, "language": "en"})
                return result
        return VOICES
