"""
Pocket TTS adapter — small, CPU-friendly, bundled model.

Uses the pocket_tts library (kyutai/pocket-tts) for real TTS generation.
Model weights and voice embeddings are loaded from local files.
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from scipy.signal import resample
import yaml

logger = logging.getLogger(__name__)

# Locate the bundled model directory relative to the engine
_ENGINE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS_DIR = _ENGINE_DIR.parent / "src-tauri" / "resources" / "models"

VOICES = [
    {"id": "alba", "name": "Alba (Male, Neutral)", "language": "en"},
    {"id": "cosette", "name": "Cosette (Female, Gentle)", "language": "en"},
    {"id": "fantine", "name": "Fantine (Female, Expressive)", "language": "en"},
    {"id": "eponine", "name": "Eponine (Female, British)", "language": "en"},
    {"id": "azelma", "name": "Azelma (Female, Youthful)", "language": "en"},
    {"id": "jean", "name": "Jean (Male, Warm)", "language": "en"},
    {"id": "marius", "name": "Marius (Male, Casual)", "language": "en"},
    {"id": "javert", "name": "Javert (Male, Authoritative)", "language": "en"},
]


def _find_model_dir(models_dir=None):
    """Resolve the pocket-tts model directory."""
    candidates = []
    if models_dir:
        candidates.append(Path(models_dir) / "pocket-tts")
    candidates.append(_DEFAULT_MODELS_DIR / "pocket-tts")

    for d in candidates:
        weights = d / "tts_b6369a24.safetensors"
        if weights.exists():
            return d

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Pocket TTS model files not found. Searched: {searched}"
    )


def _create_local_config(model_dir):
    """Create a temporary YAML config pointing to local model files."""
    # Read the default config shipped with pocket_tts to get architecture params
    import pocket_tts
    pkg_config = Path(pocket_tts.__file__).parent / "config" / "b6369a24.yaml"
    with open(pkg_config) as f:
        config = yaml.safe_load(f)

    weights_path = str(model_dir / "tts_b6369a24.safetensors")
    tokenizer_path = str(model_dir / "tokenizer.model")

    # Override paths to use local files
    config["weights_path"] = weights_path
    config["weights_path_without_voice_cloning"] = weights_path
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = tokenizer_path

    # Write temporary config
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="pocket_tts_", delete=False
    ) as tmp:
        yaml.dump(config, tmp, default_flow_style=False)
        return tmp.name


class PocketTTSModel:
    def __init__(self):
        self._model = None
        self._voice_states = {}
        self._model_dir = None
        self._config_path = None

    @staticmethod
    def display_name():
        return "Pocket TTS"

    @staticmethod
    def supports_clone():
        return True

    def load(self, models_dir=None):
        """Load the pocket-tts model from local weights."""
        from pocket_tts import TTSModel

        self._model_dir = _find_model_dir(models_dir)
        logger.info("Loading Pocket TTS from %s", self._model_dir)

        self._config_path = _create_local_config(self._model_dir)
        self._model = TTSModel.load_model(config=self._config_path)

        # Pre-load the default voice
        self._preload_voice("alba")
        logger.info("Pocket TTS loaded successfully")

    def unload(self):
        """Unload model and free resources."""
        self._model = None
        self._voice_states = {}
        if self._config_path and os.path.exists(self._config_path):
            os.unlink(self._config_path)
            self._config_path = None
        self._model_dir = None

    def _preload_voice(self, voice_id):
        """Pre-load a voice state from its v2 embedding file."""
        if voice_id in self._voice_states:
            return
        # Use v2 embeddings (compatible with pocket-tts 1.1+)
        emb_path = self._model_dir / "embeddings_v2" / f"{voice_id}.safetensors"
        if emb_path.exists():
            logger.info("Loading voice embedding: %s", voice_id)
            self._voice_states[voice_id] = self._model.get_state_for_audio_prompt(
                str(emb_path)
            )

    def _get_voice_state(self, voice_id):
        """Get a voice state, loading it if needed."""
        if voice_id not in self._voice_states:
            self._preload_voice(voice_id)
        if voice_id not in self._voice_states:
            raise ValueError(
                f"Voice '{voice_id}' not found. Available: "
                + ", ".join(v["id"] for v in VOICES)
            )
        return self._voice_states[voice_id]

    @staticmethod
    def _apply_speed(audio_array, speed):
        """Resample audio to change playback speed. speed>1 = faster."""
        if abs(speed - 1.0) < 0.05:
            return audio_array
        new_length = int(len(audio_array) / speed)
        return resample(audio_array, new_length).astype(audio_array.dtype)

    def _write_pcm16(self, audio_tensor, output_path):
        """Write audio tensor as 16-bit PCM WAV (compatible with all players)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        samples = audio_tensor.numpy()
        # Clamp and convert float32 → int16 PCM
        samples = np.clip(samples, -1.0, 1.0)
        pcm16 = (samples * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, self._model.sample_rate, pcm16)

    def generate(self, text, voice="alba", speed=1.0, output_path="output.wav", voice_prompt=None):
        """Generate speech audio from text."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        voice_state = self._get_voice_state(voice)
        audio = self._model.generate_audio(voice_state, text)

        # Apply speed via resampling
        if abs(speed - 1.0) >= 0.05:
            samples = audio.numpy()
            original_len = len(samples)
            samples = self._apply_speed(samples, speed)
            logger.info("Speed %.2fx applied: %d -> %d samples", speed, original_len, len(samples))
            # Convert back: _write_pcm16 expects tensor-like with .numpy()
            import torch
            audio = torch.from_numpy(samples)

        self._write_pcm16(audio, output_path)
        return output_path

    def clone_from_audio(self, reference_audio, text, output_path):
        """Generate speech using a cloned voice from reference audio."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        voice_state = self._model.get_state_for_audio_prompt(reference_audio)
        audio = self._model.generate_audio(voice_state, text)
        self._write_pcm16(audio, output_path)
        return output_path

    def get_voices(self):
        return VOICES
