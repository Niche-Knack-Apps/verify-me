"""
TTS orchestration — manages models and dispatches generation requests.
"""

import logging
import os

from models import MODEL_REGISTRY
from voice_clone import clone_voice

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self):
        self._current_model_id = None
        self._current_model = None
        self._models_dir = os.environ.get("VERIFY_ME_MODELS_DIR")
        logger.info("TTSEngine initialized, models_dir=%s", self._models_dir)
        logger.info("Registered models: %s", list(MODEL_REGISTRY.keys()))

    def list_models(self):
        """List all registered models with their status."""
        models = []
        for model_id, model_class in MODEL_REGISTRY.items():
            status = "loaded" if model_id == self._current_model_id else "available"
            models.append({
                "id": model_id,
                "name": model_class.display_name(),
                "status": status,
                "supports_clone": model_class.supports_clone(),
            })
        logger.info("list_models: %d models, loaded=%s", len(models), self._current_model_id)
        return models

    def load_model(self, model_id):
        """Load a model by its registry ID."""
        logger.info("load_model requested: %s (current: %s)", model_id, self._current_model_id)

        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(MODEL_REGISTRY.keys())}")

        if self._current_model_id == model_id:
            logger.info("Model %s already loaded, skipping", model_id)
            return

        self.unload_model()

        model_class = MODEL_REGISTRY[model_id]
        logger.info("Instantiating model class: %s", model_class.__name__)
        self._current_model = model_class()

        logger.info("Loading model %s with models_dir=%s", model_id, self._models_dir)
        self._current_model.load(models_dir=self._models_dir)
        self._current_model_id = model_id
        logger.info("Model %s loaded successfully", model_id)

    def unload_model(self):
        """Unload the currently loaded model."""
        if self._current_model is not None:
            logger.info("Unloading model: %s", self._current_model_id)
            self._current_model.unload()
            self._current_model = None
            self._current_model_id = None
            logger.info("Model unloaded")

    def _ensure_model(self, model_id):
        """Ensure the requested model is loaded."""
        if self._current_model_id != model_id:
            logger.info("Model switch needed: %s -> %s", self._current_model_id, model_id)
            self.load_model(model_id)

    def generate(self, text, model_id, voice="default", speed=1.0, output_path=None,
                 voice_prompt=None, voice_mode=None, voice_description=None):
        """Generate speech audio from text."""
        logger.info(
            "generate() called: model=%s, voice=%s, speed=%s, voice_mode=%s, text_len=%d, output=%s",
            model_id,
            voice,
            speed,
            voice_mode,
            len(text),
            output_path,
        )

        self._ensure_model(model_id)

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "output.wav")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            self._current_model.generate(
                text, voice, speed, output_path,
                voice_prompt=voice_prompt,
                voice_mode=voice_mode,
                voice_description=voice_description,
            )
            logger.info("generate() complete: %s", output_path)
            return output_path
        except Exception:
            logger.exception("generate() failed for model=%s, voice=%s", model_id, voice)
            raise

    def get_voices(self, model_id):
        """Get available voices for a model."""
        logger.info("get_voices() called: model=%s", model_id)
        self._ensure_model(model_id)
        voices = self._current_model.get_voices()
        logger.info("get_voices() returned %d voices", len(voices) if voices else 0)
        return voices

    def clone_voice(self, text, reference_audio, model_id, output_path):
        """Generate speech using a cloned voice from reference audio."""
        logger.info(
            "clone_voice() called: model=%s, ref=%s, text_len=%d, output=%s",
            model_id,
            reference_audio,
            len(text),
            output_path,
        )

        self._ensure_model(model_id)

        if not self._current_model.supports_clone():
            raise ValueError(f"Model {model_id} does not support voice cloning")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            result = clone_voice(reference_audio, text, self._current_model, output_path)
            logger.info("clone_voice() complete: %s", result)
            return result
        except Exception:
            logger.exception("clone_voice() failed for model=%s", model_id)
            raise
