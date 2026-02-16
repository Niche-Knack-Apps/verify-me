"""
TTS orchestration — manages models and dispatches generation requests.
"""

import os

from models import MODEL_REGISTRY
from voice_clone import clone_voice


class TTSEngine:
    def __init__(self):
        self._current_model_id = None
        self._current_model = None

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
        return models

    def load_model(self, model_id):
        """Load a model by its registry ID."""
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_id}")

        if self._current_model_id == model_id:
            return  # Already loaded

        self.unload_model()

        model_class = MODEL_REGISTRY[model_id]
        self._current_model = model_class()
        self._current_model.load()
        self._current_model_id = model_id

    def unload_model(self):
        """Unload the currently loaded model."""
        if self._current_model is not None:
            self._current_model.unload()
            self._current_model = None
            self._current_model_id = None

    def _ensure_model(self, model_id):
        """Ensure the requested model is loaded."""
        if self._current_model_id != model_id:
            self.load_model(model_id)

    def generate(self, text, model_id, voice="default", speed=1.0, output_path=None, voice_prompt=None):
        """Generate speech audio from text."""
        self._ensure_model(model_id)

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "output.wav")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        self._current_model.generate(text, voice, speed, output_path, voice_prompt=voice_prompt)
        return output_path

    def get_voices(self, model_id):
        """Get available voices for a model."""
        self._ensure_model(model_id)
        return self._current_model.get_voices()

    def clone_voice(self, text, reference_audio, model_id, output_path):
        """Generate speech using a cloned voice from reference audio."""
        self._ensure_model(model_id)

        if not self._current_model.supports_clone():
            raise ValueError(f"Model {model_id} does not support voice cloning")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        return clone_voice(reference_audio, text, self._current_model, output_path)
