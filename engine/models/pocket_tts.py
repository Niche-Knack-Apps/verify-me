"""
Pocket TTS adapter — small, CPU-friendly, bundled model.

Placeholder implementation: generates a silent WAV file until the actual
model is integrated.
"""

import struct
import wave


class PocketTTSModel:
    def __init__(self):
        self._loaded = False

    @staticmethod
    def display_name():
        return "Pocket TTS"

    @staticmethod
    def supports_clone():
        return True

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def generate(self, text, voice="default", speed=1.0, output_path="output.wav", voice_prompt=None):
        """Generate speech audio. Placeholder: writes a silent WAV."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        sample_rate = 22050
        duration_sec = max(0.5, len(text) * 0.06)  # ~60ms per character
        num_samples = int(sample_rate * duration_sec / speed)

        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * num_samples)

        return output_path

    def get_voices(self):
        return [
            {"id": "alba", "name": "Alba (Female, Neutral)", "language": "en"},
            {"id": "cosette", "name": "Cosette (Female, Gentle)", "language": "en"},
            {"id": "fantine", "name": "Fantine (Female, Expressive)", "language": "en"},
            {"id": "eponine", "name": "Eponine (Female, British)", "language": "en"},
            {"id": "azelma", "name": "Azelma (Female, Youthful)", "language": "en"},
            {"id": "jean", "name": "Jean (Male, Warm)", "language": "en"},
            {"id": "marius", "name": "Marius (Male, Casual)", "language": "en"},
            {"id": "javert", "name": "Javert (Male, Authoritative)", "language": "en"},
        ]
