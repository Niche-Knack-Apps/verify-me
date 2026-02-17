"""
Voice cloning logic — uses pocket-tts voice cloning via audio prompt.
"""


def clone_voice(reference_audio_path, text, model, output_path):
    """
    Clone a voice from reference audio and generate speech.

    Args:
        reference_audio_path: Path to the reference audio WAV file.
        text: Text to synthesize with the cloned voice.
        model: A loaded PocketTTSModel instance that supports cloning.
        output_path: Where to write the output WAV.

    Returns:
        The output_path of the generated audio.
    """
    return model.clone_from_audio(reference_audio_path, text, output_path)
