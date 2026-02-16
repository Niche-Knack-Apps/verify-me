from .pocket_tts import PocketTTSModel
from .qwen3_tts import Qwen3TTSModel

MODEL_REGISTRY = {
    "pocket-tts": PocketTTSModel,
    "qwen3-tts": Qwen3TTSModel,
}
