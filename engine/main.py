"""
Verify Me TTS Engine — JSON-RPC 2.0 stdio server.

Reads JSON lines from stdin, writes JSON responses to stdout.
"""

import json
import sys
import os
import signal
import traceback

from tts_engine import TTSEngine
from device_manager import get_device, get_device_info

VERSION = "0.1.0"

engine = TTSEngine()
running = True


def success_response(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def error_response(req_id, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def handle_engine_health(req_id, _params):
    return success_response(req_id, {
        "status": "ok",
        "version": VERSION,
        "device": get_device(),
    })


def handle_engine_shutdown(req_id, _params):
    global running
    running = False
    return success_response(req_id, {"status": "shutting_down"})


def handle_models_list(req_id, _params):
    models = engine.list_models()
    return success_response(req_id, {"models": models})


def handle_models_load(req_id, params):
    model_id = params.get("model_id")
    if not model_id:
        return error_response(req_id, INVALID_PARAMS, "model_id is required")
    try:
        engine.load_model(model_id)
        return success_response(req_id, {"status": "loaded", "model_id": model_id})
    except Exception as e:
        return error_response(req_id, INTERNAL_ERROR, str(e))


def handle_models_unload(req_id, _params):
    try:
        engine.unload_model()
        return success_response(req_id, {"status": "unloaded"})
    except Exception as e:
        return error_response(req_id, INTERNAL_ERROR, str(e))


def handle_tts_generate(req_id, params):
    required = ["text", "model_id", "output_path"]
    for key in required:
        if key not in params:
            return error_response(req_id, INVALID_PARAMS, f"{key} is required")
    try:
        audio_path = engine.generate(
            text=params["text"],
            model_id=params["model_id"],
            voice=params.get("voice", "default"),
            speed=params.get("speed", 1.0),
            output_path=params["output_path"],
        )
        return success_response(req_id, {"audio_path": audio_path})
    except Exception as e:
        return error_response(req_id, INTERNAL_ERROR, str(e))


def handle_tts_voices(req_id, params):
    model_id = params.get("model_id")
    if not model_id:
        return error_response(req_id, INVALID_PARAMS, "model_id is required")
    try:
        voices = engine.get_voices(model_id)
        return success_response(req_id, {"voices": voices})
    except Exception as e:
        return error_response(req_id, INTERNAL_ERROR, str(e))


def handle_voice_clone(req_id, params):
    required = ["text", "reference_audio", "model_id", "output_path"]
    for key in required:
        if key not in params:
            return error_response(req_id, INVALID_PARAMS, f"{key} is required")
    try:
        audio_path = engine.clone_voice(
            text=params["text"],
            reference_audio=params["reference_audio"],
            model_id=params["model_id"],
            output_path=params["output_path"],
        )
        return success_response(req_id, {"audio_path": audio_path})
    except Exception as e:
        return error_response(req_id, INTERNAL_ERROR, str(e))


METHOD_HANDLERS = {
    "engine.health": handle_engine_health,
    "engine.shutdown": handle_engine_shutdown,
    "models.list": handle_models_list,
    "models.load": handle_models_load,
    "models.unload": handle_models_unload,
    "tts.generate": handle_tts_generate,
    "tts.voices": handle_tts_voices,
    "voice.clone": handle_voice_clone,
}


def dispatch(request):
    if not isinstance(request, dict):
        return error_response(None, INVALID_REQUEST, "Request must be a JSON object")

    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if not method or not isinstance(method, str):
        return error_response(req_id, INVALID_REQUEST, "method is required")

    handler = METHOD_HANDLERS.get(method)
    if not handler:
        return error_response(req_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

    return handler(req_id, params)


def write_response(response):
    line = json.dumps(response)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main():
    global running

    # Ensure engine module path is available
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)

    signal.signal(signal.SIGINT, lambda *_: setattr(sys.modules[__name__], "running", False))
    signal.signal(signal.SIGTERM, lambda *_: setattr(sys.modules[__name__], "running", False))

    while running:
        try:
            line = sys.stdin.readline()
            if not line:
                break  # EOF

            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                write_response(error_response(None, PARSE_ERROR, f"Parse error: {e}"))
                continue

            response = dispatch(request)
            write_response(response)

        except Exception as e:
            write_response(error_response(None, INTERNAL_ERROR, f"Internal error: {e}"))

    engine.unload_model()


if __name__ == "__main__":
    # Ensure we can import sibling modules
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)

    main()
