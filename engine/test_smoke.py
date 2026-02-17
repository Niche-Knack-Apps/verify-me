"""
Engine smoke tests — validates the JSON-RPC server without loading models.

Run:  python engine/test_smoke.py
Exit: 0 on pass, 1 on failure
"""

import json
import os
import subprocess
import sys
import time

ENGINE_PATH = os.path.join(os.path.dirname(__file__), "main.py")
PYTHON = sys.executable
TIMEOUT = 10  # seconds per request


def rpc(proc, method, params=None, req_id=1):
    """Send a JSON-RPC request and return the parsed response."""
    request = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        request["params"] = params
    line = json.dumps(request) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()

    # Read one response line
    response_line = proc.stdout.readline()
    if not response_line:
        raise RuntimeError(f"No response for method {method} — engine may have crashed")
    return json.loads(response_line)


def assert_ok(resp, msg=""):
    """Assert that a JSON-RPC response has a result (no error)."""
    if "error" in resp and resp["error"] is not None:
        raise AssertionError(f"RPC error{' (' + msg + ')' if msg else ''}: {resp['error']}")
    if "result" not in resp or resp["result"] is None:
        raise AssertionError(f"Missing result{' (' + msg + ')' if msg else ''}: {resp}")


def assert_error(resp, expected_code=None, msg=""):
    """Assert that a JSON-RPC response is an error."""
    if "error" not in resp or resp["error"] is None:
        raise AssertionError(f"Expected error{' (' + msg + ')' if msg else ''}, got: {resp}")
    if expected_code is not None and resp["error"].get("code") != expected_code:
        raise AssertionError(
            f"Expected error code {expected_code}, got {resp['error'].get('code')}"
        )


def main():
    passed = 0
    failed = 0
    errors = []

    # Start engine process
    proc = subprocess.Popen(
        [PYTHON, ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "VERIFY_ME_FORCE_CPU": "1"},
    )

    # Give it a moment to initialize
    time.sleep(0.5)

    if proc.poll() is not None:
        print(f"FAIL: Engine exited immediately with code {proc.returncode}")
        stderr = proc.stderr.read()
        if stderr:
            print(f"  stderr: {stderr[:500]}")
        sys.exit(1)

    tests = [
        ("engine.health returns ok", test_health),
        ("engine.device_info returns device", test_device_info),
        ("models.list returns list", test_models_list),
        ("unknown method returns error", test_unknown_method),
        ("missing params returns error", test_missing_params),
        ("stdout purity — no non-JSON output", test_stdout_purity),
        ("engine.shutdown graceful", test_shutdown),
    ]

    for name, test_fn in tests:
        try:
            test_fn(proc)
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1
            errors.append((name, str(e)))

    # Clean up
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()

    print(f"\n{passed} passed, {failed} failed")
    if errors:
        for name, err in errors:
            print(f"  - {name}: {err}")
    sys.exit(1 if failed else 0)


def test_health(proc):
    resp = rpc(proc, "engine.health")
    assert_ok(resp, "health")
    result = resp["result"]
    assert result["status"] == "ok", f"Expected status 'ok', got '{result.get('status')}'"
    assert "version" in result, "Missing version in health response"
    assert "device" in result, "Missing device in health response"


def test_device_info(proc):
    resp = rpc(proc, "engine.device_info", req_id=2)
    assert_ok(resp, "device_info")
    result = resp["result"]
    assert "device" in result, "Missing 'device' key"
    assert "name" in result, "Missing 'name' key"


def test_models_list(proc):
    resp = rpc(proc, "models.list", req_id=3)
    assert_ok(resp, "models.list")
    result = resp["result"]
    assert "models" in result, "Missing 'models' key"
    assert isinstance(result["models"], list), "models should be a list"


def test_unknown_method(proc):
    resp = rpc(proc, "nonexistent.method", req_id=4)
    assert_error(resp, expected_code=-32601, msg="unknown method")


def test_missing_params(proc):
    # tts.generate without required params
    resp = rpc(proc, "tts.generate", params={}, req_id=5)
    assert_error(resp, msg="missing params")


def test_stdout_purity(proc):
    """Verify no stray output leaked onto stdout between requests."""
    # Send two rapid requests and check we get exactly two JSON responses
    r1 = rpc(proc, "engine.health", req_id=100)
    r2 = rpc(proc, "engine.health", req_id=101)
    assert_ok(r1, "purity check 1")
    assert_ok(r2, "purity check 2")
    assert r1["id"] == 100, f"Expected id 100, got {r1['id']}"
    assert r2["id"] == 101, f"Expected id 101, got {r2['id']}"


def test_shutdown(proc):
    resp = rpc(proc, "engine.shutdown", req_id=999)
    assert_ok(resp, "shutdown")
    result = resp["result"]
    assert result["status"] == "shutting_down", f"Expected 'shutting_down', got {result}"


if __name__ == "__main__":
    main()
