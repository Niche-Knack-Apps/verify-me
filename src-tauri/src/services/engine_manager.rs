use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Default timeout for JSON-RPC responses from the Python engine.
/// TTS generation can take 30+ seconds on CPU, so we use a generous limit.
const ENGINE_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    #[allow(dead_code)]
    id: u64,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i64,
    message: String,
}

pub struct EngineManager {
    process: Option<Child>,
    stdin: Option<Mutex<ChildStdin>>,
    stdout: Option<Mutex<BufReader<std::process::ChildStdout>>>,
}

impl EngineManager {
    pub fn new() -> Self {
        Self {
            process: None,
            stdin: None,
            stdout: None,
        }
    }

    pub fn is_running(&self) -> bool {
        self.process.is_some()
    }

    pub fn start(
        &mut self,
        python_path: &str,
        engine_path: &str,
        env_vars: Vec<(&str, String)>,
    ) -> Result<(), String> {
        if self.is_running() {
            return Err("Engine is already running".into());
        }

        let mut cmd = Command::new(python_path);
        cmd.arg(engine_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        for (key, val) in &env_vars {
            cmd.env(key, val);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn engine process: {}", e))?;

        // Check for immediate crash (e.g., Python not found, import error)
        std::thread::sleep(Duration::from_millis(100));
        if let Some(ref mut proc) = Some(&mut child) {
            if let Ok(Some(status)) = proc.try_wait() {
                return Err(format!(
                    "Engine exited immediately with status: {}. Check Python installation and dependencies.",
                    status
                ));
            }
        }

        let stdin = child.stdin.take().ok_or("Failed to capture engine stdin")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("Failed to capture engine stdout")?;

        self.stdin = Some(Mutex::new(stdin));
        self.stdout = Some(Mutex::new(BufReader::new(stdout)));
        self.process = Some(child);

        // Verify engine started with a health check
        let health = self.send_request("engine.health", None)?;
        log::info!("Engine started successfully: {:?}", health);

        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), String> {
        if !self.is_running() {
            return Ok(());
        }

        // Try graceful shutdown first
        let _ = self.send_request("engine.shutdown", None);

        // Clean up process
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
            let _ = process.wait();
        }

        self.stdin = None;
        self.stdout = None;

        log::info!("Engine stopped");
        Ok(())
    }

    pub fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value, String> {
        let stdin_lock = self.stdin.as_ref().ok_or("Engine not running")?;
        let stdout_lock = self.stdout.as_ref().ok_or("Engine not running")?;

        let id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);

        let request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        };

        let mut request_json =
            serde_json::to_string(&request).map_err(|e| format!("Serialize error: {}", e))?;
        request_json.push('\n');

        // Write request
        {
            let mut stdin = stdin_lock
                .lock()
                .map_err(|e| format!("Stdin lock error: {}", e))?;
            stdin
                .write_all(request_json.as_bytes())
                .map_err(|e| format!("Failed to write to engine: {}", e))?;
            stdin
                .flush()
                .map_err(|e| format!("Failed to flush engine stdin: {}", e))?;
        }

        // Read response with timeout.
        // We use a scoped thread for the blocking read_line() with a channel
        // + recv_timeout to avoid hanging indefinitely if the engine dies.
        let response_line = {
            let (tx, rx) = std::sync::mpsc::channel();

            std::thread::scope(|s| {
                s.spawn(|| {
                    let Ok(mut stdout) = stdout_lock.lock() else {
                        let _ = tx.send(Err("Stdout lock poisoned".to_string()));
                        return;
                    };
                    let mut line = String::new();
                    match stdout.read_line(&mut line) {
                        Ok(_) => {
                            let _ = tx.send(Ok(line));
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Failed to read from engine: {}", e)));
                        }
                    }
                });

                match rx.recv_timeout(ENGINE_TIMEOUT) {
                    Ok(Ok(line)) => {
                        if line.is_empty() {
                            Err(format!(
                                "Engine closed connection (method: {}). The Python process may have crashed.",
                                method
                            ))
                        } else {
                            Ok(line)
                        }
                    }
                    Ok(Err(e)) => Err(e),
                    Err(_) => {
                        Err(format!(
                            "Engine timed out after {}s (method: {}). The engine may be overloaded or hung.",
                            ENGINE_TIMEOUT.as_secs(),
                            method
                        ))
                    }
                }
            })?
        };

        let response: JsonRpcResponse = serde_json::from_str(&response_line).map_err(|e| {
            format!(
                "Failed to parse engine response: {} (raw: {})",
                e,
                response_line.trim()
            )
        })?;

        if let Some(error) = response.error {
            return Err(format!("Engine error: {}", error.message));
        }

        response
            .result
            .ok_or_else(|| "Empty response from engine".into())
    }

    pub fn health_check(&self) -> Result<Value, String> {
        self.send_request("engine.health", None)
    }
}

impl Drop for EngineManager {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
