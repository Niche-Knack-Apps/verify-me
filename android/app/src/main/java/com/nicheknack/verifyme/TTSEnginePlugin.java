package com.nicheknack.verifyme;

import android.util.Log;
import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@CapacitorPlugin(name = "TTSEngine")
public class TTSEnginePlugin extends Plugin {

    private static final String TAG = "TTSEngine";

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private PocketTTSInference pocketTTS = null;
    private Qwen3TTSInference qwen3TTS = null;
    private String currentModelId = null;
    private boolean engineRunning = false;

    @PluginMethod
    public void initialize(PluginCall call) {
        String modelId = call.getString("modelId");

        if (modelId == null || modelId.isEmpty()) {
            call.reject("modelId is required");
            return;
        }

        Log.d(TAG, "initialize: modelId=" + modelId);

        executor.submit(() -> {
            try {
                // Shutdown any existing engine
                shutdownInternal();

                File modelsDir = new File(getContext().getFilesDir(), "models");
                File modelDir = new File(modelsDir, modelId);

                if (!modelDir.isDirectory()) {
                    Log.e(TAG, "Model directory not found: " + modelDir.getAbsolutePath());
                    JSObject ret = new JSObject();
                    ret.put("success", false);
                    ret.put("error", "Model not downloaded: " + modelId);
                    call.resolve(ret);
                    return;
                }

                Log.d(TAG, "Model dir: " + modelDir.getAbsolutePath());

                if (modelId.equals("pocket-tts")) {
                    pocketTTS = new PocketTTSInference();
                    pocketTTS.initialize(modelDir);
                } else if (modelId.startsWith("qwen3-tts")) {
                    qwen3TTS = new Qwen3TTSInference();
                    qwen3TTS.initialize(modelDir);
                } else {
                    JSObject ret = new JSObject();
                    ret.put("success", false);
                    ret.put("error", "Unknown model type: " + modelId);
                    call.resolve(ret);
                    return;
                }

                currentModelId = modelId;
                engineRunning = true;

                Log.d(TAG, "Engine initialized successfully: " + modelId);
                JSObject ret = new JSObject();
                ret.put("success", true);
                ret.put("modelId", modelId);
                call.resolve(ret);

            } catch (Throwable e) {
                Log.e(TAG, "Failed to initialize: " + e.getMessage(), e);
                engineRunning = false;
                JSObject ret = new JSObject();
                ret.put("success", false);
                ret.put("error", "Failed to initialize: " + e.getMessage());
                call.resolve(ret);
            }
        });
    }

    @PluginMethod
    public void generateSpeech(PluginCall call) {
        String text = call.getString("text");
        String voice = call.getString("voice", "alba");
        Float speed = call.getFloat("speed", 1.0f);

        if (text == null || text.trim().isEmpty()) {
            call.reject("text is required");
            return;
        }

        if (!engineRunning) {
            call.reject("Engine not running. Call initialize() first.");
            return;
        }

        Log.d(TAG, "generateSpeech: text='" + text.substring(0, Math.min(text.length(), 50))
            + "' voice=" + voice + " speed=" + speed);

        executor.submit(() -> {
            try {
                File outputDir = new File(getContext().getCacheDir(), "output");
                if (!outputDir.exists()) {
                    outputDir.mkdirs();
                }

                String timestamp = String.valueOf(System.currentTimeMillis());
                File outputFile = new File(outputDir, "tts_" + timestamp + ".wav");

                if (pocketTTS != null && pocketTTS.isInitialized()) {
                    pocketTTS.generateSpeech(text, voice, speed, outputFile);
                } else if (qwen3TTS != null && qwen3TTS.isInitialized()) {
                    qwen3TTS.generateSpeech(text, voice, speed, outputFile);
                } else {
                    Log.e(TAG, "No model loaded");
                    call.reject("No model loaded");
                    return;
                }

                Log.d(TAG, "Speech generated: " + outputFile.getAbsolutePath()
                    + " (" + outputFile.length() + " bytes)");

                JSObject ret = new JSObject();
                ret.put("success", true);
                ret.put("filePath", outputFile.getAbsolutePath());
                call.resolve(ret);

            } catch (Throwable e) {
                Log.e(TAG, "Speech generation failed: " + e.getMessage(), e);
                JSObject ret = new JSObject();
                ret.put("success", false);
                ret.put("error", "Speech generation failed: " + e.getMessage());
                call.resolve(ret);
            }
        });
    }

    @PluginMethod
    public void cloneVoice(PluginCall call) {
        String text = call.getString("text");
        String referenceAudioPath = call.getString("referenceAudioPath");

        if (text == null || text.trim().isEmpty()) {
            call.reject("text is required");
            return;
        }

        if (referenceAudioPath == null || referenceAudioPath.isEmpty()) {
            call.reject("referenceAudioPath is required");
            return;
        }

        if (!engineRunning) {
            call.reject("Engine not running. Call initialize() first.");
            return;
        }

        Log.d(TAG, "cloneVoice: text='" + text.substring(0, Math.min(text.length(), 50))
            + "' ref=" + referenceAudioPath);

        executor.submit(() -> {
            try {
                File refAudio = new File(referenceAudioPath);
                if (!refAudio.exists()) {
                    Log.e(TAG, "Reference audio not found: " + referenceAudioPath);
                    call.reject("Reference audio file not found: " + referenceAudioPath);
                    return;
                }

                File outputDir = new File(getContext().getCacheDir(), "output");
                if (!outputDir.exists()) {
                    outputDir.mkdirs();
                }

                String timestamp = String.valueOf(System.currentTimeMillis());
                File outputFile = new File(outputDir, "clone_" + timestamp + ".wav");

                if (pocketTTS != null && pocketTTS.isInitialized()) {
                    pocketTTS.cloneVoice(text, refAudio, outputFile);
                } else if (qwen3TTS != null && qwen3TTS.isInitialized()) {
                    qwen3TTS.cloneVoice(text, refAudio, outputFile);
                } else {
                    Log.e(TAG, "No model loaded");
                    call.reject("No model loaded");
                    return;
                }

                Log.d(TAG, "Voice cloned: " + outputFile.getAbsolutePath()
                    + " (" + outputFile.length() + " bytes)");

                JSObject ret = new JSObject();
                ret.put("success", true);
                ret.put("filePath", outputFile.getAbsolutePath());
                call.resolve(ret);

            } catch (Throwable e) {
                Log.e(TAG, "Voice cloning failed: " + e.getMessage(), e);
                JSObject ret = new JSObject();
                ret.put("success", false);
                ret.put("error", "Voice cloning failed: " + e.getMessage());
                call.resolve(ret);
            }
        });
    }

    @PluginMethod
    public void getVoices(PluginCall call) {
        String modelId = call.getString("modelId", currentModelId);

        JSArray voices = new JSArray();

        if (pocketTTS != null && pocketTTS.isInitialized() &&
            (modelId == null || modelId.equals("pocket-tts"))) {
            for (String v : pocketTTS.getAvailableVoices()) {
                JSObject voice = new JSObject();
                voice.put("id", v);
                voice.put("name", v.substring(0, 1).toUpperCase() + v.substring(1));
                voices.put(voice);
            }
        } else if (qwen3TTS != null && qwen3TTS.isInitialized()) {
            for (String v : qwen3TTS.getAvailableVoices()) {
                JSObject voice = new JSObject();
                voice.put("id", v);
                voice.put("name", v.replace("_", " "));
                voices.put(voice);
            }
        }

        JSObject ret = new JSObject();
        ret.put("voices", voices);
        call.resolve(ret);
    }

    @PluginMethod
    public void getHealth(PluginCall call) {
        JSObject ret = new JSObject();
        ret.put("engineRunning", engineRunning);
        ret.put("device", "CPU (ONNX)");
        if (currentModelId != null) {
            ret.put("modelId", currentModelId);
        }
        call.resolve(ret);
    }

    @PluginMethod
    public void getDeviceInfo(PluginCall call) {
        JSObject ret = new JSObject();
        ret.put("device", "cpu");
        ret.put("name", "ONNX Runtime Android");
        call.resolve(ret);
    }

    @PluginMethod
    public void shutdown(PluginCall call) {
        Log.d(TAG, "shutdown requested");
        shutdownInternal();

        JSObject ret = new JSObject();
        ret.put("success", true);
        call.resolve(ret);
    }

    private void shutdownInternal() {
        if (pocketTTS != null) {
            pocketTTS.shutdown();
            pocketTTS = null;
        }
        if (qwen3TTS != null) {
            qwen3TTS.shutdown();
            qwen3TTS = null;
        }
        currentModelId = null;
        engineRunning = false;
        Log.d(TAG, "Engine shut down");
    }
}
