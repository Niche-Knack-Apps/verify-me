package com.nicheknack.verifyme;

import android.content.res.AssetManager;
import android.os.StatFs;
import android.util.Log;
import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

@CapacitorPlugin(name = "ModelManager")
public class ModelManagerPlugin extends Plugin {

    private static final String TAG = "ModelManager";
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean cancelRequested = new AtomicBoolean(false);

    // ── Model catalog ───────────────────────────────────────────
    // pocket-tts: bundled in APK assets, extracted on first launch
    // qwen3-tts-0.6b: downloaded from sivasub987/Qwen3-TTS-0.6B-ONNX-INT8

    private static final ModelEntry[] KNOWN_MODELS = {
        new ModelEntry(
            "pocket-tts",
            "Pocket TTS",
            "~240 MB",
            true, false, false,
            true, // bundled in APK assets
            new String[][] {
                {"alba", "Alba (Male, Neutral)"},
                {"cosette", "Cosette (Female, Gentle)"},
                {"fantine", "Fantine (Female, Expressive)"},
                {"eponine", "Eponine (Female, British)"},
                {"azelma", "Azelma (Female, Youthful)"},
                {"jean", "Jean (Male, Warm)"},
                {"marius", "Marius (Male, Casual)"},
                {"javert", "Javert (Male, Authoritative)"},
            },
            new String[] {
                "text_conditioner.onnx",
                "mimi_encoder.onnx",
                "flow_lm_main_int8.onnx",
                "flow_lm_flow_int8.onnx",
                "mimi_decoder_int8.onnx",
                "tokenizer.model",
            },
            null, // no HF repo — bundled
            null  // no HF subdirectory
        ),
        new ModelEntry(
            "qwen3-tts-0.6b",
            "Qwen3 TTS 0.6B",
            "~2.1 GB",
            true, false, false,
            false, // not bundled — downloadable
            new String[][] {
                {"Aiden", "Aiden (Male, American English)"},
                {"Ryan", "Ryan (Male, English)"},
                {"Vivian", "Vivian (Female, Chinese)"},
                {"Serena", "Serena (Female, Chinese)"},
                {"Dylan", "Dylan (Male, Chinese)"},
                {"Eric", "Eric (Male, Chinese/Sichuan)"},
                {"Uncle_Fu", "Uncle Fu (Male, Chinese)"},
                {"Ono_Anna", "Ono Anna (Female, Japanese)"},
                {"Sohee", "Sohee (Female, Korean)"},
            },
            new String[] {
                "codec_embed_q.onnx",
                "speaker_encoder_q.onnx",
                "code_predictor_embed_q.onnx",
                "code_predictor_q.onnx",
                "tokenizer12hz_encode_q.onnx",
                "tokenizer12hz_decode_q.onnx",
                "text_project_q.onnx",
                "talker_decode_q.onnx",
                "talker_prefill_q.onnx",
            },
            "sivasub987/Qwen3-TTS-0.6B-ONNX-INT8",
            null // files at repo root
        ),
    };

    // Tokenizer files for Qwen3 (from zukky repo, different from ONNX model files)
    private static final String[] QWEN3_TOKENIZER_FILES = {
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    };
    private static final String QWEN3_TOKENIZER_REPO = "zukky/Qwen3-TTS-ONNX-DLL";
    private static final String QWEN3_TOKENIZER_SUBDIR = "models/Qwen3-TTS-12Hz-0.6B-Base/";

    // ── Plugin methods ──────────────────────────────────────────

    @PluginMethod
    public void listModels(PluginCall call) {
        // No extraction here — return the catalog instantly
        File modelsDir = getModelsDir();
        JSArray result = new JSArray();

        for (ModelEntry entry : KNOWN_MODELS) {
            JSObject model = new JSObject();
            model.put("id", entry.id);
            model.put("name", entry.name);
            model.put("size", entry.size);
            model.put("supportsClone", entry.supportsClone);
            model.put("supportsVoicePrompt", entry.supportsVoicePrompt);
            model.put("supportsVoiceDesign", entry.supportsVoiceDesign);

            boolean available = isModelDownloaded(modelsDir, entry);
            if (available) {
                model.put("status", "available");
            } else if (entry.bundled) {
                model.put("status", "bundled");
            } else {
                model.put("status", "downloadable");
            }

            JSArray voices = new JSArray();
            for (String[] voice : entry.voices) {
                JSObject v = new JSObject();
                v.put("id", voice[0]);
                v.put("name", voice[1]);
                voices.put(v);
            }
            model.put("voices", voices);

            if (entry.hfRepo != null) {
                model.put("hfRepo", entry.hfRepo);
            }

            result.put(model);
        }

        JSObject ret = new JSObject();
        ret.put("models", result);
        call.resolve(ret);
    }

    @PluginMethod
    public void extractBundledModels(PluginCall call) {
        executor.submit(() -> {
            try {
                for (ModelEntry entry : KNOWN_MODELS) {
                    if (!entry.bundled) continue;
                    if (isModelDownloaded(getModelsDir(), entry)) continue;
                    extractBundledModel(entry);
                    JSObject data = new JSObject();
                    data.put("modelId", entry.id);
                    data.put("status", "available");
                    notifyListeners("model-extracted", data);
                }
                JSObject ret = new JSObject();
                ret.put("success", true);
                call.resolve(ret);
            } catch (Exception e) {
                call.reject("Extraction failed: " + e.getMessage());
            }
        });
    }

    @PluginMethod
    public void downloadModel(PluginCall call) {
        String modelId = call.getString("modelId");
        String hfToken = call.getString("hfToken", null);

        if (modelId == null || modelId.isEmpty()) {
            call.reject("modelId is required");
            return;
        }

        ModelEntry entry = findModel(modelId);
        if (entry == null) {
            call.reject("Unknown model: " + modelId);
            return;
        }

        if (entry.bundled) {
            // Bundled models are extracted from assets, not downloaded
            executor.submit(() -> {
                try {
                    extractBundledModel(entry);
                    JSObject data = new JSObject();
                    data.put("modelId", modelId);
                    data.put("status", "available");
                    notifyListeners("model-extracted", data);
                    JSObject ret = new JSObject();
                    ret.put("success", true);
                    ret.put("modelId", modelId);
                    call.resolve(ret);
                } catch (Exception e) {
                    call.reject("Extraction failed: " + e.getMessage());
                }
            });
            return;
        }

        // Check storage
        File modelsDir = getModelsDir();
        if (!modelsDir.exists()) {
            modelsDir.mkdirs();
        }

        StatFs stat = new StatFs(modelsDir.getAbsolutePath());
        long availableBytes = stat.getAvailableBytes();
        long requiredBytes = estimateModelSize(entry);

        if (availableBytes < requiredBytes) {
            call.reject("Insufficient storage. Need ~" + (requiredBytes / (1024 * 1024)) +
                " MB, available: " + (availableBytes / (1024 * 1024)) + " MB");
            return;
        }

        cancelRequested.set(false);

        executor.submit(() -> {
            try {
                File modelDir = new File(modelsDir, modelId);
                if (!modelDir.exists()) {
                    modelDir.mkdirs();
                }

                // Download ONNX model files
                int totalFiles = entry.files.length;
                for (int i = 0; i < totalFiles; i++) {
                    if (cancelRequested.get()) {
                        emitProgress(modelId, "cancelled", 0);
                        call.reject("Download cancelled");
                        return;
                    }

                    String filename = entry.files[i];
                    File destFile = new File(modelDir, filename);

                    if (destFile.exists() && destFile.length() > 1024) {
                        emitProgress(modelId, filename, ((i + 1) * 100) / totalFiles);
                        continue;
                    }

                    String subdir = entry.hfSubdir != null ? entry.hfSubdir : "";
                    String fileUrl = "https://huggingface.co/" + entry.hfRepo +
                        "/resolve/main/" + subdir + filename;

                    downloadFile(fileUrl, destFile, hfToken, modelId, filename,
                        i, totalFiles);
                }

                // Also download tokenizer files for Qwen3 models
                if (modelId.startsWith("qwen3-tts")) {
                    downloadQwen3Tokenizer(modelDir, hfToken, modelId);
                }

                emitProgress(modelId, "complete", 100);

                JSObject ret = new JSObject();
                ret.put("success", true);
                ret.put("modelId", modelId);
                ret.put("path", modelDir.getAbsolutePath());
                call.resolve(ret);

            } catch (Exception e) {
                call.reject("Download failed: " + e.getMessage());
            }
        });
    }

    @PluginMethod
    public void cancelDownload(PluginCall call) {
        cancelRequested.set(true);
        JSObject ret = new JSObject();
        ret.put("success", true);
        call.resolve(ret);
    }

    @PluginMethod
    public void deleteModel(PluginCall call) {
        String modelId = call.getString("modelId");
        if (modelId == null || modelId.isEmpty()) {
            call.reject("modelId is required");
            return;
        }

        // Don't allow deleting bundled models
        ModelEntry entry = findModel(modelId);
        if (entry != null && entry.bundled) {
            call.reject("Cannot delete bundled model: " + modelId);
            return;
        }

        File modelsDir = getModelsDir();
        File modelDir = new File(modelsDir, modelId);

        if (!modelDir.exists()) {
            call.reject("Model not found: " + modelId);
            return;
        }

        boolean deleted = deleteRecursive(modelDir);

        JSObject ret = new JSObject();
        ret.put("success", deleted);
        if (!deleted) {
            ret.put("error", "Failed to delete some files");
        }
        call.resolve(ret);
    }

    @PluginMethod
    public void getModelsDirectory(PluginCall call) {
        JSObject ret = new JSObject();
        ret.put("path", getModelsDir().getAbsolutePath());
        call.resolve(ret);
    }

    // ── Bundled model extraction ────────────────────────────────

    private void extractBundledModelsIfNeeded() {
        for (ModelEntry entry : KNOWN_MODELS) {
            if (entry.bundled) {
                File modelsDir = getModelsDir();
                if (!isModelDownloaded(modelsDir, entry)) {
                    try {
                        extractBundledModel(entry);
                    } catch (Exception e) {
                        Log.e(TAG, "Failed to extract bundled model " + entry.id, e);
                    }
                }
            }
        }
    }

    private void extractBundledModel(ModelEntry entry) throws IOException {
        AssetManager assets = getContext().getAssets();
        File modelsDir = getModelsDir();
        File modelDir = new File(modelsDir, entry.id);

        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }

        String assetBase = "models/" + entry.id;

        // Copy ONNX files and tokenizer
        for (String filename : entry.files) {
            File destFile = new File(modelDir, filename);
            if (destFile.exists() && destFile.length() > 1024) {
                continue; // Already extracted
            }
            copyAssetFile(assets, assetBase + "/" + filename, destFile);
        }

        // Copy embeddings directory (for pocket-tts voices)
        copyAssetDirectory(assets, assetBase + "/embeddings_v2",
            new File(modelDir, "embeddings_v2"));

        Log.i(TAG, "Extracted bundled model: " + entry.id);
    }

    private void copyAssetFile(AssetManager assets, String assetPath,
                                File destFile) throws IOException {
        try (InputStream in = assets.open(assetPath);
             FileOutputStream out = new FileOutputStream(destFile)) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        }
    }

    private void copyAssetDirectory(AssetManager assets, String assetDir,
                                     File destDir) throws IOException {
        String[] children = assets.list(assetDir);
        if (children == null || children.length == 0) {
            return;
        }

        if (!destDir.exists()) {
            destDir.mkdirs();
        }

        for (String child : children) {
            File destFile = new File(destDir, child);
            if (destFile.exists() && destFile.length() > 1024) {
                continue;
            }

            String[] subChildren = assets.list(assetDir + "/" + child);
            if (subChildren != null && subChildren.length > 0) {
                // Subdirectory — recurse
                copyAssetDirectory(assets, assetDir + "/" + child, destFile);
            } else {
                // File — copy
                copyAssetFile(assets, assetDir + "/" + child, destFile);
            }
        }
    }

    // ── Qwen3 tokenizer download ────────────────────────────────

    private void downloadQwen3Tokenizer(File modelDir, String hfToken,
                                         String modelId) throws Exception {
        for (String filename : QWEN3_TOKENIZER_FILES) {
            File destFile = new File(modelDir, filename);
            if (destFile.exists() && destFile.length() > 100) {
                continue;
            }

            String fileUrl = "https://huggingface.co/" + QWEN3_TOKENIZER_REPO +
                "/resolve/main/" + QWEN3_TOKENIZER_SUBDIR + filename;

            downloadFile(fileUrl, destFile, hfToken, modelId, filename, 0, 1);
        }
    }

    // ── Helpers ──────────────────────────────────────────────────

    private File getModelsDir() {
        return new File(getContext().getFilesDir(), "models");
    }

    private ModelEntry findModel(String modelId) {
        for (ModelEntry e : KNOWN_MODELS) {
            if (e.id.equals(modelId)) {
                return e;
            }
        }
        return null;
    }

    private boolean isModelDownloaded(File modelsDir, ModelEntry entry) {
        File modelDir = new File(modelsDir, entry.id);
        if (!modelDir.isDirectory()) return false;

        // Check that all required ONNX files exist
        for (String filename : entry.files) {
            File f = new File(modelDir, filename);
            if (!f.exists() || f.length() < 1024) {
                return false;
            }
        }
        return true;
    }

    private long estimateModelSize(ModelEntry entry) {
        String size = entry.size.replaceAll("[^0-9.]", "");
        try {
            double value = Double.parseDouble(size);
            if (entry.size.contains("GB")) {
                return (long) (value * 1024 * 1024 * 1024);
            }
            return (long) (value * 1024 * 1024);
        } catch (NumberFormatException e) {
            return 500L * 1024 * 1024;
        }
    }

    private void downloadFile(String urlString, File destFile, String hfToken,
                              String modelId, String filename,
                              int fileIndex, int totalFiles) throws Exception {
        URL url = new URL(urlString);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(30000);
        conn.setReadTimeout(60000);
        conn.setInstanceFollowRedirects(true);

        if (hfToken != null && !hfToken.isEmpty()) {
            conn.setRequestProperty("Authorization", "Bearer " + hfToken);
        }

        // Resume support
        long existingBytes = 0;
        if (destFile.exists()) {
            existingBytes = destFile.length();
            conn.setRequestProperty("Range", "bytes=" + existingBytes + "-");
        }

        int responseCode = conn.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_MOVED_TEMP ||
            responseCode == HttpURLConnection.HTTP_MOVED_PERM ||
            responseCode == 307 || responseCode == 308) {
            String redirect = conn.getHeaderField("Location");
            conn.disconnect();
            conn = (HttpURLConnection) new URL(redirect).openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(60000);
            if (existingBytes > 0) {
                conn.setRequestProperty("Range", "bytes=" + existingBytes + "-");
            }
            responseCode = conn.getResponseCode();
        }

        if (responseCode != HttpURLConnection.HTTP_OK &&
            responseCode != HttpURLConnection.HTTP_PARTIAL) {
            conn.disconnect();
            throw new Exception("HTTP " + responseCode + " for " + filename);
        }

        long totalBytes = conn.getContentLengthLong() + existingBytes;
        boolean append = responseCode == HttpURLConnection.HTTP_PARTIAL;

        try (InputStream in = conn.getInputStream();
             FileOutputStream out = new FileOutputStream(destFile, append)) {

            byte[] buffer = new byte[8192];
            long downloaded = existingBytes;
            int bytesRead;

            while ((bytesRead = in.read(buffer)) != -1) {
                if (cancelRequested.get()) {
                    throw new Exception("Download cancelled");
                }

                out.write(buffer, 0, bytesRead);
                downloaded += bytesRead;

                double fileProgress = totalBytes > 0 ?
                    (double) downloaded / totalBytes : 0;
                int overallPercent = (int) (((fileIndex + fileProgress) / totalFiles) * 100);

                emitProgress(modelId, filename, overallPercent);
            }
        } finally {
            conn.disconnect();
        }
    }

    private void emitProgress(String modelId, String filename, int percent) {
        JSObject data = new JSObject();
        data.put("modelId", modelId);
        data.put("filename", filename);
        data.put("percent", percent);
        notifyListeners("model-download-progress", data);
    }

    private boolean deleteRecursive(File fileOrDirectory) {
        if (fileOrDirectory.isDirectory()) {
            File[] children = fileOrDirectory.listFiles();
            if (children != null) {
                for (File child : children) {
                    deleteRecursive(child);
                }
            }
        }
        return fileOrDirectory.delete();
    }

    // ── Model catalog entry ─────────────────────────────────────

    private static class ModelEntry {
        final String id;
        final String name;
        final String size;
        final boolean supportsClone;
        final boolean supportsVoicePrompt;
        final boolean supportsVoiceDesign;
        final boolean bundled;
        final String[][] voices;
        final String[] files;
        final String hfRepo;
        final String hfSubdir;

        ModelEntry(String id, String name, String size,
                   boolean supportsClone, boolean supportsVoicePrompt,
                   boolean supportsVoiceDesign, boolean bundled,
                   String[][] voices, String[] files,
                   String hfRepo, String hfSubdir) {
            this.id = id;
            this.name = name;
            this.size = size;
            this.supportsClone = supportsClone;
            this.supportsVoicePrompt = supportsVoicePrompt;
            this.supportsVoiceDesign = supportsVoiceDesign;
            this.bundled = bundled;
            this.voices = voices;
            this.files = files;
            this.hfRepo = hfRepo;
            this.hfSubdir = hfSubdir;
        }
    }
}
