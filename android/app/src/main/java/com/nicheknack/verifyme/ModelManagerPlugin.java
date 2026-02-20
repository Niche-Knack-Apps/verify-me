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
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPInputStream;

@CapacitorPlugin(name = "ModelManager")
public class ModelManagerPlugin extends Plugin {

    private static final String TAG = "ModelManager";
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean cancelRequested = new AtomicBoolean(false);

    // ── Model catalog ───────────────────────────────────────────
    // pocket-tts: bundled in APK assets, extracted on first launch
    // qwen3-tts-0.6b: downloaded from nicheknack.app

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
            null // no download URL — bundled
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
            "https://nicheknack.app/downloads/verify-me/models/qwen3-tts-0.6b.tar.gz"
        ),
    };

    // ── Plugin methods ──────────────────────────────────────────

    @PluginMethod
    public void listModels(PluginCall call) {
        try {
            File modelsDir = getModelsDir();
            Log.i(TAG, "listModels called, modelsDir=" + modelsDir.getAbsolutePath()
                + " exists=" + modelsDir.exists());

            // Verify asset access is working
            try {
                String[] assetList = getContext().getAssets().list("models");
                Log.i(TAG, "Assets under models/: " +
                    (assetList != null ? String.join(", ", assetList) : "null"));
            } catch (Exception ae) {
                Log.e(TAG, "Failed to list assets: " + ae.getMessage());
            }

            JSArray result = new JSArray();

            for (ModelEntry entry : KNOWN_MODELS) {
                try {
                    JSObject model = new JSObject();
                    model.put("id", entry.id);
                    model.put("name", entry.name);
                    model.put("size", entry.size);
                    model.put("supportsClone", entry.supportsClone);
                    model.put("supportsVoicePrompt", entry.supportsVoicePrompt);
                    model.put("supportsVoiceDesign", entry.supportsVoiceDesign);

                    boolean available = isModelDownloaded(modelsDir, entry);
                    String status;
                    if (available) {
                        status = "available";
                    } else if (entry.bundled) {
                        status = "bundled";
                    } else {
                        status = "downloadable";
                    }
                    model.put("status", status);

                    JSArray voices = new JSArray();
                    for (String[] voice : entry.voices) {
                        JSObject v = new JSObject();
                        v.put("id", voice[0]);
                        v.put("name", voice[1]);
                        voices.put(v);
                    }
                    model.put("voices", voices);

                    if (entry.downloadUrl != null) {
                        model.put("downloadUrl", entry.downloadUrl);
                    }

                    result.put(model);
                    Log.i(TAG, "  model: " + entry.id + " status=" + status
                        + " bundled=" + entry.bundled + " files=" + entry.files.length);
                } catch (Exception me) {
                    Log.e(TAG, "Error building model entry " + entry.id + ": " + me.getMessage());
                }
            }

            Log.i(TAG, "listModels returning " + result.length() + " models");
            JSObject ret = new JSObject();
            ret.put("models", result);
            call.resolve(ret);
        } catch (Exception e) {
            Log.e(TAG, "listModels FAILED: " + e.getMessage(), e);
            call.reject("listModels failed: " + e.getMessage());
        }
    }

    @PluginMethod
    public void extractBundledModels(PluginCall call) {
        executor.submit(() -> {
            int extracted = 0;
            int failed = 0;
            StringBuilder errors = new StringBuilder();

            for (ModelEntry entry : KNOWN_MODELS) {
                if (!entry.bundled) continue;
                if (isModelDownloaded(getModelsDir(), entry)) {
                    Log.i(TAG, "extractBundledModels: " + entry.id + " already available, skipping");
                    continue;
                }
                try {
                    Log.i(TAG, "extractBundledModels: extracting " + entry.id + "...");
                    extractBundledModel(entry);

                    // Verify extraction succeeded
                    if (isModelDownloaded(getModelsDir(), entry)) {
                        extracted++;
                        JSObject data = new JSObject();
                        data.put("modelId", entry.id);
                        data.put("status", "available");
                        notifyListeners("model-extracted", data);
                        Log.i(TAG, "extractBundledModels: " + entry.id + " extracted and verified");
                    } else {
                        failed++;
                        String msg = entry.id + ": extracted but verification failed (files too small — likely Git LFS pointers in APK)";
                        errors.append(msg).append("; ");
                        Log.e(TAG, "extractBundledModels: " + msg);
                    }
                } catch (Exception e) {
                    failed++;
                    errors.append(entry.id).append(": ").append(e.getMessage()).append("; ");
                    Log.e(TAG, "extractBundledModels: failed for " + entry.id, e);
                }
            }

            JSObject ret = new JSObject();
            ret.put("success", failed == 0);
            ret.put("extracted", extracted);
            ret.put("failed", failed);
            if (errors.length() > 0) {
                ret.put("errors", errors.toString());
            }
            call.resolve(ret);
        });
    }

    @PluginMethod
    public void downloadModel(PluginCall call) {
        String modelId = call.getString("modelId");

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

        if (entry.downloadUrl == null) {
            call.reject("No download URL for model: " + modelId);
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

                // Download tar.gz to temp file
                File tempFile = new File(modelsDir, modelId + ".tar.gz.tmp");
                downloadFile(entry.downloadUrl, tempFile, modelId, 0, 1);

                if (cancelRequested.get()) {
                    tempFile.delete();
                    emitProgress(modelId, "cancelled", 0);
                    call.reject("Download cancelled");
                    return;
                }

                // Extract tar.gz
                emitProgress(modelId, "extracting", 95);
                extractTarGz(tempFile, modelsDir);
                tempFile.delete();

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

    private void extractBundledModel(ModelEntry entry) throws IOException {
        AssetManager assets = getContext().getAssets();
        File modelsDir = getModelsDir();
        File modelDir = new File(modelsDir, entry.id);

        if (!modelDir.exists()) {
            boolean created = modelDir.mkdirs();
            Log.i(TAG, "Created model dir: " + modelDir.getAbsolutePath() + " success=" + created);
        }

        String assetBase = "models/" + entry.id;

        // Log available assets for this model
        try {
            String[] assetFiles = assets.list(assetBase);
            Log.i(TAG, "Assets in " + assetBase + ": " +
                (assetFiles != null ? String.join(", ", assetFiles) : "null/empty"));
        } catch (Exception e) {
            Log.e(TAG, "Cannot list assets for " + assetBase + ": " + e.getMessage());
        }

        // Copy ONNX files and tokenizer
        for (String filename : entry.files) {
            File destFile = new File(modelDir, filename);
            if (destFile.exists() && destFile.length() > 1024) {
                Log.i(TAG, "  skip (exists): " + filename + " (" + destFile.length() + " bytes)");
                continue;
            }
            Log.i(TAG, "  extracting: " + assetBase + "/" + filename + " → " + destFile.getAbsolutePath());
            copyAssetFile(assets, assetBase + "/" + filename, destFile);
            long size = destFile.length();
            Log.i(TAG, "  extracted: " + filename + " (" + size + " bytes)");
            if (size < 1024) {
                Log.w(TAG, "  WARNING: " + filename + " is only " + size +
                    " bytes — may be a Git LFS pointer, not a real model file!");
            }
        }

        // Copy embeddings directory (for pocket-tts voices)
        Log.i(TAG, "  extracting embeddings_v2/ ...");
        copyAssetDirectory(assets, assetBase + "/embeddings_v2",
            new File(modelDir, "embeddings_v2"));

        Log.i(TAG, "Extracted bundled model: " + entry.id + " to " + modelDir.getAbsolutePath());
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

    // ── Tar.gz extraction ───────────────────────────────────────

    /**
     * Extract a tar.gz archive to a destination directory.
     * Implements a minimal tar parser — handles regular files and directories.
     */
    private void extractTarGz(File tarGzFile, File destDir) throws IOException {
        try (InputStream fis = new java.io.FileInputStream(tarGzFile);
             BufferedInputStream bis = new BufferedInputStream(fis, 65536);
             GZIPInputStream gzis = new GZIPInputStream(bis, 65536)) {

            byte[] header = new byte[512];

            while (true) {
                int totalRead = 0;
                while (totalRead < 512) {
                    int n = gzis.read(header, totalRead, 512 - totalRead);
                    if (n < 0) {
                        if (totalRead == 0) return; // clean EOF
                        break;
                    }
                    totalRead += n;
                }
                if (totalRead < 512) break;

                // Check for end-of-archive (zero block)
                boolean allZero = true;
                for (byte b : header) {
                    if (b != 0) { allZero = false; break; }
                }
                if (allZero) break;

                // Parse tar header
                String name = parseTarString(header, 0, 100);
                long size = parseTarOctal(header, 124, 12);
                byte typeFlag = header[156];

                // Handle USTAR prefix (bytes 345-500)
                String prefix = parseTarString(header, 345, 155);
                if (!prefix.isEmpty()) {
                    name = prefix + "/" + name;
                }

                // Security: prevent path traversal
                if (name.contains("..")) {
                    Log.w(TAG, "Skipping path with ..: " + name);
                    skipBytes(gzis, roundUp512(size));
                    continue;
                }

                File destFile = new File(destDir, name);

                if (typeFlag == '5' || name.endsWith("/")) {
                    // Directory
                    destFile.mkdirs();
                } else if (typeFlag == '0' || typeFlag == 0) {
                    // Regular file
                    if (destFile.getParentFile() != null) {
                        destFile.getParentFile().mkdirs();
                    }
                    try (FileOutputStream fos = new FileOutputStream(destFile)) {
                        long remaining = size;
                        byte[] buf = new byte[8192];
                        while (remaining > 0) {
                            int toRead = (int) Math.min(buf.length, remaining);
                            int n = gzis.read(buf, 0, toRead);
                            if (n < 0) break;
                            fos.write(buf, 0, n);
                            remaining -= n;
                        }
                    }
                    // Skip padding to 512-byte boundary
                    long padding = roundUp512(size) - size;
                    skipBytes(gzis, padding);
                } else {
                    // Skip unknown entry types (symlinks, etc.)
                    skipBytes(gzis, roundUp512(size));
                }
            }
        }
        Log.i(TAG, "Extracted tar.gz to " + destDir.getAbsolutePath());
    }

    private static String parseTarString(byte[] header, int offset, int length) {
        int end = offset;
        while (end < offset + length && header[end] != 0) {
            end++;
        }
        return new String(header, offset, end - offset).trim();
    }

    private static long parseTarOctal(byte[] header, int offset, int length) {
        String s = parseTarString(header, offset, length).trim();
        if (s.isEmpty()) return 0;
        try {
            return Long.parseLong(s, 8);
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private static long roundUp512(long size) {
        return ((size + 511) / 512) * 512;
    }

    private static void skipBytes(InputStream in, long count) throws IOException {
        byte[] skip = new byte[8192];
        long remaining = count;
        while (remaining > 0) {
            int toRead = (int) Math.min(skip.length, remaining);
            int n = in.read(skip, 0, toRead);
            if (n < 0) break;
            remaining -= n;
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
        if (!modelDir.isDirectory()) {
            Log.d(TAG, "isModelDownloaded(" + entry.id + "): dir not found at " + modelDir.getAbsolutePath());
            return false;
        }

        // Check that all required files exist with reasonable sizes
        for (String filename : entry.files) {
            File f = new File(modelDir, filename);
            if (!f.exists()) {
                Log.d(TAG, "isModelDownloaded(" + entry.id + "): missing " + filename);
                return false;
            }
            if (f.length() < 1024) {
                Log.d(TAG, "isModelDownloaded(" + entry.id + "): " + filename +
                    " too small (" + f.length() + " bytes, probably LFS pointer)");
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

    private void downloadFile(String urlString, File destFile,
                              String modelId, int fileIndex,
                              int totalFiles) throws Exception {
        URL url = new URL(urlString);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(30000);
        conn.setReadTimeout(60000);
        conn.setInstanceFollowRedirects(true);

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
            throw new Exception("HTTP " + responseCode + " for " + urlString);
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
                // Reserve last 10% for extraction
                int overallPercent = (int) (fileProgress * 90);

                emitProgress(modelId, urlString, overallPercent);
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
        final String downloadUrl;

        ModelEntry(String id, String name, String size,
                   boolean supportsClone, boolean supportsVoicePrompt,
                   boolean supportsVoiceDesign, boolean bundled,
                   String[][] voices, String[] files,
                   String downloadUrl) {
            this.id = id;
            this.name = name;
            this.size = size;
            this.supportsClone = supportsClone;
            this.supportsVoicePrompt = supportsVoicePrompt;
            this.supportsVoiceDesign = supportsVoiceDesign;
            this.bundled = bundled;
            this.voices = voices;
            this.files = files;
            this.downloadUrl = downloadUrl;
        }
    }
}
