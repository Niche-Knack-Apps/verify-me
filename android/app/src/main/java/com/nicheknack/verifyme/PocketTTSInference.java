package com.nicheknack.verifyme;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ONNX inference engine for pocket-tts model.
 *
 * Pipeline: tokenize → text_conditioner → load voice KV-cache →
 *           flow_lm_main autoregressive loop → flow_lm_flow → mimi_decoder → WAV
 */
public class PocketTTSInference {

    private static final int SAMPLE_RATE = 24000;
    private static final int MAX_STEPS = 2048;
    private static final int LATENT_DIM = 128;
    private static final int NUM_CODEBOOKS = 8;

    private OrtEnvironment env;
    private OrtSession textConditioner;
    private OrtSession mimiEncoder;
    private OrtSession flowLmMain;
    private OrtSession flowLmFlow;
    private OrtSession mimiDecoder;

    private File modelDir;
    private boolean initialized = false;

    // SentencePiece vocab (loaded from tokenizer.model protobuf)
    private Map<String, Integer> vocab;
    private int bosId = 1;
    private int eosId = 2;

    public PocketTTSInference() {
        this.env = OrtEnvironment.getEnvironment();
    }

    public void initialize(File modelDir) throws Exception {
        this.modelDir = modelDir;

        SessionOptions opts = new SessionOptions();
        opts.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
        opts.setIntraOpNumThreads(4);

        textConditioner = env.createSession(
            new File(modelDir, "text_conditioner.onnx").getAbsolutePath(), opts);
        mimiEncoder = env.createSession(
            new File(modelDir, "mimi_encoder.onnx").getAbsolutePath(), opts);
        flowLmMain = env.createSession(
            new File(modelDir, "flow_lm_main_int8.onnx").getAbsolutePath(), opts);
        flowLmFlow = env.createSession(
            new File(modelDir, "flow_lm_flow_int8.onnx").getAbsolutePath(), opts);
        mimiDecoder = env.createSession(
            new File(modelDir, "mimi_decoder_int8.onnx").getAbsolutePath(), opts);

        loadTokenizer(new File(modelDir, "tokenizer.model"));
        initialized = true;
    }

    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Generate speech from text using a preset voice.
     *
     * @param text      Input text to synthesize
     * @param voiceId   Voice embedding ID (e.g., "alba", "cosette")
     * @param speed     Speed factor (1.0 = normal)
     * @param outputFile Output WAV file
     */
    public void generateSpeech(String text, String voiceId, float speed,
                                File outputFile) throws Exception {
        if (!initialized) {
            throw new IllegalStateException("Model not initialized");
        }

        // 1. Tokenize text
        long[] tokenIds = tokenize(text);

        // 2. Run text conditioner
        float[][][] textEmbeddings = runTextConditioner(tokenIds);

        // 3. Load voice embedding from safetensors
        Map<String, float[]> voiceKvCache = loadVoiceEmbedding(voiceId);

        // 4. Autoregressive generation loop
        float[][] latents = autoregressiveGenerate(textEmbeddings, voiceKvCache);

        // 5. Decode latents to audio via mimi_decoder
        float[] audio = runMimiDecoder(latents);

        // 6. Apply speed adjustment
        if (Math.abs(speed - 1.0f) > 0.01f) {
            audio = resampleAudio(audio, speed);
        }

        // 7. Write WAV
        writeWav(audio, SAMPLE_RATE, outputFile);
    }

    /**
     * Generate speech using a reference audio for voice cloning.
     */
    public void cloneVoice(String text, File referenceAudio,
                           File outputFile) throws Exception {
        if (!initialized) {
            throw new IllegalStateException("Model not initialized");
        }

        // 1. Tokenize text
        long[] tokenIds = tokenize(text);

        // 2. Run text conditioner
        float[][][] textEmbeddings = runTextConditioner(tokenIds);

        // 3. Encode reference audio through mimi_encoder to get voice conditioning
        float[] refAudioSamples = loadAudioSamples(referenceAudio);
        Map<String, float[]> voiceKvCache = encodeReferenceAudio(refAudioSamples);

        // 4. Autoregressive generation loop
        float[][] latents = autoregressiveGenerate(textEmbeddings, voiceKvCache);

        // 5. Decode latents to audio
        float[] audio = runMimiDecoder(latents);

        // 6. Write WAV
        writeWav(audio, SAMPLE_RATE, outputFile);
    }

    public List<String> getAvailableVoices() {
        List<String> voices = new ArrayList<>();
        File embeddingsDir = new File(modelDir, "embeddings_v2");
        if (embeddingsDir.isDirectory()) {
            File[] files = embeddingsDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    String name = f.getName();
                    if (name.endsWith(".safetensors")) {
                        voices.add(name.replace(".safetensors", ""));
                    }
                }
            }
        }
        Collections.sort(voices);
        return voices;
    }

    public void shutdown() {
        initialized = false;
        closeSession(textConditioner);
        closeSession(mimiEncoder);
        closeSession(flowLmMain);
        closeSession(flowLmFlow);
        closeSession(mimiDecoder);
        textConditioner = null;
        mimiEncoder = null;
        flowLmMain = null;
        flowLmFlow = null;
        mimiDecoder = null;
    }

    // ── Tokenizer ───────────────────────────────────────────────

    private void loadTokenizer(File tokenizerFile) throws IOException {
        // Simple SentencePiece model loader — reads the protobuf format
        // to extract vocabulary pieces and their IDs.
        // For a full implementation, use sentencepiece-jni or a protobuf parser.
        vocab = new HashMap<>();

        // Read raw bytes and extract string pieces heuristically
        // (SentencePiece protobuf: field 1 = pieces, each piece has field 1 = string, field 2 = score)
        byte[] data = readFileBytes(tokenizerFile);
        extractVocabFromProtobuf(data);

        if (vocab.isEmpty()) {
            // Fallback: build a basic character-level tokenizer
            for (int i = 0; i < 256; i++) {
                vocab.put(String.valueOf((char) i), i + 3);
            }
        }
    }

    private void extractVocabFromProtobuf(byte[] data) {
        // Lightweight protobuf parser for SentencePiece model
        int idx = 0;
        int pieceIndex = 0;

        while (idx < data.length) {
            try {
                // Read field tag
                int tag = readVarint(data, idx);
                int tagSize = varintSize(data, idx);
                idx += tagSize;

                int fieldNumber = tag >>> 3;
                int wireType = tag & 0x7;

                if (fieldNumber == 1 && wireType == 2) {
                    // Length-delimited: this is a piece message
                    int msgLen = readVarint(data, idx);
                    int msgLenSize = varintSize(data, idx);
                    idx += msgLenSize;

                    // Parse inner message to find the string piece (field 1)
                    int msgEnd = idx + msgLen;
                    String piece = null;

                    int inner = idx;
                    while (inner < msgEnd) {
                        int innerTag = readVarint(data, inner);
                        int innerTagSize = varintSize(data, inner);
                        inner += innerTagSize;

                        int innerField = innerTag >>> 3;
                        int innerWire = innerTag & 0x7;

                        if (innerField == 1 && innerWire == 2) {
                            int strLen = readVarint(data, inner);
                            int strLenSize = varintSize(data, inner);
                            inner += strLenSize;

                            if (inner + strLen <= msgEnd) {
                                piece = new String(data, inner, strLen, StandardCharsets.UTF_8);
                            }
                            inner += strLen;
                        } else if (innerWire == 0) {
                            // Varint — skip
                            inner += varintSize(data, inner);
                        } else if (innerWire == 2) {
                            int len = readVarint(data, inner);
                            inner += varintSize(data, inner) + len;
                        } else if (innerWire == 5) {
                            inner += 4; // 32-bit fixed
                        } else if (innerWire == 1) {
                            inner += 8; // 64-bit fixed
                        } else {
                            break;
                        }
                    }

                    if (piece != null) {
                        vocab.put(piece, pieceIndex);
                    }
                    pieceIndex++;
                    idx = msgEnd;
                } else if (wireType == 0) {
                    idx += varintSize(data, idx);
                } else if (wireType == 2) {
                    int len = readVarint(data, idx);
                    idx += varintSize(data, idx) + len;
                } else if (wireType == 5) {
                    idx += 4;
                } else if (wireType == 1) {
                    idx += 8;
                } else {
                    break;
                }
            } catch (Exception e) {
                break;
            }
        }
    }

    private int readVarint(byte[] data, int offset) {
        int result = 0;
        int shift = 0;
        while (offset < data.length) {
            byte b = data[offset++];
            result |= (b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }

    private int varintSize(byte[] data, int offset) {
        int size = 0;
        while (offset < data.length) {
            size++;
            if ((data[offset++] & 0x80) == 0) break;
        }
        return size;
    }

    private long[] tokenize(String text) {
        // SentencePiece-style tokenization: try longest match first
        List<Integer> tokens = new ArrayList<>();
        tokens.add(bosId);

        String normalized = text.replace("\n", " ").trim();
        // Add SentencePiece prefix space marker
        normalized = "\u2581" + normalized.replace(" ", "\u2581");

        int i = 0;
        while (i < normalized.length()) {
            int bestLen = 0;
            Integer bestId = null;

            // Try matches from longest to shortest
            for (int len = Math.min(normalized.length() - i, 32); len >= 1; len--) {
                String candidate = normalized.substring(i, i + len);
                Integer id = vocab.get(candidate);
                if (id != null) {
                    bestLen = len;
                    bestId = id;
                    break;
                }
            }

            if (bestId != null) {
                tokens.add(bestId);
                i += bestLen;
            } else {
                // Unknown character — use byte fallback
                char c = normalized.charAt(i);
                Integer charId = vocab.get(String.valueOf(c));
                if (charId != null) {
                    tokens.add(charId);
                } else {
                    // Skip unknown
                }
                i++;
            }
        }

        tokens.add(eosId);

        long[] result = new long[tokens.size()];
        for (int j = 0; j < tokens.size(); j++) {
            result[j] = tokens.get(j);
        }
        return result;
    }

    // ── ONNX inference steps ────────────────────────────────────

    private float[][][] runTextConditioner(long[] tokenIds) throws Exception {
        long[][] inputIds = new long[1][tokenIds.length];
        inputIds[0] = tokenIds;

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputIds);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputTensor);

        try (Result result = textConditioner.run(inputs)) {
            float[][][] output = (float[][][]) result.get(0).getValue();
            return output;
        } finally {
            inputTensor.close();
        }
    }

    private Map<String, float[]> loadVoiceEmbedding(String voiceId) throws Exception {
        File embFile = new File(modelDir, "embeddings_v2/" + voiceId + ".safetensors");
        if (!embFile.exists()) {
            throw new Exception("Voice embedding not found: " + voiceId);
        }

        return loadSafetensors(embFile);
    }

    private Map<String, float[]> encodeReferenceAudio(float[] audioSamples) throws Exception {
        // Shape: [1, 1, num_samples]
        float[][][] audioInput = new float[1][1][audioSamples.length];
        audioInput[0][0] = audioSamples;

        OnnxTensor audioTensor = OnnxTensor.createTensor(env, audioInput);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("audio", audioTensor);

        try (Result result = mimiEncoder.run(inputs)) {
            // Extract latent representations as voice conditioning
            float[][][] encoded = (float[][][]) result.get(0).getValue();
            Map<String, float[]> kvCache = new HashMap<>();

            // Flatten to 1D for storage
            int dim1 = encoded[0].length;
            int dim2 = encoded[0][0].length;
            float[] flat = new float[dim1 * dim2];
            for (int i = 0; i < dim1; i++) {
                System.arraycopy(encoded[0][i], 0, flat, i * dim2, dim2);
            }
            kvCache.put("voice_conditioning", flat);

            return kvCache;
        } finally {
            audioTensor.close();
        }
    }

    private float[][] autoregressiveGenerate(float[][][] textEmbeddings,
                                              Map<String, float[]> voiceKvCache) throws Exception {
        List<float[]> latentFrames = new ArrayList<>();

        int seqLen = textEmbeddings[0].length;
        int embedDim = textEmbeddings[0][0].length;

        // Prepare initial KV-cache tensors from voice embedding
        // The exact tensor shapes depend on the model architecture
        Map<String, OnnxTensor> inputs = new HashMap<>();

        OnnxTensor textEmbTensor = OnnxTensor.createTensor(env, textEmbeddings);
        inputs.put("text_embeddings", textEmbTensor);

        // Add voice conditioning
        for (Map.Entry<String, float[]> entry : voiceKvCache.entrySet()) {
            float[][] shaped = new float[1][entry.getValue().length];
            shaped[0] = entry.getValue();
            OnnxTensor kvTensor = OnnxTensor.createTensor(env, shaped);
            inputs.put(entry.getKey(), kvTensor);
        }

        // Autoregressive loop: generate one latent frame at a time
        float[] prevLatent = new float[LATENT_DIM];

        for (int step = 0; step < MAX_STEPS; step++) {
            // Add current step position
            long[][] stepInput = new long[1][1];
            stepInput[0][0] = step;
            OnnxTensor stepTensor = OnnxTensor.createTensor(env, stepInput);
            inputs.put("step", stepTensor);

            // Add previous latent
            float[][] prevLatentShaped = new float[1][LATENT_DIM];
            prevLatentShaped[0] = prevLatent;
            OnnxTensor prevLatentTensor = OnnxTensor.createTensor(env, prevLatentShaped);
            inputs.put("prev_latent", prevLatentTensor);

            try (Result result = flowLmMain.run(inputs)) {
                float[][] mainOutput = (float[][]) result.get(0).getValue();

                // Run through flow model
                float[][] flowInput = new float[1][mainOutput[0].length];
                flowInput[0] = mainOutput[0];
                OnnxTensor flowTensor = OnnxTensor.createTensor(env, flowInput);

                Map<String, OnnxTensor> flowInputs = new HashMap<>();
                flowInputs.put("input", flowTensor);

                try (Result flowResult = flowLmFlow.run(flowInputs)) {
                    float[][] flowOutput = (float[][]) flowResult.get(0).getValue();

                    // Check for EOS condition (all zeros or very small values)
                    float maxVal = 0;
                    for (float v : flowOutput[0]) {
                        maxVal = Math.max(maxVal, Math.abs(v));
                    }
                    if (step > 10 && maxVal < 0.001f) {
                        break; // End of speech
                    }

                    // Accumulate latent
                    float[] latent = new float[LATENT_DIM];
                    int copyLen = Math.min(flowOutput[0].length, LATENT_DIM);
                    System.arraycopy(flowOutput[0], 0, latent, 0, copyLen);
                    latentFrames.add(latent);
                    prevLatent = latent;
                } finally {
                    flowTensor.close();
                }
            } finally {
                stepTensor.close();
                prevLatentTensor.close();
            }
        }

        // Clean up persistent tensors
        textEmbTensor.close();
        for (Map.Entry<String, float[]> entry : voiceKvCache.entrySet()) {
            // Tensors created above are consumed
        }

        return latentFrames.toArray(new float[0][]);
    }

    private float[] runMimiDecoder(float[][] latents) throws Exception {
        if (latents.length == 0) {
            return new float[0];
        }

        // Shape: [1, latent_dim, num_frames]
        int numFrames = latents.length;
        int dim = latents[0].length;
        float[][][] decoderInput = new float[1][dim][numFrames];

        for (int f = 0; f < numFrames; f++) {
            for (int d = 0; d < dim; d++) {
                decoderInput[0][d][f] = latents[f][d];
            }
        }

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, decoderInput);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("latents", inputTensor);

        try (Result result = mimiDecoder.run(inputs)) {
            // Output shape: [1, 1, num_samples]
            Object output = result.get(0).getValue();
            float[][][] audioOut = (float[][][]) output;
            return audioOut[0][0];
        } finally {
            inputTensor.close();
        }
    }

    // ── Safetensors loader ──────────────────────────────────────

    private Map<String, float[]> loadSafetensors(File file) throws Exception {
        Map<String, float[]> tensors = new HashMap<>();

        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            // Read header length (8 bytes, little-endian u64)
            byte[] headerLenBytes = new byte[8];
            raf.readFully(headerLenBytes);
            long headerLen = ByteBuffer.wrap(headerLenBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();

            // Read header JSON
            byte[] headerBytes = new byte[(int) headerLen];
            raf.readFully(headerBytes);
            String headerJson = new String(headerBytes, StandardCharsets.UTF_8);

            long dataOffset = 8 + headerLen;

            // Parse header JSON to find tensor metadata
            // Format: {"tensor_name": {"dtype": "F32", "shape": [...], "data_offsets": [start, end]}, ...}
            // Simple JSON parser for this specific format
            parseSafetensorsHeader(headerJson, raf, dataOffset, tensors);
        }

        return tensors;
    }

    private void parseSafetensorsHeader(String json, RandomAccessFile raf,
                                         long dataOffset,
                                         Map<String, float[]> tensors) throws Exception {
        // Remove outer braces
        json = json.trim();
        if (json.startsWith("{")) json = json.substring(1);
        if (json.endsWith("}")) json = json.substring(0, json.length() - 1);

        // Split by top-level entries
        int depth = 0;
        int start = 0;
        String currentKey = null;

        for (int i = 0; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') {
                depth--;
                if (depth == 0 && currentKey != null) {
                    String value = json.substring(start, i + 1).trim();
                    if (!currentKey.equals("__metadata__")) {
                        parseTensorEntry(currentKey, value, raf, dataOffset, tensors);
                    }
                    currentKey = null;
                }
            } else if (c == '"' && depth == 0 && currentKey == null) {
                int end = json.indexOf('"', i + 1);
                currentKey = json.substring(i + 1, end);
                i = end;
                // Skip to the colon and opening brace
                int colonIdx = json.indexOf(':', i + 1);
                start = json.indexOf('{', colonIdx);
                i = start;
                depth = 1;
            }
        }
    }

    private void parseTensorEntry(String name, String json, RandomAccessFile raf,
                                   long dataOffset,
                                   Map<String, float[]> tensors) throws Exception {
        // Extract dtype
        String dtype = extractJsonString(json, "dtype");
        if (!"F32".equals(dtype) && !"F16".equals(dtype)) {
            return; // Only handle float types
        }

        // Extract data_offsets
        int offsetsStart = json.indexOf("data_offsets");
        if (offsetsStart < 0) return;
        int bracketStart = json.indexOf('[', offsetsStart);
        int bracketEnd = json.indexOf(']', bracketStart);
        String offsetsStr = json.substring(bracketStart + 1, bracketEnd);
        String[] parts = offsetsStr.split(",");
        long startOffset = Long.parseLong(parts[0].trim());
        long endOffset = Long.parseLong(parts[1].trim());

        long byteLen = endOffset - startOffset;
        int numFloats = (int) (byteLen / ("F16".equals(dtype) ? 2 : 4));

        // Read tensor data
        raf.seek(dataOffset + startOffset);
        byte[] rawData = new byte[(int) byteLen];
        raf.readFully(rawData);

        float[] floatData = new float[numFloats];
        ByteBuffer buf = ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN);

        if ("F32".equals(dtype)) {
            buf.asFloatBuffer().get(floatData);
        } else {
            // F16 conversion
            for (int i = 0; i < numFloats; i++) {
                floatData[i] = halfToFloat(buf.getShort());
            }
        }

        tensors.put(name, floatData);
    }

    private String extractJsonString(String json, String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) return null;
        int colonIdx = json.indexOf(':', idx);
        int quoteStart = json.indexOf('"', colonIdx + 1);
        int quoteEnd = json.indexOf('"', quoteStart + 1);
        return json.substring(quoteStart + 1, quoteEnd);
    }

    // ── Audio utilities ─────────────────────────────────────────

    private float[] loadAudioSamples(File audioFile) throws Exception {
        // Read WAV or M4A file — for simplicity, handle raw WAV PCM16
        try (RandomAccessFile raf = new RandomAccessFile(audioFile, "r")) {
            // Read RIFF header
            byte[] header = new byte[44];
            raf.readFully(header);

            ByteBuffer hdr = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
            // Skip to data chunk size at offset 40
            hdr.position(40);
            int dataSize = hdr.getInt();
            int numSamples = dataSize / 2; // PCM16

            byte[] audioData = new byte[dataSize];
            raf.readFully(audioData);

            float[] samples = new float[numSamples];
            ByteBuffer audioBuf = ByteBuffer.wrap(audioData).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < numSamples; i++) {
                samples[i] = audioBuf.getShort() / 32768.0f;
            }
            return samples;
        }
    }

    private float[] resampleAudio(float[] audio, float speed) {
        int newLen = (int) (audio.length / speed);
        float[] resampled = new float[newLen];

        for (int i = 0; i < newLen; i++) {
            float srcIdx = i * speed;
            int idx0 = (int) srcIdx;
            int idx1 = Math.min(idx0 + 1, audio.length - 1);
            float frac = srcIdx - idx0;
            resampled[i] = audio[idx0] * (1 - frac) + audio[idx1] * frac;
        }

        return resampled;
    }

    private void writeWav(float[] audio, int sampleRate, File outputFile) throws IOException {
        int numSamples = audio.length;
        int dataSize = numSamples * 2; // PCM16

        try (FileOutputStream fos = new FileOutputStream(outputFile)) {
            ByteBuffer header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);

            // RIFF header
            header.put("RIFF".getBytes(StandardCharsets.US_ASCII));
            header.putInt(36 + dataSize);
            header.put("WAVE".getBytes(StandardCharsets.US_ASCII));

            // fmt chunk
            header.put("fmt ".getBytes(StandardCharsets.US_ASCII));
            header.putInt(16);          // chunk size
            header.putShort((short) 1); // PCM format
            header.putShort((short) 1); // mono
            header.putInt(sampleRate);
            header.putInt(sampleRate * 2); // byte rate
            header.putShort((short) 2);    // block align
            header.putShort((short) 16);   // bits per sample

            // data chunk
            header.put("data".getBytes(StandardCharsets.US_ASCII));
            header.putInt(dataSize);

            fos.write(header.array());

            // Write PCM16 samples
            ByteBuffer samples = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN);
            for (float sample : audio) {
                float clamped = Math.max(-1.0f, Math.min(1.0f, sample));
                samples.putShort((short) (clamped * 32767));
            }
            fos.write(samples.array());
        }
    }

    private static float halfToFloat(short half) {
        int h = half & 0xFFFF;
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) return sign == 1 ? -0.0f : 0.0f;
            // Denormalized
            float val = (float) (Math.pow(2, -14) * (mant / 1024.0));
            return sign == 1 ? -val : val;
        } else if (exp == 31) {
            if (mant == 0) return sign == 1 ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
            return Float.NaN;
        }

        float val = (float) (Math.pow(2, exp - 15) * (1 + mant / 1024.0));
        return sign == 1 ? -val : val;
    }

    private byte[] readFileBytes(File file) throws IOException {
        byte[] data = new byte[(int) file.length()];
        try (FileInputStream fis = new FileInputStream(file)) {
            int offset = 0;
            while (offset < data.length) {
                int read = fis.read(data, offset, data.length - offset);
                if (read < 0) break;
                offset += read;
            }
        }
        return data;
    }

    private void closeSession(OrtSession session) {
        if (session != null) {
            try {
                session.close();
            } catch (Exception e) {
                // Ignore
            }
        }
    }
}
