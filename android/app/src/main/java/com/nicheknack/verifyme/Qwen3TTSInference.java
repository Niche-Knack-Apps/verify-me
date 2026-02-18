package com.nicheknack.verifyme;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ONNX inference engine for Qwen3-TTS model (0.6B and 1.7B variants).
 *
 * Pipeline:
 *   tokenize → text_project → talker_prefill →
 *   talker_decode (autoregressive) → code_predictor →
 *   tokenizer12hz_decode → audio output
 *
 * Voice cloning uses speaker_encoder to extract speaker embeddings from reference audio.
 */
public class Qwen3TTSInference {

    private static final int SAMPLE_RATE = 24000;
    private static final int MAX_STEPS = 4096;

    private OrtEnvironment env;
    private OrtSession codecEmbed;
    private OrtSession speakerEncoder;
    private OrtSession codePredictorEmbed;
    private OrtSession codePredictor;
    private OrtSession tokenizer12hzEncode;
    private OrtSession tokenizer12hzDecode;
    private OrtSession textProject;
    private OrtSession talkerDecode;
    private OrtSession talkerPrefill;

    private File modelDir;
    private boolean initialized = false;

    // BPE tokenizer
    private Map<String, Integer> vocabMap;
    private List<String[]> merges;
    private int bosId = 0;
    private int eosId = 1;
    private int padId = 2;

    public Qwen3TTSInference() {
        this.env = OrtEnvironment.getEnvironment();
    }

    public void initialize(File modelDir) throws Exception {
        this.modelDir = modelDir;

        SessionOptions opts = new SessionOptions();
        opts.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
        opts.setIntraOpNumThreads(4);

        codecEmbed = loadSession("codec_embed_q.onnx", opts);
        speakerEncoder = loadSession("speaker_encoder_q.onnx", opts);
        codePredictorEmbed = loadSession("code_predictor_embed_q.onnx", opts);
        codePredictor = loadSession("code_predictor_q.onnx", opts);
        tokenizer12hzEncode = loadSession("tokenizer12hz_encode_q.onnx", opts);
        tokenizer12hzDecode = loadSession("tokenizer12hz_decode_q.onnx", opts);
        textProject = loadSession("text_project_q.onnx", opts);
        talkerDecode = loadSession("talker_decode_q.onnx", opts);
        talkerPrefill = loadSession("talker_prefill_q.onnx", opts);

        loadBpeTokenizer();
        initialized = true;
    }

    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Generate speech from text using a preset voice.
     */
    public void generateSpeech(String text, String voiceId, float speed,
                                File outputFile) throws Exception {
        if (!initialized) {
            throw new IllegalStateException("Model not initialized");
        }

        // 1. Tokenize text with BPE
        long[] tokenIds = tokenize(text);

        // 2. Project text tokens to embeddings
        float[][][] textEmbeddings = runTextProject(tokenIds);

        // 3. Prefill with text embeddings + voice conditioning
        float[][][] prefillState = runTalkerPrefill(textEmbeddings, voiceId);

        // 4. Autoregressive generation
        long[][] codes = autoregressiveGenerate(prefillState);

        // 5. Decode audio codes via tokenizer12hz_decode
        float[] audio = decodeAudio(codes);

        // 6. Speed adjustment
        if (Math.abs(speed - 1.0f) > 0.01f) {
            audio = resampleAudio(audio, speed);
        }

        // 7. Write WAV
        writeWav(audio, SAMPLE_RATE, outputFile);
    }

    /**
     * Generate speech using reference audio for voice cloning.
     */
    public void cloneVoice(String text, File referenceAudio,
                           File outputFile) throws Exception {
        if (!initialized) {
            throw new IllegalStateException("Model not initialized");
        }

        // 1. Tokenize text
        long[] tokenIds = tokenize(text);

        // 2. Project text tokens
        float[][][] textEmbeddings = runTextProject(tokenIds);

        // 3. Encode reference audio for speaker embedding
        float[] refSamples = loadAudioSamples(referenceAudio);
        float[][] speakerEmbedding = runSpeakerEncoder(refSamples);

        // 4. Prefill with text + speaker embedding
        float[][][] prefillState = runTalkerPrefillWithSpeaker(textEmbeddings, speakerEmbedding);

        // 5. Autoregressive generation
        long[][] codes = autoregressiveGenerate(prefillState);

        // 6. Decode audio
        float[] audio = decodeAudio(codes);

        // 7. Write WAV
        writeWav(audio, SAMPLE_RATE, outputFile);
    }

    public List<String> getAvailableVoices() {
        // Qwen3-TTS uses hardcoded voice IDs
        List<String> voices = new ArrayList<>();
        voices.add("Aiden");
        voices.add("Ryan");
        voices.add("Vivian");
        voices.add("Serena");
        voices.add("Dylan");
        voices.add("Eric");
        voices.add("Uncle_Fu");
        voices.add("Ono_Anna");
        voices.add("Sohee");
        return voices;
    }

    public void shutdown() {
        initialized = false;
        closeSession(codecEmbed);
        closeSession(speakerEncoder);
        closeSession(codePredictorEmbed);
        closeSession(codePredictor);
        closeSession(tokenizer12hzEncode);
        closeSession(tokenizer12hzDecode);
        closeSession(textProject);
        closeSession(talkerDecode);
        closeSession(talkerPrefill);
        codecEmbed = null;
        speakerEncoder = null;
        codePredictorEmbed = null;
        codePredictor = null;
        tokenizer12hzEncode = null;
        tokenizer12hzDecode = null;
        textProject = null;
        talkerDecode = null;
        talkerPrefill = null;
    }

    // ── BPE Tokenizer ───────────────────────────────────────────

    private void loadBpeTokenizer() throws Exception {
        vocabMap = new HashMap<>();
        merges = new ArrayList<>();

        // Load vocab.json
        File vocabFile = new File(modelDir, "vocab.json");
        if (vocabFile.exists()) {
            String vocabJson = readFileString(vocabFile);
            parseVocabJson(vocabJson);
        }

        // Load merges.txt
        File mergesFile = new File(modelDir, "merges.txt");
        if (mergesFile.exists()) {
            String mergesContent = readFileString(mergesFile);
            String[] lines = mergesContent.split("\n");
            for (String line : lines) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#version")) continue;
                String[] parts = line.split(" ", 2);
                if (parts.length == 2) {
                    merges.add(parts);
                }
            }
        }
    }

    private void parseVocabJson(String json) {
        // Simple JSON object parser: {"token": id, ...}
        json = json.trim();
        if (json.startsWith("{")) json = json.substring(1);
        if (json.endsWith("}")) json = json.substring(0, json.length() - 1);

        int i = 0;
        while (i < json.length()) {
            // Find key
            int keyStart = json.indexOf('"', i);
            if (keyStart < 0) break;
            int keyEnd = findUnescapedQuote(json, keyStart + 1);
            if (keyEnd < 0) break;

            String key = unescapeJson(json.substring(keyStart + 1, keyEnd));

            // Find value
            int colonIdx = json.indexOf(':', keyEnd + 1);
            if (colonIdx < 0) break;

            int valStart = colonIdx + 1;
            while (valStart < json.length() && json.charAt(valStart) == ' ') valStart++;

            int valEnd = valStart;
            while (valEnd < json.length() && json.charAt(valEnd) != ',' && json.charAt(valEnd) != '}') {
                valEnd++;
            }

            String valStr = json.substring(valStart, valEnd).trim();
            try {
                int id = Integer.parseInt(valStr);
                vocabMap.put(key, id);
            } catch (NumberFormatException e) {
                // Skip non-integer values
            }

            i = valEnd + 1;
        }
    }

    private int findUnescapedQuote(String s, int start) {
        for (int i = start; i < s.length(); i++) {
            if (s.charAt(i) == '"' && (i == 0 || s.charAt(i - 1) != '\\')) {
                return i;
            }
        }
        return -1;
    }

    private String unescapeJson(String s) {
        return s.replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\/", "/");
    }

    private long[] tokenize(String text) {
        List<Integer> tokens = new ArrayList<>();

        // BPE tokenization
        // 1. Pre-tokenize: split on whitespace and punctuation
        List<String> words = preTokenize(text);

        // 2. Apply BPE merges to each word
        for (String word : words) {
            List<String> wordTokens = applyBpe(word);
            for (String tok : wordTokens) {
                Integer id = vocabMap.get(tok);
                if (id != null) {
                    tokens.add(id);
                }
            }
        }

        long[] result = new long[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            result[i] = tokens.get(i);
        }
        return result;
    }

    private List<String> preTokenize(String text) {
        // GPT-style pre-tokenization with byte-level encoding
        List<String> result = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == ' ') {
                if (current.length() > 0) {
                    result.add(current.toString());
                    current = new StringBuilder();
                }
                current.append('\u0120'); // GPT2 space prefix
            } else {
                current.append(c);
            }
        }

        if (current.length() > 0) {
            result.add(current.toString());
        }

        return result;
    }

    private List<String> applyBpe(String word) {
        List<String> symbols = new ArrayList<>();
        for (int i = 0; i < word.length(); i++) {
            symbols.add(String.valueOf(word.charAt(i)));
        }

        for (String[] merge : merges) {
            List<String> newSymbols = new ArrayList<>();
            int i = 0;
            while (i < symbols.size()) {
                if (i < symbols.size() - 1 &&
                    symbols.get(i).equals(merge[0]) &&
                    symbols.get(i + 1).equals(merge[1])) {
                    newSymbols.add(merge[0] + merge[1]);
                    i += 2;
                } else {
                    newSymbols.add(symbols.get(i));
                    i++;
                }
            }
            symbols = newSymbols;
            if (symbols.size() == 1) break;
        }

        return symbols;
    }

    // ── ONNX inference steps ────────────────────────────────────

    private float[][][] runTextProject(long[] tokenIds) throws Exception {
        long[][] inputIds = new long[1][tokenIds.length];
        inputIds[0] = tokenIds;

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputIds);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputTensor);

        try (Result result = textProject.run(inputs)) {
            float[][][] output = (float[][][]) result.get(0).getValue();
            return output;
        } finally {
            inputTensor.close();
        }
    }

    private float[][] runSpeakerEncoder(float[] audioSamples) throws Exception {
        // Shape: [1, 1, num_samples]
        float[][][] audioInput = new float[1][1][audioSamples.length];
        audioInput[0][0] = audioSamples;

        OnnxTensor audioTensor = OnnxTensor.createTensor(env, audioInput);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("audio", audioTensor);

        try (Result result = speakerEncoder.run(inputs)) {
            float[][] embedding = (float[][]) result.get(0).getValue();
            return embedding;
        } finally {
            audioTensor.close();
        }
    }

    private float[][][] runTalkerPrefill(float[][][] textEmbeddings,
                                          String voiceId) throws Exception {
        OnnxTensor textTensor = OnnxTensor.createTensor(env, textEmbeddings);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("text_embeddings", textTensor);

        // Pass voice ID as a simple string token
        long[][] voiceTokens = new long[1][1];
        Integer voiceTokenId = vocabMap.get(voiceId);
        voiceTokens[0][0] = voiceTokenId != null ? voiceTokenId : 0;
        OnnxTensor voiceTensor = OnnxTensor.createTensor(env, voiceTokens);
        inputs.put("speaker_id", voiceTensor);

        try (Result result = talkerPrefill.run(inputs)) {
            float[][][] state = (float[][][]) result.get(0).getValue();
            return state;
        } finally {
            textTensor.close();
            voiceTensor.close();
        }
    }

    private float[][][] runTalkerPrefillWithSpeaker(float[][][] textEmbeddings,
                                                     float[][] speakerEmbedding) throws Exception {
        OnnxTensor textTensor = OnnxTensor.createTensor(env, textEmbeddings);
        OnnxTensor speakerTensor = OnnxTensor.createTensor(env, speakerEmbedding);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("text_embeddings", textTensor);
        inputs.put("speaker_embedding", speakerTensor);

        try (Result result = talkerPrefill.run(inputs)) {
            float[][][] state = (float[][][]) result.get(0).getValue();
            return state;
        } finally {
            textTensor.close();
            speakerTensor.close();
        }
    }

    private long[][] autoregressiveGenerate(float[][][] prefillState) throws Exception {
        List<long[]> codeFrames = new ArrayList<>();

        Map<String, OnnxTensor> inputs = new HashMap<>();
        OnnxTensor stateTensor = OnnxTensor.createTensor(env, prefillState);
        inputs.put("state", stateTensor);

        for (int step = 0; step < MAX_STEPS; step++) {
            long[][] stepInput = new long[1][1];
            stepInput[0][0] = step;
            OnnxTensor stepTensor = OnnxTensor.createTensor(env, stepInput);
            inputs.put("step", stepTensor);

            try (Result result = talkerDecode.run(inputs)) {
                float[][] logits = (float[][]) result.get(0).getValue();

                // Get predicted code via argmax
                int codeIdx = argmax(logits[0]);

                // Check EOS
                if (codeIdx == eosId && step > 10) {
                    break;
                }

                // Run through code predictor
                long[][] codeInput = new long[1][1];
                codeInput[0][0] = codeIdx;
                OnnxTensor codeTensor = OnnxTensor.createTensor(env, codeInput);

                Map<String, OnnxTensor> cpInputs = new HashMap<>();
                cpInputs.put("code", codeTensor);

                try (Result cpResult = codePredictor.run(cpInputs)) {
                    long[][] predictedCodes = (long[][]) cpResult.get(0).getValue();
                    codeFrames.add(predictedCodes[0]);
                } finally {
                    codeTensor.close();
                }

                // Update state for next step (use codec_embed for feedback)
                long[][] embedInput = new long[1][1];
                embedInput[0][0] = codeIdx;
                OnnxTensor embedTensor = OnnxTensor.createTensor(env, embedInput);

                Map<String, OnnxTensor> embedInputs = new HashMap<>();
                embedInputs.put("codes", embedTensor);

                try (Result embedResult = codecEmbed.run(embedInputs)) {
                    float[][] newEmbed = (float[][]) embedResult.get(0).getValue();
                    // Feed back into next decode step
                    float[][][] feedbackState = new float[1][1][newEmbed[0].length];
                    feedbackState[0][0] = newEmbed[0];
                    stateTensor.close();
                    stateTensor = OnnxTensor.createTensor(env, feedbackState);
                    inputs.put("state", stateTensor);
                } finally {
                    embedTensor.close();
                }
            } finally {
                stepTensor.close();
            }
        }

        stateTensor.close();

        return codeFrames.toArray(new long[0][]);
    }

    private float[] decodeAudio(long[][] codes) throws Exception {
        if (codes.length == 0) {
            return new float[0];
        }

        // Shape codes for decoder: [1, num_codebooks, num_frames]
        int numFrames = codes.length;
        int numCodes = codes[0].length;

        long[][][] decoderInput = new long[1][numCodes][numFrames];
        for (int f = 0; f < numFrames; f++) {
            for (int c = 0; c < numCodes; c++) {
                decoderInput[0][c][f] = codes[f][c];
            }
        }

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, decoderInput);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("codes", inputTensor);

        try (Result result = tokenizer12hzDecode.run(inputs)) {
            float[][][] audioOut = (float[][][]) result.get(0).getValue();
            return audioOut[0][0];
        } finally {
            inputTensor.close();
        }
    }

    // ── Utilities ───────────────────────────────────────────────

    private int argmax(float[] arr) {
        int maxIdx = 0;
        float maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private OrtSession loadSession(String filename, SessionOptions opts) throws Exception {
        File file = new File(modelDir, filename);
        if (!file.exists()) {
            throw new Exception("Model file not found: " + filename);
        }
        return env.createSession(file.getAbsolutePath(), opts);
    }

    private float[] loadAudioSamples(File audioFile) throws Exception {
        try (RandomAccessFile raf = new RandomAccessFile(audioFile, "r")) {
            byte[] header = new byte[44];
            raf.readFully(header);

            ByteBuffer hdr = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
            hdr.position(40);
            int dataSize = hdr.getInt();
            int numSamples = dataSize / 2;

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
        int dataSize = numSamples * 2;

        try (FileOutputStream fos = new FileOutputStream(outputFile)) {
            ByteBuffer header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);

            header.put("RIFF".getBytes(StandardCharsets.US_ASCII));
            header.putInt(36 + dataSize);
            header.put("WAVE".getBytes(StandardCharsets.US_ASCII));

            header.put("fmt ".getBytes(StandardCharsets.US_ASCII));
            header.putInt(16);
            header.putShort((short) 1);
            header.putShort((short) 1);
            header.putInt(sampleRate);
            header.putInt(sampleRate * 2);
            header.putShort((short) 2);
            header.putShort((short) 16);

            header.put("data".getBytes(StandardCharsets.US_ASCII));
            header.putInt(dataSize);

            fos.write(header.array());

            ByteBuffer samples = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN);
            for (float sample : audio) {
                float clamped = Math.max(-1.0f, Math.min(1.0f, sample));
                samples.putShort((short) (clamped * 32767));
            }
            fos.write(samples.array());
        }
    }

    private String readFileString(File file) throws IOException {
        byte[] data = new byte[(int) file.length()];
        try (FileInputStream fis = new FileInputStream(file)) {
            int offset = 0;
            while (offset < data.length) {
                int read = fis.read(data, offset, data.length - offset);
                if (read < 0) break;
                offset += read;
            }
        }
        return new String(data, StandardCharsets.UTF_8);
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
