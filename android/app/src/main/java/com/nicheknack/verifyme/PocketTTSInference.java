package com.nicheknack.verifyme;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.TensorInfo;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * ONNX inference engine for pocket-tts model.
 *
 * Pipeline: tokenize -> text_conditioner -> load voice KV-cache ->
 *           flow_lm_main autoregressive loop -> flow_lm_flow -> mimi_decoder -> WAV
 *
 * Port of the working Rust implementation in pocket_tts.rs.
 */
public class PocketTTSInference {

    private static final String TAG = "PocketTTSInference";

    private static final int SAMPLE_RATE = 24000;
    private static final int MAX_STEPS = 2048;
    private static final int LATENT_DIM = 32;
    private static final int NUM_HEADS = 16;
    private static final int HEAD_DIM = 64;
    private static final int NUM_LAYERS = 6;
    private static final int KV_CACHE_LEN = 1000;
    private static final float EOS_THRESHOLD = -4.0f;

    /** Target reference audio duration in seconds for voice cloning. */
    private static final float TARGET_REFERENCE_SECONDS = 12.0f;
    /** Minimum usable reference audio duration (seconds). */
    private static final float MIN_REFERENCE_SECONDS = 3.0f;
    /** Maximum reference audio duration before smart selection kicks in. */
    private static final float MAX_REFERENCE_SECONDS = 15.0f;

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

    // ── Typed state tensor for ONNX state carry-over ────────────

    /**
     * Represents a typed tensor value used for stateful ONNX model carry-over.
     * Mirrors the Rust StateValue enum: F32, I64, or Bool.
     */
    private static class StateValue {
        enum Type { F32, I64, BOOL }

        final Type type;
        final long[] shape;
        float[] floatData;   // for F32
        long[] longData;     // for I64
        boolean[] boolData;  // for BOOL

        StateValue(long[] shape, float[] data) {
            this.type = Type.F32;
            this.shape = shape;
            this.floatData = data;
        }

        StateValue(long[] shape, long[] data) {
            this.type = Type.I64;
            this.shape = shape;
            this.longData = data;
        }

        StateValue(long[] shape, boolean[] data) {
            this.type = Type.BOOL;
            this.shape = shape;
            this.boolData = data;
        }

        /**
         * Create an OnnxTensor from this state value.
         * Handles zero-size dimensions by using flat buffer + shape.
         */
        OnnxTensor toOnnxTensor(OrtEnvironment env) throws OrtException {
            // Compute element count from shape (treating negatives as 0)
            long numElements = 1;
            boolean hasZero = false;
            for (long d : shape) {
                if (d <= 0) { hasZero = true; break; }
                numElements *= d;
            }

            switch (type) {
                case F32: {
                    if (hasZero || numElements == 0) {
                        // Zero-element tensor: use empty buffer with shape
                        FloatBuffer buf = FloatBuffer.allocate(0);
                        return OnnxTensor.createTensor(env, buf, shape);
                    }
                    FloatBuffer buf = FloatBuffer.wrap(floatData);
                    return OnnxTensor.createTensor(env, buf, shape);
                }
                case I64: {
                    if (hasZero || numElements == 0) {
                        LongBuffer buf = LongBuffer.allocate(0);
                        return OnnxTensor.createTensor(env, buf, shape);
                    }
                    LongBuffer buf = LongBuffer.wrap(longData);
                    return OnnxTensor.createTensor(env, buf, shape);
                }
                case BOOL: {
                    // ORT Java requires boolean multidimensional arrays, not ByteBuffer.
                    // createTensor(env, Object) accepts boolean[], boolean[][], etc.
                    if (hasZero || numElements == 0) {
                        // Empty bool tensor: create a scalar false as fallback
                        // (zero-size bool tensors are uncommon in mimi states)
                        return OnnxTensor.createTensor(env, false);
                    }
                    // Reshape boolData into the correct dimensionality
                    Object shaped = reshapeBoolArray(boolData, shape);
                    return OnnxTensor.createTensor(env, shaped);
                }
                default:
                    throw new OrtException("Unsupported state type: " + type);
            }
        }
    }

    /**
     * Reshape a flat boolean array into a multidimensional boolean array matching
     * the given shape. ORT Java's createTensor(env, Object) requires the correct
     * array nesting (boolean[] for 1D, boolean[][] for 2D, etc.).
     */
    private static Object reshapeBoolArray(boolean[] data, long[] shape) {
        if (shape.length == 0) {
            // Scalar
            return data.length > 0 ? data[0] : false;
        }
        if (shape.length == 1) {
            // 1D: boolean[]
            boolean[] result = new boolean[(int) shape[0]];
            System.arraycopy(data, 0, result, 0, Math.min(data.length, result.length));
            return result;
        }
        if (shape.length == 2) {
            // 2D: boolean[][]
            int d0 = (int) shape[0];
            int d1 = (int) shape[1];
            boolean[][] result = new boolean[d0][d1];
            int idx = 0;
            for (int i = 0; i < d0 && idx < data.length; i++) {
                for (int j = 0; j < d1 && idx < data.length; j++) {
                    result[i][j] = data[idx++];
                }
            }
            return result;
        }
        if (shape.length == 3) {
            // 3D: boolean[][][]
            int d0 = (int) shape[0];
            int d1 = (int) shape[1];
            int d2 = (int) shape[2];
            boolean[][][] result = new boolean[d0][d1][d2];
            int idx = 0;
            for (int i = 0; i < d0 && idx < data.length; i++) {
                for (int j = 0; j < d1 && idx < data.length; j++) {
                    for (int k = 0; k < d2 && idx < data.length; k++) {
                        result[i][j][k] = data[idx++];
                    }
                }
            }
            return result;
        }
        // Fallback: 4D+ is uncommon for bool states, use 1D
        boolean[] result = new boolean[(int) data.length];
        System.arraycopy(data, 0, result, 0, data.length);
        return result;
    }

    /**
     * Holds tensor data loaded from a safetensors file, including shape.
     */
    private static class TensorData {
        final long[] shape;
        final float[] data;

        TensorData(long[] shape, float[] data) {
            this.shape = shape;
            this.data = data;
        }
    }

    // ── Constructor / Lifecycle ──────────────────────────────────

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

    // ── Public API ──────────────────────────────────────────────

    /**
     * Generate speech from text using a preset voice.
     *
     * Flow (matching Rust generate_speech):
     * 1. Prepare text (normalize, capitalize, add punctuation, pad short texts)
     * 2. Tokenize
     * 3. Run text_conditioner -> text embeddings [1, T, 1024]
     * 4. Initialize flow_lm states (18 tensors), load voice
     * 5. Text conditioning pass: run_flow_lm_step(empty_seq, 0, text_embeddings, text_len, states)
     * 6. AR generation loop
     * 7. Decode latents frame-by-frame through mimi_decoder (with state carry-over)
     * 8. Speed adjustment + write WAV
     */
    public void generateSpeech(String text, String voiceId, float speed,
                                File outputFile) throws Exception {
        if (!initialized) {
            throw new IllegalStateException("Model not initialized");
        }

        // 1. Prepare text
        String prepared = prepareText(text);
        int maxGenFrames = computeMaxFrames(prepared);
        int framesAfterEos = computeFramesAfterEos(prepared);

        // 2. Tokenize
        long[] tokenIds = tokenize(prepared);

        // 3. Run text conditioner -> text embeddings [1, T, 1024] (flat)
        float[] textEmbeddings = runTextConditioner(tokenIds);
        int textLen = tokenIds.length;

        // 4. Initialize flow_lm states and load voice
        List<StateValue> flowStates = initFlowLmStates();
        Map<String, TensorData> voiceTensors = loadVoiceEmbedding(voiceId);
        loadVoiceIntoStates(flowStates, voiceTensors);

        // 5. Text conditioning pass: process text through backbone, updating KV cache
        FlowLmResult condResult = runFlowLmStep(
            new float[0], 0,       // empty sequence
            textEmbeddings, textLen,
            flowStates
        );
        flowStates = condResult.states;

        // 6. Autoregressive generation
        List<float[]> latents = autoregressiveGenerate(flowStates, maxGenFrames, framesAfterEos);

        // 7. Decode latents to audio
        float[] audio = decodeLatents(latents);

        // 8. Speed adjustment
        if (Math.abs(speed - 1.0f) > 0.01f) {
            audio = resampleAudio(audio, speed);
        }

        // 9. Write WAV
        writeWav(audio, SAMPLE_RATE, outputFile);
    }

    /**
     * Generate speech using reference audio for voice cloning.
     *
     * Flow (matching Rust clone_voice):
     * 1. Prepare text + tokenize + text conditioner
     * 2. Load + validate reference audio (3-15 seconds)
     * 3. Encode reference audio via mimi_encoder -> audio conditioning [1, T', 1024]
     * 4. Initialize fresh flow_lm states
     * 5. Voice conditioning pass: run_flow_lm_step(empty, 0, audio_conditioning, T', states)
     * 6. Text conditioning pass: run_flow_lm_step(empty, 0, text_embeddings, T, states)
     * 7. AR generation + decode + write WAV
     */
    public void cloneVoice(String text, File referenceAudio,
                           File outputFile) throws Exception {
        if (!initialized) {
            throw new IllegalStateException("Model not initialized");
        }

        // 1. Prepare text + tokenize + text conditioner
        String prepared = prepareText(text);
        int maxGenFrames = computeMaxFrames(prepared);
        int framesAfterEos = computeFramesAfterEos(prepared);

        long[] tokenIds = tokenize(prepared);
        float[] textEmbeddings = runTextConditioner(tokenIds);
        int textLen = tokenIds.length;

        // 2. Load + validate reference audio
        float[] refSamples = loadAudioSamples(referenceAudio);
        if (refSamples.length == 0) {
            throw new Exception("Reference audio is empty");
        }
        float originalDuration = refSamples.length / (float) SAMPLE_RATE;
        if (originalDuration < MIN_REFERENCE_SECONDS) {
            throw new Exception(String.format(
                "Reference audio too short (%.1fs) - need at least %.0fs",
                originalDuration, MIN_REFERENCE_SECONDS));
        }

        // Smart-select the best window for voice cloning
        refSamples = selectBestAudioWindow(refSamples, SAMPLE_RATE);

        // 3. Encode reference audio via mimi_encoder -> audio conditioning [1, T', 1024]
        EncodeResult encResult = encodeReferenceAudio(refSamples);
        float[] audioConditioning = encResult.data;
        int audioCondLen = encResult.numFrames;

        // 4. Initialize fresh flow_lm states (no predefined voice)
        List<StateValue> flowStates = initFlowLmStates();

        // 5. Voice conditioning pass: audio conditioning through backbone
        FlowLmResult voiceCondResult = runFlowLmStep(
            new float[0], 0,                  // empty sequence
            audioConditioning, audioCondLen,   // audio conditioning as text_embeddings
            flowStates
        );
        flowStates = voiceCondResult.states;

        // Free audio conditioning memory
        audioConditioning = null;
        refSamples = null;

        // 6. Text conditioning pass
        FlowLmResult textCondResult = runFlowLmStep(
            new float[0], 0,            // empty sequence
            textEmbeddings, textLen,
            flowStates
        );
        flowStates = textCondResult.states;

        // Free text embeddings before generation loop
        textEmbeddings = null;

        // 7. AR generation + decode + write WAV
        List<float[]> latents = autoregressiveGenerate(flowStates, maxGenFrames, framesAfterEos);
        float[] audio = decodeLatents(latents);
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

    // ── Flow LM state management ────────────────────────────────

    /**
     * Initialize 18 flow_lm state tensors (6 layers x 3 per layer).
     *
     * Per layer:
     *   state_{layer*3}:   KV cache float[2][1][1000][16][64] initialized to NaN
     *   state_{layer*3+1}: current_end float[0] (empty array, shape [0])
     *   state_{layer*3+2}: step long[1] = {0}
     */
    private List<StateValue> initFlowLmStates() {
        List<StateValue> states = new ArrayList<>(18);
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            // cache [2, 1, 1000, 16, 64] filled with NaN
            int cacheSize = 2 * KV_CACHE_LEN * NUM_HEADS * HEAD_DIM;
            float[] cacheData = new float[cacheSize];
            Arrays.fill(cacheData, Float.NaN);
            states.add(new StateValue(
                new long[]{2, 1, KV_CACHE_LEN, NUM_HEADS, HEAD_DIM},
                cacheData
            ));

            // current_end: empty f32 tensor shape [0]
            states.add(new StateValue(new long[]{0}, new float[0]));

            // step: i64 [1] = {0}
            states.add(new StateValue(new long[]{1}, new long[]{0}));
        }
        return states;
    }

    /**
     * Load predefined voice embedding into flow_lm states.
     * Voice embeddings contain cache [2,1,N,16,64] and current_end [N] per layer.
     * We pad cache to [2,1,1000,16,64] with NaN and set step = N.
     */
    private void loadVoiceIntoStates(List<StateValue> states,
                                      Map<String, TensorData> voiceTensors) {
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            String cacheKey = "transformer.layers." + layer + ".self_attn/cache";

            TensorData cacheTensor = voiceTensors.get(cacheKey);
            if (cacheTensor == null) continue;

            // Shape is [2, 1, N, 16, 64] - extract N (voice frames)
            int voiceFrames;
            if (cacheTensor.shape.length >= 3) {
                voiceFrames = (int) cacheTensor.shape[2];
            } else {
                // Compute from data size
                voiceFrames = cacheTensor.data.length / (2 * NUM_HEADS * HEAD_DIM);
            }

            // Pad cache to [2, 1, 1000, 16, 64] with NaN
            int total = 2 * KV_CACHE_LEN * NUM_HEADS * HEAD_DIM;
            float[] padded = new float[total];
            Arrays.fill(padded, Float.NaN);

            int framesPerKv = Math.min(voiceFrames, KV_CACHE_LEN);
            int elemsPerFrame = NUM_HEADS * HEAD_DIM; // 1024

            for (int kv = 0; kv < 2; kv++) {
                int srcBase = kv * voiceFrames * elemsPerFrame;
                int dstBase = kv * KV_CACHE_LEN * elemsPerFrame;
                int copyLen = framesPerKv * elemsPerFrame;

                if (srcBase + copyLen <= cacheTensor.data.length
                    && dstBase + copyLen <= padded.length) {
                    System.arraycopy(cacheTensor.data, srcBase, padded, dstBase, copyLen);
                }
            }

            states.set(layer * 3, new StateValue(
                new long[]{2, 1, KV_CACHE_LEN, NUM_HEADS, HEAD_DIM},
                padded
            ));

            // Set step = voiceFrames
            states.set(layer * 3 + 2, new StateValue(
                new long[]{1}, new long[]{voiceFrames}
            ));

            // current_end stays as empty [0] - not used by ONNX patched attention
        }
    }

    /**
     * Extract updated flow_lm states from session output.
     * Outputs at indices [offset..offset+17] are the 18 state tensors.
     */
    private List<StateValue> extractFlowLmStates(Result result, int offset) throws OrtException {
        List<StateValue> states = new ArrayList<>(18);
        for (int i = 0; i < 18; i++) {
            int idx = offset + i;
            int stateType = i % 3; // 0=cache(f32), 1=current_end(f32), 2=step(i64)

            if (stateType == 0 || stateType == 1) {
                // Float32 state
                OnnxTensor tensor = (OnnxTensor) result.get(idx);
                long[] shape = tensor.getInfo().getShape();
                // Handle zero-size dimensions
                long numElements = 1;
                boolean hasZero = false;
                for (long d : shape) {
                    if (d <= 0) { hasZero = true; break; }
                    numElements *= d;
                }
                float[] data;
                if (hasZero || numElements == 0) {
                    data = new float[0];
                } else {
                    FloatBuffer buf = tensor.getFloatBuffer();
                    data = new float[buf.remaining()];
                    buf.get(data);
                }
                states.add(new StateValue(shape, data));
            } else {
                // Int64 state (step)
                OnnxTensor tensor = (OnnxTensor) result.get(idx);
                long[] shape = tensor.getInfo().getShape();
                LongBuffer buf = tensor.getLongBuffer();
                long[] data = new long[buf.remaining()];
                buf.get(data);
                states.add(new StateValue(shape, data));
            }
        }
        return states;
    }

    // ── Mimi decoder state management ───────────────────────────

    /**
     * Initialize mimi decoder states from model input metadata.
     * Reads the session's input names to find state_0, state_1, ...
     * and creates appropriately typed zero/true-initialized tensors.
     */
    private List<StateValue> initMimiStates(OrtSession session) throws OrtException {
        Map<String, NodeInfo> inputInfo = session.getInputInfo();
        List<StateValue> states = new ArrayList<>();

        int i = 0;
        while (true) {
            String name = "state_" + i;
            NodeInfo info = inputInfo.get(name);
            if (info == null) break;

            TensorInfo tensorInfo = (TensorInfo) info.getInfo();
            long[] modelShape = tensorInfo.getShape();
            OnnxJavaType javaType = tensorInfo.type;

            // Replace negative dimensions (dynamic) with 0
            long[] shape = new long[modelShape.length];
            for (int d = 0; d < modelShape.length; d++) {
                shape[d] = modelShape[d] < 0 ? 0 : modelShape[d];
            }

            // Compute number of elements
            long numElements = 1;
            boolean hasZero = false;
            for (long d : shape) {
                if (d <= 0) { hasZero = true; break; }
                numElements *= d;
            }
            int count = hasZero ? 0 : (int) numElements;

            switch (javaType) {
                case FLOAT:
                    states.add(new StateValue(shape, new float[count]));
                    break;
                case INT64:
                    states.add(new StateValue(shape, new long[count]));
                    break;
                case BOOL:
                    // Bool states are "first" flags - initialized to true
                    boolean[] bools = new boolean[count];
                    Arrays.fill(bools, true);
                    states.add(new StateValue(shape, bools));
                    break;
                default:
                    throw new OrtException("Unsupported mimi state type: " + javaType
                        + " for " + name);
            }

            i++;
        }

        return states;
    }

    /**
     * Extract updated mimi states from outputs, using input state types as template.
     */
    private List<StateValue> extractMimiStates(Result result, int offset,
                                                List<StateValue> template) throws OrtException {
        List<StateValue> states = new ArrayList<>(template.size());
        for (int i = 0; i < template.size(); i++) {
            int idx = offset + i;
            StateValue tmpl = template.get(i);
            OnnxTensor tensor = (OnnxTensor) result.get(idx);
            long[] shape = tensor.getInfo().getShape();

            switch (tmpl.type) {
                case F32: {
                    long numElements = 1;
                    boolean hasZero = false;
                    for (long d : shape) {
                        if (d <= 0) { hasZero = true; break; }
                        numElements *= d;
                    }
                    float[] data;
                    if (hasZero || numElements == 0) {
                        data = new float[0];
                    } else {
                        FloatBuffer buf = tensor.getFloatBuffer();
                        data = new float[buf.remaining()];
                        buf.get(data);
                    }
                    states.add(new StateValue(shape, data));
                    break;
                }
                case I64: {
                    LongBuffer buf = tensor.getLongBuffer();
                    long[] data = new long[buf.remaining()];
                    buf.get(data);
                    states.add(new StateValue(shape, data));
                    break;
                }
                case BOOL: {
                    // ORT returns bool as bytes
                    ByteBuffer buf = tensor.getByteBuffer();
                    boolean[] data = new boolean[buf.remaining()];
                    for (int b = 0; b < data.length; b++) {
                        data[b] = buf.get() != 0;
                    }
                    states.add(new StateValue(shape, data));
                    break;
                }
            }
        }
        return states;
    }

    // ── ONNX inference steps ────────────────────────────────────

    /**
     * Result holder for run_flow_lm_step.
     */
    private static class FlowLmResult {
        final float[] conditioning; // [1024]
        final float eosLogit;
        final List<StateValue> states;

        FlowLmResult(float[] conditioning, float eosLogit, List<StateValue> states) {
            this.conditioning = conditioning;
            this.eosLogit = eosLogit;
            this.states = states;
        }
    }

    /**
     * Result holder for encode_reference_audio.
     */
    private static class EncodeResult {
        final float[] data;
        final int numFrames;

        EncodeResult(float[] data, int numFrames) {
            this.data = data;
            this.numFrames = numFrames;
        }
    }

    /**
     * Run text conditioner: token_ids -> text embeddings [1, T, 1024].
     * Returns flattened embedding data.
     *
     * ONNX input name: "token_ids" (NOT "input_ids")
     */
    private float[] runTextConditioner(long[] tokenIds) throws Exception {
        long[][] inputIds = new long[1][tokenIds.length];
        inputIds[0] = tokenIds;

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputIds);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("token_ids", inputTensor);

        try (Result result = textConditioner.run(inputs)) {
            OnnxTensor outTensor = (OnnxTensor) result.get(0);
            FloatBuffer buf = outTensor.getFloatBuffer();
            float[] data = new float[buf.remaining()];
            buf.get(data);
            return data;
        } finally {
            inputTensor.close();
        }
    }

    /**
     * Load voice embedding from safetensors, returning tensors with shapes.
     */
    private Map<String, TensorData> loadVoiceEmbedding(String voiceId) throws Exception {
        File embFile = new File(modelDir, "embeddings_v2/" + voiceId + ".safetensors");
        if (!embFile.exists()) {
            throw new Exception("Voice embedding not found: " + voiceId);
        }
        return loadSafetensorsWithShapes(embFile);
    }

    /**
     * Encode reference audio -> conditioning embeddings [1, T', 1024].
     * Returns (flat_data, num_frames).
     *
     * ONNX input name: "audio"
     */
    private EncodeResult encodeReferenceAudio(float[] audioSamples) throws Exception {
        // Shape: [1, 1, num_samples]
        float[][][] audioInput = new float[1][1][audioSamples.length];
        audioInput[0][0] = audioSamples;

        OnnxTensor audioTensor = OnnxTensor.createTensor(env, audioInput);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("audio", audioTensor);

        try (Result result = mimiEncoder.run(inputs)) {
            OnnxTensor outTensor = (OnnxTensor) result.get(0);
            long[] shape = outTensor.getInfo().getShape();

            FloatBuffer buf = outTensor.getFloatBuffer();
            float[] data = new float[buf.remaining()];
            buf.get(data);

            // shape is [1, T', 1024]
            int numFrames = shape.length >= 2 ? (int) shape[1] : 1;
            return new EncodeResult(data, numFrames);
        } finally {
            audioTensor.close();
        }
    }

    /**
     * Run one step of flow_lm_main.
     *
     * ONNX inputs:
     *   "sequence": float [1, seq_len, 32]
     *   "text_embeddings": float [1, text_len, 1024]
     *   "state_0" through "state_17": the 18 state tensors
     *
     * ONNX outputs:
     *   output[0]: conditioning [1, 1024]
     *   output[1]: eos_logit [1, 1]
     *   output[2..19]: updated 18 state tensors
     *
     * Returns (conditioning [1024], eos_logit, updated_states).
     */
    private FlowLmResult runFlowLmStep(float[] sequenceData, int seqLen,
                                        float[] textEmbData, int textLen,
                                        List<StateValue> states) throws Exception {
        List<OnnxTensor> toClose = new ArrayList<>();
        try {
            Map<String, OnnxTensor> inputs = new LinkedHashMap<>();

            // Sequence input [1, seqLen, 32]
            OnnxTensor seqTensor;
            if (seqLen == 0) {
                // Zero-length sequence: create tensor with shape [1, 0, 32]
                FloatBuffer emptyBuf = FloatBuffer.allocate(0);
                seqTensor = OnnxTensor.createTensor(env, emptyBuf, new long[]{1, 0, LATENT_DIM});
            } else {
                FloatBuffer seqBuf = FloatBuffer.wrap(sequenceData);
                seqTensor = OnnxTensor.createTensor(env, seqBuf,
                    new long[]{1, seqLen, LATENT_DIM});
            }
            toClose.add(seqTensor);
            inputs.put("sequence", seqTensor);

            // Text embeddings input [1, textLen, 1024]
            OnnxTensor textTensor;
            if (textLen == 0) {
                FloatBuffer emptyBuf = FloatBuffer.allocate(0);
                textTensor = OnnxTensor.createTensor(env, emptyBuf, new long[]{1, 0, 1024});
            } else {
                FloatBuffer textBuf = FloatBuffer.wrap(textEmbData);
                textTensor = OnnxTensor.createTensor(env, textBuf,
                    new long[]{1, textLen, 1024});
            }
            toClose.add(textTensor);
            inputs.put("text_embeddings", textTensor);

            // State inputs
            for (int i = 0; i < states.size(); i++) {
                OnnxTensor stateTensor = states.get(i).toOnnxTensor(env);
                toClose.add(stateTensor);
                inputs.put("state_" + i, stateTensor);
            }

            // Run
            try (Result result = flowLmMain.run(inputs)) {
                // Extract conditioning [1, 1024]
                OnnxTensor condTensor = (OnnxTensor) result.get(0);
                FloatBuffer condBuf = condTensor.getFloatBuffer();
                float[] conditioning = new float[condBuf.remaining()];
                condBuf.get(conditioning);

                // Extract eos_logit [1, 1]
                OnnxTensor eosTensor = (OnnxTensor) result.get(1);
                FloatBuffer eosBuf = eosTensor.getFloatBuffer();
                float eosLogit = eosBuf.get();

                // Extract updated states (outputs 2..19)
                List<StateValue> newStates = extractFlowLmStates(result, 2);

                return new FlowLmResult(conditioning, eosLogit, newStates);
            }
        } finally {
            for (OnnxTensor t : toClose) {
                try { t.close(); } catch (Exception e) { /* ignore */ }
            }
        }
    }

    /**
     * Run flow matching: conditioning -> latent via LSD decode (1 step).
     *
     * ONNX inputs:
     *   "c": float [1, 1024] (conditioning)
     *   "s": float [1, 1] = {0.0}
     *   "t": float [1, 1] = {1.0}
     *   "x": float [1, 32] = zeros
     *
     * Output: flow_dir [1, 32] (the latent)
     */
    private float[] runFlowMatching(float[] conditioning) throws Exception {
        OnnxTensor cTensor = null, sTensor = null, tTensor = null, xTensor = null;
        try {
            float[][] cData = new float[1][1024];
            System.arraycopy(conditioning, 0, cData[0], 0,
                Math.min(conditioning.length, 1024));
            cTensor = OnnxTensor.createTensor(env, cData);

            float[][] sData = {{0.0f}};
            sTensor = OnnxTensor.createTensor(env, sData);

            float[][] tData = {{1.0f}};
            tTensor = OnnxTensor.createTensor(env, tData);

            float[][] xData = new float[1][LATENT_DIM];
            // xData already initialized to zeros
            xTensor = OnnxTensor.createTensor(env, xData);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("c", cTensor);
            inputs.put("s", sTensor);
            inputs.put("t", tTensor);
            inputs.put("x", xTensor);

            try (Result result = flowLmFlow.run(inputs)) {
                OnnxTensor outTensor = (OnnxTensor) result.get(0);
                FloatBuffer buf = outTensor.getFloatBuffer();
                float[] latent = new float[buf.remaining()];
                buf.get(latent);
                return latent;
            }
        } finally {
            if (cTensor != null) cTensor.close();
            if (sTensor != null) sTensor.close();
            if (tTensor != null) tTensor.close();
            if (xTensor != null) xTensor.close();
        }
    }

    /**
     * Autoregressive generation loop.
     * Returns generated latent frames.
     *
     * 1. backbone_input starts as NaN[32] (signals BOS position)
     * 2. Each step: run_flow_lm_step(backbone_input, 1, empty, 0, states) -> conditioning + eos
     * 3. EOS check: eos_logit > -4.0 means EOS detected
     * 4. run_flow_matching(conditioning) -> latent[32]
     * 5. latent becomes next backbone_input
     */
    private List<float[]> autoregressiveGenerate(List<StateValue> flowStates,
                                                  int maxFrames,
                                                  int framesAfterEos) throws Exception {
        List<float[]> latents = new ArrayList<>();

        // First backbone input is NaN (signals BOS position)
        float[] backboneInput = new float[LATENT_DIM];
        Arrays.fill(backboneInput, Float.NaN);

        Integer eosStep = null;

        for (int step = 0; step < maxFrames; step++) {
            // Run flow_lm_main with backbone input [1, 1, 32]
            FlowLmResult result = runFlowLmStep(
                backboneInput, 1,   // seq_len = 1
                new float[0], 0,    // empty text embeddings, text_len = 0
                flowStates
            );
            flowStates = result.states;

            // Check EOS
            if (result.eosLogit > EOS_THRESHOLD && eosStep == null) {
                eosStep = step;
            }
            if (eosStep != null && step >= eosStep + framesAfterEos) {
                break;
            }

            // Run flow matching: conditioning -> latent
            float[] latent = runFlowMatching(result.conditioning);
            latents.add(latent);
            backboneInput = latent;
        }

        return latents;
    }

    /**
     * Decode latent frames to audio via mimi decoder (streaming, frame by frame).
     *
     * The mimi_decoder is stateful: it takes latent [1, 1, 32] plus N state tensors,
     * and outputs audio plus updated states.
     */
    private float[] decodeLatents(List<float[]> latents) throws Exception {
        if (latents.isEmpty()) {
            return new float[0];
        }

        List<StateValue> mimiStates = initMimiStates(mimiDecoder);
        List<Float> audioSamples = new ArrayList<>();

        for (int i = 0; i < latents.size(); i++) {
            float[] latent = latents.get(i);

            List<OnnxTensor> toClose = new ArrayList<>();
            try {
                Map<String, OnnxTensor> inputs = new LinkedHashMap<>();

                // latent [1, 1, 32]
                float[][][] latentArr = new float[1][1][LATENT_DIM];
                System.arraycopy(latent, 0, latentArr[0][0], 0,
                    Math.min(latent.length, LATENT_DIM));
                OnnxTensor latentTensor = OnnxTensor.createTensor(env, latentArr);
                toClose.add(latentTensor);
                inputs.put("latent", latentTensor);

                // State inputs
                for (int j = 0; j < mimiStates.size(); j++) {
                    OnnxTensor stateTensor = mimiStates.get(j).toOnnxTensor(env);
                    toClose.add(stateTensor);
                    inputs.put("state_" + j, stateTensor);
                }

                // Run mimi decoder
                try (Result result = mimiDecoder.run(inputs)) {
                    // Extract audio frame [1, 1, N]
                    OnnxTensor audioTensor = (OnnxTensor) result.get(0);
                    FloatBuffer buf = audioTensor.getFloatBuffer();
                    while (buf.hasRemaining()) {
                        audioSamples.add(buf.get());
                    }

                    // Extract updated states (output 1 onward)
                    mimiStates = extractMimiStates(result, 1, mimiStates);
                }
            } finally {
                for (OnnxTensor t : toClose) {
                    try { t.close(); } catch (Exception e) { /* ignore */ }
                }
            }
        }

        // Convert List<Float> to float[]
        float[] audio = new float[audioSamples.size()];
        for (int i = 0; i < audioSamples.size(); i++) {
            audio[i] = audioSamples.get(i);
        }
        return audio;
    }

    // ── Text preparation (matching Rust prepare_text) ───────────

    private String prepareText(String text) {
        String t = text.trim();
        if (t.isEmpty()) return t;

        // Normalize whitespace
        t = t.replace('\n', ' ').replace('\r', ' ');
        while (t.contains("  ")) {
            t = t.replace("  ", " ");
        }

        // Capitalize first letter
        if (!t.isEmpty() && Character.isLowerCase(t.charAt(0))) {
            t = Character.toUpperCase(t.charAt(0)) + t.substring(1);
        }

        // Ensure ends with punctuation
        if (!t.isEmpty()) {
            char last = t.charAt(t.length() - 1);
            if (Character.isLetterOrDigit(last)) {
                t = t + ".";
            }
        }

        // Pad short texts (model performs poorly with very few tokens)
        int wordCount = t.trim().split("\\s+").length;
        if (wordCount < 5) {
            t = "        " + t; // 8 spaces
        }

        return t;
    }

    private int computeMaxFrames(String text) {
        int words = text.trim().split("\\s+").length;
        double genLenSec = words + 2.0;
        return (int) (genLenSec * 12.5);
    }

    private int computeFramesAfterEos(String text) {
        int words = text.trim().split("\\s+").length;
        return words <= 4 ? 5 : 3;
    }

    // ── Tokenizer ───────────────────────────────────────────────

    private void loadTokenizer(File tokenizerFile) throws IOException {
        // Simple SentencePiece model loader - reads the protobuf format
        // to extract vocabulary pieces and their IDs.
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
                            // Varint - skip
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
                // Unknown character - use byte fallback
                char c = normalized.charAt(i);
                Integer charId = vocab.get(String.valueOf(c));
                if (charId != null) {
                    tokens.add(charId);
                }
                // Skip unknown
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

    // ── Safetensors loader (with shape info) ────────────────────

    /**
     * Load safetensors file, returning tensors with shape information.
     * Keys like "transformer.layers.{i}.self_attn/cache" hold shape + float data.
     */
    private Map<String, TensorData> loadSafetensorsWithShapes(File file) throws Exception {
        Map<String, TensorData> tensors = new HashMap<>();

        byte[] fileData = readFileBytes(file);
        if (fileData.length < 8) {
            throw new Exception("Safetensors file too small");
        }

        long headerLen = ByteBuffer.wrap(fileData, 0, 8)
            .order(ByteOrder.LITTLE_ENDIAN).getLong();
        if (fileData.length < 8 + headerLen) {
            throw new Exception("Safetensors file truncated");
        }

        String headerJson = new String(fileData, 8, (int) headerLen, StandardCharsets.UTF_8);
        int dataOffset = 8 + (int) headerLen;

        // Parse the header JSON to extract tensor metadata
        parseSafetensorsHeaderWithShapes(headerJson, fileData, dataOffset, tensors);

        return tensors;
    }

    private void parseSafetensorsHeaderWithShapes(String json, byte[] fileData,
                                                   int dataOffset,
                                                   Map<String, TensorData> tensors) throws Exception {
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
                        parseTensorEntryWithShape(currentKey, value, fileData,
                            dataOffset, tensors);
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

    private void parseTensorEntryWithShape(String name, String json, byte[] fileData,
                                            int dataOffset,
                                            Map<String, TensorData> tensors) throws Exception {
        // Extract dtype
        String dtype = extractJsonString(json, "dtype");
        if (!"F32".equals(dtype) && !"F16".equals(dtype)) {
            return; // Only handle float types
        }

        // Extract shape
        long[] shape = extractJsonLongArray(json, "shape");

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

        int absStart = dataOffset + (int) startOffset;
        int absEnd = dataOffset + (int) endOffset;
        if (absEnd > fileData.length) {
            throw new Exception("Tensor '" + name + "' data out of bounds");
        }

        float[] floatData = new float[numFloats];
        ByteBuffer buf = ByteBuffer.wrap(fileData, absStart, (int) byteLen)
            .order(ByteOrder.LITTLE_ENDIAN);

        if ("F32".equals(dtype)) {
            buf.asFloatBuffer().get(floatData);
        } else {
            // F16 conversion
            for (int i = 0; i < numFloats; i++) {
                floatData[i] = halfToFloat(buf.getShort());
            }
        }

        tensors.put(name, new TensorData(shape, floatData));
    }

    private String extractJsonString(String json, String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) return null;
        int colonIdx = json.indexOf(':', idx);
        int quoteStart = json.indexOf('"', colonIdx + 1);
        int quoteEnd = json.indexOf('"', quoteStart + 1);
        return json.substring(quoteStart + 1, quoteEnd);
    }

    private long[] extractJsonLongArray(String json, String key) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) return new long[0];
        int bracketStart = json.indexOf('[', idx);
        if (bracketStart < 0) return new long[0];
        int bracketEnd = json.indexOf(']', bracketStart);
        if (bracketEnd < 0) return new long[0];

        String content = json.substring(bracketStart + 1, bracketEnd).trim();
        if (content.isEmpty()) return new long[0];

        String[] parts = content.split(",");
        long[] result = new long[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Long.parseLong(parts[i].trim());
        }
        return result;
    }

    // ── Audio utilities ─────────────────────────────────────────

    private float[] loadAudioSamples(File audioFile) throws Exception {
        // Read WAV file - handle PCM16
        try (RandomAccessFile raf = new RandomAccessFile(audioFile, "r")) {
            // Read RIFF header
            byte[] header = new byte[44];
            raf.readFully(header);

            ByteBuffer hdr = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);

            // Read channels (offset 22, 2 bytes)
            hdr.position(22);
            short channels = hdr.getShort();

            // Read sample rate (offset 24, 4 bytes)
            int fileSampleRate = hdr.getInt();

            // Skip to data chunk size at offset 40
            hdr.position(40);
            int dataSize = hdr.getInt();
            int numSamples = dataSize / 2; // PCM16

            byte[] audioData = new byte[dataSize];
            raf.readFully(audioData);

            float[] rawSamples = new float[numSamples];
            ByteBuffer audioBuf = ByteBuffer.wrap(audioData).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < numSamples; i++) {
                rawSamples[i] = audioBuf.getShort() / 32768.0f;
            }

            // Mix to mono if stereo/multi-channel
            float[] mono;
            if (channels > 1) {
                int monoLen = numSamples / channels;
                mono = new float[monoLen];
                for (int i = 0; i < monoLen; i++) {
                    float sum = 0;
                    for (int ch = 0; ch < channels; ch++) {
                        sum += rawSamples[i * channels + ch];
                    }
                    mono[i] = sum / channels;
                }
            } else {
                mono = rawSamples;
            }

            // Resample to SAMPLE_RATE if needed
            if (fileSampleRate != SAMPLE_RATE) {
                double ratio = (double) SAMPLE_RATE / fileSampleRate;
                int newLen = (int) (mono.length * ratio);
                float[] resampled = new float[newLen];
                for (int i = 0; i < newLen; i++) {
                    double srcIdx = i / ratio;
                    int idx0 = (int) srcIdx;
                    int idx1 = Math.min(idx0 + 1, mono.length - 1);
                    float frac = (float) (srcIdx - idx0);
                    resampled[i] = mono[idx0] * (1 - frac) + mono[idx1] * frac;
                }
                return resampled;
            }

            return mono;
        }
    }

    /**
     * Select the best audio window for voice cloning.
     * The model works best with 10-15 seconds of clean speech.
     */
    private float[] selectBestAudioWindow(float[] samples, int sampleRate) {
        int targetSamples = (int) (TARGET_REFERENCE_SECONDS * sampleRate);
        int maxSamples = (int) (MAX_REFERENCE_SECONDS * sampleRate);

        // Short enough already - just trim silence and return
        if (samples.length <= maxSamples) {
            return trimSilence(samples, sampleRate);
        }

        // Audio is too long - find the best window
        // Compute RMS energy in 0.5s frames
        int frameSize = sampleRate / 2;
        int numFrames = samples.length / frameSize;
        if (numFrames == 0) return samples;

        float[] frameRms = new float[numFrames];
        for (int i = 0; i < numFrames; i++) {
            int start = i * frameSize;
            int end = Math.min(start + frameSize, samples.length);
            float sumSq = 0;
            for (int j = start; j < end; j++) {
                sumSq += samples[j] * samples[j];
            }
            frameRms[i] = (float) Math.sqrt(sumSq / (end - start));
        }

        // Compute overall RMS threshold to distinguish speech from silence
        float meanRms = 0;
        for (float rms : frameRms) meanRms += rms;
        meanRms /= frameRms.length;
        float silenceThreshold = meanRms * 0.15f;

        // Find the best window of targetSamples length
        int windowFrames = Math.min((int) (TARGET_REFERENCE_SECONDS * 2), numFrames);

        float bestScore = Float.NEGATIVE_INFINITY;
        int bestStartFrame = 0;

        int searchEnd = Math.max(1, numFrames - windowFrames + 1);
        for (int start = 0; start < searchEnd; start++) {
            int end = Math.min(start + windowFrames, numFrames);
            float speechEnergy = 0;
            int silenceCount = 0;

            for (int f = start; f < end; f++) {
                if (frameRms[f] > silenceThreshold) {
                    speechEnergy += frameRms[f];
                } else {
                    silenceCount++;
                }
            }

            float silenceRatio = (float) silenceCount / (end - start);
            float score = speechEnergy * (1.0f - silenceRatio * 0.5f);

            if (score > bestScore) {
                bestScore = score;
                bestStartFrame = start;
            }
        }

        int startSample = bestStartFrame * frameSize;
        int endSample = Math.min(startSample + targetSamples, samples.length);

        float[] selected = Arrays.copyOfRange(samples, startSample, endSample);
        return trimSilence(selected, sampleRate);
    }

    /**
     * Trim leading and trailing silence from audio.
     */
    private float[] trimSilence(float[] samples, int sampleRate) {
        if (samples.length == 0) return new float[0];

        int frameSize = sampleRate / 20; // 50ms frames
        int numFrames = samples.length / frameSize;
        if (numFrames == 0) return samples.clone();

        // Compute per-frame RMS
        float[] frameRms = new float[numFrames];
        float peakRms = 0;
        for (int i = 0; i < numFrames; i++) {
            int start = i * frameSize;
            int end = Math.min(start + frameSize, samples.length);
            float sumSq = 0;
            for (int j = start; j < end; j++) {
                sumSq += samples[j] * samples[j];
            }
            frameRms[i] = (float) Math.sqrt(sumSq / (end - start));
            peakRms = Math.max(peakRms, frameRms[i]);
        }

        if (peakRms < 1e-6f) return samples.clone();

        float threshold = peakRms * 0.05f;

        // Find first and last speech frames
        int firstSpeech = 0;
        for (int i = 0; i < numFrames; i++) {
            if (frameRms[i] > threshold) { firstSpeech = i; break; }
        }
        int lastSpeech = numFrames - 1;
        for (int i = numFrames - 1; i >= 0; i--) {
            if (frameRms[i] > threshold) { lastSpeech = i; break; }
        }

        // Add a small margin (100ms)
        int marginFrames = 2;
        int startFrame = Math.max(0, firstSpeech - marginFrames);
        int endFrame = Math.min(numFrames, lastSpeech + marginFrames + 1);

        int startSample = startFrame * frameSize;
        int endSample = Math.min(endFrame * frameSize, samples.length);

        return Arrays.copyOfRange(samples, startSample, endSample);
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
