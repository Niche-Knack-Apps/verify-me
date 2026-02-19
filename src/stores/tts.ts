import { defineStore } from 'pinia';
import { ref, computed, watch } from 'vue';
import { useModelsStore } from './models';
import type { Voice } from './models';

// Non-reactive recording internals
let durationInterval: ReturnType<typeof setInterval> | null = null;
let levelPollInterval: ReturnType<typeof setInterval> | null = null;
let generatingTimer: ReturnType<typeof setInterval> | null = null;

function isCapacitor(): boolean {
  return 'Capacitor' in window;
}

let _ttsEngine: any = null;
async function getTTSEngine() {
  if (!_ttsEngine) {
    const { registerPlugin } = await import('@capacitor/core');
    _ttsEngine = registerPlugin('TTSEngine');
  }
  return _ttsEngine;
}

export const useTTSStore = defineStore('tts', () => {
  const modelsStore = useModelsStore();

  const text = ref('');
  const selectedModelId = ref('pocket-tts');
  const selectedVoice = ref('alba');
  const voicePrompt = ref('');
  const voiceMode = ref<'speaker' | 'design'>('speaker');
  const voiceDescription = ref('');
  const speed = ref(1.0);
  const outputAudioPath = ref<string | null>(null);
  const isGenerating = ref(false);
  const generatingElapsed = ref(0);
  const error = ref<string | null>(null);
  const initializedModelId = ref<string | null>(null);

  // Voice clone state
  const referenceAudioPath = ref<string | null>(null);
  const isRecording = ref(false);
  const recordingDuration = ref(0);
  const currentLevel = ref(0);

  const selectedModel = computed(() => {
    return modelsStore.models.find(m => m.id === selectedModelId.value);
  });

  const voices = computed<Voice[]>(() => {
    return selectedModel.value?.voices ?? [{ id: 'default', name: 'Default' }];
  });

  // Reset voice, prompt, and mode when model changes
  watch(selectedModelId, () => {
    selectedVoice.value = voices.value[0]?.id ?? 'alba';
    voicePrompt.value = '';
    voiceMode.value = 'speaker';
    voiceDescription.value = '';
  });

  // ── Recording: Capacitor (Android) ──────────────────────────

  async function startCapacitorRecording() {
    try {
      const { Capacitor, registerPlugin } = await import('@capacitor/core');
      const AudioRecorder = registerPlugin<{
        startRecording: () => Promise<{ success: boolean; error?: string }>;
        stopRecording: () => Promise<{ success: boolean; filePath?: string; error?: string }>;
      }>('AudioRecorder');

      const result = await AudioRecorder.startRecording();
      if (!result.success) {
        error.value = result.error ?? 'Failed to start recording';
        return;
      }

      isRecording.value = true;
      recordingDuration.value = 0;

      // Duration timer (no level metering on Android — native plugin doesn't expose it)
      const startTime = Date.now();
      durationInterval = setInterval(() => {
        recordingDuration.value = Math.floor((Date.now() - startTime) / 1000);
      }, 100);
    } catch (e) {
      error.value = `Recording failed: ${e instanceof Error ? e.message : e}`;
      isRecording.value = false;
    }
  }

  async function stopCapacitorRecording() {
    try {
      const { registerPlugin } = await import('@capacitor/core');
      const AudioRecorder = registerPlugin<{
        stopRecording: () => Promise<{ success: boolean; filePath?: string; error?: string }>;
      }>('AudioRecorder');

      const result = await AudioRecorder.stopRecording();
      if (result.success && result.filePath) {
        referenceAudioPath.value = result.filePath;
      } else {
        error.value = result.error ?? 'Failed to stop recording';
      }
    } catch (e) {
      error.value = `Stop recording failed: ${e instanceof Error ? e.message : e}`;
    }

    if (durationInterval !== null) {
      clearInterval(durationInterval);
      durationInterval = null;
    }
    isRecording.value = false;
    currentLevel.value = 0;
  }

  // ── Recording: Native CPAL (Tauri desktop) ─────────────────

  async function startNativeRecording() {
    const { invoke } = await import('@tauri-apps/api/core');
    await invoke('start_recording');

    isRecording.value = true;
    recordingDuration.value = 0;

    // Poll audio level from Rust every 100ms
    levelPollInterval = setInterval(async () => {
      try {
        currentLevel.value = await invoke<number>('get_recording_level');
      } catch {
        // Ignore polling errors if recording stopped
      }
    }, 100);

    // Duration timer
    const startTime = Date.now();
    durationInterval = setInterval(() => {
      recordingDuration.value = Math.floor((Date.now() - startTime) / 1000);
    }, 100);
  }

  async function stopNativeRecording() {
    if (levelPollInterval !== null) {
      clearInterval(levelPollInterval);
      levelPollInterval = null;
    }
    if (durationInterval !== null) {
      clearInterval(durationInterval);
      durationInterval = null;
    }

    const { invoke } = await import('@tauri-apps/api/core');
    const filePath = await invoke<string>('stop_recording');
    referenceAudioPath.value = filePath;

    isRecording.value = false;
    currentLevel.value = 0;
  }

  // ── Public recording API (platform-aware) ───────────────────

  async function startRecording() {
    error.value = null;
    try {
      if (isCapacitor()) {
        await startCapacitorRecording();
      } else {
        await startNativeRecording();
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      error.value = `Microphone access failed: ${msg}`;
      isRecording.value = false;
    }
  }

  async function stopRecording() {
    try {
      if (isCapacitor()) {
        await stopCapacitorRecording();
      } else {
        await stopNativeRecording();
      }
    } catch (e) {
      error.value = `Stop recording failed: ${e instanceof Error ? e.message : String(e)}`;
      isRecording.value = false;
      currentLevel.value = 0;
    }
  }

  // ── Engine auto-init (Android) ──────────────────────────────

  async function ensureEngineInitialized() {
    if (!isCapacitor()) return;
    if (initializedModelId.value === selectedModelId.value) return;
    const TTSEngine = await getTTSEngine();
    const result = await TTSEngine.initialize({ modelId: selectedModelId.value });
    if (result.success) {
      initializedModelId.value = selectedModelId.value;
    } else {
      throw new Error(result.error ?? 'Failed to initialize engine');
    }
  }

  // ── TTS generation ──────────────────────────────────────────

  async function generateSpeech() {
    if (!text.value.trim()) return;
    isGenerating.value = true;
    generatingElapsed.value = 0;
    error.value = null;
    const startTime = Date.now();
    generatingTimer = setInterval(() => {
      generatingElapsed.value = Math.floor((Date.now() - startTime) / 1000);
    }, 1000);
    try {
      if (isCapacitor()) {
        await ensureEngineInitialized();
        const TTSEngine = await getTTSEngine();
        const result = await TTSEngine.generateSpeech({
          text: text.value,
          voice: selectedVoice.value,
          speed: speed.value,
        });
        if (!result.success) {
          throw new Error(result.error ?? 'Speech generation failed');
        }
        // Read the generated file as a blob URL
        const { Filesystem, Directory } = await import('@capacitor/filesystem');
        const fileData = await Filesystem.readFile({ path: result.filePath });
        const byteString = atob(fileData.data as string);
        const bytes = new Uint8Array(byteString.length);
        for (let i = 0; i < byteString.length; i++) {
          bytes[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'audio/wav' });
        if (outputAudioPath.value?.startsWith('blob:')) {
          URL.revokeObjectURL(outputAudioPath.value);
        }
        outputAudioPath.value = URL.createObjectURL(blob);
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        const params: Record<string, unknown> = {
          text: text.value,
          modelId: selectedModelId.value,
          voice: selectedVoice.value,
          speed: speed.value,
        };
        if (voiceMode.value === 'design') {
          params.voiceMode = 'design';
          params.voiceDescription = voiceDescription.value.trim();
        } else if (voicePrompt.value.trim()) {
          params.voicePrompt = voicePrompt.value.trim();
        }
        const result = await invoke<string>('generate_speech', params);
        const { readFile } = await import('@tauri-apps/plugin-fs');
        const bytes = await readFile(result);
        const blob = new Blob([bytes], { type: 'audio/wav' });
        if (outputAudioPath.value?.startsWith('blob:')) {
          URL.revokeObjectURL(outputAudioPath.value);
        }
        outputAudioPath.value = URL.createObjectURL(blob);
      }
    } catch (e) {
      error.value = String(e);
    } finally {
      if (generatingTimer !== null) {
        clearInterval(generatingTimer);
        generatingTimer = null;
      }
      isGenerating.value = false;
    }
  }

  async function cloneVoice() {
    if (!text.value.trim() || !referenceAudioPath.value) return;
    isGenerating.value = true;
    generatingElapsed.value = 0;
    error.value = null;
    const cloneStartTime = Date.now();
    generatingTimer = setInterval(() => {
      generatingElapsed.value = Math.floor((Date.now() - cloneStartTime) / 1000);
    }, 1000);
    try {
      if (isCapacitor()) {
        await ensureEngineInitialized();
        const TTSEngine = await getTTSEngine();
        const result = await TTSEngine.cloneVoice({
          text: text.value,
          referenceAudioPath: referenceAudioPath.value,
        });
        if (!result.success) {
          throw new Error(result.error ?? 'Voice cloning failed');
        }
        const { Filesystem, Directory } = await import('@capacitor/filesystem');
        const fileData = await Filesystem.readFile({ path: result.filePath });
        const byteString = atob(fileData.data as string);
        const bytes = new Uint8Array(byteString.length);
        for (let i = 0; i < byteString.length; i++) {
          bytes[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'audio/wav' });
        if (outputAudioPath.value?.startsWith('blob:')) {
          URL.revokeObjectURL(outputAudioPath.value);
        }
        outputAudioPath.value = URL.createObjectURL(blob);
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        const result = await invoke<string>('voice_clone', {
          text: text.value,
          referenceAudio: referenceAudioPath.value,
          modelId: selectedModelId.value,
        });
        const { readFile } = await import('@tauri-apps/plugin-fs');
        const bytes = await readFile(result);
        const blob = new Blob([bytes], { type: 'audio/wav' });
        if (outputAudioPath.value?.startsWith('blob:')) {
          URL.revokeObjectURL(outputAudioPath.value);
        }
        outputAudioPath.value = URL.createObjectURL(blob);
      }
    } catch (e) {
      error.value = String(e);
    } finally {
      if (generatingTimer !== null) {
        clearInterval(generatingTimer);
        generatingTimer = null;
      }
      isGenerating.value = false;
    }
  }

  return {
    text, selectedModelId, selectedVoice, voicePrompt, voiceMode, voiceDescription, speed,
    outputAudioPath, isGenerating, generatingElapsed, error,
    referenceAudioPath, isRecording, recordingDuration, currentLevel,
    selectedModel, voices,
    startRecording, stopRecording,
    generateSpeech, cloneVoice,
  };
});
