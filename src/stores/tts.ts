import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useTTSStore = defineStore('tts', () => {
  const text = ref('');
  const selectedModelId = ref('pocket-tts');
  const selectedVoice = ref('default');
  const speed = ref(1.0);
  const outputAudioPath = ref<string | null>(null);
  const isGenerating = ref(false);
  const error = ref<string | null>(null);

  // Voice clone state
  const referenceAudioPath = ref<string | null>(null);
  const isRecording = ref(false);

  async function generateSpeech() {
    if (!text.value.trim()) return;
    isGenerating.value = true;
    error.value = null;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const result = await invoke<string>('generate_speech', {
        text: text.value,
        modelId: selectedModelId.value,
        voice: selectedVoice.value,
        speed: speed.value,
      });
      outputAudioPath.value = result;
    } catch (e) {
      error.value = String(e);
    } finally {
      isGenerating.value = false;
    }
  }

  async function cloneVoice() {
    if (!text.value.trim() || !referenceAudioPath.value) return;
    isGenerating.value = true;
    error.value = null;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const result = await invoke<string>('voice_clone', {
        text: text.value,
        referenceAudio: referenceAudioPath.value,
        modelId: selectedModelId.value,
      });
      outputAudioPath.value = result;
    } catch (e) {
      error.value = String(e);
    } finally {
      isGenerating.value = false;
    }
  }

  return {
    text, selectedModelId, selectedVoice, speed,
    outputAudioPath, isGenerating, error,
    referenceAudioPath, isRecording,
    generateSpeech, cloneVoice,
  };
});
