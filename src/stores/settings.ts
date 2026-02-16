import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useSettingsStore = defineStore('settings', () => {
  const activeTab = ref<'tts' | 'clone'>('tts');
  const showSettings = ref(false);
  const engineRunning = ref(false);
  const deviceType = ref('CPU');
  const outputDirectory = ref('');
  const modelsDirectory = ref('');

  async function loadModelsDirectory() {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      modelsDirectory.value = await invoke<string>('get_models_directory');
    } catch (e) {
      console.error('Failed to load models directory:', e);
    }
  }

  return {
    activeTab,
    showSettings,
    engineRunning,
    deviceType,
    outputDirectory,
    modelsDirectory,
    loadModelsDirectory,
  };
});
