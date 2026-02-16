import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useSettingsStore = defineStore('settings', () => {
  const activeTab = ref<'tts' | 'clone'>('tts');
  const showSettings = ref(false);
  const engineRunning = ref(false);
  const deviceType = ref('CPU');
  const outputDirectory = ref('');

  return {
    activeTab,
    showSettings,
    engineRunning,
    deviceType,
    outputDirectory,
  };
});
