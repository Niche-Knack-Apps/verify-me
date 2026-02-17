import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

export type ThemeMode = 'modern' | 'eighties';

const STORAGE_KEY = 'verify-me-settings';

interface PersistedSettings {
  theme: ThemeMode;
  hfToken: string;
}

function loadSettings(): PersistedSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed.theme === 'modern' || parsed.theme === 'eighties') {
        return { theme: parsed.theme, hfToken: parsed.hfToken ?? '' };
      }
    }
  } catch {
    // Ignore invalid storage
  }
  return { theme: 'modern', hfToken: '' };
}

function saveSettings(settings: PersistedSettings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Ignore storage errors
  }
}

export const useSettingsStore = defineStore('settings', () => {
  const persisted = loadSettings();

  const activeTab = ref<'tts' | 'clone'>('tts');
  const showSettings = ref(false);
  const engineRunning = ref(false);
  const deviceType = ref('CPU');
  const outputDirectory = ref('');
  const modelsDirectory = ref('');
  const theme = ref<ThemeMode>(persisted.theme);
  const hfToken = ref(persisted.hfToken);

  const isEighties = computed(() => theme.value === 'eighties');

  function applyTheme() {
    if (theme.value === 'eighties') {
      document.documentElement.setAttribute('data-theme', 'eighties');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  }

  function persistAll() {
    saveSettings({ theme: theme.value, hfToken: hfToken.value });
  }

  function setTheme(mode: ThemeMode) {
    theme.value = mode;
    applyTheme();
    persistAll();
  }

  function setHfToken(token: string) {
    hfToken.value = token;
    persistAll();
  }

  function toggleTheme() {
    setTheme(theme.value === 'modern' ? 'eighties' : 'modern');
  }

  // Apply theme immediately at store creation (before first render)
  applyTheme();

  async function startEngine() {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke<string>('start_engine');
      engineRunning.value = true;
      await updateDeviceInfo();
    } catch (e) {
      console.error('Failed to start engine:', e);
      engineRunning.value = false;
    }
  }

  async function stopEngine() {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke<string>('stop_engine');
      engineRunning.value = false;
    } catch (e) {
      console.error('Failed to stop engine:', e);
    }
  }

  async function checkEngineHealth() {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const health = await invoke<{ engine_running?: boolean; status?: string; device?: string }>('engine_health');
      engineRunning.value = health.engine_running ?? false;
      if (health.device) {
        deviceType.value = health.device;
      }
    } catch (e) {
      console.error('Failed to check engine health:', e);
      engineRunning.value = false;
    }
  }

  async function updateDeviceInfo() {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const info = await invoke<{ device?: string }>('get_device_info');
      if (info.device) {
        deviceType.value = info.device;
      }
    } catch {
      // Device info not available yet
    }
  }

  async function initEngine() {
    await startEngine();
  }

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
    theme,
    hfToken,
    isEighties,
    setTheme,
    setHfToken,
    toggleTheme,
    startEngine,
    stopEngine,
    checkEngineHealth,
    initEngine,
    loadModelsDirectory,
  };
});
