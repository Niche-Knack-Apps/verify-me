import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

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

let _modelManager: any = null;
async function getModelManager() {
  if (!_modelManager) {
    const { registerPlugin } = await import('@capacitor/core');
    _modelManager = registerPlugin('ModelManager');
  }
  return _modelManager;
}

export type ThemeMode = 'modern' | 'eighties';

const STORAGE_KEY = 'verify-me-settings';

interface PersistedSettings {
  theme: ThemeMode;
  hfToken: string;
  forceCpu: boolean;
}

function loadSettings(): PersistedSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed.theme === 'modern' || parsed.theme === 'eighties') {
        return { theme: parsed.theme, hfToken: parsed.hfToken ?? '', forceCpu: parsed.forceCpu ?? false };
      }
    }
  } catch {
    // Ignore invalid storage
  }
  return { theme: 'modern', hfToken: '', forceCpu: false };
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
  const engineStarting = ref(false);
  const engineError = ref<string | null>(null);
  const deviceType = ref('CPU');
  const outputDirectory = ref('');
  const modelsDirectory = ref('');
  const theme = ref<ThemeMode>(persisted.theme);
  const hfToken = ref(persisted.hfToken);
  const forceCpu = ref(persisted.forceCpu);

  const isEighties = computed(() => theme.value === 'eighties');

  function applyTheme() {
    if (theme.value === 'eighties') {
      document.documentElement.setAttribute('data-theme', 'eighties');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  }

  function persistAll() {
    saveSettings({ theme: theme.value, hfToken: hfToken.value, forceCpu: forceCpu.value });
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

  function setForceCpu(force: boolean) {
    forceCpu.value = force;
    persistAll();
  }

  function toggleTheme() {
    setTheme(theme.value === 'modern' ? 'eighties' : 'modern');
  }

  // Apply theme immediately at store creation (before first render)
  applyTheme();

  async function startEngine() {
    engineStarting.value = true;
    engineError.value = null;
    try {
      if (isCapacitor()) {
        const TTSEngine = await getTTSEngine();
        const ModelManager = await getModelManager();
        const listResult = await ModelManager.listModels();
        const allModels = listResult.models ?? [];
        let available = allModels.find((m: any) => m.status === 'available');

        // If no available models but bundled exist, extract them first
        if (!available) {
          const hasBundled = allModels.some((m: any) => m.status === 'bundled');
          if (hasBundled) {
            console.log('[Engine] No available models, extracting bundled models...');
            await ModelManager.extractBundledModels();
            const refreshed = await ModelManager.listModels();
            available = (refreshed.models ?? []).find((m: any) => m.status === 'available');
          }
        }

        if (available) {
          console.log('[Engine] Initializing with model:', available.id);
          const result = await TTSEngine.initialize({ modelId: available.id });
          engineRunning.value = result.success === true;
          if (engineRunning.value) {
            deviceType.value = 'CPU (ONNX)';
          }
        } else {
          engineRunning.value = false;
          deviceType.value = 'No model';
          engineError.value = 'No models available. Extract bundled models or download a model in Settings.';
        }
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        await invoke<string>('start_engine', { forceCpu: forceCpu.value });
        engineRunning.value = true;
        await updateDeviceInfo();
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error('Failed to start engine:', msg);
      engineError.value = msg;
      engineRunning.value = false;
    } finally {
      engineStarting.value = false;
    }
  }

  async function stopEngine() {
    try {
      if (isCapacitor()) {
        const TTSEngine = await getTTSEngine();
        await TTSEngine.shutdown();
        engineRunning.value = false;
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        await invoke<string>('stop_engine');
        engineRunning.value = false;
      }
    } catch (e) {
      console.error('Failed to stop engine:', e);
    }
  }

  async function checkEngineHealth() {
    try {
      if (isCapacitor()) {
        const TTSEngine = await getTTSEngine();
        const health = await TTSEngine.getHealth();
        engineRunning.value = health.engineRunning === true;
        if (health.device) {
          deviceType.value = health.device;
        }
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        const health = await invoke<{ engine_running?: boolean; status?: string; device?: string }>('engine_health');
        engineRunning.value = health.engine_running ?? false;
        if (health.device) {
          deviceType.value = health.device;
        }
      }
    } catch (e) {
      console.error('Failed to check engine health:', e);
      engineRunning.value = false;
    }
  }

  async function updateDeviceInfo() {
    try {
      if (isCapacitor()) {
        const TTSEngine = await getTTSEngine();
        const info = await TTSEngine.getDeviceInfo();
        if (info.name) {
          deviceType.value = info.name;
        }
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        const info = await invoke<{ device?: string }>('get_device_info');
        if (info.device) {
          deviceType.value = info.device;
        }
      }
    } catch {
      // Device info not available yet
    }
  }

  async function restartEngine() {
    await stopEngine();
    await new Promise(resolve => setTimeout(resolve, 500));
    await startEngine();
  }

  async function loadModelsDirectory() {
    try {
      if (isCapacitor()) {
        const ModelManager = await getModelManager();
        const result = await ModelManager.getModelsDirectory();
        modelsDirectory.value = result.path ?? '';
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        modelsDirectory.value = await invoke<string>('get_models_directory');
      }
    } catch (e) {
      console.error('Failed to load models directory:', e);
    }
  }

  return {
    activeTab,
    showSettings,
    engineRunning,
    engineStarting,
    engineError,
    deviceType,
    outputDirectory,
    modelsDirectory,
    theme,
    hfToken,
    forceCpu,
    isEighties,
    setTheme,
    setHfToken,
    setForceCpu,
    toggleTheme,
    startEngine,
    stopEngine,
    restartEngine,
    checkEngineHealth,
    loadModelsDirectory,
  };
});
