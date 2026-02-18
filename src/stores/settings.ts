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
  const pythonEnvReady = ref<boolean | null>(null);
  const pythonEnvIssue = ref<string | null>(null);
  const settingUpPython = ref(false);
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
        // On Android, initialize with the first available model
        const ModelManager = await getModelManager();
        const listResult = await ModelManager.listModels();
        const available = listResult.models?.find((m: any) => m.status === 'available');
        if (available) {
          const result = await TTSEngine.initialize({ modelId: available.id });
          engineRunning.value = result.success === true;
          if (engineRunning.value) {
            deviceType.value = 'CPU (ONNX)';
          }
        } else {
          engineRunning.value = false;
          deviceType.value = 'No model';
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

  async function checkPythonEnvironment() {
    if (isCapacitor()) return;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const result = await invoke<{ ready: boolean; pythonPath: string; issue?: string }>('check_python_environment');
      pythonEnvReady.value = result.ready;
      pythonEnvIssue.value = result.issue ?? null;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error('Failed to check Python environment:', msg);
      pythonEnvReady.value = false;
      pythonEnvIssue.value = msg;
    }
  }

  async function setupPythonEnvironment() {
    if (isCapacitor()) return;
    settingUpPython.value = true;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke<string>('setup_python_environment');
      await checkPythonEnvironment();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error('Failed to set up Python environment:', msg);
      pythonEnvIssue.value = msg;
    } finally {
      settingUpPython.value = false;
    }
  }

  async function initEngine() {
    await checkPythonEnvironment();
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
    pythonEnvReady,
    pythonEnvIssue,
    settingUpPython,
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
    checkPythonEnvironment,
    setupPythonEnvironment,
    initEngine,
    loadModelsDirectory,
  };
});
