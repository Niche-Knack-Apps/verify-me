import { defineStore } from 'pinia';
import { ref } from 'vue';
import { isCapacitor, getModelManager } from '@/services/capacitor-plugins';

export interface Voice {
  id: string;
  name: string;
}

export interface TTSModel {
  id: string;
  name: string;
  size: string;
  status: 'available' | 'downloadable' | 'bundled' | 'extracting' | 'downloading';
  supportsClone: boolean;
  supportsVoicePrompt: boolean;
  supportsVoiceDesign: boolean;
  voices: Voice[];
  downloadUrl?: string;
}

export const useModelsStore = defineStore('models', () => {
  const models = ref<TTSModel[]>([]);
  const loading = ref(false);
  const loadError = ref<string | null>(null);
  const downloadProgress = ref<Record<string, number>>({});
  const downloading = ref<string | null>(null);
  const downloadErrors = ref<Record<string, string>>({});
  const downloadFilename = ref<Record<string, string>>({});

  async function loadModels() {
    if (loading.value) return; // prevent concurrent calls
    loading.value = true;
    loadError.value = null;
    try {
      if (isCapacitor()) {
        console.log('[models] Getting ModelManager plugin...');
        let ModelManager: any;
        try {
          ModelManager = await getModelManager();
        } catch (pluginErr) {
          throw new Error(`Plugin init failed: ${pluginErr}`);
        }
        console.log('[models] Calling ModelManager.listModels...');

        // Timeout guard: if plugin doesn't respond in 10s, fail gracefully
        const result = await Promise.race([
          ModelManager.listModels(),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Plugin timeout — ModelManager.listModels() did not respond in 10s')), 10000)
          ),
        ]) as any;

        console.log('[models] listModels raw result:', JSON.stringify(result));
        const raw = result?.models ?? [];
        if (raw.length === 0) {
          loadError.value = 'Plugin returned 0 models. Check logcat for ModelManager errors.';
        }
        // Map from plugin format to TTSModel[]
        models.value = raw.map((m: any) => ({
          id: m.id,
          name: m.name,
          size: m.size,
          status: m.status,
          supportsClone: m.supportsClone ?? false,
          supportsVoicePrompt: m.supportsVoicePrompt ?? false,
          supportsVoiceDesign: m.supportsVoiceDesign ?? false,
          voices: m.voices ?? [],
          downloadUrl: m.downloadUrl,
        }));
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        const result = await invoke<TTSModel[]>('list_models');
        models.value = result;
      }
    } catch (e: unknown) {
      const err = e instanceof Error ? e : new Error(String(e));
      loadError.value = `Failed to load models: ${err.message}`;
      console.error('[models] Failed to load models:', err.message);
      console.error('[models] Stack:', err.stack);
    } finally {
      loading.value = false;
    }
  }

  async function extractBundledModels() {
    if (!isCapacitor()) return;
    models.value = models.value.map(m =>
      m.status === 'bundled' ? { ...m, status: 'extracting' as const } : m
    );
    try {
      const ModelManager = await getModelManager();
      const result = await ModelManager.extractBundledModels();
      console.log('[models] extractBundledModels result:', JSON.stringify(result));
      if (result.errors) {
        console.error('[models] Extraction errors:', result.errors);
        loadError.value = `Extraction: ${result.errors}`;
      }
    } catch (e) {
      console.error('[models] Bundled extraction failed:', e);
      loadError.value = `Extraction failed: ${e instanceof Error ? e.message : String(e)}`;
      models.value = models.value.map(m =>
        m.status === 'extracting' ? { ...m, status: 'bundled' as const } : m
      );
    }
    await loadModels();
  }

  async function downloadModel(modelId: string) {
    const model = models.value.find(m => m.id === modelId);
    if (!model) {
      console.error('Model not found:', modelId);
      return;
    }

    if (!model.downloadUrl) {
      console.error('No download URL for model:', modelId);
      return;
    }

    try {
      downloading.value = modelId;
      downloadProgress.value[modelId] = 0;
      delete downloadErrors.value[modelId];
      models.value = models.value.map(m =>
        m.id === modelId ? { ...m, status: 'downloading' as const } : m
      );

      if (isCapacitor()) {
        const ModelManager = await getModelManager();
        await ModelManager.downloadModel({ modelId });
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        await invoke('download_model', { url: model.downloadUrl, modelId });
      }

      delete downloadProgress.value[modelId];
      downloading.value = null;
      await loadModels();
    } catch (e) {
      delete downloadProgress.value[modelId];
      delete downloadFilename.value[modelId];
      downloading.value = null;
      downloadErrors.value[modelId] = String(e);
      models.value = models.value.map(m =>
        m.id === modelId ? { ...m, status: 'downloadable' as const } : m
      );
      console.error('Failed to download model:', e);
      throw e;
    }
  }

  async function cancelDownload(modelId: string) {
    try {
      if (isCapacitor()) {
        const ModelManager = await getModelManager();
        await ModelManager.cancelDownload();
      }
      delete downloadProgress.value[modelId];
      delete downloadFilename.value[modelId];
      downloading.value = null;
      models.value = models.value.map(m =>
        m.id === modelId ? { ...m, status: 'downloadable' as const } : m
      );
    } catch (e) {
      console.error('Failed to cancel download:', e);
    }
  }

  async function deleteModel(id: string) {
    try {
      if (isCapacitor()) {
        const ModelManager = await getModelManager();
        await ModelManager.deleteModel({ modelId: id });
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        await invoke('delete_model', { modelId: id });
      }
      await loadModels();
    } catch (e) {
      console.error('Failed to delete model:', e);
    }
  }

  async function initEventListeners() {
    try {
      if (isCapacitor()) {
        const ModelManager = await getModelManager();
        await ModelManager.addListener('model-download-progress',
          (data: { modelId: string; filename: string; percent: number }) => {
            downloadProgress.value[data.modelId] = data.percent;
            if (data.filename) {
              downloadFilename.value[data.modelId] = data.filename;
            }
          }
        );
        await ModelManager.addListener('model-extracted',
          (data: { modelId: string; status: string }) => {
            const idx = models.value.findIndex(m => m.id === data.modelId);
            if (idx !== -1) {
              models.value[idx] = { ...models.value[idx], status: 'available' };
            }
          }
        );
      } else {
        const { listen } = await import('@tauri-apps/api/event');
        await listen<{ filename: string; percent: number }>('model-download-progress', (event) => {
          downloadProgress.value[event.payload.filename] = event.payload.percent;
        });
      }
    } catch (e) {
      console.error('Failed to init model event listeners:', e);
    }
  }

  initEventListeners();

  return {
    models,
    loading,
    loadError,
    downloadProgress,
    downloading,
    downloadErrors,
    downloadFilename,
    loadModels,
    extractBundledModels,
    downloadModel,
    cancelDownload,
    deleteModel,
  };
});
