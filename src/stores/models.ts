import { defineStore } from 'pinia';
import { ref } from 'vue';

function isCapacitor(): boolean {
  return 'Capacitor' in window;
}

let _modelManager: any = null;
async function getModelManager() {
  if (!_modelManager) {
    const { registerPlugin } = await import('@capacitor/core');
    _modelManager = registerPlugin('ModelManager');
  }
  return _modelManager;
}

export interface Voice {
  id: string;
  name: string;
}

export interface TTSModel {
  id: string;
  name: string;
  size: string;
  status: 'available' | 'downloadable' | 'bundled' | 'extracting';
  supportsClone: boolean;
  supportsVoicePrompt: boolean;
  supportsVoiceDesign: boolean;
  voices: Voice[];
  downloadUrl?: string;
  hfRepo?: string;
}

export const useModelsStore = defineStore('models', () => {
  const models = ref<TTSModel[]>([]);
  const downloadProgress = ref<Record<string, number>>({});
  const downloading = ref<string | null>(null);

  async function loadModels() {
    try {
      if (isCapacitor()) {
        const ModelManager = await getModelManager();
        const result = await ModelManager.listModels();
        // Map from plugin format to TTSModel[]
        models.value = (result.models ?? []).map((m: any) => ({
          id: m.id,
          name: m.name,
          size: m.size,
          status: m.status,
          supportsClone: m.supportsClone ?? false,
          supportsVoicePrompt: m.supportsVoicePrompt ?? false,
          supportsVoiceDesign: m.supportsVoiceDesign ?? false,
          voices: m.voices ?? [],
          hfRepo: m.hfRepo,
        }));
      } else {
        const { invoke } = await import('@tauri-apps/api/core');
        const result = await invoke<TTSModel[]>('list_models');
        models.value = result;
      }
    } catch (e) {
      console.error('Failed to load models:', e);
    }
  }

  async function extractBundledModels() {
    if (!isCapacitor()) return;
    models.value = models.value.map(m =>
      m.status === 'bundled' ? { ...m, status: 'extracting' as const } : m
    );
    try {
      const ModelManager = await getModelManager();
      await ModelManager.extractBundledModels();
    } catch (e) {
      console.error('Bundled extraction failed:', e);
      models.value = models.value.map(m =>
        m.status === 'extracting' ? { ...m, status: 'bundled' as const } : m
      );
    }
    await loadModels();
  }

  async function downloadModel(modelId: string, hfToken?: string) {
    const model = models.value.find(m => m.id === modelId);
    if (!model) {
      console.error('Model not found:', modelId);
      return;
    }

    try {
      downloading.value = modelId;
      downloadProgress.value[modelId] = 0;

      if (isCapacitor()) {
        const ModelManager = await getModelManager();
        await ModelManager.downloadModel({
          modelId,
          hfToken: hfToken || null,
        });
      } else {
        const { invoke } = await import('@tauri-apps/api/core');

        if (model.hfRepo) {
          await invoke('download_hf_model', {
            repoId: model.hfRepo,
            modelId,
            token: hfToken || null,
          });
        } else if (model.downloadUrl) {
          await invoke('download_model', { url: model.downloadUrl, filename: modelId });
        } else {
          throw new Error('No download source for model');
        }
      }

      delete downloadProgress.value[modelId];
      downloading.value = null;
      await loadModels();
    } catch (e) {
      delete downloadProgress.value[modelId];
      downloading.value = null;
      console.error('Failed to download model:', e);
      throw e;
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
    downloadProgress,
    downloading,
    loadModels,
    extractBundledModels,
    downloadModel,
    deleteModel,
  };
});
