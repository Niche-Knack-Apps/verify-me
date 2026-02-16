import { defineStore } from 'pinia';
import { ref } from 'vue';

export interface Voice {
  id: string;
  name: string;
}

export interface TTSModel {
  id: string;
  name: string;
  size: string;
  status: 'available' | 'downloadable';
  supportsClone: boolean;
  supportsVoicePrompt: boolean;
  voices: Voice[];
  downloadUrl?: string;
}

export const useModelsStore = defineStore('models', () => {
  const models = ref<TTSModel[]>([]);
  const downloadProgress = ref<Record<string, number>>({});
  const downloading = ref<string | null>(null);

  async function loadModels() {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const result = await invoke<TTSModel[]>('list_models');
      models.value = result;
    } catch (e) {
      console.error('Failed to load models:', e);
    }
  }

  async function downloadModel(modelId: string) {
    const model = models.value.find(m => m.id === modelId);
    if (!model?.downloadUrl) {
      console.error('No download URL for model:', modelId);
      return;
    }

    try {
      const { invoke } = await import('@tauri-apps/api/core');
      downloading.value = modelId;
      downloadProgress.value[modelId] = 0;
      await invoke('download_model', { url: model.downloadUrl, filename: modelId });
      delete downloadProgress.value[modelId];
      downloading.value = null;
      await loadModels();
    } catch (e) {
      delete downloadProgress.value[modelId];
      downloading.value = null;
      console.error('Failed to download model:', e);
    }
  }

  async function deleteModel(id: string) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('delete_model', { modelId: id });
      await loadModels();
    } catch (e) {
      console.error('Failed to delete model:', e);
    }
  }

  async function initEventListeners() {
    try {
      const { listen } = await import('@tauri-apps/api/event');
      await listen<{ filename: string; percent: number }>('model-download-progress', (event) => {
        downloadProgress.value[event.payload.filename] = event.payload.percent;
      });
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
    downloadModel,
    deleteModel,
  };
});
