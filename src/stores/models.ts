import { defineStore } from 'pinia';
import { ref } from 'vue';

export interface TTSModel {
  id: string;
  name: string;
  size: string;
  status: 'loaded' | 'available' | 'downloadable';
  supportsClone: boolean;
}

export const useModelsStore = defineStore('models', () => {
  const models = ref<TTSModel[]>([]);
  const downloadProgress = ref<Record<string, number>>({});

  async function loadModels() {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const result = await invoke<TTSModel[]>('list_models');
      models.value = result;
    } catch (e) {
      console.error('Failed to load models:', e);
    }
  }

  async function downloadModel(url: string, filename: string) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      downloadProgress.value[filename] = 0;
      await invoke('download_model', { url, filename });
      delete downloadProgress.value[filename];
      await loadModels();
    } catch (e) {
      delete downloadProgress.value[filename];
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

  return {
    models,
    downloadProgress,
    loadModels,
    downloadModel,
    deleteModel,
  };
});
