<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useModelsStore } from '@/stores/models';
import type { TTSModel } from '@/stores/models';

const props = defineProps<{
  modelValue?: string;
  modelFilter?: (model: TTSModel) => boolean;
}>();

const emit = defineEmits<{
  'update:modelValue': [value: string];
}>();

const modelsStore = useModelsStore();

const filteredModels = computed(() => {
  if (props.modelFilter) {
    return modelsStore.models.filter(props.modelFilter);
  }
  return modelsStore.models;
});

function statusLabel(status: TTSModel['status']): string {
  switch (status) {
    case 'available': return 'Available';
    case 'downloadable': return 'Download';
    default: return '';
  }
}

function statusClass(status: TTSModel['status']): string {
  switch (status) {
    case 'available': return 'status-available';
    case 'downloadable': return 'status-download';
    default: return '';
  }
}

onMounted(() => {
  if (modelsStore.models.length === 0) {
    modelsStore.loadModels();
  }
});
</script>

<template>
  <div class="model-selector">
    <label class="selector-label">Model</label>
    <div class="selector-list">
      <div
        v-for="model in filteredModels"
        :key="model.id"
        class="model-item"
        :class="{ selected: modelValue === model.id, disabled: modelsStore.downloading === model.id }"
        @click="model.status === 'available' && modelsStore.downloading !== model.id && emit('update:modelValue', model.id)"
      >
        <div class="model-info">
          <span class="model-name">{{ model.name }}</span>
          <span class="model-size">{{ model.size }}</span>
        </div>
        <span v-if="modelsStore.downloading === model.id" class="download-progress">
          {{ Math.round(modelsStore.downloadProgress[model.id] ?? 0) }}%
        </span>
        <span v-else-if="model.status === 'available'" class="model-status" :class="statusClass(model.status)">
          {{ statusLabel(model.status) }}
        </span>
        <button
          v-else
          class="download-btn"
          @click.stop="modelsStore.downloadModel(model.id)"
          title="Download model"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download
        </button>
      </div>
      <div v-if="filteredModels.length === 0" class="no-models">
        No models available
      </div>
    </div>
  </div>
</template>

<style scoped>
.model-selector {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.selector-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #9ca3af;
}

.selector-list {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
  max-height: 12rem;
  overflow-y: auto;
}

.model-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.625rem 0.75rem;
  min-height: 44px;
  background: var(--color-surface);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.375rem;
  cursor: pointer;
  transition: border-color 0.15s;
}
.model-item:hover {
  border-color: rgba(255, 255, 255, 0.25);
}
.model-item.selected {
  border-color: var(--color-accent);
  background: rgba(34, 211, 238, 0.08);
}

.model-info {
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
}

.model-name {
  font-size: 0.875rem;
  font-weight: 500;
}

.model-size {
  font-size: 0.75rem;
  color: #6b7280;
}

.model-status {
  font-size: 0.75rem;
  padding: 0.125rem 0.5rem;
  border-radius: 9999px;
}
.status-available {
  background: rgba(234, 179, 8, 0.15);
  color: #eab308;
}
.status-download {
  color: #6b7280;
}

.download-btn {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.375rem 0.75rem;
  min-height: 36px;
  font-size: 0.75rem;
  background: transparent;
  color: var(--color-accent);
  border: 1px solid var(--color-accent);
  border-radius: 0.375rem;
  cursor: pointer;
  transition: background 0.15s;
}
.download-btn:hover {
  background: rgba(34, 211, 238, 0.1);
}

.model-item.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.download-progress {
  font-size: 0.75rem;
  color: var(--color-accent);
}

.no-models {
  padding: 1rem;
  text-align: center;
  color: #6b7280;
  font-size: 0.875rem;
}
</style>
