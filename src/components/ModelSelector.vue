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

onMounted(() => {
  if (modelsStore.models.length === 0) {
    modelsStore.loadModels();
  }
});
</script>

<template>
  <div class="model-selector">
    <label class="selector-label">&gt; MODEL</label>
    <div class="selector-list">
      <div
        v-for="model in filteredModels"
        :key="model.id"
        class="model-item"
        :class="{ selected: modelValue === model.id, disabled: modelsStore.downloading === model.id }"
        @click="model.status === 'available' && modelsStore.downloading !== model.id && emit('update:modelValue', model.id)"
      >
        <div class="model-info">
          <span class="model-indicator">{{ modelValue === model.id ? '[>]' : '[ ]' }}</span>
          <span class="model-name">{{ model.name }}</span>
          <span class="model-size">{{ model.size }}</span>
        </div>
        <span v-if="modelsStore.downloading === model.id" class="download-progress">
          {{ Math.round(modelsStore.downloadProgress[model.id] ?? 0) }}%
        </span>
        <span v-else-if="model.status === 'available'" class="model-status">
          READY
        </span>
        <button
          v-else
          class="download-btn"
          @click.stop="modelsStore.downloadModel(model.id)"
          title="Download model"
        >
          [GET]
        </button>
      </div>
      <div v-if="filteredModels.length === 0" class="no-models">
        NO MODELS AVAILABLE
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
  font-size: 16px;
  color: var(--crt-dim);
  letter-spacing: 0.05em;
}

.selector-list {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  max-height: 12rem;
  overflow-y: auto;
}

.model-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0.75rem;
  min-height: 44px;
  background: transparent;
  border: 1px solid var(--crt-border);
  border-radius: 0;
  cursor: pointer;
  transition: border-color 0.15s;
}
.model-item:hover {
  border-color: var(--crt-text);
}
.model-item.selected {
  border-color: var(--crt-bright);
  background: rgba(51, 255, 0, 0.05);
}

.model-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.model-indicator {
  color: var(--crt-bright);
  flex-shrink: 0;
}

.model-name {
  font-size: 18px;
  color: var(--crt-text);
}

.model-size {
  font-size: 16px;
  color: var(--crt-dim);
}

.model-status {
  font-size: 14px;
  color: var(--crt-dim);
  letter-spacing: 0.05em;
}

.download-btn {
  padding: 0.25rem 0.75rem;
  min-height: 32px;
  font-family: 'VT323', monospace;
  font-size: 16px;
  background: transparent;
  color: var(--crt-bright);
  border: 1px solid var(--crt-bright);
  border-radius: 0;
  cursor: pointer;
  transition: background 0.15s;
}
.download-btn:hover {
  background: rgba(51, 255, 0, 0.08);
}

.model-item.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.download-progress {
  font-size: 16px;
  color: var(--crt-bright);
}

.no-models {
  padding: 1rem;
  text-align: center;
  color: var(--crt-dim);
  font-size: 16px;
}
</style>
