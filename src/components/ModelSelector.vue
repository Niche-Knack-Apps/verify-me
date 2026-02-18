<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useModelsStore } from '@/stores/models';
import { useSettingsStore } from '@/stores/settings';
import type { TTSModel } from '@/stores/models';

const props = defineProps<{
  modelValue?: string;
  modelFilter?: (model: TTSModel) => boolean;
}>();

const emit = defineEmits<{
  'update:modelValue': [value: string];
}>();

const modelsStore = useModelsStore();
const settings = useSettingsStore();

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
    <label class="selector-label">
      {{ settings.isEighties ? '> MODEL' : 'Model' }}
    </label>
    <div class="selector-list">
      <div
        v-for="model in filteredModels"
        :key="model.id"
        class="model-item"
        :class="{ selected: modelValue === model.id, disabled: model.status === 'downloading' || model.status === 'extracting' || model.status === 'bundled' || modelsStore.downloading === model.id }"
        @click="model.status === 'available' && modelsStore.downloading !== model.id && emit('update:modelValue', model.id)"
      >
        <div class="model-info">
          <span v-if="settings.isEighties" class="model-indicator">{{ modelValue === model.id ? '[>]' : '[ ]' }}</span>
          <span v-else class="model-indicator">
            <span class="radio-dot" :class="{ 'radio-dot--active': modelValue === model.id }" />
          </span>
          <span class="model-name">{{ model.name }}</span>
          <span class="model-size">{{ model.size }}</span>
        </div>
        <span v-if="model.status === 'downloading' || modelsStore.downloading === model.id" class="download-progress">
          {{ Math.round(modelsStore.downloadProgress[model.id] ?? 0) }}%
        </span>
        <span v-else-if="model.status === 'extracting'" class="download-progress">
          {{ settings.isEighties ? 'EXTRACTING...' : 'Extracting...' }}
        </span>
        <button
          v-else-if="model.status === 'bundled'"
          class="download-btn"
          @click.stop="modelsStore.extractBundledModels()"
          title="Extract bundled model"
        >
          {{ settings.isEighties ? '[EXTRACT]' : 'Extract' }}
        </button>
        <span v-else-if="model.status === 'available'" class="model-status">
          {{ settings.isEighties ? 'READY' : 'Ready' }}
        </span>
        <button
          v-else
          class="download-btn"
          @click.stop="modelsStore.downloadModel(model.id)"
          title="Download model"
        >
          {{ settings.isEighties ? '[GET]' : 'Download' }}
        </button>
      </div>
      <div v-if="filteredModels.length === 0" class="no-models">
        {{ settings.isEighties ? 'NO MODELS AVAILABLE' : 'No models available' }}
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
  font-size: 0.8125rem;
  color: var(--app-muted);
  font-weight: 500;
}

[data-theme="eighties"] .selector-label {
  font-size: 16px;
  font-weight: 400;
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
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}
.model-item:hover {
  border-color: var(--app-muted);
  background: var(--app-accent-hover-bg);
}
.model-item.selected {
  border-color: var(--app-accent);
  background: var(--app-accent-hover-bg);
}

[data-theme="eighties"] .model-item:hover {
  background: transparent;
}
[data-theme="eighties"] .model-item.selected {
  background: rgba(51, 255, 0, 0.05);
}

.model-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.model-indicator {
  color: var(--app-accent);
  flex-shrink: 0;
}

.radio-dot {
  display: block;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 2px solid var(--app-border);
  transition: border-color 0.15s;
}
.radio-dot--active {
  border-color: var(--app-accent);
  background: var(--app-accent);
  box-shadow: inset 0 0 0 3px var(--app-bg);
}

.model-name {
  color: var(--app-text);
}

.model-size {
  font-size: 0.8125rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .model-size {
  font-size: 16px;
}

.model-status {
  font-size: 0.75rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .model-status {
  font-size: 14px;
  letter-spacing: 0.05em;
}

.download-btn {
  padding: 0.25rem 0.75rem;
  min-height: 32px;
  font-family: var(--app-font);
  font-size: 0.8125rem;
  background: var(--app-accent);
  color: #fff;
  border: none;
  border-radius: var(--app-radius);
  cursor: pointer;
  transition: filter 0.15s;
}
.download-btn:hover {
  filter: brightness(1.1);
}

[data-theme="eighties"] .download-btn {
  font-size: 16px;
  background: transparent;
  color: var(--app-accent);
  border: 1px solid var(--app-accent);
  border-radius: 0;
}
[data-theme="eighties"] .download-btn:hover {
  background: rgba(51, 255, 0, 0.08);
  filter: none;
}

.model-item.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.download-progress {
  font-size: 0.8125rem;
  color: var(--app-accent);
}

[data-theme="eighties"] .download-progress {
  font-size: 16px;
}

.no-models {
  padding: 1rem;
  text-align: center;
  color: var(--app-muted);
  font-size: 0.875rem;
}

[data-theme="eighties"] .no-models {
  font-size: 16px;
}
</style>
