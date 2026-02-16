<script setup lang="ts">
import { onMounted } from 'vue';
import { useSettingsStore } from '@/stores/settings';
import { useModelsStore } from '@/stores/models';
import Button from '@/components/ui/Button.vue';
import LoggingPanel from '@/components/settings/LoggingPanel.vue';
import AboutPanel from '@/components/settings/AboutPanel.vue';

const APP_VERSION = '0.1.0';

const emit = defineEmits<{
  close: [];
}>();

const settings = useSettingsStore();
const modelsStore = useModelsStore();

async function openModelsDirectory() {
  try {
    const { invoke } = await import('@tauri-apps/api/core');
    await invoke('open_models_directory');
  } catch (e) {
    console.error('Failed to open models directory:', e);
  }
}

onMounted(() => {
  modelsStore.loadModels();
  settings.loadModelsDirectory();
});
</script>

<template>
  <div class="modal-overlay" @click.self="emit('close')">
    <div class="modal-container">
      <!-- Header -->
      <div class="modal-header">
        <h2 class="modal-title">&gt; SYSTEM CONFIGURATION_</h2>
        <button class="close-btn" @click="emit('close')">[X]</button>
      </div>

      <!-- Content -->
      <div class="modal-content">
        <!-- Models -->
        <div class="config-section">
          <h3 class="section-title">// MODELS</h3>
          <div class="config-items">
            <div>
              <label class="config-label">Models Directory</label>
              <div class="dir-row">
                <div class="dir-display">
                  {{ settings.modelsDirectory || 'Loading...' }}
                </div>
                <Button variant="secondary" size="sm" @click="openModelsDirectory">
                  [OPEN]
                </Button>
              </div>
            </div>

            <div>
              <label class="config-label">Available Models</label>
              <div v-if="modelsStore.models.length === 0" class="empty-text">
                No models found
              </div>
              <div v-else class="model-list">
                <div
                  v-for="model in modelsStore.models"
                  :key="model.id"
                  class="model-entry"
                >
                  <span class="model-entry-info">
                    <span v-if="model.status === 'available'" class="status-ready">[*]</span>
                    <span v-else class="status-missing">[ ]</span>
                    <span :class="model.status === 'available' ? 'text-ready' : 'text-missing'">
                      {{ model.name }} ({{ model.size }})
                    </span>
                  </span>
                  <div class="model-entry-actions">
                    <!-- Download progress -->
                    <div v-if="modelsStore.downloading === model.id" class="download-status">
                      <span class="download-bar">
                        {{ '\u2588'.repeat(Math.round((modelsStore.downloadProgress[model.id] ?? 0) / 5)) }}{{ '\u2591'.repeat(20 - Math.round((modelsStore.downloadProgress[model.id] ?? 0) / 5)) }}
                      </span>
                      <span class="download-pct">{{ Math.round(modelsStore.downloadProgress[model.id] ?? 0) }}%</span>
                    </div>
                    <!-- Download button -->
                    <button
                      v-else-if="model.status === 'downloadable'"
                      class="action-link"
                      @click="modelsStore.downloadModel(model.id)"
                    >
                      [GET]
                    </button>
                    <!-- Delete button -->
                    <button
                      v-else-if="model.status === 'available' && model.downloadUrl"
                      class="action-link action-link--danger"
                      @click="modelsStore.deleteModel(model.id)"
                    >
                      [DEL]
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Engine -->
        <div class="config-section">
          <h3 class="section-title">// ENGINE</h3>
          <div class="config-items">
            <div class="config-row">
              <span class="config-key">Device</span>
              <span class="config-value">{{ settings.deviceType.toUpperCase() }}</span>
            </div>
            <div class="config-row">
              <span class="config-key">Status</span>
              <span :class="settings.engineRunning ? 'config-value--ok' : 'config-value--off'">
                {{ settings.engineRunning ? 'RUNNING' : 'STOPPED' }}
              </span>
            </div>
          </div>
        </div>

        <!-- Logging -->
        <div class="config-section">
          <h3 class="section-title">// LOGGING</h3>
          <LoggingPanel />
        </div>

        <!-- About -->
        <div class="config-section">
          <h3 class="section-title">// ABOUT</h3>
          <AboutPanel
            app-name="Verify Me"
            :app-version="APP_VERSION"
          />
        </div>
      </div>

      <!-- Footer -->
      <div class="modal-footer">
        <Button variant="primary" size="sm" @click="emit('close')">
          [DONE]
        </Button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
  padding: 1rem;
}

.modal-container {
  background: var(--crt-bg);
  border: 1px solid var(--crt-bright);
  border-radius: 0;
  box-shadow: 0 0 20px rgba(51, 255, 0, 0.15);
  width: 100%;
  max-width: 28rem;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--crt-border);
}

.modal-title {
  font-size: 20px;
  font-weight: 400;
  color: var(--crt-bright);
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}

.close-btn {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: var(--crt-dim);
  border: 1px solid var(--crt-border);
  border-radius: 0;
  cursor: pointer;
  font-family: 'VT323', monospace;
  font-size: 18px;
  transition: color 0.15s, border-color 0.15s;
}
.close-btn:hover {
  color: var(--crt-error);
  border-color: var(--crt-error);
}

.modal-content {
  padding: 1rem 1.5rem;
  overflow-y: auto;
  flex: 1;
}

.config-section {
  margin-bottom: 1.5rem;
}
.config-section:last-child {
  margin-bottom: 0;
}

.section-title {
  font-size: 16px;
  font-weight: 400;
  color: var(--crt-dim);
  margin-bottom: 0.75rem;
  letter-spacing: 0.05em;
}

.config-items {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.config-label {
  display: block;
  font-size: 14px;
  color: var(--crt-dim);
  margin-bottom: 0.25rem;
}

.dir-row {
  display: flex;
  gap: 0.5rem;
}

.dir-display {
  flex: 1;
  padding: 0.25rem 0.5rem;
  font-size: 16px;
  background: var(--crt-surface);
  border: 1px solid var(--crt-border);
  color: var(--crt-dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: flex;
  align-items: center;
}

.empty-text {
  font-size: 14px;
  color: var(--crt-dim);
  font-style: italic;
}

.model-list {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  max-height: 10rem;
  overflow-y: auto;
}

.model-entry {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.375rem 0.5rem;
  background: var(--crt-surface);
  border: 1px solid var(--crt-border);
  font-size: 16px;
}

.model-entry-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.status-ready {
  color: var(--crt-bright);
}
.status-missing {
  color: var(--crt-dim);
}

.text-ready {
  color: var(--crt-text);
}
.text-missing {
  color: var(--crt-dim);
}

.model-entry-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.download-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.download-bar {
  font-size: 12px;
  color: var(--crt-text);
  text-shadow: none;
  letter-spacing: 0;
}

.download-pct {
  font-size: 14px;
  color: var(--crt-dim);
}

.action-link {
  font-family: 'VT323', monospace;
  font-size: 16px;
  background: transparent;
  color: var(--crt-bright);
  border: none;
  cursor: pointer;
  padding: 0;
  transition: text-shadow 0.15s;
}
.action-link:hover {
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.5);
}
.action-link--danger {
  color: var(--crt-dim);
}
.action-link--danger:hover {
  color: var(--crt-error);
  text-shadow: 0 0 6px rgba(255, 51, 51, 0.4);
}

.config-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 16px;
}

.config-key {
  color: var(--crt-dim);
}

.config-value {
  color: var(--crt-text);
}

.config-value--ok {
  color: var(--crt-bright);
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}
.config-value--off {
  color: var(--crt-dim);
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  padding: 0.75rem 1rem;
  border-top: 1px solid var(--crt-border);
}
</style>
