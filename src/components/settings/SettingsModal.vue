<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useSettingsStore } from '@/stores/settings';
import { useModelsStore } from '@/stores/models';
import Button from '@/components/ui/Button.vue';
import Toggle from '@/components/ui/Toggle.vue';
import LoggingPanel from '@/components/settings/LoggingPanel.vue';
import AboutPanel from '@/components/settings/AboutPanel.vue';

const APP_VERSION = '0.2.0';

const emit = defineEmits<{
  close: [];
}>();

const settings = useSettingsStore();
const modelsStore = useModelsStore();
const downloadError = ref<string | null>(null);
const showLargeDownloadConfirm = ref<string | null>(null);
const isAndroid = 'Capacitor' in window;

const LARGE_DOWNLOAD_THRESHOLD_MB = 1024;

async function openModelsDirectory() {
  try {
    const { invoke } = await import('@tauri-apps/api/core');
    await invoke('open_models_directory');
  } catch (e) {
    console.error('Failed to open models directory:', e);
  }
}

function parseSizeMB(size: string): number {
  const match = size.match(/([\d.]+)\s*(GB|MB)/i);
  if (!match) return 0;
  const val = parseFloat(match[1]);
  return match[2].toUpperCase() === 'GB' ? val * 1024 : val;
}

async function handleDownload(modelId: string) {
  const model = modelsStore.models.find(m => m.id === modelId);
  if (isAndroid && model && parseSizeMB(model.size) >= LARGE_DOWNLOAD_THRESHOLD_MB) {
    showLargeDownloadConfirm.value = modelId;
    return;
  }
  await startDownload(modelId);
}

async function startDownload(modelId: string) {
  downloadError.value = null;
  showLargeDownloadConfirm.value = null;
  try {
    await modelsStore.downloadModel(modelId, settings.hfToken || undefined);
  } catch (e) {
    downloadError.value = String(e);
  }
}

onMounted(() => {
  modelsStore.loadModels();
  settings.loadModelsDirectory();
  settings.checkEngineHealth();
});
</script>

<template>
  <div class="modal-overlay" @click.self="emit('close')">
    <div class="modal-container">
      <!-- Header -->
      <div class="modal-header">
        <h2 v-if="settings.isEighties" class="modal-title">&gt; SYSTEM CONFIGURATION_</h2>
        <h2 v-else class="modal-title">Settings</h2>
        <button class="close-btn" @click="emit('close')">
          <template v-if="settings.isEighties">[X]</template>
          <svg v-else xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>

      <!-- Content -->
      <div class="modal-content">
        <!-- Appearance -->
        <div class="config-section">
          <h3 class="section-title">
            {{ settings.isEighties ? '// APPEARANCE' : 'Appearance' }}
          </h3>
          <div class="config-items">
            <div class="config-row">
              <span class="config-key">
                {{ settings.isEighties ? "80's Mode" : "80's Mode" }}
              </span>
              <Toggle
                :model-value="settings.isEighties"
                @update:model-value="settings.toggleTheme()"
              />
            </div>
          </div>
        </div>

        <!-- Models -->
        <div class="config-section">
          <h3 class="section-title">
            {{ settings.isEighties ? '// MODELS' : 'Models' }}
          </h3>
          <div class="config-items">
            <div v-if="!isAndroid">
              <label class="config-label">Models Directory</label>
              <div class="dir-row">
                <div class="dir-display">
                  {{ settings.modelsDirectory || 'Loading...' }}
                </div>
                <Button variant="secondary" size="sm" @click="openModelsDirectory">
                  {{ settings.isEighties ? '[OPEN]' : 'Open' }}
                </Button>
              </div>
            </div>

            <div>
              <label class="config-label">HuggingFace Token</label>
              <div class="hf-token-row">
                <input
                  :type="settings.isEighties ? 'text' : 'password'"
                  :value="settings.hfToken"
                  @input="settings.setHfToken(($event.target as HTMLInputElement).value)"
                  class="hf-token-input"
                  placeholder="hf_..."
                />
                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  class="action-link"
                  title="Create a token at huggingface.co"
                >
                  {{ settings.isEighties ? '[?]' : 'Get token' }}
                </a>
              </div>
              <p class="config-hint">Required to download Qwen 3 TTS model</p>
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
                    <template v-if="settings.isEighties">
                      <span v-if="model.status === 'available'" class="status-ready">[*]</span>
                      <span v-else-if="model.status === 'extracting' || model.status === 'downloading'" class="status-ready">[~]</span>
                      <span v-else class="status-missing">[ ]</span>
                    </template>
                    <template v-else>
                      <span v-if="model.status === 'available'" class="status-dot status-dot--ready" />
                      <span v-else-if="model.status === 'extracting' || model.status === 'downloading'" class="status-dot status-dot--ready" style="opacity:0.5" />
                      <span v-else class="status-dot status-dot--missing" />
                    </template>
                    <span :class="model.status === 'available' ? 'text-ready' : 'text-missing'">
                      {{ model.name }} ({{ model.size }})
                    </span>
                  </span>
                  <div class="model-entry-actions">
                    <!-- Download progress -->
                    <div v-if="modelsStore.downloading === model.id || model.status === 'downloading'" class="download-status">
                      <template v-if="settings.isEighties">
                        <span class="download-bar">
                          {{ '\u2588'.repeat(Math.round((modelsStore.downloadProgress[model.id] ?? 0) / 5)) }}{{ '\u2591'.repeat(20 - Math.round((modelsStore.downloadProgress[model.id] ?? 0) / 5)) }}
                        </span>
                      </template>
                      <template v-else>
                        <div class="download-bar-modern">
                          <div class="download-bar-fill" :style="{ width: `${modelsStore.downloadProgress[model.id] ?? 0}%` }" />
                        </div>
                      </template>
                      <span class="download-pct">{{ Math.round(modelsStore.downloadProgress[model.id] ?? 0) }}%</span>
                      <button
                        v-if="isAndroid"
                        class="action-link action-link--danger cancel-btn"
                        @click="modelsStore.cancelDownload(model.id)"
                      >
                        {{ settings.isEighties ? '[X]' : 'Cancel' }}
                      </button>
                    </div>
                    <!-- Extracting -->
                    <span v-else-if="model.status === 'extracting'" class="action-link" style="cursor:default;opacity:0.7">
                      {{ settings.isEighties ? '[SETUP...]' : 'Extracting...' }}
                    </span>
                    <!-- Bundled but not yet extracted -->
                    <button
                      v-else-if="model.status === 'bundled'"
                      class="action-link"
                      @click="modelsStore.extractBundledModels()"
                    >
                      {{ settings.isEighties ? '[EXTRACT]' : 'Extract' }}
                    </button>
                    <!-- Download button -->
                    <button
                      v-else-if="model.status === 'downloadable'"
                      class="action-link"
                      @click="handleDownload(model.id)"
                    >
                      {{ settings.isEighties ? '[GET]' : 'Download' }}
                    </button>
                    <!-- Delete button -->
                    <button
                      v-else-if="model.status === 'available' && (model.downloadUrl || model.hfRepo)"
                      class="action-link action-link--danger"
                      @click="modelsStore.deleteModel(model.id)"
                    >
                      {{ settings.isEighties ? '[DEL]' : 'Delete' }}
                    </button>
                  </div>
                  <!-- Download filename -->
                  <div v-if="(modelsStore.downloading === model.id || model.status === 'downloading') && modelsStore.downloadFilename[model.id]" class="download-filename">
                    {{ modelsStore.downloadFilename[model.id] }}
                  </div>
                  <!-- Per-model download error with retry -->
                  <div v-if="modelsStore.downloadErrors[model.id]" class="download-error-inline">
                    <span>{{ settings.isEighties ? `ERROR: ${modelsStore.downloadErrors[model.id]}` : modelsStore.downloadErrors[model.id] }}</span>
                    <button class="action-link" @click="handleDownload(model.id)">
                      {{ settings.isEighties ? '[RETRY]' : 'Retry' }}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <p v-if="downloadError" class="download-error">
              {{ settings.isEighties ? `ERROR: ${downloadError}` : downloadError }}
            </p>

            <!-- Large download confirmation -->
            <div v-if="showLargeDownloadConfirm" class="large-download-warning">
              <p class="large-download-text">
                {{ settings.isEighties
                  ? '// WARNING: LARGE DOWNLOAD. USE WIFI.'
                  : 'This model is over 1 GB. Downloading on mobile data may incur charges.'
                }}
              </p>
              <div class="large-download-actions">
                <Button variant="primary" size="sm" @click="startDownload(showLargeDownloadConfirm)">
                  {{ settings.isEighties ? '[PROCEED]' : 'Download anyway' }}
                </Button>
                <Button variant="secondary" size="sm" @click="showLargeDownloadConfirm = null">
                  {{ settings.isEighties ? '[CANCEL]' : 'Cancel' }}
                </Button>
              </div>
            </div>
          </div>
        </div>

        <!-- Engine -->
        <div class="config-section">
          <h3 class="section-title">
            {{ settings.isEighties ? '// ENGINE' : 'Engine' }}
          </h3>
          <div class="config-items">
            <div class="config-row">
              <span class="config-key">Device</span>
              <span class="config-value">
                {{ settings.isEighties ? settings.deviceType.toUpperCase() : settings.deviceType }}
              </span>
            </div>
            <div class="config-row">
              <span class="config-key">Status</span>
              <span :class="settings.engineRunning ? 'config-value--ok' : 'config-value--off'">
                {{ settings.isEighties
                  ? (settings.engineRunning ? 'RUNNING' : 'STOPPED')
                  : (settings.engineRunning ? 'Running' : 'Stopped')
                }}
              </span>
            </div>
            <div class="config-row">
              <span class="config-key">Force CPU</span>
              <Toggle
                :model-value="settings.forceCpu"
                @update:model-value="settings.setForceCpu($event)"
              />
            </div>
            <p class="config-hint">
              {{ settings.isEighties
                ? '// REQUIRES ENGINE RESTART TO TAKE EFFECT'
                : 'Requires engine restart to take effect'
              }}
            </p>

            <!-- Engine Controls -->
            <div class="engine-controls">
              <Button
                v-if="!settings.engineRunning"
                variant="primary"
                size="sm"
                :disabled="settings.engineStarting"
                @click="settings.startEngine()"
              >
                {{ settings.isEighties
                  ? (settings.engineStarting ? '[STARTING...]' : '[START]')
                  : (settings.engineStarting ? 'Starting...' : 'Start Engine')
                }}
              </Button>
              <Button
                v-if="settings.engineRunning"
                variant="secondary"
                size="sm"
                @click="settings.restartEngine()"
              >
                {{ settings.isEighties ? '[RESTART]' : 'Restart' }}
              </Button>
              <Button
                v-if="settings.engineRunning"
                variant="secondary"
                size="sm"
                @click="settings.stopEngine()"
              >
                {{ settings.isEighties ? '[STOP]' : 'Stop' }}
              </Button>
            </div>

            <!-- Engine Error -->
            <p v-if="settings.engineError" class="engine-error">
              {{ settings.isEighties ? `ERROR: ${settings.engineError}` : settings.engineError }}
            </p>

          </div>
        </div>

        <!-- Logging -->
        <div class="config-section">
          <h3 class="section-title">
            {{ settings.isEighties ? '// LOGGING' : 'Logging' }}
          </h3>
          <LoggingPanel />
        </div>

        <!-- About -->
        <div class="config-section">
          <h3 class="section-title">
            {{ settings.isEighties ? '// ABOUT' : 'About' }}
          </h3>
          <AboutPanel
            app-name="Verify Me"
            :app-version="APP_VERSION"
          />
        </div>
      </div>

      <!-- Footer -->
      <div class="modal-footer">
        <Button variant="primary" size="sm" @click="emit('close')">
          {{ settings.isEighties ? '[DONE]' : 'Done' }}
        </Button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: var(--app-overlay-bg);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
  padding: 1rem;
  backdrop-filter: blur(4px);
}

[data-theme="eighties"] .modal-overlay {
  backdrop-filter: none;
}

.modal-container {
  background: var(--app-bg);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  width: 100%;
  max-width: 28rem;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

[data-theme="eighties"] .modal-container {
  border-color: var(--app-accent);
  border-radius: 0;
  box-shadow: 0 0 20px rgba(51, 255, 0, 0.15);
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--app-border);
}

.modal-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--app-text);
}

[data-theme="eighties"] .modal-title {
  font-size: 20px;
  font-weight: 400;
  color: var(--app-accent);
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}

.close-btn {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: var(--app-muted);
  border: 1px solid transparent;
  border-radius: var(--app-radius);
  cursor: pointer;
  font-family: var(--app-font);
  font-size: inherit;
  transition: color 0.15s, background 0.15s, border-color 0.15s;
}
.close-btn:hover {
  color: var(--app-error);
  background: rgba(239, 68, 68, 0.08);
}

[data-theme="eighties"] .close-btn {
  border: 1px solid var(--app-border);
  border-radius: 0;
}
[data-theme="eighties"] .close-btn:hover {
  background: transparent;
  border-color: var(--app-error);
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
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--app-muted);
  margin-bottom: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

[data-theme="eighties"] .section-title {
  font-size: 16px;
  font-weight: 400;
  text-transform: none;
}

.config-items {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.config-label {
  display: block;
  font-size: 0.8125rem;
  color: var(--app-muted);
  margin-bottom: 0.25rem;
}

[data-theme="eighties"] .config-label {
  font-size: 14px;
}

.dir-row {
  display: flex;
  gap: 0.5rem;
}

.dir-display {
  flex: 1;
  padding: 0.25rem 0.5rem;
  font-size: 0.8125rem;
  background: var(--app-surface);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  color: var(--app-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: flex;
  align-items: center;
}

[data-theme="eighties"] .dir-display {
  font-size: 16px;
  border-radius: 0;
}

.empty-text {
  font-size: 0.8125rem;
  color: var(--app-muted);
  font-style: italic;
}

[data-theme="eighties"] .empty-text {
  font-size: 14px;
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
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  padding: 0.375rem 0.5rem;
  background: var(--app-surface);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  font-size: 0.875rem;
}

[data-theme="eighties"] .model-entry {
  font-size: 16px;
  border-radius: 0;
}

.model-entry-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.status-ready {
  color: var(--app-accent);
}
.status-missing {
  color: var(--app-muted);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.status-dot--ready {
  background: var(--app-success);
}
.status-dot--missing {
  background: var(--app-muted);
}

.text-ready {
  color: var(--app-text);
}
.text-missing {
  color: var(--app-muted);
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
  color: var(--app-text);
  text-shadow: none;
  letter-spacing: 0;
}

.download-bar-modern {
  width: 60px;
  height: 4px;
  background: var(--app-border);
  border-radius: 2px;
  overflow: hidden;
}

.download-bar-fill {
  height: 100%;
  background: var(--app-accent);
  border-radius: 2px;
  transition: width 0.3s ease;
}

.download-pct {
  font-size: 0.75rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .download-pct {
  font-size: 14px;
}

.action-link {
  font-family: var(--app-font);
  font-size: 0.8125rem;
  background: transparent;
  color: var(--app-accent);
  border: none;
  cursor: pointer;
  padding: 0;
  transition: opacity 0.15s;
}
.action-link:hover {
  opacity: 0.8;
}

[data-theme="eighties"] .action-link {
  font-size: 16px;
}
[data-theme="eighties"] .action-link:hover {
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.5);
  opacity: 1;
}

.action-link--danger {
  color: var(--app-muted);
}
.action-link--danger:hover {
  color: var(--app-error);
}

[data-theme="eighties"] .action-link--danger:hover {
  text-shadow: 0 0 6px rgba(255, 51, 51, 0.4);
}

.config-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.875rem;
}

[data-theme="eighties"] .config-row {
  font-size: 16px;
}

.config-key {
  color: var(--app-muted);
}

.config-value {
  color: var(--app-text);
}

.config-value--ok {
  color: var(--app-success);
}

[data-theme="eighties"] .config-value--ok {
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}

.config-value--off {
  color: var(--app-muted);
}

.hf-token-row {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.hf-token-input {
  flex: 1;
  padding: 0.375rem 0.5rem;
  min-height: 36px;
  font-family: var(--app-font);
  font-size: 0.8125rem;
  background: var(--app-bg);
  color: var(--app-text);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  caret-color: var(--app-accent);
}
.hf-token-input:focus {
  outline: none;
  border-color: var(--app-accent);
  box-shadow: var(--app-focus-ring);
}
.hf-token-input::placeholder {
  color: var(--app-muted);
}

[data-theme="eighties"] .hf-token-input {
  font-size: 16px;
  border-radius: 0;
  text-shadow: var(--app-glow);
}

.config-hint {
  font-size: 0.6875rem;
  color: var(--app-muted);
  margin-top: 0.25rem;
}

[data-theme="eighties"] .config-hint {
  font-size: 12px;
}

.download-error {
  font-size: 0.8125rem;
  color: var(--app-error);
  padding: 0.375rem 0.5rem;
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--app-radius);
}

[data-theme="eighties"] .download-error {
  font-size: 14px;
  border-radius: 0;
}

.engine-controls {
  display: flex;
  gap: 0.5rem;
}

.engine-error {
  font-size: 0.8125rem;
  color: var(--app-error);
  padding: 0.375rem 0.5rem;
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--app-radius);
}

[data-theme="eighties"] .engine-error {
  font-size: 14px;
  border-radius: 0;
}

.cancel-btn {
  margin-left: 0.25rem;
}

.download-filename {
  width: 100%;
  font-size: 0.6875rem;
  color: var(--app-muted);
  padding: 0.125rem 0.5rem 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

[data-theme="eighties"] .download-filename {
  font-size: 12px;
}

.download-error-inline {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  font-size: 0.75rem;
  color: var(--app-error);
  padding: 0.25rem 0.5rem 0;
}

[data-theme="eighties"] .download-error-inline {
  font-size: 12px;
}

.large-download-warning {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.5rem;
  background: rgba(234, 179, 8, 0.08);
  border: 1px solid rgba(234, 179, 8, 0.3);
  border-radius: var(--app-radius);
}

[data-theme="eighties"] .large-download-warning {
  border-radius: 0;
}

.large-download-text {
  font-size: 0.8125rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .large-download-text {
  font-size: 14px;
}

.large-download-actions {
  display: flex;
  gap: 0.5rem;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  padding: 0.75rem 1rem;
  border-top: 1px solid var(--app-border);
}
</style>
