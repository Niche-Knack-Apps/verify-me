<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue';
import { getLogger, type LogStats, type LogEntry } from '@/services/debug-logger';
import { useSettingsStore } from '@/stores/settings';
import Button from '@/components/ui/Button.vue';

const settings = useSettingsStore();
const stats = ref<LogStats | null>(null);
const recentLogs = ref<LogEntry[]>([]);
const loading = ref(false);
const filterLevel = ref<'all' | 'info' | 'warn' | 'error' | 'debug'>('all');
const autoRefresh = ref(false);
const clearing = ref(false);
const downloading = ref(false);
const downloadResult = ref<{ success: boolean; filename?: string; error?: string } | null>(null);
const isDev = import.meta.env.DEV;

// Checkpoint comparison state (dev only)
interface CheckpointEntry {
  engine: string;
  stage: string;
  timestamp: number;
  data: Record<string, unknown>;
}
const onnxCheckpoints = ref<CheckpointEntry[]>([]);
const safetensorsCheckpoints = ref<CheckpointEntry[]>([]);

let refreshInterval: number | null = null;

const filteredLogs = computed(() => {
  if (filterLevel.value === 'all') {
    return recentLogs.value;
  }
  return recentLogs.value.filter((log) => log.level === filterLevel.value);
});

async function refreshCheckpoints() {
  if (!isDev) return;
  try {
    const { getCheckpoints } = await import('@/services/checkpoint-listener');
    onnxCheckpoints.value = getCheckpoints('onnx');
    safetensorsCheckpoints.value = getCheckpoints('safetensors');
  } catch {
    // Checkpoint listener may not be initialized
  }
}

async function loadData() {
  const logger = getLogger();
  if (!logger) return;

  loading.value = true;
  try {
    stats.value = await logger.getStats();
    recentLogs.value = await logger.getLogs({ limit: 50 });
    await refreshCheckpoints();
  } finally {
    loading.value = false;
  }
}

async function handleDownload() {
  const logger = getLogger();
  if (!logger) return;

  downloading.value = true;
  downloadResult.value = null;
  try {
    const result = await logger.downloadLogs();
    downloadResult.value = result;
    if (!result.success) {
      console.error('Failed to download logs:', result.error);
    }
    if (result.success) {
      setTimeout(() => {
        if (downloadResult.value?.success) {
          downloadResult.value = null;
        }
      }, 5000);
    }
  } catch (e) {
    downloadResult.value = { success: false, error: String(e) };
  } finally {
    downloading.value = false;
  }
}

async function handleClear() {
  const logger = getLogger();
  if (!logger) return;

  clearing.value = true;
  try {
    await logger.clearLogs();
    await loadData();
  } finally {
    clearing.value = false;
  }
}

function toggleAutoRefresh() {
  autoRefresh.value = !autoRefresh.value;
  if (autoRefresh.value) {
    refreshInterval = window.setInterval(loadData, 3000);
  } else if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}

function formatTimestamp(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleTimeString();
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getLevelTag(level: string): string {
  return settings.isEighties ? `[${level.toUpperCase()}]` : level.toUpperCase();
}

function formatCheckpointData(data: Record<string, unknown>): string {
  const str = JSON.stringify(data, null, 0);
  return str.length > 120 ? str.slice(0, 120) + '...' : str;
}

function getLevelClass(level: string): string {
  switch (level) {
    case 'error': return 'log-error';
    case 'warn': return 'log-warn';
    case 'debug': return 'log-debug';
    default: return 'log-info';
  }
}

onMounted(() => {
  loadData();
});

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
});
</script>

<template>
  <div class="logging-panel">
    <!-- Stats -->
    <div v-if="stats" class="stats-row">
      <span class="stat">
        {{ settings.isEighties ? 'ENTRIES:' : 'Entries:' }}
        <span class="stat-val">{{ stats.totalCount }}</span>
      </span>
      <span class="stat">
        {{ settings.isEighties ? 'SESSIONS:' : 'Sessions:' }}
        <span class="stat-val">{{ stats.sessionCount }}</span>
      </span>
      <span class="stat">
        {{ settings.isEighties ? 'SIZE:' : 'Size:' }}
        <span class="stat-val">{{ formatSize(stats.estimatedSize) }}</span>
      </span>
    </div>

    <!-- Level breakdown -->
    <div v-if="stats" class="level-breakdown">
      <span class="level-tag log-info">{{ settings.isEighties ? 'INFO' : 'Info' }}:{{ stats.byLevel.info }}</span>
      <span class="level-tag log-warn">{{ settings.isEighties ? 'WARN' : 'Warn' }}:{{ stats.byLevel.warn }}</span>
      <span class="level-tag log-error">{{ settings.isEighties ? 'ERR' : 'Err' }}:{{ stats.byLevel.error }}</span>
      <span class="level-tag log-debug">{{ settings.isEighties ? 'DBG' : 'Debug' }}:{{ stats.byLevel.debug }}</span>
    </div>

    <!-- Actions -->
    <div class="actions">
      <Button variant="secondary" size="sm" :disabled="downloading" @click="handleDownload">
        {{ downloading ? '...' : (settings.isEighties ? '[EXPORT]' : 'Export') }}
      </Button>
      <Button variant="ghost" size="sm" :disabled="clearing" @click="handleClear">
        {{ clearing ? '...' : (settings.isEighties ? '[CLEAR]' : 'Clear') }}
      </Button>
      <Button
        variant="ghost"
        size="sm"
        @click="toggleAutoRefresh"
      >
        {{ settings.isEighties
          ? (autoRefresh ? '[AUTO:ON]' : '[AUTO:OFF]')
          : (autoRefresh ? 'Auto: On' : 'Auto: Off')
        }}
      </Button>
    </div>

    <!-- Download result feedback -->
    <div v-if="downloadResult" class="feedback" :class="downloadResult.success ? 'feedback--ok' : 'feedback--err'">
      <span v-if="downloadResult.success">Saved: {{ downloadResult.filename }}</span>
      <span v-else>{{ settings.isEighties ? 'ERROR:' : 'Error:' }} {{ downloadResult.error }}</span>
    </div>

    <!-- Filter -->
    <div class="filter-row">
      <span class="filter-label">{{ settings.isEighties ? 'FILTER:' : 'Filter:' }}</span>
      <select v-model="filterLevel" class="filter-select">
        <option value="all">{{ settings.isEighties ? 'ALL' : 'All' }}</option>
        <option value="info">{{ settings.isEighties ? 'INFO' : 'Info' }}</option>
        <option value="warn">{{ settings.isEighties ? 'WARN' : 'Warn' }}</option>
        <option value="error">{{ settings.isEighties ? 'ERROR' : 'Error' }}</option>
        <option value="debug">{{ settings.isEighties ? 'DEBUG' : 'Debug' }}</option>
      </select>
      <button class="refresh-btn" @click="loadData" :disabled="loading">
        {{ loading ? '...' : (settings.isEighties ? '[REF]' : 'Refresh') }}
      </button>
    </div>

    <!-- Recent logs -->
    <div class="logs-container">
      <div v-if="filteredLogs.length === 0" class="no-logs">
        {{ settings.isEighties ? '-- NO LOGS --' : 'No logs yet' }}
      </div>
      <div
        v-for="(log, index) in filteredLogs"
        :key="index"
        class="log-entry"
      >
        <span class="log-time">{{ formatTimestamp(log.timestamp) }}</span>
        <span class="log-level" :class="getLevelClass(log.level)">{{ getLevelTag(log.level) }}</span>
        <span class="log-message">{{ log.message }}</span>
      </div>
    </div>

    <!-- Checkpoint Comparison (dev only) -->
    <template v-if="isDev && (onnxCheckpoints.length > 0 || safetensorsCheckpoints.length > 0)">
      <div class="checkpoint-section">
        <h4 class="checkpoint-title">
          {{ settings.isEighties ? '> CHECKPOINTS' : 'Checkpoints' }}
        </h4>
        <div class="checkpoint-columns">
          <div class="checkpoint-col">
            <div class="checkpoint-col-header">
              {{ settings.isEighties ? 'ONNX' : 'ONNX' }}
              <span class="checkpoint-count">({{ onnxCheckpoints.length }})</span>
            </div>
            <div v-if="onnxCheckpoints.length === 0" class="no-checkpoints">
              {{ settings.isEighties ? '-- NONE --' : 'No checkpoints' }}
            </div>
            <div
              v-for="(cp, i) in onnxCheckpoints"
              :key="'onnx-' + i"
              class="checkpoint-entry"
            >
              <span class="checkpoint-stage">{{ cp.stage }}</span>
              <span class="checkpoint-data">{{ formatCheckpointData(cp.data) }}</span>
            </div>
          </div>
          <div class="checkpoint-col">
            <div class="checkpoint-col-header">
              {{ settings.isEighties ? 'SAFETENSORS' : 'Safetensors' }}
              <span class="checkpoint-count">({{ safetensorsCheckpoints.length }})</span>
            </div>
            <div v-if="safetensorsCheckpoints.length === 0" class="no-checkpoints">
              {{ settings.isEighties ? '-- NONE --' : 'No checkpoints' }}
            </div>
            <div
              v-for="(cp, i) in safetensorsCheckpoints"
              :key="'st-' + i"
              class="checkpoint-entry"
            >
              <span class="checkpoint-stage">{{ cp.stage }}</span>
              <span class="checkpoint-data">{{ formatCheckpointData(cp.data) }}</span>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.logging-panel {
  padding: 0.5rem 0;
}

.stats-row {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
  font-size: 0.8125rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .stats-row {
  font-size: 14px;
}

.stat-val {
  color: var(--app-accent);
}

.level-breakdown {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
  font-size: 0.8125rem;
}

[data-theme="eighties"] .level-breakdown {
  font-size: 14px;
}

.level-tag {
  letter-spacing: 0.05em;
}

.log-info { color: var(--app-text); }
.log-warn { color: var(--app-warn); }
.log-error { color: var(--app-error); }
.log-debug { color: var(--app-muted); }

.actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
}

.feedback {
  font-size: 0.8125rem;
  padding: 0.375rem 0.5rem;
  margin-bottom: 0.5rem;
  word-break: break-all;
  border-radius: var(--app-radius);
}

[data-theme="eighties"] .feedback {
  font-size: 14px;
  border-radius: 0;
}

.feedback--ok {
  background: rgba(34, 197, 94, 0.08);
  border: 1px solid var(--app-border);
  color: var(--app-text);
}

[data-theme="eighties"] .feedback--ok {
  background: rgba(51, 255, 0, 0.05);
}

.feedback--err {
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: var(--app-error);
}

[data-theme="eighties"] .feedback--err {
  background: rgba(255, 51, 51, 0.05);
  border-color: rgba(255, 51, 51, 0.3);
}

.filter-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.filter-label {
  font-size: 0.8125rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .filter-label {
  font-size: 14px;
}

.filter-select {
  flex: 1;
  height: 28px;
  padding: 0 0.5rem;
  font-size: 0.8125rem;
  font-family: var(--app-font);
  background: var(--app-bg);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  color: var(--app-text);
}

[data-theme="eighties"] .filter-select {
  font-size: 16px;
  border-radius: 0;
}

.refresh-btn {
  height: 28px;
  padding: 0 0.5rem;
  font-family: var(--app-font);
  font-size: 0.8125rem;
  background: transparent;
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  color: var(--app-muted);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}
.refresh-btn:hover:not(:disabled) {
  border-color: var(--app-accent);
  color: var(--app-accent);
}
.refresh-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

[data-theme="eighties"] .refresh-btn {
  font-size: 16px;
  border-radius: 0;
}

.logs-container {
  max-height: 200px;
  overflow-y: auto;
  background: var(--app-surface);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  padding: 0.5rem;
  font-size: 0.75rem;
}

[data-theme="eighties"] .logs-container {
  font-size: 14px;
  border-radius: 0;
}

.no-logs {
  text-align: center;
  padding: 1rem;
  color: var(--app-muted);
}

.log-entry {
  display: flex;
  gap: 0.5rem;
  padding: 0.125rem 0;
  border-bottom: 1px solid var(--app-border);
}

.log-entry:last-child {
  border-bottom: none;
}

.log-time {
  color: var(--app-muted);
  flex-shrink: 0;
}

.log-level {
  width: 3.5rem;
  flex-shrink: 0;
}

.log-message {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--app-text);
}

/* Checkpoint comparison styles */
.checkpoint-section {
  margin-top: 1rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--app-border);
}

.checkpoint-title {
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--app-text);
  margin: 0 0 0.5rem 0;
}

[data-theme="eighties"] .checkpoint-title {
  font-size: 14px;
  font-weight: 400;
  letter-spacing: 0.05em;
}

.checkpoint-columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
}

.checkpoint-col {
  background: var(--app-surface);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  padding: 0.375rem;
  max-height: 200px;
  overflow-y: auto;
  font-size: 0.75rem;
}

[data-theme="eighties"] .checkpoint-col {
  font-size: 14px;
  border-radius: 0;
}

.checkpoint-col-header {
  font-weight: 600;
  color: var(--app-accent);
  padding-bottom: 0.25rem;
  border-bottom: 1px solid var(--app-border);
  margin-bottom: 0.25rem;
}

[data-theme="eighties"] .checkpoint-col-header {
  font-weight: 400;
}

.checkpoint-count {
  color: var(--app-muted);
  font-weight: 400;
}

.no-checkpoints {
  color: var(--app-muted);
  text-align: center;
  padding: 0.5rem;
}

.checkpoint-entry {
  display: flex;
  flex-direction: column;
  padding: 0.125rem 0;
  border-bottom: 1px solid var(--app-border);
}

.checkpoint-entry:last-child {
  border-bottom: none;
}

.checkpoint-stage {
  color: var(--app-accent);
  font-weight: 500;
}

[data-theme="eighties"] .checkpoint-stage {
  font-weight: 400;
}

.checkpoint-data {
  color: var(--app-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
