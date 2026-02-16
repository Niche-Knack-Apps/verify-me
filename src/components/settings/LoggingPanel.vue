<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue';
import { getLogger, type LogStats, type LogEntry } from '@/services/debug-logger';
import Button from '@/components/ui/Button.vue';

const stats = ref<LogStats | null>(null);
const recentLogs = ref<LogEntry[]>([]);
const loading = ref(false);
const filterLevel = ref<'all' | 'info' | 'warn' | 'error' | 'debug'>('all');
const autoRefresh = ref(false);
const clearing = ref(false);
const downloading = ref(false);
const downloadResult = ref<{ success: boolean; filename?: string; error?: string } | null>(null);

let refreshInterval: number | null = null;

const filteredLogs = computed(() => {
  if (filterLevel.value === 'all') {
    return recentLogs.value;
  }
  return recentLogs.value.filter((log) => log.level === filterLevel.value);
});

async function loadData() {
  const logger = getLogger();
  if (!logger) return;

  loading.value = true;
  try {
    stats.value = await logger.getStats();
    recentLogs.value = await logger.getLogs({ limit: 50 });
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
  return `[${level.toUpperCase()}]`;
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
      <span class="stat">ENTRIES: <span class="stat-val">{{ stats.totalCount }}</span></span>
      <span class="stat">SESSIONS: <span class="stat-val">{{ stats.sessionCount }}</span></span>
      <span class="stat">SIZE: <span class="stat-val">{{ formatSize(stats.estimatedSize) }}</span></span>
    </div>

    <!-- Level breakdown -->
    <div v-if="stats" class="level-breakdown">
      <span class="level-tag log-info">INFO:{{ stats.byLevel.info }}</span>
      <span class="level-tag log-warn">WARN:{{ stats.byLevel.warn }}</span>
      <span class="level-tag log-error">ERR:{{ stats.byLevel.error }}</span>
      <span class="level-tag log-debug">DBG:{{ stats.byLevel.debug }}</span>
    </div>

    <!-- Actions -->
    <div class="actions">
      <Button variant="secondary" size="sm" :disabled="downloading" @click="handleDownload">
        {{ downloading ? '[...]' : '[EXPORT]' }}
      </Button>
      <Button variant="ghost" size="sm" :disabled="clearing" @click="handleClear">
        {{ clearing ? '[...]' : '[CLEAR]' }}
      </Button>
      <Button
        variant="ghost"
        size="sm"
        @click="toggleAutoRefresh"
      >
        {{ autoRefresh ? '[AUTO:ON]' : '[AUTO:OFF]' }}
      </Button>
    </div>

    <!-- Download result feedback -->
    <div v-if="downloadResult" class="feedback" :class="downloadResult.success ? 'feedback--ok' : 'feedback--err'">
      <span v-if="downloadResult.success">Saved: {{ downloadResult.filename }}</span>
      <span v-else>ERROR: {{ downloadResult.error }}</span>
    </div>

    <!-- Filter -->
    <div class="filter-row">
      <span class="filter-label">FILTER:</span>
      <select v-model="filterLevel" class="filter-select">
        <option value="all">ALL</option>
        <option value="info">INFO</option>
        <option value="warn">WARN</option>
        <option value="error">ERROR</option>
        <option value="debug">DEBUG</option>
      </select>
      <button class="refresh-btn" @click="loadData" :disabled="loading">
        {{ loading ? '[...]' : '[REF]' }}
      </button>
    </div>

    <!-- Recent logs -->
    <div class="logs-container">
      <div v-if="filteredLogs.length === 0" class="no-logs">
        -- NO LOGS --
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
  font-size: 14px;
  color: var(--crt-dim);
}

.stat-val {
  color: var(--crt-bright);
}

.level-breakdown {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
  font-size: 14px;
}

.level-tag {
  letter-spacing: 0.05em;
}

.log-info { color: var(--crt-text); }
.log-warn { color: var(--crt-warn); }
.log-error { color: var(--crt-error); }
.log-debug { color: var(--crt-dim); }

.actions {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
}

.feedback {
  font-size: 14px;
  padding: 0.375rem 0.5rem;
  margin-bottom: 0.5rem;
  word-break: break-all;
}

.feedback--ok {
  background: rgba(51, 255, 0, 0.05);
  border: 1px solid var(--crt-border);
  color: var(--crt-text);
}

.feedback--err {
  background: rgba(255, 51, 51, 0.05);
  border: 1px solid rgba(255, 51, 51, 0.3);
  color: var(--crt-error);
}

.filter-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.filter-label {
  font-size: 14px;
  color: var(--crt-dim);
}

.filter-select {
  flex: 1;
  height: 28px;
  padding: 0 0.5rem;
  font-size: 16px;
  font-family: 'VT323', monospace;
  background: var(--crt-bg);
  border: 1px solid var(--crt-border);
  border-radius: 0;
  color: var(--crt-text);
}

.refresh-btn {
  height: 28px;
  padding: 0 0.5rem;
  font-family: 'VT323', monospace;
  font-size: 16px;
  background: transparent;
  border: 1px solid var(--crt-border);
  border-radius: 0;
  color: var(--crt-dim);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}
.refresh-btn:hover:not(:disabled) {
  border-color: var(--crt-text);
  color: var(--crt-text);
}
.refresh-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.logs-container {
  max-height: 200px;
  overflow-y: auto;
  background: var(--crt-surface);
  border: 1px solid var(--crt-border);
  padding: 0.5rem;
  font-size: 14px;
}

.no-logs {
  text-align: center;
  padding: 1rem;
  color: var(--crt-dim);
}

.log-entry {
  display: flex;
  gap: 0.5rem;
  padding: 0.125rem 0;
  border-bottom: 1px solid rgba(26, 58, 26, 0.5);
}

.log-entry:last-child {
  border-bottom: none;
}

.log-time {
  color: var(--crt-dim);
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
  color: var(--crt-text);
}
</style>
