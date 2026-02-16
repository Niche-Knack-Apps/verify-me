<script setup lang="ts">
import { useSettingsStore } from '@/stores/settings';

defineEmits<{
  close: [];
}>();

const settings = useSettingsStore();
</script>

<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-panel">
      <div class="modal-header">
        <h2 class="modal-title">Settings</h2>
        <button class="close-btn" @click="$emit('close')" title="Close">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div class="modal-body">
        <div class="setting-group">
          <label class="setting-label">Output Directory</label>
          <input
            type="text"
            class="setting-input"
            v-model="settings.outputDirectory"
            placeholder="Default output directory"
            readonly
          />
        </div>

        <div class="setting-group">
          <label class="setting-label">Device</label>
          <p class="setting-value">{{ settings.deviceType }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
  padding: 1rem;
}

.modal-panel {
  background: var(--color-bg);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.75rem;
  width: 100%;
  max-width: 28rem;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0;
}

.close-btn {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: var(--color-text);
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
}
.close-btn:hover {
  color: var(--color-accent);
}

.modal-body {
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.setting-group {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.setting-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #9ca3af;
}

.setting-input {
  padding: 0.625rem 0.75rem;
  min-height: 44px;
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.375rem;
  font-size: 0.875rem;
}

.setting-value {
  margin: 0;
  font-size: 0.875rem;
  color: var(--color-text);
}
</style>
