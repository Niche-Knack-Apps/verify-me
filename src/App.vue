<script setup lang="ts">
import { useSettingsStore } from '@/stores/settings';
import TTSTab from '@/components/TTSTab.vue';
import VoiceCloneTab from '@/components/VoiceCloneTab.vue';
import SettingsModal from '@/components/settings/SettingsModal.vue';

const settings = useSettingsStore();
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <h1 class="app-title">Verify Me</h1>
      <button class="settings-btn" @click="settings.showSettings = true" title="Settings">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </button>
    </header>

    <!-- Tab Bar -->
    <nav class="tab-bar">
      <button
        class="tab-btn"
        :class="{ active: settings.activeTab === 'tts' }"
        @click="settings.activeTab = 'tts'"
      >
        Text to Speech
      </button>
      <button
        class="tab-btn"
        :class="{ active: settings.activeTab === 'clone' }"
        @click="settings.activeTab = 'clone'"
      >
        Voice Clone
      </button>
    </nav>

    <!-- Content -->
    <main class="content-area">
      <TTSTab v-if="settings.activeTab === 'tts'" />
      <VoiceCloneTab v-else />
    </main>

    <!-- Footer -->
    <footer class="app-footer">
      <span class="engine-status">
        <span class="status-dot" :class="settings.engineRunning ? 'running' : 'stopped'" />
        Engine: {{ settings.engineRunning ? 'Running' : 'Stopped' }}
        ({{ settings.deviceType }})
      </span>
    </footer>

    <!-- Settings Modal -->
    <SettingsModal v-if="settings.showSettings" @close="settings.showSettings = false" />
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.app-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--color-accent);
  margin: 0;
}

.settings-btn {
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
  transition: color 0.15s;
}
.settings-btn:hover {
  color: var(--color-accent);
}

.tab-bar {
  display: flex;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.tab-btn {
  flex: 1;
  min-height: 44px;
  padding: 0.625rem 1rem;
  font-size: 0.9rem;
  font-weight: 500;
  background: transparent;
  color: #9ca3af;
  border: none;
  border-bottom: 2px solid transparent;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}
.tab-btn:hover {
  color: var(--color-text);
}
.tab-btn.active {
  color: var(--color-accent);
  border-bottom-color: var(--color-accent);
}

.content-area {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.app-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.engine-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8rem;
  color: #9ca3af;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.status-dot.running {
  background: #22c55e;
  box-shadow: 0 0 6px rgba(34, 197, 94, 0.5);
}
.status-dot.stopped {
  background: #6b7280;
}
</style>
