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
      <h1 class="app-title">&gt; VERIFY ME_</h1>
      <button class="settings-btn" @click="settings.showSettings = true" title="Settings">
        [CFG]
      </button>
    </header>

    <!-- Tab Bar -->
    <nav class="tab-bar">
      <button
        class="tab-btn"
        :class="{ active: settings.activeTab === 'tts' }"
        @click="settings.activeTab = 'tts'"
      >
        {{ settings.activeTab === 'tts' ? '[*]' : '[ ]' }} TEXT-TO-SPEECH
      </button>
      <button
        class="tab-btn"
        :class="{ active: settings.activeTab === 'clone' }"
        @click="settings.activeTab = 'clone'"
      >
        {{ settings.activeTab === 'clone' ? '[*]' : '[ ]' }} VOICE CLONE
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
        <span class="status-indicator">{{ settings.engineRunning ? '[ONLINE]' : '[OFFLINE]' }}</span>
        ENGINE: {{ settings.engineRunning ? 'RUNNING' : 'STOPPED' }}
        // {{ settings.deviceType.toUpperCase() }}
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
  border-bottom: 1px solid var(--crt-border);
  flex-shrink: 0;
}

.app-title {
  font-size: 1.5rem;
  font-weight: 400;
  color: var(--crt-bright);
  margin: 0;
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.6), 0 0 16px rgba(51, 255, 0, 0.3);
  letter-spacing: 0.1em;
}

.settings-btn {
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
.settings-btn:hover {
  color: var(--crt-bright);
  border-color: var(--crt-bright);
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}

.tab-bar {
  display: flex;
  border-bottom: 1px solid var(--crt-border);
  flex-shrink: 0;
}

.tab-btn {
  flex: 1;
  min-height: 44px;
  padding: 0.625rem 1rem;
  font-family: 'VT323', monospace;
  font-size: 18px;
  background: transparent;
  color: var(--crt-dim);
  border: none;
  border-bottom: 2px solid transparent;
  border-radius: 0;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
  text-shadow: none;
  letter-spacing: 0.05em;
}
.tab-btn:hover {
  color: var(--crt-text);
  text-shadow: var(--crt-glow);
}
.tab-btn.active {
  color: var(--crt-bright);
  border-bottom-color: var(--crt-bright);
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
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
  border-top: 1px solid var(--crt-border);
  flex-shrink: 0;
}

.engine-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 16px;
  color: var(--crt-dim);
  letter-spacing: 0.05em;
}

.status-indicator {
  color: var(--crt-text);
}
</style>
