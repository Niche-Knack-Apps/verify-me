<script setup lang="ts">
import { onMounted } from 'vue';
import { useSettingsStore } from '@/stores/settings';
import { useModelsStore } from '@/stores/models';
import TTSTab from '@/components/TTSTab.vue';
import VoiceCloneTab from '@/components/VoiceCloneTab.vue';
import SettingsModal from '@/components/settings/SettingsModal.vue';

const settings = useSettingsStore();
const modelsStore = useModelsStore();

onMounted(async () => {
  // Load model catalog first so UI renders immediately
  await modelsStore.loadModels();

  if ('Capacitor' in window) {
    // Extract bundled models if needed, then start engine
    const hasBundled = modelsStore.models.some(m => m.status === 'bundled');
    if (hasBundled) {
      modelsStore.extractBundledModels().then(() => settings.startEngine());
    } else {
      await settings.startEngine();
    }
  } else {
    // Desktop: Python env check + engine start
    await settings.initEngine();
  }
});
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <h1 v-if="settings.isEighties" class="app-title">&gt; VERIFY ME_</h1>
      <h1 v-else class="app-title">"My voice is my passport. Verify me."</h1>
      <button class="settings-btn" @click="settings.showSettings = true" title="Settings">
        <template v-if="settings.isEighties">[CFG]</template>
        <svg v-else xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
      </button>
    </header>

    <!-- Tab Bar -->
    <nav class="tab-bar">
      <button
        class="tab-btn"
        :class="{ active: settings.activeTab === 'tts' }"
        @click="settings.activeTab = 'tts'"
      >
        <template v-if="settings.isEighties">{{ settings.activeTab === 'tts' ? '[*]' : '[ ]' }} TEXT-TO-SPEECH</template>
        <template v-else>Text to Speech</template>
      </button>
      <button
        class="tab-btn"
        :class="{ active: settings.activeTab === 'clone' }"
        @click="settings.activeTab = 'clone'"
      >
        <template v-if="settings.isEighties">{{ settings.activeTab === 'clone' ? '[*]' : '[ ]' }} VOICE CLONE</template>
        <template v-else>Voice Clone</template>
      </button>
    </nav>

    <!-- Engine Offline Banner -->
    <div v-if="!settings.engineRunning && !settings.engineStarting" class="engine-banner">
      <span class="engine-banner-text">
        {{ settings.isEighties ? '// ENGINE OFFLINE' : 'Engine is not running' }}
      </span>
      <button class="engine-banner-btn" @click="settings.startEngine()">
        {{ settings.isEighties ? '[START ENGINE]' : 'Start Engine' }}
      </button>
    </div>

    <!-- Content -->
    <main class="content-area">
      <TTSTab v-if="settings.activeTab === 'tts'" />
      <VoiceCloneTab v-else />
    </main>

    <!-- Footer -->
    <footer
      class="app-footer"
      :class="{ 'app-footer--clickable': !settings.engineRunning }"
      @click="!settings.engineRunning && (settings.showSettings = true)"
    >
      <span class="engine-status">
        <template v-if="settings.isEighties">
          <span class="status-indicator">{{ settings.engineRunning ? '[ONLINE]' : '[OFFLINE]' }}</span>
          ENGINE: {{ settings.engineRunning ? 'RUNNING' : 'STOPPED' }}
          // {{ settings.deviceType.toUpperCase() }}
        </template>
        <template v-else>
          <span class="status-dot" :class="settings.engineRunning ? 'status-dot--on' : 'status-dot--off'" />
          Engine: {{ settings.engineRunning ? 'Running' : 'Stopped' }}
          <span class="device-badge">({{ settings.deviceType }})</span>
        </template>
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
  border-bottom: 1px solid var(--app-border);
  flex-shrink: 0;
}

.app-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--app-accent);
  margin: 0;
  font-style: italic;
}

[data-theme="eighties"] .app-title {
  font-size: 1.5rem;
  font-weight: 400;
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
  color: var(--app-muted);
  border: 1px solid transparent;
  border-radius: var(--app-radius);
  cursor: pointer;
  font-family: var(--app-font);
  font-size: inherit;
  transition: color 0.15s, background 0.15s, border-color 0.15s;
}
.settings-btn:hover {
  color: var(--app-accent);
  background: var(--app-accent-hover-bg);
}

[data-theme="eighties"] .settings-btn {
  border: 1px solid var(--app-border);
  border-radius: 0;
}
[data-theme="eighties"] .settings-btn:hover {
  color: var(--app-accent);
  border-color: var(--app-accent);
  background: transparent;
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}

.tab-bar {
  display: flex;
  border-bottom: 1px solid var(--app-border);
  flex-shrink: 0;
}

.tab-btn {
  flex: 1;
  min-height: 44px;
  padding: 0.625rem 1rem;
  font-family: var(--app-font);
  font-size: inherit;
  font-weight: 500;
  background: transparent;
  color: var(--app-muted);
  border: none;
  border-bottom: 2px solid transparent;
  border-radius: 0;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}
.tab-btn:hover {
  color: var(--app-text);
}
.tab-btn.active {
  color: var(--app-accent);
  border-bottom-color: var(--app-accent);
}

[data-theme="eighties"] .tab-btn {
  font-weight: 400;
  text-shadow: none;
  letter-spacing: 0.05em;
}
[data-theme="eighties"] .tab-btn:hover {
  text-shadow: var(--app-glow);
}
[data-theme="eighties"] .tab-btn.active {
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}

.engine-banner {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  padding: 0.5rem 1rem;
  background: rgba(234, 179, 8, 0.08);
  border-bottom: 1px solid rgba(234, 179, 8, 0.3);
  flex-shrink: 0;
}

[data-theme="eighties"] .engine-banner {
  background: rgba(255, 200, 0, 0.05);
  border-bottom-color: rgba(255, 200, 0, 0.3);
}

.engine-banner-text {
  font-size: 0.8125rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .engine-banner-text {
  font-size: 14px;
  letter-spacing: 0.05em;
}

.engine-banner-btn {
  font-family: var(--app-font);
  font-size: 0.8125rem;
  padding: 0.25rem 0.75rem;
  background: var(--app-accent);
  color: var(--app-bg);
  border: none;
  border-radius: var(--app-radius);
  cursor: pointer;
  transition: opacity 0.15s;
}
.engine-banner-btn:hover {
  opacity: 0.85;
}

[data-theme="eighties"] .engine-banner-btn {
  font-size: 14px;
  border-radius: 0;
  background: transparent;
  color: var(--app-accent);
  border: 1px solid var(--app-accent);
}
[data-theme="eighties"] .engine-banner-btn:hover {
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.6);
  box-shadow: 0 0 8px rgba(51, 255, 0, 0.2);
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
  border-top: 1px solid var(--app-border);
  flex-shrink: 0;
  transition: background 0.15s;
}

.app-footer--clickable {
  cursor: pointer;
}
.app-footer--clickable:hover {
  background: var(--app-accent-hover-bg, rgba(99, 102, 241, 0.06));
}

.engine-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8125rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .engine-status {
  font-size: 16px;
  letter-spacing: 0.05em;
}

.status-indicator {
  color: var(--app-text);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.status-dot--on {
  background: var(--app-success);
  box-shadow: 0 0 6px var(--app-success);
}
.status-dot--off {
  background: var(--app-muted);
}

.device-badge {
  color: var(--app-muted);
}
</style>
