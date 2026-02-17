<script setup lang="ts">
import { computed } from 'vue';
import { useTTSStore } from '@/stores/tts';
import { useSettingsStore } from '@/stores/settings';
import ModelSelector from '@/components/ModelSelector.vue';
import AudioPlayer from '@/components/AudioPlayer.vue';
import type { TTSModel } from '@/stores/models';

const tts = useTTSStore();
const settings = useSettingsStore();

function cloneFilter(model: TTSModel): boolean {
  return model.supportsClone;
}

async function selectReferenceAudio() {
  try {
    const { open } = await import('@tauri-apps/plugin-dialog');
    const selected = await open({
      multiple: false,
      filters: [{ name: 'Audio', extensions: ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm'] }],
    });
    if (selected) {
      tts.referenceAudioPath = selected as string;
    }
  } catch (e) {
    console.error('Failed to open file dialog:', e);
  }
}

function getFileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// Block-character level meter for 80's mode
const levelMeter = computed(() => {
  const total = 20;
  const filled = Math.round(tts.currentLevel * total);
  const empty = total - filled;
  return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
});

// Percentage for modern mode level bar
const levelPercent = computed(() => Math.round(tts.currentLevel * 100));

function levelColor(level: number): string {
  if (level > 0.9) return 'var(--app-error)';
  if (level > 0.75) return 'var(--app-warn)';
  return 'var(--app-accent)';
}
</script>

<template>
  <div class="clone-tab">
    <!-- Reference Audio -->
    <div class="reference-section">
      <label class="field-label">
        {{ settings.isEighties ? '> REFERENCE AUDIO' : 'Reference Audio' }}
      </label>

      <!-- Recording active -->
      <div v-if="tts.isRecording" class="recording-panel">
        <div class="recording-header">
          <span v-if="settings.isEighties" class="recording-indicator">[ REC ]</span>
          <span v-else class="recording-indicator recording-indicator--modern">
            <span class="rec-dot" />
            Recording
          </span>
          <span class="recording-time">{{ formatTime(tts.recordingDuration) }}</span>
        </div>

        <!-- Level meter -->
        <div v-if="settings.isEighties" class="level-meter" :style="{ color: levelColor(tts.currentLevel) }">
          {{ levelMeter }}
        </div>
        <div v-else class="level-bar-track">
          <div
            class="level-bar-fill"
            :style="{ width: `${levelPercent}%`, background: levelColor(tts.currentLevel) }"
          />
        </div>

        <button class="stop-btn" @click="tts.stopRecording()">
          {{ settings.isEighties ? '[ STOP RECORDING ]' : 'Stop Recording' }}
        </button>
      </div>

      <!-- Not recording -->
      <template v-else>
        <div class="reference-controls">
          <button class="upload-btn" @click="selectReferenceAudio">
            {{ settings.isEighties ? '[ CHOOSE FILE ]' : 'Choose File' }}
          </button>
          <button class="record-btn" @click="tts.startRecording()">
            {{ settings.isEighties ? '[ RECORD ]' : 'Record' }}
          </button>
        </div>
        <p v-if="tts.referenceAudioPath" class="reference-file">
          <template v-if="settings.isEighties">&gt; </template>{{ getFileName(tts.referenceAudioPath) }}
        </p>
      </template>
    </div>

    <ModelSelector v-model:modelValue="tts.selectedModelId" :model-filter="cloneFilter" />

    <div class="input-section">
      <label class="field-label">
        {{ settings.isEighties ? '> TEXT TO SYNTHESIZE' : 'Text to Synthesize' }}
      </label>
      <textarea
        v-model="tts.text"
        class="text-input"
        placeholder="Enter text to speak with cloned voice..."
        rows="5"
      />
    </div>

    <button
      class="generate-btn"
      :disabled="tts.isGenerating || !tts.text.trim() || !tts.referenceAudioPath"
      @click="tts.cloneVoice()"
    >
      <template v-if="tts.isGenerating">
        <span v-if="settings.isEighties" class="spinner-text">[||||] GENERATING...</span>
        <template v-else>
          <svg class="spinner-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
          Generating...
        </template>
      </template>
      <template v-else>
        {{ settings.isEighties ? '[ CLONE VOICE ]' : 'Clone Voice' }}
      </template>
    </button>

    <p v-if="tts.error" class="error-msg">
      {{ settings.isEighties ? `ERROR: ${tts.error}` : tts.error }}
    </p>

    <AudioPlayer
      :audio-src="tts.outputAudioPath ?? undefined"
      @save="() => {}"
    />
  </div>
</template>

<style scoped>
.clone-tab {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.field-label {
  display: block;
  font-size: 0.8125rem;
  color: var(--app-muted);
  margin-bottom: 0.375rem;
  font-weight: 500;
}

[data-theme="eighties"] .field-label {
  font-size: 16px;
  font-weight: 400;
  letter-spacing: 0.05em;
}

.reference-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.reference-controls {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.upload-btn,
.record-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.625rem 1rem;
  min-height: 44px;
  font-family: var(--app-font);
  font-size: inherit;
  background: transparent;
  color: var(--app-text);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s, background 0.15s;
}
.upload-btn:hover,
.record-btn:hover {
  border-color: var(--app-accent);
  color: var(--app-accent);
  background: var(--app-accent-hover-bg);
}

[data-theme="eighties"] .upload-btn:hover,
[data-theme="eighties"] .record-btn:hover {
  background: transparent;
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}

.reference-file {
  font-size: 0.875rem;
  color: var(--app-accent);
  padding: 0.375rem 0.75rem;
  background: var(--app-accent-hover-bg);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  word-break: break-all;
}

[data-theme="eighties"] .reference-file {
  font-size: 16px;
  background: rgba(51, 255, 0, 0.05);
  border-radius: 0;
}

/* Recording panel */
.recording-panel {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 0.875rem;
  background: var(--app-surface);
  border: 1px solid rgba(239, 68, 68, 0.4);
  border-radius: var(--app-radius);
}

[data-theme="eighties"] .recording-panel {
  border-color: rgba(255, 51, 51, 0.4);
  border-radius: 0;
}

.recording-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.recording-indicator {
  color: var(--app-error);
  animation: blink-cursor 1.2s step-end infinite;
}

[data-theme="eighties"] .recording-indicator {
  text-shadow: 0 0 8px rgba(255, 51, 51, 0.5);
}

.recording-indicator--modern {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 500;
}

.rec-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--app-error);
  animation: blink-cursor 1.2s step-end infinite;
}

.recording-time {
  font-variant-numeric: tabular-nums;
  color: var(--app-text);
}

.level-meter {
  font-size: 14px;
  letter-spacing: 0;
  line-height: 1;
  text-shadow: none;
  transition: color 150ms;
}

.level-bar-track {
  height: 6px;
  background: var(--app-border);
  border-radius: 3px;
  overflow: hidden;
}

.level-bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 100ms ease-out;
}

.stop-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.625rem 1rem;
  min-height: 44px;
  font-family: var(--app-font);
  font-size: inherit;
  font-weight: 500;
  background: transparent;
  color: var(--app-error);
  border: 1px solid rgba(239, 68, 68, 0.4);
  border-radius: var(--app-radius);
  cursor: pointer;
  transition: background 0.15s;
}
.stop-btn:hover {
  background: rgba(239, 68, 68, 0.1);
}

[data-theme="eighties"] .stop-btn {
  font-weight: 400;
  border-radius: 0;
  border-color: rgba(255, 51, 51, 0.4);
}
[data-theme="eighties"] .stop-btn:hover {
  text-shadow: 0 0 8px rgba(255, 51, 51, 0.4);
}

.text-input {
  width: 100%;
  padding: 0.75rem;
  background: var(--app-bg);
  color: var(--app-text);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  font-family: var(--app-font);
  font-size: inherit;
  line-height: 1.5;
  resize: vertical;
  caret-color: var(--app-accent);
}
.text-input:focus {
  outline: none;
  border-color: var(--app-accent);
  box-shadow: var(--app-focus-ring);
}
.text-input::placeholder {
  color: var(--app-muted);
  text-shadow: none;
}

[data-theme="eighties"] .text-input {
  text-shadow: var(--app-glow);
}

.generate-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0.75rem;
  min-height: 48px;
  font-family: var(--app-font);
  font-size: 1rem;
  font-weight: 500;
  background: var(--app-accent);
  color: #fff;
  border: none;
  border-radius: var(--app-radius);
  cursor: pointer;
  transition: opacity 0.15s, filter 0.15s;
}
.generate-btn:hover:not(:disabled) {
  filter: brightness(1.1);
}
.generate-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

[data-theme="eighties"] .generate-btn {
  font-size: 22px;
  font-weight: 400;
  background: transparent;
  color: var(--app-accent);
  border: 1px solid var(--app-accent);
  border-radius: 0;
  letter-spacing: 0.1em;
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}
[data-theme="eighties"] .generate-btn:hover:not(:disabled) {
  background: rgba(51, 255, 0, 0.08);
  text-shadow: 0 0 12px rgba(51, 255, 0, 0.6);
  filter: none;
}

.spinner-text {
  animation: blink-cursor 0.8s step-end infinite;
}

.spinner-icon {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-msg {
  color: var(--app-error);
  font-size: 0.875rem;
  padding: 0.5rem 0.75rem;
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--app-radius);
}

[data-theme="eighties"] .error-msg {
  font-size: 16px;
  border-radius: 0;
  background: rgba(255, 51, 51, 0.08);
  border-color: rgba(255, 51, 51, 0.3);
}
</style>
