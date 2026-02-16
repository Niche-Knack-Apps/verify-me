<script setup lang="ts">
import { computed } from 'vue';
import { useTTSStore } from '@/stores/tts';
import ModelSelector from '@/components/ModelSelector.vue';
import AudioPlayer from '@/components/AudioPlayer.vue';
import type { TTSModel } from '@/stores/models';

const tts = useTTSStore();

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

// Block-character level meter: ████░░░░░░
const levelMeter = computed(() => {
  const total = 20;
  const filled = Math.round(tts.currentLevel * total);
  const empty = total - filled;
  return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
});

function levelColor(level: number): string {
  if (level > 0.9) return 'var(--crt-error)';
  if (level > 0.75) return 'var(--crt-warn)';
  return 'var(--crt-text)';
}
</script>

<template>
  <div class="clone-tab">
    <!-- Reference Audio -->
    <div class="reference-section">
      <label class="field-label">&gt; REFERENCE AUDIO</label>

      <!-- Recording active -->
      <div v-if="tts.isRecording" class="recording-panel">
        <div class="recording-header">
          <span class="recording-indicator">[ REC ]</span>
          <span class="recording-time">{{ formatTime(tts.recordingDuration) }}</span>
        </div>
        <div class="level-meter" :style="{ color: levelColor(tts.currentLevel) }">
          {{ levelMeter }}
        </div>
        <button class="stop-btn" @click="tts.stopRecording()">
          [ STOP RECORDING ]
        </button>
      </div>

      <!-- Not recording -->
      <template v-else>
        <div class="reference-controls">
          <button class="upload-btn" @click="selectReferenceAudio">
            [ CHOOSE FILE ]
          </button>
          <button class="record-btn" @click="tts.startRecording()">
            [ RECORD ]
          </button>
        </div>
        <p v-if="tts.referenceAudioPath" class="reference-file">
          &gt; {{ getFileName(tts.referenceAudioPath) }}
        </p>
      </template>
    </div>

    <ModelSelector v-model:modelValue="tts.selectedModelId" :model-filter="cloneFilter" />

    <div class="input-section">
      <label class="field-label">&gt; TEXT TO SYNTHESIZE</label>
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
      <span v-if="tts.isGenerating" class="spinner-text">[||||] GENERATING...</span>
      <span v-else>[ CLONE VOICE ]</span>
    </button>

    <p v-if="tts.error" class="error-msg">ERROR: {{ tts.error }}</p>

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
  font-size: 16px;
  color: var(--crt-dim);
  margin-bottom: 0.375rem;
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
  font-family: 'VT323', monospace;
  font-size: 18px;
  background: transparent;
  color: var(--crt-text);
  border: 1px solid var(--crt-border);
  border-radius: 0;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s;
}
.upload-btn:hover,
.record-btn:hover {
  border-color: var(--crt-bright);
  color: var(--crt-bright);
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}

.reference-file {
  font-size: 16px;
  color: var(--crt-bright);
  padding: 0.375rem 0.75rem;
  background: rgba(51, 255, 0, 0.05);
  border: 1px solid var(--crt-border);
  border-radius: 0;
  word-break: break-all;
}

/* Recording panel */
.recording-panel {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 0.875rem;
  background: var(--crt-surface);
  border: 1px solid rgba(255, 51, 51, 0.4);
  border-radius: 0;
}

.recording-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.recording-indicator {
  color: var(--crt-error);
  animation: blink-cursor 1.2s step-end infinite;
  text-shadow: 0 0 8px rgba(255, 51, 51, 0.5);
}

.recording-time {
  font-variant-numeric: tabular-nums;
  color: var(--crt-text);
}

.level-meter {
  font-size: 14px;
  letter-spacing: 0;
  line-height: 1;
  text-shadow: none;
  transition: color 150ms;
}

.stop-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.625rem 1rem;
  min-height: 44px;
  font-family: 'VT323', monospace;
  font-size: 18px;
  background: transparent;
  color: var(--crt-error);
  border: 1px solid rgba(255, 51, 51, 0.4);
  border-radius: 0;
  cursor: pointer;
  transition: background 0.15s;
}
.stop-btn:hover {
  background: rgba(255, 51, 51, 0.1);
  text-shadow: 0 0 8px rgba(255, 51, 51, 0.4);
}

.text-input {
  width: 100%;
  padding: 0.75rem;
  background: var(--crt-bg);
  color: var(--crt-text);
  border: 1px solid var(--crt-border);
  border-radius: 0;
  font-family: 'VT323', monospace;
  font-size: 18px;
  line-height: 1.5;
  resize: vertical;
  text-shadow: var(--crt-glow);
  caret-color: var(--crt-bright);
}
.text-input:focus {
  outline: none;
  border-color: var(--crt-bright);
  box-shadow: 0 0 8px rgba(51, 255, 0, 0.2);
}
.text-input::placeholder {
  color: var(--crt-dim);
  text-shadow: none;
}

.generate-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  padding: 0.75rem;
  min-height: 48px;
  font-family: 'VT323', monospace;
  font-size: 22px;
  background: transparent;
  color: var(--crt-bright);
  border: 1px solid var(--crt-bright);
  border-radius: 0;
  cursor: pointer;
  letter-spacing: 0.1em;
  transition: background 0.15s, text-shadow 0.15s;
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}
.generate-btn:hover:not(:disabled) {
  background: rgba(51, 255, 0, 0.08);
  text-shadow: 0 0 12px rgba(51, 255, 0, 0.6);
}
.generate-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
  text-shadow: none;
}

.spinner-text {
  animation: blink-cursor 0.8s step-end infinite;
}

.error-msg {
  color: var(--crt-error);
  font-size: 16px;
  padding: 0.5rem 0.75rem;
  background: rgba(255, 51, 51, 0.08);
  border: 1px solid rgba(255, 51, 51, 0.3);
  border-radius: 0;
}
</style>
