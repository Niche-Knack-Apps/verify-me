<script setup lang="ts">
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

function levelColor(level: number): string {
  if (level > 0.9) return '#ef4444';
  if (level > 0.75) return '#eab308';
  return '#22c55e';
}
</script>

<template>
  <div class="clone-tab">
    <!-- Reference Audio -->
    <div class="reference-section">
      <label class="field-label">Reference Audio</label>

      <!-- Recording active -->
      <div v-if="tts.isRecording" class="recording-panel">
        <div class="recording-header">
          <span class="recording-dot" />
          <span class="recording-label">Recording</span>
          <span class="recording-time">{{ formatTime(tts.recordingDuration) }}</span>
        </div>
        <div class="level-meter">
          <div
            class="level-fill"
            :style="{ width: `${tts.currentLevel * 100}%`, backgroundColor: levelColor(tts.currentLevel) }"
          />
        </div>
        <button class="stop-btn" @click="tts.stopRecording()">
          <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="6" width="12" height="12" rx="1" />
          </svg>
          <span>Stop Recording</span>
        </button>
      </div>

      <!-- Not recording -->
      <template v-else>
        <div class="reference-controls">
          <button class="upload-btn" @click="selectReferenceAudio">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <span>Choose Audio File</span>
          </button>
          <button class="record-btn" @click="tts.startRecording()">
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="6" />
            </svg>
            <span>Record</span>
          </button>
        </div>
        <p v-if="tts.referenceAudioPath" class="reference-file">
          {{ getFileName(tts.referenceAudioPath) }}
        </p>
      </template>
    </div>

    <ModelSelector v-model:modelValue="tts.selectedModelId" :model-filter="cloneFilter" />

    <div class="input-section">
      <label class="field-label">Text to synthesize</label>
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
      <svg v-if="tts.isGenerating" class="spinner w-5 h-5" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-dasharray="31.4 31.4" />
      </svg>
      <span v-if="tts.isGenerating">Generating...</span>
      <span v-else>Clone Voice</span>
    </button>

    <p v-if="tts.error" class="error-msg">{{ tts.error }}</p>

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
  font-size: 0.875rem;
  font-weight: 500;
  color: #9ca3af;
  margin-bottom: 0.375rem;
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
  font-size: 0.875rem;
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 0.375rem;
  cursor: pointer;
  transition: border-color 0.15s;
}
.upload-btn:hover,
.record-btn:hover {
  border-color: var(--color-accent);
}

.reference-file {
  font-size: 0.8rem;
  color: var(--color-accent);
  padding: 0.375rem 0.75rem;
  background: rgba(34, 211, 238, 0.08);
  border-radius: 0.25rem;
  word-break: break-all;
}

/* Recording panel */
.recording-panel {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 0.875rem;
  background: var(--color-surface);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 0.375rem;
}

.recording-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.recording-dot {
  width: 0.625rem;
  height: 0.625rem;
  border-radius: 50%;
  background: #ef4444;
  animation: pulse-dot 1.2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.recording-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #ef4444;
}

.recording-time {
  margin-left: auto;
  font-size: 0.875rem;
  font-variant-numeric: tabular-nums;
  color: #d1d5db;
}

.level-meter {
  height: 6px;
  background: #374151;
  border-radius: 3px;
  overflow: hidden;
}

.level-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 75ms linear, background-color 150ms;
  min-width: 2px;
}

.stop-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.625rem 1rem;
  min-height: 44px;
  font-size: 0.875rem;
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 0.375rem;
  cursor: pointer;
  transition: background 0.15s;
}
.stop-btn:hover {
  background: rgba(239, 68, 68, 0.2);
}

.text-input {
  width: 100%;
  padding: 0.75rem;
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.375rem;
  font-size: 0.9rem;
  line-height: 1.5;
  resize: vertical;
  font-family: inherit;
}
.text-input:focus {
  outline: none;
  border-color: var(--color-accent);
}

.generate-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0.75rem;
  min-height: 48px;
  font-size: 1rem;
  font-weight: 600;
  background: var(--color-accent);
  color: #111827;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: opacity 0.15s;
}
.generate-btn:hover:not(:disabled) {
  opacity: 0.9;
}
.generate-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.spinner {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-msg {
  color: #f87171;
  font-size: 0.875rem;
  padding: 0.5rem 0.75rem;
  background: rgba(248, 113, 113, 0.1);
  border-radius: 0.375rem;
}
</style>
