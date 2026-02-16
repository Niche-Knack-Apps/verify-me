<script setup lang="ts">
import { useTTSStore } from '@/stores/tts';
import ModelSelector from '@/components/ModelSelector.vue';
import AudioPlayer from '@/components/AudioPlayer.vue';

const tts = useTTSStore();
</script>

<template>
  <div class="tts-tab">
    <ModelSelector v-model:modelValue="tts.selectedModelId" />

    <div class="input-section">
      <label class="field-label">Text</label>
      <textarea
        v-model="tts.text"
        class="text-input"
        placeholder="Enter text to speak..."
        rows="6"
      />
    </div>

    <div class="controls-row">
      <div class="control-group">
        <label class="field-label" for="voice-select">Voice</label>
        <select id="voice-select" v-model="tts.selectedVoice" class="voice-select">
          <option v-for="v in tts.voices" :key="v.id" :value="v.id">{{ v.name }}</option>
        </select>
      </div>

      <div class="control-group">
        <label class="field-label">
          Speed: {{ tts.speed.toFixed(1) }}x
        </label>
        <input
          type="range"
          class="speed-slider"
          min="0.5"
          max="2.0"
          step="0.1"
          v-model.number="tts.speed"
        />
      </div>
    </div>

    <div v-if="tts.selectedModel?.supportsVoicePrompt" class="input-section">
      <label class="field-label">Voice Prompt <span class="optional-tag">optional</span></label>
      <input
        type="text"
        v-model="tts.voicePrompt"
        class="voice-prompt-input"
        placeholder="Describe the voice, e.g. 'warm female narrator' or 'deep male with British accent'"
      />
    </div>

    <button
      class="generate-btn"
      :disabled="tts.isGenerating || !tts.text.trim()"
      @click="tts.generateSpeech()"
    >
      <svg v-if="tts.isGenerating" class="spinner w-5 h-5" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-dasharray="31.4 31.4" />
      </svg>
      <span v-if="tts.isGenerating">Generating...</span>
      <span v-else>Generate Speech</span>
    </button>

    <p v-if="tts.error" class="error-msg">{{ tts.error }}</p>

    <AudioPlayer
      :audio-src="tts.outputAudioPath ?? undefined"
      @save="() => {}"
    />
  </div>
</template>

<style scoped>
.tts-tab {
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

.optional-tag {
  font-weight: 400;
  font-size: 0.75rem;
  color: #6b7280;
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

.controls-row {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}

.control-group {
  flex: 1;
  min-width: 10rem;
}

.voice-select {
  width: 100%;
  padding: 0.625rem 0.75rem;
  min-height: 44px;
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.375rem;
  font-size: 0.875rem;
  cursor: pointer;
}
.voice-select:focus {
  outline: none;
  border-color: var(--color-accent);
}

.voice-prompt-input {
  width: 100%;
  padding: 0.625rem 0.75rem;
  min-height: 44px;
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-family: inherit;
}
.voice-prompt-input:focus {
  outline: none;
  border-color: var(--color-accent);
}
.voice-prompt-input::placeholder {
  color: #6b7280;
}

.speed-slider {
  width: 100%;
  min-height: 44px;
  accent-color: var(--color-accent);
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
