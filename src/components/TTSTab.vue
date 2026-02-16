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
      <label class="field-label">&gt; TEXT INPUT</label>
      <textarea
        v-model="tts.text"
        class="text-input"
        placeholder="Enter text to speak..."
        rows="6"
      />
    </div>

    <div class="controls-row">
      <div class="control-group">
        <label class="field-label" for="voice-select">&gt; VOICE</label>
        <select id="voice-select" v-model="tts.selectedVoice" class="voice-select">
          <option v-for="v in tts.voices" :key="v.id" :value="v.id">{{ v.name }}</option>
        </select>
      </div>

      <div class="control-group">
        <label class="field-label">
          &gt; SPEED: {{ tts.speed.toFixed(1) }}x
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
      <label class="field-label">&gt; VOICE PROMPT <span class="optional-tag">(optional)</span></label>
      <input
        type="text"
        v-model="tts.voicePrompt"
        class="voice-prompt-input"
        placeholder="Describe the voice, e.g. 'warm female narrator'"
      />
    </div>

    <button
      class="generate-btn"
      :disabled="tts.isGenerating || !tts.text.trim()"
      @click="tts.generateSpeech()"
    >
      <span v-if="tts.isGenerating" class="spinner-text">[||||] GENERATING...</span>
      <span v-else>[ GENERATE SPEECH ]</span>
    </button>

    <p v-if="tts.error" class="error-msg">ERROR: {{ tts.error }}</p>

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
  font-size: 16px;
  color: var(--crt-dim);
  margin-bottom: 0.375rem;
  letter-spacing: 0.05em;
}

.optional-tag {
  font-size: 14px;
  color: var(--crt-dim);
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
  background: var(--crt-bg);
  color: var(--crt-text);
  border: 1px solid var(--crt-border);
  border-radius: 0;
  font-family: 'VT323', monospace;
  font-size: 18px;
  cursor: pointer;
  text-shadow: var(--crt-glow);
}
.voice-select:focus {
  outline: none;
  border-color: var(--crt-bright);
  box-shadow: 0 0 8px rgba(51, 255, 0, 0.2);
}

.voice-prompt-input {
  width: 100%;
  padding: 0.625rem 0.75rem;
  min-height: 44px;
  background: var(--crt-bg);
  color: var(--crt-text);
  border: 1px solid var(--crt-border);
  border-radius: 0;
  font-family: 'VT323', monospace;
  font-size: 18px;
  text-shadow: var(--crt-glow);
  caret-color: var(--crt-bright);
}
.voice-prompt-input:focus {
  outline: none;
  border-color: var(--crt-bright);
  box-shadow: 0 0 8px rgba(51, 255, 0, 0.2);
}
.voice-prompt-input::placeholder {
  color: var(--crt-dim);
  text-shadow: none;
}

.speed-slider {
  width: 100%;
  min-height: 44px;
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
