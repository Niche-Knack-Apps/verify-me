<script setup lang="ts">
import { useTTSStore } from '@/stores/tts';
import { useSettingsStore } from '@/stores/settings';
import ModelSelector from '@/components/ModelSelector.vue';
import AudioPlayer from '@/components/AudioPlayer.vue';

const tts = useTTSStore();
const settings = useSettingsStore();
</script>

<template>
  <div class="tts-tab">
    <ModelSelector v-model:modelValue="tts.selectedModelId" />

    <div class="input-section">
      <label class="field-label">
        {{ settings.isEighties ? '> TEXT INPUT' : 'Text Input' }}
      </label>
      <textarea
        v-model="tts.text"
        class="text-input"
        placeholder="Enter text to speak..."
        rows="6"
      />
    </div>

    <div v-if="tts.selectedModel?.supportsVoiceDesign" class="voice-mode-toggle">
      <button
        class="mode-btn"
        :class="{ active: tts.voiceMode === 'speaker' }"
        @click="tts.voiceMode = 'speaker'"
      >
        {{ settings.isEighties ? '[ PREDEFINED SPEAKER ]' : 'Predefined Speaker' }}
      </button>
      <button
        class="mode-btn"
        :class="{ active: tts.voiceMode === 'design' }"
        @click="tts.voiceMode = 'design'"
      >
        {{ settings.isEighties ? '[ DESIGN VOICE ]' : 'Design Voice' }}
      </button>
    </div>

    <div v-if="tts.voiceMode === 'design' && tts.selectedModel?.supportsVoiceDesign" class="input-section">
      <label class="field-label">
        {{ settings.isEighties ? '> VOICE DESCRIPTION' : 'Voice Description' }}
      </label>
      <textarea
        v-model="tts.voiceDescription"
        class="text-input"
        placeholder="Describe the voice you want, e.g. 'A young woman with a gentle, breathy voice and subtle British accent'"
        rows="3"
      />
    </div>

    <div class="controls-row">
      <div v-if="tts.voiceMode === 'speaker' || !tts.selectedModel?.supportsVoiceDesign" class="control-group">
        <label class="field-label" for="voice-select">
          {{ settings.isEighties ? '> VOICE' : 'Voice' }}
        </label>
        <select id="voice-select" v-model="tts.selectedVoice" class="voice-select">
          <option v-for="v in tts.voices" :key="v.id" :value="v.id">{{ v.name }}</option>
        </select>
      </div>

      <div class="control-group">
        <label class="field-label">
          {{ settings.isEighties ? '> SPEED' : 'Speed' }}
        </label>
        <div class="speed-stepper">
          <button
            class="stepper-btn"
            :disabled="tts.speed <= 0.5"
            @click="tts.speed = Math.round((tts.speed - 0.1) * 10) / 10"
          >−</button>
          <span class="stepper-value">{{ tts.speed.toFixed(1) }}x</span>
          <button
            class="stepper-btn"
            :disabled="tts.speed >= 2.0"
            @click="tts.speed = Math.round((tts.speed + 0.1) * 10) / 10"
          >+</button>
        </div>
      </div>
    </div>

    <div v-if="tts.selectedModel?.supportsVoicePrompt && tts.voiceMode === 'speaker'" class="input-section">
      <label class="field-label">
        {{ settings.isEighties ? '> VOICE INSTRUCTIONS' : 'Voice Instructions' }}
        <span class="optional-tag">(optional)</span>
      </label>
      <input
        type="text"
        v-model="tts.voicePrompt"
        class="voice-prompt-input"
        placeholder="e.g. 'Speak in a whisper', 'Sound excited'"
      />
    </div>

    <button
      class="generate-btn"
      :disabled="tts.isGenerating || !tts.text.trim()"
      @click="tts.generateSpeech()"
    >
      <template v-if="tts.isGenerating">
        <span v-if="settings.isEighties" class="spinner-text">[||||] GENERATING... {{ tts.generatingElapsed }}s</span>
        <template v-else>
          <svg class="spinner-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
          Generating... {{ tts.generatingElapsed }}s
        </template>
      </template>
      <template v-else>
        {{ settings.isEighties ? '[ GENERATE SPEECH ]' : 'Generate Speech' }}
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
.tts-tab {
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

.optional-tag {
  font-size: 0.75rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .optional-tag {
  font-size: 14px;
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

.voice-mode-toggle {
  display: flex;
  gap: 0;
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  overflow: hidden;
}

.mode-btn {
  flex: 1;
  padding: 0.5rem 0.75rem;
  min-height: 40px;
  font-family: var(--app-font);
  font-size: 0.8125rem;
  font-weight: 500;
  background: var(--app-bg);
  color: var(--app-muted);
  border: none;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}

.mode-btn + .mode-btn {
  border-left: 1px solid var(--app-border);
}

.mode-btn.active {
  background: var(--app-accent);
  color: #fff;
}

.mode-btn:hover:not(.active) {
  background: var(--app-surface);
}

[data-theme="eighties"] .voice-mode-toggle {
  border-radius: 0;
}

[data-theme="eighties"] .mode-btn {
  font-size: 16px;
  font-weight: 400;
  letter-spacing: 0.05em;
}

[data-theme="eighties"] .mode-btn.active {
  background: transparent;
  color: var(--app-accent);
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
  border-bottom: 2px solid var(--app-accent);
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
  background: var(--app-bg);
  color: var(--app-text);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  font-family: var(--app-font);
  font-size: inherit;
  cursor: pointer;
}
.voice-select:focus {
  outline: none;
  border-color: var(--app-accent);
  box-shadow: var(--app-focus-ring);
}

[data-theme="eighties"] .voice-select {
  text-shadow: var(--app-glow);
}

.voice-prompt-input {
  width: 100%;
  padding: 0.625rem 0.75rem;
  min-height: 44px;
  background: var(--app-bg);
  color: var(--app-text);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  font-family: var(--app-font);
  font-size: inherit;
  caret-color: var(--app-accent);
}
.voice-prompt-input:focus {
  outline: none;
  border-color: var(--app-accent);
  box-shadow: var(--app-focus-ring);
}
.voice-prompt-input::placeholder {
  color: var(--app-muted);
  text-shadow: none;
}

[data-theme="eighties"] .voice-prompt-input {
  text-shadow: var(--app-glow);
}

.speed-stepper {
  display: flex;
  align-items: center;
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  overflow: hidden;
  min-height: 44px;
}

.stepper-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  min-height: 44px;
  font-family: var(--app-font);
  font-size: 1.25rem;
  font-weight: 600;
  background: var(--app-surface);
  color: var(--app-text);
  border: none;
  cursor: pointer;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  transition: background 0.15s;
}

.stepper-btn:hover:not(:disabled) {
  background: var(--app-border);
}

.stepper-btn:active:not(:disabled) {
  background: var(--app-accent);
  color: #fff;
}

.stepper-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.stepper-value {
  flex: 1;
  text-align: center;
  font-size: 0.9375rem;
  font-weight: 500;
  color: var(--app-text);
  padding: 0 0.5rem;
}

[data-theme="eighties"] .speed-stepper {
  border-radius: 0;
}

[data-theme="eighties"] .stepper-btn {
  font-weight: 400;
  background: transparent;
  color: var(--app-accent);
}

[data-theme="eighties"] .stepper-btn:hover:not(:disabled) {
  background: rgba(51, 255, 0, 0.08);
}

[data-theme="eighties"] .stepper-value {
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
