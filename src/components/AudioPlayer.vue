<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue';
import { useSettingsStore } from '@/stores/settings';

const props = defineProps<{
  audioSrc?: string;
}>();

const emit = defineEmits<{
  save: [];
}>();

const settings = useSettingsStore();
const audioEl = ref<HTMLAudioElement | null>(null);
const isPlaying = ref(false);
const currentTime = ref(0);
const duration = ref(0);
const seekValue = ref(0);

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// ASCII progress bar for 80's mode
const progressBar = computed(() => {
  const total = 20;
  const filled = duration.value > 0
    ? Math.round((currentTime.value / duration.value) * total)
    : 0;
  const empty = total - filled;
  return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
});

// Percentage for modern mode
const progressPercent = computed(() => {
  if (duration.value <= 0) return 0;
  return (currentTime.value / duration.value) * 100;
});

function togglePlay() {
  if (!audioEl.value) return;
  if (isPlaying.value) {
    audioEl.value.pause();
  } else {
    audioEl.value.play();
  }
}

function onTimeUpdate() {
  if (!audioEl.value) return;
  currentTime.value = audioEl.value.currentTime;
  if (duration.value > 0) {
    seekValue.value = (currentTime.value / duration.value) * 100;
  }
}

function onLoadedMetadata() {
  if (!audioEl.value) return;
  duration.value = audioEl.value.duration;
}

function onSeek(event: Event) {
  const target = event.target as HTMLInputElement;
  const pct = parseFloat(target.value);
  if (audioEl.value && duration.value > 0) {
    audioEl.value.currentTime = (pct / 100) * duration.value;
  }
}

function onEnded() {
  isPlaying.value = false;
  seekValue.value = 0;
  currentTime.value = 0;
}

watch(() => props.audioSrc, () => {
  isPlaying.value = false;
  currentTime.value = 0;
  seekValue.value = 0;
  duration.value = 0;
});

onUnmounted(() => {
  if (audioEl.value) {
    audioEl.value.pause();
  }
});
</script>

<template>
  <div v-if="audioSrc" class="audio-player">
    <audio
      ref="audioEl"
      :src="audioSrc"
      @timeupdate="onTimeUpdate"
      @loadedmetadata="onLoadedMetadata"
      @play="isPlaying = true"
      @pause="isPlaying = false"
      @ended="onEnded"
    />

    <div class="player-row">
      <button class="play-btn" @click="togglePlay" :title="isPlaying ? 'Pause' : 'Play'">
        <template v-if="settings.isEighties">{{ isPlaying ? '[||]' : '[>]' }}</template>
        <template v-else>
          <!-- Play icon -->
          <svg v-if="!isPlaying" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          <!-- Pause icon -->
          <svg v-else xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
        </template>
      </button>

      <div class="progress-section">
        <!-- 80's ASCII bar -->
        <div v-if="settings.isEighties" class="progress-bar-ascii">{{ progressBar }}</div>
        <!-- Modern bar -->
        <div v-else class="progress-bar-modern">
          <div class="progress-bar-fill" :style="{ width: `${progressPercent}%` }" />
        </div>
        <input
          type="range"
          class="seek-bar"
          min="0"
          max="100"
          step="0.1"
          :value="seekValue"
          @input="onSeek"
        />
      </div>

      <span class="time-display">
        {{ formatTime(currentTime) }}/{{ formatTime(duration) }}
      </span>

      <button class="save-btn" @click="emit('save')" title="Save audio">
        <template v-if="settings.isEighties">[SAVE]</template>
        <svg v-else xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.audio-player {
  padding: 0.75rem;
  background: var(--app-surface);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
}

.player-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.play-btn {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--app-accent);
  color: #fff;
  border: none;
  border-radius: var(--app-radius);
  cursor: pointer;
  font-family: var(--app-font);
  font-size: inherit;
  flex-shrink: 0;
  transition: filter 0.15s;
}
.play-btn:hover {
  filter: brightness(1.1);
}

[data-theme="eighties"] .play-btn {
  background: transparent;
  color: var(--app-accent);
  border: 1px solid var(--app-accent);
  border-radius: 0;
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}
[data-theme="eighties"] .play-btn:hover {
  background: rgba(51, 255, 0, 0.08);
  filter: none;
}

.progress-section {
  flex: 1;
  position: relative;
  min-width: 0;
}

.progress-bar-ascii {
  font-size: 14px;
  color: var(--app-text);
  letter-spacing: 0;
  line-height: 1;
  text-shadow: none;
  pointer-events: none;
}

.progress-bar-modern {
  height: 4px;
  background: var(--app-border);
  border-radius: 2px;
  overflow: hidden;
  pointer-events: none;
}

.progress-bar-fill {
  height: 100%;
  background: var(--app-accent);
  border-radius: 2px;
  transition: width 0.1s linear;
}

.seek-bar {
  position: absolute;
  inset: 0;
  width: 100%;
  opacity: 0;
  cursor: pointer;
  min-height: 44px;
}

.time-display {
  font-size: 0.8125rem;
  color: var(--app-muted);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

[data-theme="eighties"] .time-display {
  font-size: 16px;
}

.save-btn {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: var(--app-muted);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  cursor: pointer;
  font-family: var(--app-font);
  font-size: inherit;
  flex-shrink: 0;
  transition: border-color 0.15s, color 0.15s;
}
.save-btn:hover {
  border-color: var(--app-accent);
  color: var(--app-accent);
}

[data-theme="eighties"] .save-btn:hover {
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}
</style>
