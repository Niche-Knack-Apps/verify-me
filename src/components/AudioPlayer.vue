<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue';

const props = defineProps<{
  audioSrc?: string;
}>();

const emit = defineEmits<{
  save: [];
}>();

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

// ASCII progress bar: [████████░░░░░░░░░░░░]
const progressBar = computed(() => {
  const total = 20;
  const filled = duration.value > 0
    ? Math.round((currentTime.value / duration.value) * total)
    : 0;
  const empty = total - filled;
  return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
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
        {{ isPlaying ? '[||]' : '[>]' }}
      </button>

      <div class="progress-section">
        <div class="progress-bar">{{ progressBar }}</div>
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
        [SAVE]
      </button>
    </div>
  </div>
</template>

<style scoped>
.audio-player {
  padding: 0.75rem;
  background: var(--crt-surface);
  border: 1px solid var(--crt-border);
  border-radius: 0;
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
  background: transparent;
  color: var(--crt-bright);
  border: 1px solid var(--crt-bright);
  border-radius: 0;
  cursor: pointer;
  font-family: 'VT323', monospace;
  font-size: 18px;
  flex-shrink: 0;
  transition: background 0.15s;
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}
.play-btn:hover {
  background: rgba(51, 255, 0, 0.08);
}

.progress-section {
  flex: 1;
  position: relative;
  min-width: 0;
}

.progress-bar {
  font-size: 14px;
  color: var(--crt-text);
  letter-spacing: 0;
  line-height: 1;
  text-shadow: none;
  pointer-events: none;
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
  font-size: 16px;
  color: var(--crt-dim);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.save-btn {
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
  font-size: 16px;
  flex-shrink: 0;
  transition: border-color 0.15s, color 0.15s;
}
.save-btn:hover {
  border-color: var(--crt-bright);
  color: var(--crt-bright);
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}
</style>
