<script setup lang="ts">
import { ref, watch, onUnmounted } from 'vue';

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

    <button class="play-btn" @click="togglePlay" :title="isPlaying ? 'Pause' : 'Play'">
      <!-- Play icon -->
      <svg v-if="!isPlaying" class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
        <path d="M8 5v14l11-7z" />
      </svg>
      <!-- Pause icon -->
      <svg v-else class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
        <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
      </svg>
    </button>

    <input
      type="range"
      class="seek-bar"
      min="0"
      max="100"
      step="0.1"
      :value="seekValue"
      @input="onSeek"
    />

    <span class="time-display">
      {{ formatTime(currentTime) }} / {{ formatTime(duration) }}
    </span>

    <button class="save-btn" @click="emit('save')" title="Save audio">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
    </button>
  </div>
</template>

<style scoped>
.audio-player {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: var(--color-surface);
  border-radius: 0.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.play-btn {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-accent);
  color: #111827;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  flex-shrink: 0;
  transition: opacity 0.15s;
}
.play-btn:hover {
  opacity: 0.85;
}

.seek-bar {
  flex: 1;
  min-height: 44px;
  accent-color: var(--color-accent);
  cursor: pointer;
}

.time-display {
  font-size: 0.8rem;
  color: #9ca3af;
  white-space: nowrap;
  min-width: 5rem;
  text-align: center;
}

.save-btn {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: var(--color-text);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 0.375rem;
  cursor: pointer;
  flex-shrink: 0;
  transition: border-color 0.15s;
}
.save-btn:hover {
  border-color: var(--color-accent);
  color: var(--color-accent);
}
</style>
