<script setup lang="ts">
import { useSettingsStore } from '@/stores/settings';

interface Props {
  modelValue: boolean;
  disabled?: boolean;
  label?: string;
}

const props = withDefaults(defineProps<Props>(), {
  disabled: false,
});

const emit = defineEmits<{
  'update:modelValue': [value: boolean];
}>();

const settings = useSettingsStore();

function toggle() {
  if (!props.disabled) {
    emit('update:modelValue', !props.modelValue);
  }
}
</script>

<template>
  <div
    class="app-toggle"
    :class="{ 'app-toggle--disabled': disabled }"
  >
    <!-- 80's mode: ASCII switch -->
    <button
      v-if="settings.isEighties"
      type="button"
      role="switch"
      :aria-checked="modelValue"
      class="app-toggle__ascii"
      :disabled="disabled"
      @click="toggle"
    >
      {{ modelValue ? '[ON ]' : '[OFF]' }}
    </button>

    <!-- Modern: pill-style toggle -->
    <button
      v-else
      type="button"
      role="switch"
      :aria-checked="modelValue"
      class="app-toggle__pill"
      :class="{ 'app-toggle__pill--on': modelValue }"
      :disabled="disabled"
      @click="toggle"
    >
      <span class="app-toggle__thumb" />
    </button>

    <span v-if="label" class="app-toggle__label">{{ label }}</span>
  </div>
</template>

<style scoped>
.app-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
}

.app-toggle--disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ASCII toggle (80's) */
.app-toggle__ascii {
  font-family: var(--app-font);
  font-size: 18px;
  background: transparent;
  color: var(--app-muted);
  border: 1px solid var(--app-border);
  border-radius: 0;
  padding: 0.125rem 0.375rem;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}

.app-toggle__ascii[aria-checked="true"] {
  color: var(--app-accent);
  border-color: var(--app-accent);
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}

.app-toggle__ascii:hover:not(:disabled) {
  border-color: var(--app-text);
}

/* Pill toggle (modern) */
.app-toggle__pill {
  position: relative;
  width: 44px;
  height: 24px;
  background: var(--app-border);
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: background 0.2s;
  padding: 0;
  flex-shrink: 0;
}

.app-toggle__pill--on {
  background: var(--app-accent);
}

.app-toggle__thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;
  height: 20px;
  background: #fff;
  border-radius: 50%;
  transition: transform 0.2s;
  pointer-events: none;
}

.app-toggle__pill--on .app-toggle__thumb {
  transform: translateX(20px);
}

.app-toggle__label {
  font-size: 0.875rem;
  color: var(--app-text);
}

[data-theme="eighties"] .app-toggle__label {
  font-size: 16px;
}
</style>
