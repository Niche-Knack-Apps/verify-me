<script setup lang="ts">
interface Props {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'md',
  disabled: false,
  loading: false,
});

const emit = defineEmits<{
  click: [event: MouseEvent];
}>();

function handleClick(event: MouseEvent) {
  if (!props.disabled && !props.loading) {
    emit('click', event);
  }
}
</script>

<template>
  <button
    :class="[
      'crt-btn',
      `crt-btn--${variant}`,
      `crt-btn--${size}`,
      { 'crt-btn--disabled': disabled || loading },
    ]"
    :disabled="disabled || loading"
    @click="handleClick"
  >
    <span v-if="loading" class="crt-btn__spinner">[...]</span>
    <slot />
  </button>
</template>

<style scoped>
.crt-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.375rem;
  font-family: 'VT323', monospace;
  border-radius: 0;
  cursor: pointer;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
  letter-spacing: 0.05em;
}

/* Variants */
.crt-btn--primary {
  background: transparent;
  color: var(--crt-bright);
  border: 1px solid var(--crt-bright);
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}
.crt-btn--primary:hover:not(:disabled) {
  background: rgba(51, 255, 0, 0.08);
  text-shadow: 0 0 10px rgba(51, 255, 0, 0.6);
}

.crt-btn--secondary {
  background: transparent;
  color: var(--crt-text);
  border: 1px solid var(--crt-border);
}
.crt-btn--secondary:hover:not(:disabled) {
  border-color: var(--crt-text);
  text-shadow: var(--crt-glow);
}

.crt-btn--ghost {
  background: transparent;
  color: var(--crt-dim);
  border: 1px solid transparent;
}
.crt-btn--ghost:hover:not(:disabled) {
  color: var(--crt-text);
  border-color: var(--crt-border);
  text-shadow: var(--crt-glow);
}

/* Sizes */
.crt-btn--sm {
  padding: 0.125rem 0.5rem;
  font-size: 16px;
}

.crt-btn--md {
  padding: 0.375rem 0.75rem;
  font-size: 18px;
  min-height: 44px;
}

.crt-btn--lg {
  padding: 0.5rem 1rem;
  font-size: 20px;
  min-height: 44px;
}

/* Disabled */
.crt-btn--disabled {
  opacity: 0.4;
  cursor: not-allowed;
  text-shadow: none;
}

.crt-btn__spinner {
  animation: blink-cursor 0.8s step-end infinite;
}
</style>
