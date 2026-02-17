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
      'app-btn',
      `app-btn--${variant}`,
      `app-btn--${size}`,
      { 'app-btn--disabled': disabled || loading },
    ]"
    :disabled="disabled || loading"
    @click="handleClick"
  >
    <span v-if="loading" class="app-btn__spinner">[...]</span>
    <slot />
  </button>
</template>

<style scoped>
.app-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.375rem;
  font-family: var(--app-font);
  border-radius: var(--app-radius);
  cursor: pointer;
  transition: background 0.15s, color 0.15s, border-color 0.15s, filter 0.15s;
  font-weight: 500;
}

[data-theme="eighties"] .app-btn {
  font-weight: 400;
  letter-spacing: 0.05em;
}

/* Variants — Modern */
.app-btn--primary {
  background: var(--app-accent);
  color: #fff;
  border: 1px solid var(--app-accent);
}
.app-btn--primary:hover:not(:disabled) {
  filter: brightness(1.1);
}

.app-btn--secondary {
  background: transparent;
  color: var(--app-text);
  border: 1px solid var(--app-border);
}
.app-btn--secondary:hover:not(:disabled) {
  border-color: var(--app-accent);
  color: var(--app-accent);
  background: var(--app-accent-hover-bg);
}

.app-btn--ghost {
  background: transparent;
  color: var(--app-muted);
  border: 1px solid transparent;
}
.app-btn--ghost:hover:not(:disabled) {
  color: var(--app-text);
  background: var(--app-accent-hover-bg);
}

/* 80's overrides */
[data-theme="eighties"] .app-btn--primary {
  background: transparent;
  color: var(--app-accent);
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}
[data-theme="eighties"] .app-btn--primary:hover:not(:disabled) {
  background: rgba(51, 255, 0, 0.08);
  text-shadow: 0 0 10px rgba(51, 255, 0, 0.6);
  filter: none;
}

[data-theme="eighties"] .app-btn--secondary:hover:not(:disabled) {
  background: transparent;
  text-shadow: var(--app-glow);
}

[data-theme="eighties"] .app-btn--ghost {
  border: 1px solid transparent;
}
[data-theme="eighties"] .app-btn--ghost:hover:not(:disabled) {
  color: var(--app-text);
  border-color: var(--app-border);
  background: transparent;
  text-shadow: var(--app-glow);
}

/* Sizes */
.app-btn--sm {
  padding: 0.125rem 0.5rem;
  font-size: 0.8125rem;
}

[data-theme="eighties"] .app-btn--sm {
  font-size: 16px;
}

.app-btn--md {
  padding: 0.375rem 0.75rem;
  font-size: 0.875rem;
  min-height: 44px;
}

[data-theme="eighties"] .app-btn--md {
  font-size: 18px;
}

.app-btn--lg {
  padding: 0.5rem 1rem;
  font-size: 1rem;
  min-height: 44px;
}

[data-theme="eighties"] .app-btn--lg {
  font-size: 20px;
}

/* Disabled */
.app-btn--disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.app-btn__spinner {
  animation: blink-cursor 0.8s step-end infinite;
}
</style>
