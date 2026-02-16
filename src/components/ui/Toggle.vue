<script setup lang="ts">
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

function toggle() {
  if (!props.disabled) {
    emit('update:modelValue', !props.modelValue);
  }
}
</script>

<template>
  <label
    class="crt-toggle"
    :class="{ 'crt-toggle--disabled': disabled }"
  >
    <button
      type="button"
      role="switch"
      :aria-checked="modelValue"
      class="crt-toggle__switch"
      :disabled="disabled"
      @click="toggle"
    >
      {{ modelValue ? '[ON ]' : '[OFF]' }}
    </button>
    <span v-if="label" class="crt-toggle__label">{{ label }}</span>
  </label>
</template>

<style scoped>
.crt-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
}

.crt-toggle--disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.crt-toggle__switch {
  font-family: 'VT323', monospace;
  font-size: 18px;
  background: transparent;
  color: var(--crt-dim);
  border: 1px solid var(--crt-border);
  border-radius: 0;
  padding: 0.125rem 0.375rem;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}

.crt-toggle__switch[aria-checked="true"] {
  color: var(--crt-bright);
  border-color: var(--crt-bright);
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}

.crt-toggle__switch:hover:not(:disabled) {
  border-color: var(--crt-text);
}

.crt-toggle__label {
  font-size: 16px;
  color: var(--crt-text);
}
</style>
