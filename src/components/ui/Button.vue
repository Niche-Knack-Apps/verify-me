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
      'inline-flex items-center justify-center font-medium transition-colors rounded focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900',
      {
        'bg-cyan-500 text-gray-900 hover:bg-cyan-600 focus:ring-cyan-500': variant === 'primary',
        'bg-gray-600 text-gray-100 hover:bg-gray-500 focus:ring-gray-500': variant === 'secondary',
        'bg-transparent text-gray-300 hover:bg-gray-700 focus:ring-gray-600': variant === 'ghost',
        'px-2 py-1 text-xs': size === 'sm',
        'px-3 py-2 text-sm min-h-[44px]': size === 'md',
        'px-4 py-2.5 text-base min-h-[44px]': size === 'lg',
        'opacity-50 cursor-not-allowed': disabled || loading,
      },
    ]"
    :disabled="disabled || loading"
    @click="handleClick"
  >
    <svg
      v-if="loading"
      class="animate-spin -ml-1 mr-2 h-4 w-4"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        class="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        stroke-width="4"
      />
      <path
        class="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
    <slot />
  </button>
</template>
