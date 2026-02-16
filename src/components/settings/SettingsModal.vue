<script setup lang="ts">
import { onMounted } from 'vue';
import { useSettingsStore } from '@/stores/settings';
import { useModelsStore } from '@/stores/models';
import Button from '@/components/ui/Button.vue';
import LoggingPanel from '@/components/settings/LoggingPanel.vue';
import AboutPanel from '@/components/settings/AboutPanel.vue';

const APP_VERSION = '0.1.0';

const emit = defineEmits<{
  close: [];
}>();

const settings = useSettingsStore();
const modelsStore = useModelsStore();

onMounted(() => {
  modelsStore.loadModels();
});
</script>

<template>
  <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" @click.self="emit('close')">
    <div class="bg-gray-900 rounded-lg shadow-xl w-full max-w-md max-h-[90vh] flex flex-col">
      <div class="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h2 class="text-lg font-medium">Settings</h2>
        <button
          type="button"
          class="min-w-[44px] min-h-[44px] flex items-center justify-center text-gray-400 hover:text-gray-200 transition-colors"
          @click="emit('close')"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div class="px-6 py-4 space-y-6 overflow-y-auto flex-1">
        <!-- Models -->
        <div>
          <h3 class="text-sm font-medium text-gray-300 mb-3">Models</h3>
          <div class="space-y-3">
            <div>
              <label class="block text-xs text-gray-400 mb-1">Models Directory</label>
              <div class="flex-1 h-8 px-2 text-sm bg-gray-800 border border-gray-700 rounded text-gray-300 flex items-center overflow-hidden">
                <span class="truncate text-gray-500 italic">
                  {{ settings.outputDirectory || 'Using default location' }}
                </span>
              </div>
            </div>

            <div>
              <label class="block text-xs text-gray-400 mb-2">Available Models</label>
              <div v-if="modelsStore.models.length === 0" class="text-xs text-gray-500 italic">
                No models found
              </div>
              <div v-else class="space-y-1 max-h-32 overflow-y-auto">
                <div
                  v-for="model in modelsStore.models"
                  :key="model.id"
                  class="flex items-center justify-between text-xs py-1.5 px-2 bg-gray-800 rounded"
                >
                  <span class="flex items-center gap-2">
                    <span v-if="model.status === 'loaded'" class="text-green-400">●</span>
                    <span v-else-if="model.status === 'available'" class="text-yellow-400">●</span>
                    <span v-else class="text-gray-600">○</span>
                    <span :class="model.status !== 'downloadable' ? 'text-gray-200' : 'text-gray-500'">
                      {{ model.name }} ({{ model.size }})
                    </span>
                  </span>
                  <span v-if="model.status === 'loaded'" class="text-green-400 text-xs">Loaded</span>
                  <span v-else-if="model.status === 'available'" class="text-yellow-400 text-xs">Ready</span>
                  <span v-else class="text-gray-500 text-xs">Download</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Engine -->
        <div>
          <h3 class="text-sm font-medium text-gray-300 mb-3">Engine</h3>
          <div class="space-y-2">
            <div class="flex items-center justify-between text-sm">
              <span class="text-gray-400">Device</span>
              <span class="text-gray-200">{{ settings.deviceType }}</span>
            </div>
            <div class="flex items-center justify-between text-sm">
              <span class="text-gray-400">Status</span>
              <span :class="settings.engineRunning ? 'text-green-400' : 'text-gray-500'">
                {{ settings.engineRunning ? 'Running' : 'Stopped' }}
              </span>
            </div>
          </div>
        </div>

        <!-- Logging -->
        <div>
          <h3 class="text-sm font-medium text-gray-300 mb-3">Logging</h3>
          <LoggingPanel />
        </div>

        <!-- About -->
        <div>
          <h3 class="text-sm font-medium text-gray-300 mb-3">About</h3>
          <AboutPanel
            app-name="Verify Me"
            :app-version="APP_VERSION"
          />
        </div>
      </div>

      <div class="flex justify-end px-4 py-3 border-t border-gray-800">
        <Button
          variant="primary"
          size="sm"
          @click="emit('close')"
        >
          Done
        </Button>
      </div>
    </div>
  </div>
</template>
