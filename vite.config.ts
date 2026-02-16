import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path';

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@shared': resolve(__dirname, '../_shared'),
    },
  },
  server: {
    port: 5184,
    strictPort: true,
    watch: {
      ignored: ['**/packaging/**', '**/android/**'],
    },
  },
  build: {
    target: 'esnext',
    minify: 'esbuild',
    rollupOptions: {
      output: {
        manualChunks: {
          vue: ['vue', 'pinia'],
        },
      },
    },
  },
  optimizeDeps: {
    include: ['vue', 'pinia', '@vueuse/core'],
  },
});
