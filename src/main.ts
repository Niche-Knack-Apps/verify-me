import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { DebugLogger, setLogger } from './services/debug-logger';
import { isCapacitor, initCapacitorPlugins } from './services/capacitor-plugins';
import './assets/main.css';

const app = createApp(App);
const pinia = createPinia();

app.use(pinia);

// Initialize debug logger BEFORE mounting so console interception
// captures all startup logs (onMounted, plugin registration, etc.)
const logger = new DebugLogger({ appName: 'Verify Me' });
logger.init().then(async () => {
  setLogger(logger);

  // Catch unhandled promise rejections (e.g. plugin calls that silently fail)
  window.addEventListener('unhandledrejection', (event) => {
    const msg = event.reason instanceof Error
      ? `${event.reason.message}\n${event.reason.stack}`
      : String(event.reason);
    console.error('[unhandledrejection]', msg);
  });

  // Register Capacitor plugins before mount — plugin proxies intercept .then(),
  // so they must be created synchronously (not returned from async functions).
  if (isCapacitor()) {
    await initCapacitorPlugins();
  }

  app.mount('#app');
});
