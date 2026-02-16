import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { DebugLogger, setLogger } from './services/debug-logger';
import './assets/main.css';

const app = createApp(App);
const pinia = createPinia();

app.use(pinia);

// Initialize debug logger
const logger = new DebugLogger({ appName: 'Verify Me' });
logger.init().then(() => {
  setLogger(logger);
});

app.mount('#app');
