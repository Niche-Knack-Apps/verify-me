<script setup lang="ts">
import { ref, computed } from 'vue';

// V4V Configuration
const config = {
  brand: {
    name: 'niche-knack apps',
    tagline: 'Cabinet of Curiosities for Software',
    website: 'https://nicheknack.app',
    email: 'hello@nicheknack.app',
  },
  v4v: {
    lightning: {
      label: 'Bitcoin Lightning',
      description: 'Instant, low-fee payments',
      icon: 'ZAP',
      address: 'itsmikenichols@getalby.com',
      enabled: true,
    },
    bitcoin: {
      label: 'Bitcoin On-chain',
      description: 'For larger contributions',
      icon: 'BTC',
      address: '',
      enabled: false,
    },
    kofi: {
      label: 'Ko-fi',
      description: 'Buy us a coffee',
      icon: 'KFI',
      url: 'https://ko-fi.com/nicheknack',
      enabled: true,
    },
    paypal: {
      label: 'PayPal',
      description: 'Traditional payment',
      icon: 'PAY',
      url: 'https://paypal.me/itsmikenichols',
      enabled: true,
    },
  },
};

const props = defineProps<{
  appName: string;
  appVersion: string;
}>();

const copiedKey = ref<string | null>(null);

interface V4VOption {
  key: string;
  label: string;
  description: string;
  icon: string;
  address?: string;
  url?: string;
  enabled: boolean;
}

const enabledOptions = computed<V4VOption[]>(() => {
  return Object.entries(config.v4v)
    .filter(([_, opt]) => opt.enabled)
    .map(([key, opt]) => ({ key, ...opt }));
});

async function copyToClipboard(text: string, key: string) {
  try {
    await navigator.clipboard.writeText(text);
    copiedKey.value = key;
    setTimeout(() => {
      copiedKey.value = null;
    }, 2000);
  } catch (err) {
    console.error('Failed to copy:', err);
  }
}

function openExternal(url: string) {
  window.open(url, '_blank', 'noopener');
}
</script>

<template>
  <div class="about-panel">
    <!-- Header -->
    <div class="about-header">
      <img src="@/assets/niche-knack-logo.png" alt="niche-knack apps" class="about-logo" />
      <div class="about-title">&gt; {{ appName.toUpperCase() }}</div>
      <div class="about-version">v{{ appVersion }}</div>
      <div class="about-org">
        <p>Part of <strong>niche-knack apps</strong></p>
        <a :href="config.brand.website" target="_blank" rel="noopener">nicheknack.app</a>
      </div>
    </div>

    <!-- Value for Value -->
    <div class="v4v-section">
      <h4 class="v4v-heading">// SUPPORT DEVELOPMENT</h4>
      <p class="v4v-description">
        This app is free, built on the
        <a href="https://value4value.info/" target="_blank" rel="noopener">Value for Value</a> model.
        If it brings you value, consider giving back.
      </p>

      <!-- Donation Options -->
      <div class="donation-options">
        <div
          v-for="option in enabledOptions"
          :key="option.key"
          class="donation-option"
        >
          <div class="donation-info">
            <span class="donation-icon">[{{ option.icon }}]</span>
            <div class="donation-details">
              <strong>{{ option.label }}</strong>
              <small>{{ option.address || option.description }}</small>
            </div>
          </div>
          <div class="donation-action">
            <button
              v-if="option.address"
              class="action-btn"
              :class="{ 'action-btn--copied': copiedKey === option.key }"
              @click="copyToClipboard(option.address!, option.key)"
            >
              {{ copiedKey === option.key ? '[OK!]' : '[CPY]' }}
            </button>
            <button
              v-else-if="option.url"
              class="action-btn"
              @click="openExternal(option.url!)"
            >
              [GO]
            </button>
          </div>
        </div>
      </div>

      <!-- Other ways -->
      <div class="other-ways">
        <strong>Other ways to help:</strong>
        <div class="help-list">
          <div>* Share this app with others</div>
          <div>* Report bugs (<a :href="`mailto:${config.brand.email}`">{{ config.brand.email }}</a>)</div>
          <div>* Suggest features</div>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <div class="about-footer">
      <p>Learn more at <a :href="config.brand.website" target="_blank" rel="noopener">nicheknack.app</a></p>
    </div>
  </div>
</template>

<style scoped>
.about-panel {
  padding: 0.5rem 0;
}

.about-header {
  text-align: center;
  margin-bottom: 1.5rem;
}

.about-logo {
  width: 64px;
  height: 64px;
  margin: 0 auto 0.75rem;
  border-radius: 0;
  border: 1px solid var(--crt-border);
}

.about-title {
  font-size: 22px;
  color: var(--crt-bright);
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
  margin-bottom: 0.25rem;
}

.about-version {
  font-size: 16px;
  color: var(--crt-dim);
}

.about-org {
  margin-top: 0.5rem;
  font-size: 16px;
  color: var(--crt-dim);
}

.about-org a {
  color: var(--crt-bright);
}

.v4v-section {
  padding-top: 1rem;
  border-top: 1px solid var(--crt-border);
}

.v4v-heading {
  font-size: 16px;
  font-weight: 400;
  color: var(--crt-dim);
  margin-bottom: 0.5rem;
  letter-spacing: 0.05em;
}

.v4v-description {
  font-size: 16px;
  color: var(--crt-dim);
  margin-bottom: 1rem;
}

.v4v-description a {
  color: var(--crt-bright);
}

.donation-options {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.donation-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem;
  background: var(--crt-surface);
  border: 1px solid var(--crt-border);
  border-radius: 0;
}

.donation-info {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex: 1;
  min-width: 0;
}

.donation-icon {
  font-size: 16px;
  color: var(--crt-bright);
  flex-shrink: 0;
}

.donation-details {
  flex: 1;
  min-width: 0;
}

.donation-details strong {
  display: block;
  font-size: 16px;
  font-weight: 400;
  color: var(--crt-text);
}

.donation-details small {
  display: block;
  font-size: 14px;
  color: var(--crt-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.action-btn {
  font-family: 'VT323', monospace;
  font-size: 16px;
  padding: 0.25rem 0.5rem;
  border: 1px solid var(--crt-border);
  border-radius: 0;
  background: transparent;
  color: var(--crt-text);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}

.action-btn:hover {
  color: var(--crt-bright);
  border-color: var(--crt-bright);
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}

.action-btn--copied {
  color: var(--crt-bright);
  border-color: var(--crt-bright);
}

.other-ways {
  margin-top: 1rem;
  padding: 0.75rem;
  background: var(--crt-surface);
  border: 1px solid var(--crt-border);
  border-radius: 0;
  font-size: 16px;
}

.other-ways strong {
  display: block;
  color: var(--crt-text);
  margin-bottom: 0.5rem;
}

.help-list {
  color: var(--crt-dim);
}

.help-list div {
  padding: 0.125rem 0;
}

.help-list a {
  color: var(--crt-bright);
}

.about-footer {
  margin-top: 1rem;
  text-align: center;
  font-size: 14px;
  color: var(--crt-dim);
}

.about-footer a {
  color: var(--crt-bright);
}
</style>
