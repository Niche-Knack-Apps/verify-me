<script setup lang="ts">
import { ref, computed } from 'vue';
import { useSettingsStore } from '@/stores/settings';

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
      iconEmoji: '\u26A1',
      address: 'itsmikenichols@getalby.com',
      enabled: true,
    },
    bitcoin: {
      label: 'Bitcoin On-chain',
      description: 'For larger contributions',
      icon: 'BTC',
      iconEmoji: '\u20BF',
      address: '',
      enabled: false,
    },
    kofi: {
      label: 'Ko-fi',
      description: 'Buy us a coffee',
      icon: 'KFI',
      iconEmoji: '\u2615',
      url: 'https://ko-fi.com/nicheknack',
      enabled: true,
    },
    paypal: {
      label: 'PayPal',
      description: 'Traditional payment',
      icon: 'PAY',
      iconEmoji: '\uD83D\uDCB3',
      url: 'https://paypal.me/itsmikenichols',
      enabled: true,
    },
  },
};

const props = defineProps<{
  appName: string;
  appVersion: string;
}>();

const settings = useSettingsStore();
const copiedKey = ref<string | null>(null);

interface V4VOption {
  key: string;
  label: string;
  description: string;
  icon: string;
  iconEmoji: string;
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
      <div v-if="settings.isEighties" class="about-title">&gt; {{ props.appName.toUpperCase() }}</div>
      <div v-else class="about-title">{{ props.appName }}</div>
      <div class="about-version">v{{ props.appVersion }}</div>
      <div class="about-org">
        <p>Part of <strong>niche-knack apps</strong></p>
        <a :href="config.brand.website" target="_blank" rel="noopener">nicheknack.app</a>
      </div>
    </div>

    <!-- Value for Value -->
    <div class="v4v-section">
      <h4 class="v4v-heading">
        {{ settings.isEighties ? '// SUPPORT DEVELOPMENT' : 'Support Development' }}
      </h4>
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
            <span v-if="settings.isEighties" class="donation-icon">[{{ option.icon }}]</span>
            <span v-else class="donation-icon donation-icon--modern">{{ option.iconEmoji }}</span>
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
              {{ copiedKey === option.key
                ? (settings.isEighties ? '[OK!]' : 'Copied!')
                : (settings.isEighties ? '[CPY]' : 'Copy')
              }}
            </button>
            <button
              v-else-if="option.url"
              class="action-btn"
              @click="openExternal(option.url!)"
            >
              {{ settings.isEighties ? '[GO]' : 'Open' }}
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
  border-radius: var(--app-radius);
  border: 1px solid var(--app-border);
}

.about-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--app-accent);
  margin-bottom: 0.25rem;
}

[data-theme="eighties"] .about-title {
  font-size: 22px;
  font-weight: 400;
  text-shadow: 0 0 8px rgba(51, 255, 0, 0.4);
}

.about-version {
  font-size: 0.875rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .about-version {
  font-size: 16px;
}

.about-org {
  margin-top: 0.5rem;
  font-size: 0.875rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .about-org {
  font-size: 16px;
}

.about-org a {
  color: var(--app-accent);
}

.v4v-section {
  padding-top: 1rem;
  border-top: 1px solid var(--app-border);
}

.v4v-heading {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--app-muted);
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

[data-theme="eighties"] .v4v-heading {
  font-size: 16px;
  font-weight: 400;
  text-transform: none;
}

.v4v-description {
  font-size: 0.875rem;
  color: var(--app-muted);
  margin-bottom: 1rem;
}

[data-theme="eighties"] .v4v-description {
  font-size: 16px;
}

.v4v-description a {
  color: var(--app-accent);
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
  background: var(--app-surface);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
}

[data-theme="eighties"] .donation-option {
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
  font-size: 0.875rem;
  color: var(--app-accent);
  flex-shrink: 0;
}

[data-theme="eighties"] .donation-icon {
  font-size: 16px;
}

.donation-icon--modern {
  font-size: 1.25rem;
}

.donation-details {
  flex: 1;
  min-width: 0;
}

.donation-details strong {
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--app-text);
}

[data-theme="eighties"] .donation-details strong {
  font-size: 16px;
  font-weight: 400;
}

.donation-details small {
  display: block;
  font-size: 0.75rem;
  color: var(--app-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

[data-theme="eighties"] .donation-details small {
  font-size: 14px;
}

.action-btn {
  font-family: var(--app-font);
  font-size: 0.8125rem;
  padding: 0.25rem 0.5rem;
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  background: transparent;
  color: var(--app-text);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
}

.action-btn:hover {
  color: var(--app-accent);
  border-color: var(--app-accent);
  background: var(--app-accent-hover-bg);
}

[data-theme="eighties"] .action-btn {
  font-size: 16px;
  border-radius: 0;
}
[data-theme="eighties"] .action-btn:hover {
  background: transparent;
  text-shadow: 0 0 6px rgba(51, 255, 0, 0.4);
}

.action-btn--copied {
  color: var(--app-accent);
  border-color: var(--app-accent);
}

.other-ways {
  margin-top: 1rem;
  padding: 0.75rem;
  background: var(--app-surface);
  border: 1px solid var(--app-border);
  border-radius: var(--app-radius);
  font-size: 0.875rem;
}

[data-theme="eighties"] .other-ways {
  font-size: 16px;
  border-radius: 0;
}

.other-ways strong {
  display: block;
  color: var(--app-text);
  margin-bottom: 0.5rem;
}

.help-list {
  color: var(--app-muted);
}

.help-list div {
  padding: 0.125rem 0;
}

.help-list a {
  color: var(--app-accent);
}

.about-footer {
  margin-top: 1rem;
  text-align: center;
  font-size: 0.8125rem;
  color: var(--app-muted);
}

[data-theme="eighties"] .about-footer {
  font-size: 14px;
}

.about-footer a {
  color: var(--app-accent);
}
</style>
