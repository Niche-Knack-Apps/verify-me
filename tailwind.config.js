import forms from '@tailwindcss/forms';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        app: {
          bg: 'var(--app-bg)',
          surface: 'var(--app-surface)',
          text: 'var(--app-text)',
          accent: 'var(--app-accent)',
          muted: 'var(--app-muted)',
          border: 'var(--app-border)',
          error: 'var(--app-error)',
          warn: 'var(--app-warn)',
          success: 'var(--app-success)',
        },
      },
      fontFamily: {
        app: ['var(--app-font)'],
      },
      borderRadius: {
        app: 'var(--app-radius)',
      },
    },
  },
  plugins: [forms],
};
