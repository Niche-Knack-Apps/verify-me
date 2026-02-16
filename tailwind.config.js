import forms from '@tailwindcss/forms';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        crt: {
          bg: '#0a0a0a',
          surface: '#0f1a0f',
          text: '#20c20e',
          bright: '#33ff00',
          dim: '#0a8a0a',
          border: '#1a3a1a',
          error: '#ff3333',
          warn: '#cccc00',
        },
      },
      fontFamily: {
        terminal: ["'VT323'", 'monospace'],
      },
    },
  },
  plugins: [forms],
};
