import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.nicheknack.verifyme',
  appName: 'Verify Me',
  webDir: 'dist',
  server: {
    androidScheme: 'https',
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 1500,
      backgroundColor: '#1a1a1a',
      showSpinner: true,
      spinnerColor: '#22d3ee',
    },
    Keyboard: {
      resize: 'body',
      resizeOnFullScreen: true,
    },
  },
  android: {
    allowMixedContent: false,
    captureInput: true,
    webContentsDebuggingEnabled: true,
  },
};

export default config;
