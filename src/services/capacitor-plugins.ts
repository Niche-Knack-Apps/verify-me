/**
 * Shared Capacitor plugin registry.
 * Uses a promise lock so concurrent callers share one registration.
 */

let _pluginsPromise: Promise<Record<string, any>> | null = null;

function ensurePlugins(): Promise<Record<string, any>> {
  if (!_pluginsPromise) {
    _pluginsPromise = (async () => {
      const { registerPlugin } = await import('@capacitor/core');
      return {
        ModelManager: registerPlugin('ModelManager'),
        TTSEngine: registerPlugin('TTSEngine'),
        AudioRecorder: registerPlugin('AudioRecorder'),
      };
    })();
  }
  return _pluginsPromise;
}

export async function getModelManager(): Promise<any> {
  const p = await ensurePlugins();
  return p.ModelManager;
}

export async function getTTSEngine(): Promise<any> {
  const p = await ensurePlugins();
  return p.TTSEngine;
}

export async function getAudioRecorder(): Promise<any> {
  const p = await ensurePlugins();
  return p.AudioRecorder;
}

export function isCapacitor(): boolean {
  return typeof window !== 'undefined' && 'Capacitor' in window;
}
