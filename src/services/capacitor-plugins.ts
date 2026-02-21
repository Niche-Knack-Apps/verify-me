/**
 * Shared Capacitor plugin registry.
 * Each plugin is registered exactly once and cached for reuse across all stores.
 */

let _plugins: Record<string, any> | null = null;

async function ensurePlugins(): Promise<Record<string, any>> {
  if (!_plugins) {
    const { registerPlugin } = await import('@capacitor/core');
    _plugins = {
      ModelManager: registerPlugin('ModelManager'),
      TTSEngine: registerPlugin('TTSEngine'),
      AudioRecorder: registerPlugin('AudioRecorder'),
    };
  }
  return _plugins;
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
