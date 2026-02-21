/**
 * Shared Capacitor plugin registry.
 * Plugins are initialized once at startup, then accessed synchronously.
 * IMPORTANT: Capacitor plugin proxies intercept ALL property access,
 * including .then(), so they must NEVER be returned from async functions
 * or passed through await/Promise.resolve() — that triggers thenable
 * detection which calls plugin.then() on the native side and crashes.
 */

let _plugins: Record<string, any> | null = null;

/**
 * Call once at startup (before app.mount) when running on Capacitor.
 */
export async function initCapacitorPlugins(): Promise<void> {
  if (_plugins) return;
  const { registerPlugin } = await import('@capacitor/core');
  _plugins = {
    ModelManager: registerPlugin('ModelManager'),
    TTSEngine: registerPlugin('TTSEngine'),
    AudioRecorder: registerPlugin('AudioRecorder'),
  };
}

/** Synchronous getter — call initCapacitorPlugins() first. */
export function getModelManager(): any {
  return _plugins!.ModelManager;
}

/** Synchronous getter — call initCapacitorPlugins() first. */
export function getTTSEngine(): any {
  return _plugins!.TTSEngine;
}

/** Synchronous getter — call initCapacitorPlugins() first. */
export function getAudioRecorder(): any {
  return _plugins!.AudioRecorder;
}

export function isCapacitor(): boolean {
  return typeof window !== 'undefined' && 'Capacitor' in window;
}
