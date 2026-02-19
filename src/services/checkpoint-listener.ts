/**
 * Dev-only checkpoint listener for side-by-side TTS pipeline comparison.
 *
 * Listens for 'tts-checkpoint' events from the Rust backend and stores
 * them in memory, grouped by engine (onnx vs safetensors). Pipes each
 * checkpoint to the debug logger for display in the settings panel.
 *
 * All code is gated behind import.meta.env.DEV checks in callers.
 */

import { getLogger } from './debug-logger';

export interface Checkpoint {
  engine: string;
  stage: string;
  timestamp: number;
  data: Record<string, unknown>;
}

let checkpoints: Checkpoint[] = [];
let unlisten: (() => void) | null = null;

/**
 * Initialize the checkpoint event listener.
 * Call once from App.vue onMounted (dev mode only).
 */
export async function initCheckpointListener(): Promise<void> {
  const { listen } = await import('@tauri-apps/api/event');

  unlisten = await listen<Checkpoint>('tts-checkpoint', (event) => {
    const cp = event.payload;
    checkpoints.push(cp);

    // Log full checkpoint data to debug logger (persisted to IndexedDB, included in exports)
    const logger = getLogger();
    if (logger) {
      const dataStr = JSON.stringify(cp.data, null, 2);
      logger.log('debug', `[CHECKPOINT:${cp.engine}] ${cp.stage}:\n${dataStr}`, {
        source: 'checkpoint',
        checkpointEngine: cp.engine,
        checkpointStage: cp.stage,
        checkpointData: cp.data,
      });
    }
  });

  console.log('[checkpoint-listener] Initialized');
}

/**
 * Clear stored checkpoints. Call before each generation.
 */
export function clearCheckpoints(): void {
  checkpoints = [];
}

/**
 * Get all stored checkpoints, optionally filtered by engine.
 */
export function getCheckpoints(engine?: string): Checkpoint[] {
  if (engine) {
    return checkpoints.filter((cp) => cp.engine === engine);
  }
  return [...checkpoints];
}

/**
 * Destroy the listener. Call on unmount if needed.
 */
export function destroyCheckpointListener(): void {
  if (unlisten) {
    unlisten();
    unlisten = null;
  }
  checkpoints = [];
}
