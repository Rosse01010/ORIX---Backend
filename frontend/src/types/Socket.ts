import type { Alert } from "./Alert";
import type { CameraStatus } from "./Camera";
import type { FaceCandidate } from "./Candidate";

/**
 * Events the ORIX backend pushes to the client.
 *
 * NOTE: the current backend (FastAPI native WebSocket) only pushes
 * `face_detected` payloads, which the socket service translates into
 * `alert` events. The other channels are kept here for forward
 * compatibility and for code that still emits them (e.g. in-browser
 * face detection in CameraFeed).
 */
export interface ServerToClientEvents {
  alert: (alert: Alert) => void;
  "camera-status": (payload: { cameraId: string; status: CameraStatus }) => void;
  "detection-result": (payload: {
    cameraId: string;
    boxes: BoundingBox[];
    /** Faces that need operator confirmation (off-angle or unknown) */
    candidates: FaceCandidate[];
  }) => void;
  connect: () => void;
  disconnect: (reason: string) => void;
}

/**
 * Events the client "emits". Since the backend WebSocket is push-only,
 * these are local-only (see socketService.ts `emit`).
 */
export interface ClientToServerEvents {
  "face-detected": (payload: { cameraId: string; count: number }) => void;
  "ack-alert": (payload: { alertId: string }) => void;
  "subscribe-camera": (payload: { cameraId: string }) => void;
  "unsubscribe-camera": (payload: { cameraId: string }) => void;
}

/**
 * Normalized bounding box shape used for overlays.
 */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  label?: string;
  confidence?: number;
  quality?: number;
  angle?: string;
}

/**
 * Minimal socket-ish interface the rest of the app relies on. The concrete
 * implementation lives in `services/socketService.ts` and wraps a native
 * `WebSocket`, but consumers only see this event-emitter surface.
 */
export interface OrixSocket {
  id: string | null;
  connected: boolean;
  connect(): void;
  disconnect(): void;
  on<E extends keyof ServerToClientEvents>(
    event: E,
    handler: ServerToClientEvents[E]
  ): void;
  on(event: string, handler: (...args: unknown[]) => void): void;
  off<E extends keyof ServerToClientEvents>(
    event: E,
    handler: ServerToClientEvents[E]
  ): void;
  off(event: string, handler: (...args: unknown[]) => void): void;
  emit<E extends keyof ClientToServerEvents>(
    event: E,
    ...args: Parameters<ClientToServerEvents[E]>
  ): void;
  emit(event: string, ...args: unknown[]): void;
}
