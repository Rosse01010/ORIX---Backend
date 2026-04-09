import type { OrixSocket } from "../types/Socket";
import type { Alert } from "../types/Alert";
import { envString } from "../utils/helpers";

/**
 * Native WebSocket client for the ORIX backend.
 *
 * The backend (FastAPI) exposes a raw WebSocket at
 * `/api/recognition/ws` that pushes JSON-encoded detection events. This
 * module wraps that connection behind a small event-emitter surface so
 * the rest of the app can keep calling `socket.on("alert", cb)` as if it
 * were still Socket.IO.
 *
 * Backend event shape (routes/recognition.py):
 *   {
 *     event:       "face_detected",
 *     event_id:    "<uuid>",
 *     camera_id:   "cam-00",
 *     status:      "matched" | "unknown",
 *     person_id:   "<uuid>" | null,
 *     person_name: "Juan Pérez" | null,
 *     similarity:  0.12 | null,
 *     confidence:  0.97,
 *     timestamp:   "1718000000.0"
 *   }
 */

const SOCKET_URL = envString("VITE_SOCKET_URL", "http://localhost:8000");
const WS_PATH = "/api/recognition/ws";
const RECONNECT_DELAY_MS = 1_000;
const RECONNECT_DELAY_MAX_MS = 5_000;

type Handler = (...args: unknown[]) => void;

/** Convert an http(s):// origin into a ws(s):// url pointing at the WS path. */
function buildWsUrl(httpOrigin: string, path: string): string {
  try {
    const url = new URL(httpOrigin);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = path;
    return url.toString();
  } catch {
    return `ws://localhost:8000${path}`;
  }
}

/** Map a backend detection event to a frontend Alert. */
function eventToAlert(raw: Record<string, unknown>): Alert | null {
  if (raw.event !== "face_detected") return null;

  const status = raw.status === "matched" ? "matched" : "unknown";
  const personName =
    typeof raw.person_name === "string" ? raw.person_name : null;
  const similarity =
    typeof raw.similarity === "number" ? raw.similarity : null;

  // Backend timestamp is float-seconds (as a string); convert to ISO.
  let isoTimestamp = new Date().toISOString();
  const rawTs = raw.timestamp;
  if (typeof rawTs === "string" || typeof rawTs === "number") {
    const seconds = Number(rawTs);
    if (Number.isFinite(seconds) && seconds > 0) {
      isoTimestamp = new Date(seconds * 1000).toISOString();
    }
  }

  const message =
    status === "matched"
      ? `${personName ?? "Known person"} detected${
          similarity !== null ? ` (sim ${similarity.toFixed(2)})` : ""
        }`
      : "Unknown face detected";

  return {
    id: typeof raw.event_id === "string" ? raw.event_id : crypto.randomUUID(),
    cameraId: typeof raw.camera_id === "string" ? raw.camera_id : "unknown",
    type: "face-detected",
    level: status === "matched" ? "info" : "warning",
    message,
    timestamp: isoTimestamp,
    personId: typeof raw.person_id === "string" ? raw.person_id : null,
    meta: {
      status,
      personName,
      similarity,
      confidence: raw.confidence ?? null,
    },
  };
}

/**
 * Thin event-emitter wrapper around a native WebSocket that preserves the
 * `.on / .off / .emit / .connected / .id / .connect / .disconnect` surface
 * the rest of the app (originally written against socket.io-client) uses.
 *
 * `emit` is a no-op toward the server — the current backend WS is push-only
 * — but emitted events still fan out to local listeners so code like
 * `subscribe-camera` doesn't throw.
 */
class OrixSocketClient {
  public id: string | null = null;
  public connected = false;

  private ws: WebSocket | null = null;
  private readonly listeners: Map<string, Set<Handler>> = new Map();
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private manuallyClosed = false;

  constructor(private readonly url: string) {}

  connect(): void {
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN ||
        this.ws.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    this.manuallyClosed = false;
    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[socket] failed to construct WebSocket:", err);
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.connected = true;
      this.reconnectAttempts = 0;
      this.id = crypto.randomUUID();
      // eslint-disable-next-line no-console
      console.info("[socket] connected", this.id);
      this.dispatch("connect");
    };

    this.ws.onmessage = (ev: MessageEvent<string>) => {
      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        // eslint-disable-next-line no-console
        console.warn("[socket] non-JSON message ignored:", ev.data);
        return;
      }

      const alert = eventToAlert(parsed);
      if (alert) {
        this.dispatch("alert", alert);
        return;
      }

      // Generic pass-through: if the backend ever emits { event, ... }
      // use that as the channel name.
      if (typeof parsed.event === "string") {
        this.dispatch(parsed.event, parsed);
      }
    };

    this.ws.onclose = (ev) => {
      this.connected = false;
      this.id = null;
      // eslint-disable-next-line no-console
      console.warn("[socket] disconnected:", ev.reason || ev.code);
      this.dispatch("disconnect", ev.reason || String(ev.code));
      if (!this.manuallyClosed) this.scheduleReconnect();
    };

    this.ws.onerror = (ev) => {
      // eslint-disable-next-line no-console
      console.warn("[socket] connect_error:", ev);
      this.dispatch("connect_error", ev);
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    const delay = Math.min(
      RECONNECT_DELAY_MS * 2 ** this.reconnectAttempts,
      RECONNECT_DELAY_MAX_MS
    );
    this.reconnectAttempts += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  on(event: string, handler: Handler): void {
    let set = this.listeners.get(event);
    if (!set) {
      set = new Set();
      this.listeners.set(event, set);
    }
    set.add(handler);
  }

  off(event: string, handler: Handler): void {
    this.listeners.get(event)?.delete(handler);
  }

  /**
   * No-op toward the backend (the current WS is push-only) but fires any
   * local listeners so UI code can round-trip events in dev.
   */
  emit(event: string, ...args: unknown[]): void {
    this.dispatch(event, ...args);
  }

  disconnect(): void {
    this.manuallyClosed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
    this.connected = false;
    this.id = null;
  }

  private dispatch(event: string, ...args: unknown[]): void {
    const set = this.listeners.get(event);
    if (!set) return;
    for (const handler of set) {
      try {
        handler(...args);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error(`[socket] handler for "${event}" threw:`, err);
      }
    }
  }
}

// Singleton so every hook consumer shares the same connection.
let socket: OrixSocketClient | null = null;

/**
 * Lazily creates (or returns) the shared WebSocket client with automatic
 * reconnection enabled. Safe to call from anywhere in the app.
 */
export function getSocket(): OrixSocket {
  if (socket) return socket as unknown as OrixSocket;
  const wsUrl = buildWsUrl(SOCKET_URL, WS_PATH);
  socket = new OrixSocketClient(wsUrl);
  socket.connect();
  return socket as unknown as OrixSocket;
}

export function disconnectSocket(): void {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
}

// Re-export the hook from its new location for backward compatibility.
export { useSocket } from "../hooks/useSocket";
