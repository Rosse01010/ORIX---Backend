/**
 * Alerts raised either locally (face detection) or emitted by the backend
 * via WebSocket. The UI consumes them to show toasts and banners.
 */
export type AlertLevel = "info" | "warning" | "critical";

export type AlertType =
  | "face-detected"
  | "motion"
  | "intrusion"
  | "camera-offline"
  | "system";

export interface Alert {
  id: string;
  cameraId: string;
  type: AlertType;
  level: AlertLevel;
  message: string;
  /** ISO timestamp */
  timestamp: string;
  /**
   * When the backend matched the face to a registered person, this is
   * the person's UUID. The frontend uses it to fetch the person's
   * consented social-profile links via GET /api/recognition/persons/:id.
   */
  personId?: string | null;
  /** Optional metadata (number of faces detected, confidence, etc.) */
  meta?: Record<string, unknown>;
}
