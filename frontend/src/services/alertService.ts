import { toast } from "react-toastify";
import type { Alert, AlertLevel } from "../types/Alert";
import { shortId } from "../utils/format";

/**
 * Maps an alert severity to the right toast variant.
 */
function toastForLevel(level: AlertLevel, message: string): void {
  switch (level) {
    case "critical":
      toast.error(message);
      break;
    case "warning":
      toast.warn(message);
      break;
    case "info":
    default:
      toast.info(message);
      break;
  }
}

/**
 * Build a fully-formed Alert from a partial payload (used when the UI
 * itself generates an alert, e.g. local face detection).
 */
export function buildLocalAlert(
  partial: Omit<Alert, "id" | "timestamp">
): Alert {
  return {
    ...partial,
    id: shortId(),
    timestamp: new Date().toISOString(),
  };
}

/**
 * Push an alert to the UI (toast + whatever the caller stores).
 * Kept separate from the store so services don't depend on state slices.
 */
export function notifyAlert(alert: Alert, cameraName?: string): void {
  const prefix = cameraName ? `[${cameraName}] ` : "";
  toastForLevel(alert.level, `${prefix}${alert.message}`);
}
