/**
 * Formats an ISO timestamp as HH:mm:ss for alert feeds and logs.
 */
export function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "--:--:--";
  return d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/**
 * Generates a short unique id. Not cryptographically strong — used for
 * local-only Alert objects that don't yet have a backend id.
 */
export function shortId(): string {
  return Math.random().toString(36).slice(2, 10);
}
