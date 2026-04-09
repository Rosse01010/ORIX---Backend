/**
 * Tiny type-safe localStorage wrapper. Silently no-ops when localStorage is
 * unavailable (e.g. private mode, SSR).
 */
export const storage = {
  get<T>(key: string, fallback: T): T {
    try {
      const raw = localStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : fallback;
    } catch {
      return fallback;
    }
  },

  set<T>(key: string, value: T): void {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* ignore quota / disabled storage */
    }
  },

  remove(key: string): void {
    try {
      localStorage.removeItem(key);
    } catch {
      /* ignore */
    }
  },
};

/** localStorage keys used across the app. Keep them centralized to avoid typos. */
export const STORAGE_KEYS = {
  AUTH_USER: "orix:user",
  AUTH_TOKEN: "orix:token",
} as const;
