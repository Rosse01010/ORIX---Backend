import { create } from "zustand";
import type { Alert } from "../types/Alert";

const MAX_ALERTS = 50;

interface AlertStore {
  alerts: Alert[];
  /** Add a new alert to the top of the feed (capped at MAX_ALERTS). */
  push: (alert: Alert) => void;
  /** Acknowledge (dismiss) a single alert. */
  ack: (id: string) => void;
  /** Acknowledge every alert currently in the store. */
  clear: () => void;
  /** Selector helper: all alerts for a single camera. */
  byCamera: (cameraId: string) => Alert[];
}

export const useAlertStore = create<AlertStore>((set, get) => ({
  alerts: [],

  push: (alert) =>
    set((state) => ({
      alerts: [alert, ...state.alerts].slice(0, MAX_ALERTS),
    })),

  ack: (id) =>
    set((state) => ({
      alerts: state.alerts.filter((a) => a.id !== id),
    })),

  clear: () => set({ alerts: [] }),

  byCamera: (cameraId) => get().alerts.filter((a) => a.cameraId === cameraId),
}));
