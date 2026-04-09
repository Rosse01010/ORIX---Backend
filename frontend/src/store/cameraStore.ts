import { create } from "zustand";
import type { Camera, CameraStatus } from "../types/Camera";
import { loadCameras } from "../services/cameraService";

interface CameraStore {
  cameras: Camera[];
  loading: boolean;
  error: string | null;

  /** Fetch all cameras from the backend and replace the list. */
  fetchAll: () => Promise<void>;

  /** Patch a single camera's status in-place (used by socket events). */
  setStatus: (id: string, status: CameraStatus) => void;

  /** Remove all cameras (e.g. on logout). */
  clear: () => void;
}

export const useCameraStore = create<CameraStore>((set) => ({
  cameras: [],
  loading: false,
  error: null,

  fetchAll: async () => {
    set({ loading: true, error: null });
    try {
      const cameras = await loadCameras();
      set({ cameras, loading: false });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to load cameras";
      set({ error: msg, loading: false });
    }
  },

  setStatus: (id, status) =>
    set((state) => ({
      cameras: state.cameras.map((c) =>
        c.id === id ? { ...c, status } : c
      ),
    })),

  clear: () => set({ cameras: [], error: null }),
}));
