import { create } from "zustand";
import type { FaceCandidate } from "../types/Candidate";

interface CandidateStore {
  /** Pending faces waiting for operator confirmation */
  pending: FaceCandidate[];

  /** Add new candidates (from socket detection-result event) */
  addCandidates: (candidates: FaceCandidate[]) => void;

  /** Remove a face from the panel (confirmed or dismissed) */
  remove: (faceIndex: number, cameraId: string) => void;

  /** Clear all pending candidates */
  clear: () => void;
}

const MAX_PENDING = 20;

export const useCandidateStore = create<CandidateStore>((set) => ({
  pending: [],

  addCandidates: (incoming) =>
    set((state) => {
      // Deduplicate by cameraId+face_index, keep newest
      const filtered = state.pending.filter(
        (p) =>
          !incoming.some(
            (c) => c.cameraId === p.cameraId && c.face_index === p.face_index
          )
      );
      return {
        pending: [...incoming, ...filtered].slice(0, MAX_PENDING),
      };
    }),

  remove: (faceIndex, cameraId) =>
    set((state) => ({
      pending: state.pending.filter(
        (p) => !(p.face_index === faceIndex && p.cameraId === cameraId)
      ),
    })),

  clear: () => set({ pending: [] }),
}));
