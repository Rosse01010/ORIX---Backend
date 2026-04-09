import axios from "axios";
import { envString } from "../utils/helpers";
import { getStoredToken } from "../services/authService";

const API_BASE = envString("VITE_API_URL", "http://localhost:8000");

export interface ConfirmPayload {
  person_id: string;
  embedding: number[];
  angle_hint: string;
  quality_score: number;
  camera_id: string;
}

export interface ConfirmResponse {
  person_id: string;
  name: string;
  embeddings_total: number;
  message: string;
}

/**
 * Confirm that a detected face belongs to person_id.
 * This adds the embedding to the person's profile so the
 * same angle is recognised automatically in the future.
 */
export async function confirmCandidate(
  payload: ConfirmPayload
): Promise<ConfirmResponse> {
  const token = getStoredToken();
  const { data } = await axios.post<ConfirmResponse>(
    `${API_BASE}/api/candidates/confirm`,
    payload,
    {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      timeout: 8000,
    }
  );
  return data;
}
