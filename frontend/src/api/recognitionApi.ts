import axios from "axios";
import { envString } from "../utils/helpers";

/**
 * Thin client for the ORIX backend's `/api/recognition/*` REST surface
 * (FastAPI — see backend2/app/routes/recognition.py).
 *
 * Only the endpoints that actually exist on the server are wrapped here.
 * The historical `/auth`, `/cameras`, `/users` wrappers in the sibling
 * api files are now purely client-side mocks because the backend does
 * not implement them.
 */

const API_BASE = envString("VITE_API_URL", "http://localhost:8000");
const RECOGNITION_BASE = `${API_BASE}/api/recognition`;

export interface HealthResponse {
  status: string;
  websocket_clients: number;
}

export interface DetectionEvent {
  id: string;
  camera_id: string;
  status: "matched" | "unknown" | string;
  similarity: number | null;
  detected_at: string;
  person_name: string | null;
}

/**
 * Consented social-profile links stored alongside a person's face
 * embedding. These are *provided by the person at enrollment time*, not
 * discovered automatically from the face.
 */
export interface SocialLinks {
  linkedin_url?: string | null;
  instagram_handle?: string | null;
  twitter_handle?: string | null;
  notes?: string | null;
}

export interface PersonCreatePayload extends SocialLinks {
  name: string;
  embedding: number[];
}

export interface PersonCreateResponse extends SocialLinks {
  person_id: string;
  name: string;
}

export interface PersonDetails extends SocialLinks {
  id: string;
  name: string;
  created_at: string;
}

/** GET /api/recognition/health */
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await axios.get<HealthResponse>(
    `${RECOGNITION_BASE}/health`,
    { timeout: 5000 }
  );
  return data;
}

/** GET /api/recognition/events — last N detection events. */
export async function listEvents(
  params: { limit?: number; status?: "matched" | "unknown" } = {}
): Promise<DetectionEvent[]> {
  const { data } = await axios.get<DetectionEvent[]>(
    `${RECOGNITION_BASE}/events`,
    { params, timeout: 5000 }
  );
  return data;
}

/** POST /api/recognition/persons/enroll — register a person with browser embedding. */
export async function registerPerson(
  payload: PersonCreatePayload
): Promise<PersonCreateResponse> {
  const { data } = await axios.post<PersonCreateResponse>(
    `${RECOGNITION_BASE}/persons/enroll`,
    payload,
    { timeout: 10_000 }
  );
  return data;
}

/** GET /api/recognition/persons/:id */
export async function getPerson(personId: string): Promise<PersonDetails> {
  const { data } = await axios.get<PersonDetails>(
    `${RECOGNITION_BASE}/persons/${personId}`,
    { timeout: 5000 }
  );
  return data;
}

/** POST /api/recognition/index/rebuild — admin-only: rebuild IVFFLAT index. */
export async function rebuildVectorIndex(): Promise<{ message: string }> {
  const { data } = await axios.post<{ message: string }>(
    `${RECOGNITION_BASE}/index/rebuild`,
    undefined,
    { timeout: 30_000 }
  );
  return data;
}
