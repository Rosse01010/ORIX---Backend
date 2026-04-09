import type { Camera } from "../types/Camera";
import { sleep } from "../utils/helpers";

/**
 * Local-only camera registry.
 *
 * The ORIX backend does not currently expose a `/cameras` endpoint —
 * cameras are implicit (identified by `camera_id` in detection events).
 * Until the backend grows one we keep a static list here so the UI's
 * grid, routing and alert-to-camera name mapping keep working.
 *
 * Stream URLs use placeholder mp4s so the <video> element has something
 * to attach to; replace them with real RTSP/HLS proxy URLs when the
 * backend starts serving media.
 */
const MOCK_CAMERAS: Camera[] = [
  {
    id: "cam-00",
    name: "Local Webcam",
    location: "This device",
    // `webcam://` is a custom scheme handled in CameraFeed.tsx — it
    // triggers navigator.mediaDevices.getUserMedia() instead of loading
    // a remote video URL. Handy for testing face-api.js detection.
    streamUrl: "webcam://default",
    status: "online",
  },
  {
    id: "cam-01",
    name: "Parking Lot",
    location: "Exterior — North",
    streamUrl:
      "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
    status: "online",
  },
  {
    id: "cam-02",
    name: "Warehouse",
    location: "Building B",
    streamUrl:
      "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
    status: "online",
  },
  {
    id: "cam-03",
    name: "Server Room",
    location: "Floor 2",
    streamUrl:
      "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
    status: "online",
  },
];

/** GET all cameras the user can see. */
export async function fetchCameras(_token: string | null): Promise<Camera[]> {
  void _token;
  await sleep(150);
  return [...MOCK_CAMERAS];
}

/** GET a single camera by id. */
export async function fetchCameraById(
  id: string,
  _token: string | null
): Promise<Camera | null> {
  void _token;
  return MOCK_CAMERAS.find((c) => c.id === id) ?? null;
}

/** PATCH — updates a single camera in-memory (no persistence). */
export async function updateCamera(
  id: string,
  patch: Partial<Camera>,
  _token: string | null
): Promise<Camera> {
  void _token;
  const idx = MOCK_CAMERAS.findIndex((c) => c.id === id);
  if (idx === -1) throw new Error(`Camera ${id} not found`);
  MOCK_CAMERAS[idx] = { ...MOCK_CAMERAS[idx], ...patch };
  return MOCK_CAMERAS[idx];
}
