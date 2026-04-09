import {
  fetchCameras as fetchCamerasRequest,
  fetchCameraById as fetchCameraByIdRequest,
  updateCamera as updateCameraRequest,
} from "../api/cameraApi";
import type { Camera } from "../types/Camera";
import { getStoredToken } from "./authService";

/**
 * Business-logic wrapper around cameraApi. Pages and stores should call
 * this instead of the raw API so we can swap transport later (REST, gRPC,
 * WebRTC signaling) in exactly one place.
 */
export async function loadCameras(): Promise<Camera[]> {
  return fetchCamerasRequest(getStoredToken());
}

export async function loadCameraById(id: string): Promise<Camera | null> {
  return fetchCameraByIdRequest(id, getStoredToken());
}

export async function patchCamera(
  id: string,
  patch: Partial<Camera>
): Promise<Camera> {
  return updateCameraRequest(id, patch, getStoredToken());
}
