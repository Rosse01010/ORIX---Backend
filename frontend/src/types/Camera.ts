/**
 * Representation of a physical or simulated camera in the system.
 */
export type CameraStatus = "online" | "offline" | "error";

export interface Camera {
  id: string;
  name: string;
  location: string;
  streamUrl: string;
  status: CameraStatus;
}
